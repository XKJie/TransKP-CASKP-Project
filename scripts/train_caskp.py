import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from transformers import EsmTokenizer
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import copy
import traceback
import csv
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from model_caskp import FullKcatPredictor

class TrainConfig:
    FINETUNE_DATA_PATH = "train_caskp/caskp_train_data.csv"
    PRETRAINED_MODEL_PATH = "models/deep_fusion_kcat_pretrained.pt"
    
    K_FOLDS = 5  
    RUN_FINAL_TRAINING_AFTER_CV = True
    FOLD_MODELS_DIR = "train_caskp/final_models_cv"
    FINAL_MODEL_SAVE_PATH = "models/caskp_final_model.pt"

    RESULTS_DIR = "train_caskp/results_finetune_cv"
    LOG_FILE_NAME = "train_caskp/caskp_cv_log.csv"
    PREDICTIONS_FILE_NAME = "train_caskp/caskp_cv_predictions.csv"

    ESM_MODEL_NAME = "local_models/esm2_t33_650M_UR50D/"
    
    USE_AMSFF = True 
    USE_PRETRAIN = True

    STRUCT_DIM = 1
    D_MODEL = 256
    D_MULTISCALE = 128
    NUM_HEADS = 8
    
    EPOCHS = 20
    BATCH_SIZE = 8
    LR_ESM = 1e-5
    LR_FUSION_PREDICTOR = 5e-5
    WEIGHT_DECAY = 0.01
    EARLY_STOPPING_PATIENCE = 20
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

class KcatDataset(Dataset):
    def __init__(self, tokenizer, data_df, struct_dim):
        self.data = data_df
        self.tokenizer = tokenizer
        self.struct_dim = struct_dim
    
    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        original_index = self.data.index[idx]
        
        tokenized = self.tokenizer(row['sequence'], return_tensors='pt', padding="max_length", max_length=512, truncation=True)
        
        struct_str_list = str(row['structure_features']).split(',')
        struct_features_list = [float(x) for x in struct_str_list]
        
        if len(struct_features_list) > self.struct_dim:
            struct_features_list = struct_features_list[:self.struct_dim]
        else:
            struct_features_list += [0.0] * (self.struct_dim - len(struct_features_list))

        struct_features = torch.tensor(struct_features_list, dtype=torch.float)
        kcat_val = torch.tensor(row['kcat'], dtype=torch.float)

        return {
            "input_ids": tokenized['input_ids'].squeeze(0), 
            "attention_mask": tokenized['attention_mask'].squeeze(0), 
            "struct_features": struct_features, 
            "kcat": kcat_val,
            "index": original_index
        }

def custom_collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    kcat = torch.stack([item['kcat'] for item in batch])
    struct_features = torch.stack([item['struct_features'] for item in batch])
    indices = [item['index'] for item in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "kcat": kcat,
        "struct_features": struct_features,
        "indices": indices
    }

def initialize_model(config, device):
    model = FullKcatPredictor(
        esm_model_name=config.ESM_MODEL_NAME, struct_dim=config.STRUCT_DIM,
        d_model=config.D_MODEL, d_multiscale=config.D_MULTISCALE,
        num_heads=config.NUM_HEADS, use_amsff=config.USE_AMSFF
    ).to(device)

    if config.USE_PRETRAIN and os.path.exists(config.PRETRAINED_MODEL_PATH):
        print(f"Loading pretrained weights from {config.PRETRAINED_MODEL_PATH}...")
        pretrained_dict = torch.load(config.PRETRAINED_MODEL_PATH, map_location=device)
        model_dict = model.state_dict()
        transfer_weights = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape and (k.startswith('esm_model.') or k.startswith('protein_projection.'))}
        model_dict.update(transfer_weights)
        model.load_state_dict(model_dict)
        print(f"Successfully transferred {len(transfer_weights)} weight layers.")
    return model

def train_and_validate_one_fold(fold_id, train_dataset, val_dataset, config, device, log_writer):
    print(f"\n[Fold {fold_id+1}/{config.K_FOLDS}] Starting training...")
    
    model = initialize_model(config, device)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)

    optimizer = torch.optim.AdamW([
        {'params': model.esm_model.parameters(), 'lr': config.LR_ESM},
        {'params': [p for n, p in model.named_parameters() if 'esm_model' not in n], 'lr': config.LR_FUSION_PREDICTOR}
    ], weight_decay=config.WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    best_fold_preds, best_fold_reals, best_fold_indices = [], [], []

    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Fold {fold_id+1} Epoch {epoch+1}/{config.EPOCHS}")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                preds = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    struct_features=batch['struct_features'].to(device)
                ).squeeze(-1)
                loss = criterion(preds, batch['kcat'].to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
            pbar.set_postfix({"Batch MSE": f"{loss.item():.4f}"})
        
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        current_preds, current_reals, current_indices = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    preds = model(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        struct_features=batch['struct_features'].to(device)
                    ).squeeze(-1)
                current_preds.extend(preds.cpu().numpy().tolist())
                current_reals.extend(batch['kcat'].cpu().numpy().tolist())
                current_indices.extend(batch['indices'])
        
        avg_val_loss = mean_squared_error(current_reals, current_preds)
        val_rmse = np.sqrt(avg_val_loss)
        try:
            val_pcc, _ = pearsonr(current_reals, current_preds)
        except ValueError:
            val_pcc = 0.0
        try:
            val_scc, _ = spearmanr(current_reals, current_preds)
        except ValueError:
            val_scc = 0.0

        print(f"[Fold {fold_id+1} Epoch {epoch+1}] Train MSE: {avg_train_loss:.4f}, Val MSE: {avg_val_loss:.4f}, Val PCC: {val_pcc:.4f}")
        log_writer.writerow([fold_id + 1, epoch + 1, f"{avg_train_loss:.4f}", f"{avg_val_loss:.4f}", f"{val_rmse:.4f}", f"{val_pcc:.4f}", f"{val_scc:.4f}"])

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            best_fold_preds, best_fold_reals, best_fold_indices = current_preds, current_reals, current_indices
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break
            
    if best_model_state:
        os.makedirs(config.FOLD_MODELS_DIR, exist_ok=True)
        save_path = os.path.join(config.FOLD_MODELS_DIR, f"fold_{fold_id+1}_best_model.pt")
        torch.save(best_model_state, save_path)
        print(f"Fold {fold_id+1} best model saved to {save_path} with Val MSE: {best_val_loss:.4f}")

    return best_val_loss, best_fold_preds, best_fold_reals, best_fold_indices


def train_final_model(config, device, full_dataset):
    print("\nTraining final model on all data...")
    model = initialize_model(config, device)
    train_loader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
    
    optimizer = torch.optim.AdamW([
        {'params': model.esm_model.parameters(), 'lr': config.LR_ESM},
        {'params': [p for n, p in model.named_parameters() if 'esm_model' not in n], 'lr': config.LR_FUSION_PREDICTOR}
    ], weight_decay=config.WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    for epoch in range(config.EPOCHS // 2):  # Train final model for fewer epochs
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Final Training Epoch {epoch + 1}/{config.EPOCHS // 2}")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                preds = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    struct_features=batch['struct_features'].to(device)
                ).squeeze(-1)
                loss = criterion(preds, batch['kcat'].to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
            pbar.set_postfix({"Batch MSE": f"{loss.item():.4f}"})
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"[Final Training Epoch {epoch+1}] Train MSE: {avg_train_loss:.4f}")

    torch.save(model.state_dict(), config.FINAL_MODEL_SAVE_PATH)
    print(f"Final model saved to {config.FINAL_MODEL_SAVE_PATH}")


def main():
    try:
        config = TrainConfig()
        device = config.DEVICE
        
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        log_path = os.path.join(config.RESULTS_DIR, config.LOG_FILE_NAME)
        preds_path = os.path.join(config.RESULTS_DIR, config.PREDICTIONS_FILE_NAME)

        tokenizer = EsmTokenizer.from_pretrained(config.ESM_MODEL_NAME)
        df = pd.read_csv(config.FINETUNE_DATA_PATH).dropna(subset=['sequence', 'structure_features', 'kcat']).reset_index(drop=True)

        first_valid_struct = df['structure_features'].dropna().iloc[0]
        config.STRUCT_DIM = len(str(first_valid_struct).split(','))
        print(f"Auto-inferred STRUCT_DIM: {config.STRUCT_DIM}")

        full_dataset = KcatDataset(tokenizer, df, config.STRUCT_DIM)
        kf = KFold(n_splits=config.K_FOLDS, shuffle=True, random_state=42)

        with open(log_path, 'w', newline='') as log_file, open(preds_path, 'w', newline='') as preds_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow(['Fold', 'Epoch', 'Train_MSE', 'Val_MSE', 'Val_RMSE', 'Val_PCC', 'Val_SCC'])
            
            preds_writer = csv.writer(preds_file)
            preds_writer.writerow(['original_index', 'fold', 'real_log10_kcat', 'predicted_log10_kcat', 'structure_features'])

            fold_results_mse = []
            all_preds_total, all_reals_total = [], []

            for fold_id, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
                train_subset = Subset(full_dataset, train_idx)
                val_subset = Subset(full_dataset, val_idx)
                
                mse, preds, reals, indices = train_and_validate_one_fold(fold_id, train_subset, val_subset, config, device, log_writer)

                fold_results_mse.append(mse)
                
                if preds:
                    all_preds_total.extend(preds)
                    all_reals_total.extend(reals)
                    for i in range(len(preds)):
                        structure_features = str(df['structure_features'].iloc[indices[i]]).replace("[", "").replace("]", "")
                        preds_writer.writerow([indices[i], fold_id + 1, reals[i], preds[i], structure_features])

            avg_mse = np.mean(fold_results_mse)
            std_mse = np.std(fold_results_mse)
            print("\nK-Fold Cross-Validation Summary:")
            print(f"Individual fold best MSEs: {[f'{x:.4f}' for x in fold_results_mse]}")
            print(f"Average MSE: {avg_mse:.4f} ± {std_mse:.4f}")

            if all_reals_total:
                overall_rmse = np.sqrt(mean_squared_error(all_reals_total, all_preds_total))
                overall_pcc, _ = pearsonr(all_reals_total, all_preds_total) if all_reals_total else (0.0, 0.0)
                overall_scc, _ = spearmanr(all_reals_total, all_preds_total) if all_reals_total else (0.0, 0.0)

                print("\nOverall Performance (pooled from all folds' best predictions):")
                print(f"  - Overall RMSE: {overall_rmse:.4f}")
                print(f"  - Overall PCC:  {overall_pcc:.4f}")
                print(f"  - Overall SCC:  {overall_scc:.4f}")

        if config.RUN_FINAL_TRAINING_AFTER_CV:
            train_final_model(config, device, full_dataset)

    except Exception:
        print("\nAn error occurred during training!")
        traceback.print_exc()

if __name__ == "__main__":
    main()
