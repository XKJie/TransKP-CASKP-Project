import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import EsmTokenizer
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import csv
import traceback
import copy
import os

from model_transkp import DeepFusionKcatPredictor

class PretrainConfig:
    PRETRAIN_DATA_PATH = "train_transkp/transkp_train_data.csv"
    OUTPUT_MODEL_PATH = "models/deep_fusion_kcat_pretrained.pt"
    FOLD_MODELS_DIR = "train_transkp/fold_models"
    LOG_FILE_PATH = "train_transkp/pretrain_log.csv"
    PREDICTIONS_FILE_PATH = "train_transkp/best_epoch_predictions.csv"
    ESM_MODEL_NAME = "local_models/esm2_t33_650M_UR50D/"
    K_FOLDS = 10
    COLUMN_MAPPING = {
        'sequence': 'Sequence',
        'kcat': 'kcat(s^-1)',
        'smiles': 'Smiles',
        'fold': 'fold'
    }
    EPOCHS = 20 
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RUN_FINAL_TRAINING_AFTER_CV = True

    # --- Hyperparameters for DeepFusionKcatPredictor ---
    GNN_INPUT_DIM = 18  # C,O,N,S,F,Cl,Br,I,P,Co,Fe,Cu,Zn,Mg + other
    GNN_HIDDEN_DIM = 256
    GNN_HEADS = 4
    D_MODEL = 256
    NUM_FUSION_BLOCKS = 3
    NUM_ATTN_HEADS = 8
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

class PretrainDataset(Dataset):
    def __init__(self, tokenizer, dataframe, config):
        self.tokenizer = tokenizer
        self.data = dataframe 
        self.kcat_col = config.COLUMN_MAPPING['kcat']
        self.seq_col = config.COLUMN_MAPPING['sequence']
        self.smiles_col = config.COLUMN_MAPPING['smiles']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        original_index = self.data.index[idx]
        
        tokenized = self.tokenizer(row[self.seq_col], return_tensors='pt', padding="max_length", max_length=512, truncation=True)
        kcat_val = torch.tensor(np.log10(row[self.kcat_col] + 1e-9), dtype=torch.float)
        smiles = row[self.smiles_col]

        return {
            "input_ids": tokenized['input_ids'].squeeze(0),
            "attention_mask": tokenized['attention_mask'].squeeze(0),
            "smiles": smiles,
            "kcat": kcat_val,
            "index": original_index
        }

def custom_collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    kcat = torch.stack([item['kcat'] for item in batch])
    smiles_list = [item['smiles'] for item in batch]
    indices = [item['index'] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "kcat": kcat,
        "smiles_list": smiles_list,
        "indices": indices
    }

def safe_extend(array_like, target_list):
    if isinstance(array_like, torch.Tensor):
        np_array = array_like.cpu().detach().numpy()
    else:
        np_array = np.array(array_like)
    np_array = np.atleast_1d(np_array)
    target_list.extend(np_array.tolist())

def train_and_validate_one_fold(fold_id, train_df, val_df, config, tokenizer, log_writer):
    print(f"\n[Fold {fold_id+1}] 开始训练...")
    device = config.DEVICE
    
    model = DeepFusionKcatPredictor(
        esm_model_name=config.ESM_MODEL_NAME,
        gnn_input_dim=config.GNN_INPUT_DIM,
        gnn_hidden_dim=config.GNN_HIDDEN_DIM,
        gnn_heads=config.GNN_HEADS,
        d_model=config.D_MODEL,
        num_fusion_blocks=config.NUM_FUSION_BLOCKS,
        num_attn_heads=config.NUM_ATTN_HEADS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    train_dataset = PretrainDataset(tokenizer, train_df, config)
    val_dataset = PretrainDataset(tokenizer, val_df, config)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, collate_fn=custom_collate_fn)

    best_val_mse = float('inf')
    best_model_state = None
    best_fold_preds, best_fold_reals, best_fold_indices = None, None, None

    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Fold {fold_id+1} Epoch {epoch+1}/{config.EPOCHS} [Train]")
        for batch in pbar:
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                preds = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    smiles_list=batch['smiles_list']
                ).view(-1)
                loss = criterion(preds, batch['kcat'].to(device))
            
            if device == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix({"Batch MSE": f"{loss.item():.4f}"})
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        all_preds, all_reals, all_indices = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    preds = model(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        smiles_list=batch['smiles_list']
                    ).view(-1)
                safe_extend(preds, all_preds)
                safe_extend(batch['kcat'], all_reals)
                safe_extend(batch['indices'], all_indices)
        
        current_val_mse = mean_squared_error(all_reals, all_preds)
        current_val_rmse = np.sqrt(current_val_mse)
        try:
            current_val_pcc, _ = pearsonr(all_reals, all_preds)
        except ValueError:
            current_val_pcc = 0.0
        try:
            current_val_scc, _ = spearmanr(all_reals, all_preds)
        except ValueError:
            current_val_scc = 0.0

        print(f"[Fold {fold_id+1} Epoch {epoch+1}] Train MSE: {avg_train_loss:.4f}, Val MSE: {current_val_mse:.4f}, Val RMSE: {current_val_rmse:.4f}, Val PCC: {current_val_pcc:.4f}, Val SCC: {current_val_scc:.4f}")

        log_writer.writerow([
            fold_id + 1, epoch + 1, f"{avg_train_loss:.4f}", f"{current_val_mse:.4f}",
            f"{current_val_rmse:.4f}", f"{current_val_pcc:.4f}", f"{current_val_scc:.4f}"
        ])

        if current_val_mse < best_val_mse:
            best_val_mse = current_val_mse
            best_model_state = copy.deepcopy(model.state_dict())
            best_fold_preds = all_preds
            best_fold_reals = all_reals
            best_fold_indices = all_indices

            if config.FOLD_MODELS_DIR:
                os.makedirs(config.FOLD_MODELS_DIR, exist_ok=True)
                save_path = os.path.join(config.FOLD_MODELS_DIR, f"fold_{fold_id+1}_best_model.pt")
                print(f"  -> New best validation MSE. Saving model to {save_path}")
                torch.save(best_model_state, save_path)

    print(f"[Fold {fold_id+1}] 最佳验证 MSE: {best_val_mse:.4f}")
    return best_val_mse, best_fold_preds, best_fold_reals, best_fold_indices

def train_final_model(config, tokenizer, df):
    print("\n开始在全部数据上训练最终模型...")
    device = config.DEVICE
    
    model = DeepFusionKcatPredictor(
        esm_model_name=config.ESM_MODEL_NAME,
        gnn_input_dim=config.GNN_INPUT_DIM,
        gnn_hidden_dim=config.GNN_HIDDEN_DIM,
        gnn_heads=config.GNN_HEADS,
        d_model=config.D_MODEL,
        num_fusion_blocks=config.NUM_FUSION_BLOCKS,
        num_attn_heads=config.NUM_ATTN_HEADS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    full_dataset = PretrainDataset(tokenizer, df.reset_index(drop=True), config)
    train_loader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

    model.train()
    for epoch in range(config.EPOCHS):
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Final Training Epoch {epoch+1}/{config.EPOCHS}")
        for batch in pbar:
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                preds = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    smiles_list=batch['smiles_list']
                ).view(-1)
                loss = criterion(preds, batch['kcat'].to(device))
            
            if device == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({"Batch MSE": f"{loss.item():.4f}"})
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"[Final Training Epoch {epoch+1}] Train MSE: {avg_train_loss:.4f}")
    
    print(f"最终模型训练完成。正在保存到 {config.OUTPUT_MODEL_PATH}")
    torch.save(model.state_dict(), config.OUTPUT_MODEL_PATH)
    print("模型已保存。")

def main():
    try:
        config = PretrainConfig()
        output_dir = os.path.dirname(config.OUTPUT_MODEL_PATH)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        tokenizer = EsmTokenizer.from_pretrained(config.ESM_MODEL_NAME)
        df = pd.read_csv(config.PRETRAIN_DATA_PATH)
        required_cols = list(config.COLUMN_MAPPING.values())
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV文件必须包含列: {required_cols}")

        with open(config.LOG_FILE_PATH, 'w', newline='') as log_file, \
             open(config.PREDICTIONS_FILE_PATH, 'w', newline='') as preds_file:
            
            log_writer = csv.writer(log_file)
            log_writer.writerow(['Fold', 'Epoch', 'Train_MSE', 'Validation_MSE', 'Validation_RMSE', 'Validation_PCC', 'Validation_SCC'])

            preds_writer = csv.writer(preds_file)
            preds_writer.writerow(['original_index', 'fold', 'log10_real_kcat', 'log10_predicted_kcat'])

            fold_results_mse = []
            all_preds_total, all_reals_total = [], []
            for i in range(config.K_FOLDS):
                train_df = df[df[config.COLUMN_MAPPING['fold']] != i]
                val_df = df[df[config.COLUMN_MAPPING['fold']] == i]
                
                mse, preds, reals, indices = train_and_validate_one_fold(i, train_df, val_df, config, tokenizer, log_writer)
                
                fold_results_mse.append(mse)
                
                if preds is not None:
                    all_preds_total.extend(preds)
                    all_reals_total.extend(reals)
                    for j in range(len(preds)):
                        preds_writer.writerow([indices[j], i + 1, reals[j], preds[j]])

            avg_mse = np.mean(fold_results_mse)
            std_mse = np.std(fold_results_mse)
            print("\nK折交叉验证汇总:")
            print(f"各折最佳MSE: {[f'{x:.4f}' for x in fold_results_mse]}")
            print(f"平均MSE: {avg_mse:.4f} ± {std_mse:.4f}")

            if all_reals_total:
                overall_rmse = np.sqrt(mean_squared_error(all_reals_total, all_preds_total))
                overall_pcc, _ = pearsonr(all_reals_total, all_preds_total)
                overall_scc, _ = spearmanr(all_reals_total, all_preds_total)

                print(f"\n整体性能指标 (基于所有折最佳模型的预测汇总):")
                print(f"  - 整体 RMSE: {overall_rmse:.4f}")
                print(f"  - 整体 PCC:  {overall_pcc:.4f}")
                print(f"  - 整体 SCC:  {overall_scc:.4f}")
                
                log_writer.writerow(['---', '---', '---', '---', '---', '---', '---'])
                log_writer.writerow(['Overall Metrics', 'RMSE', 'PCC', 'SCC', '', '', ''])
                log_writer.writerow(['(pool and calculate)', f"{overall_rmse:.4f}", f"{overall_pcc:.4f}", f"{overall_scc:.4f}", '', '', ''])

        if config.RUN_FINAL_TRAINING_AFTER_CV:
            train_final_model(config, tokenizer, df)

    except Exception as e:
        print("\n训练过程中发生错误!")
        traceback.print_exc()

if __name__ == "__main__":
    main()
