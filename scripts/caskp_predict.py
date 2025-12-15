import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import EsmTokenizer
from predict_caskp_model import FullKcatPredictor 

MODEL_PATH = "models/caskp_final_model.pt"
ESM_MODEL_NAME = "local_models/esm2_t33_650M_UR50D/"
INPUT_CSV = "final_predict/prediction_with_features.csv"
OUTPUT_CSV = "final_predict/predicted_kcat_output.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class InferenceDataset(Dataset):
    def __init__(self, tokenizer, data_df, struct_dim):
        self.data = data_df
        self.tokenizer = tokenizer
        self.struct_dim = struct_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokenized = self.tokenizer(
            row['sequence'],
            return_tensors='pt',
            padding="max_length",
            max_length=512,
            truncation=True
        )
        struct_list = [float(x) for x in str(row['structure_features']).split(',')]
        if len(struct_list) > self.struct_dim:
            struct_list = struct_list[:self.struct_dim]
        else:
            struct_list += [0.0] * (self.struct_dim - len(struct_list))
        struct_tensor = torch.tensor(struct_list, dtype=torch.float)
        return {
            "input_ids": tokenized['input_ids'].squeeze(0),
            "attention_mask": tokenized['attention_mask'].squeeze(0),
            "struct_features": struct_tensor
        }


def load_model(model_path, esm_model_name, struct_dim, device):
    model = FullKcatPredictor(
        esm_model_name=esm_model_name,
        struct_dim=struct_dim,
        d_model=256,
        d_multiscale=128,
        num_heads=8,
        use_amsff=True
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def main():
    print("加载输入数据...")
    df = pd.read_csv(INPUT_CSV).dropna(subset=['sequence', 'structure_features']).reset_index(drop=True)

    if df.empty:
        raise ValueError("输入数据为空或缺失必要列 sequence / structure_features。")

    first_struct = df['structure_features'].dropna().iloc[0]
    struct_dim = len(str(first_struct).split(','))
    print(f"自动推断结构特征维度：{struct_dim}")

    tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME)
    dataset = InferenceDataset(tokenizer, df, struct_dim)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    print("加载模型...")
    model = load_model(MODEL_PATH, ESM_MODEL_NAME, struct_dim, DEVICE)

    preds_log_kcat = []

    print("开始预测...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            struct_features = batch['struct_features'].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                struct_features=struct_features
            ).squeeze(-1)

            preds_log_kcat.extend(outputs.cpu().numpy().tolist())

    preds_kcat = [10 ** x for x in preds_log_kcat]
    df['predicted_log10_kcat'] = preds_log_kcat
    df['predicted_kcat'] = preds_kcat

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"预测完成，结果已保存至：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
