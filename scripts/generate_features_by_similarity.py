import torch
import pandas as pd
import numpy as np
from transformers import EsmTokenizer, EsmModel
from tqdm import tqdm
from xgboost import XGBRegressor 

CONFIG = {
    "reference_csv": "final_predict/dock.csv",
    "prediction_csv": "fnal_predict/input_for_prediction.csv",
    "output_csv": "final_predict/prediction_with_features.csv",
    "esm_model_name": "local_models/esm2_t33_650M_UR50D/",
    "batch_size": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def get_esm_model_and_tokenizer(model_name):
    print(f"正在从 {model_name} 加载模型...")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.to(CONFIG["device"])
    model.eval()
    return model, tokenizer

def generate_embeddings(sequences, model, tokenizer, batch_size):
    all_embeddings = []
    print(f"正在为 {len(sequences)} 条序列生成嵌入向量...")
    
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_seqs = sequences[i:i + batch_size]
        
        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=1024
        ).to(CONFIG["device"])

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu())
            
    return torch.cat(all_embeddings, dim=0).numpy()



def main():
    print("加载数据...")
    df_ref = pd.read_csv(CONFIG["reference_csv"])
    df_pred = pd.read_csv(CONFIG["prediction_csv"])

    df_ref.dropna(subset=['sequence', 'structure_features'], inplace=True)
    df_pred.dropna(subset=['sequence'], inplace=True)
    
    model, tokenizer = get_esm_model_and_tokenizer(CONFIG["esm_model_name"])

    print("\n--- 步骤1: 准备训练数据 (你的6000条参考序列) ---")
    print("为参考数据集生成嵌入向量作为特征 (X_train)...")
    X_train = generate_embeddings(df_ref['sequence'].tolist(), model, tokenizer, CONFIG["batch_size"])
    y_train = df_ref['structure_features'].values
    print(f"训练数据准备完毕: X_train 形状 {X_train.shape}, y_train 形状 {y_train.shape}")

    print("\n--- 步骤2: 训练 XGBoost 回归模型 ---")
    print("模型正在学习从序列指纹到结构特征的映射规律...")
    regressor = XGBRegressor(n_estimators=200, learning_rate=0.05, n_jobs=-1, random_state=42)
    regressor.fit(X_train, y_train)
    print("模型训练完成！")

    print("\n--- 步骤3: 为你的30000条新序列生成嵌入并预测其特征 ---")
    X_pred = generate_embeddings(df_pred['sequence'].tolist(), model, tokenizer, CONFIG["batch_size"])
    print(f"待预测数据准备完毕: X_pred 形状 {X_pred.shape}")

    print("开始使用训练好的模型进行预测...")
    generated_features = regressor.predict(X_pred)
    print("预测完成。")

    df_pred['structure_features'] = generated_features
    
    unique_generated = df_pred['structure_features'].nunique()
    print(f"\n--- 最终结果检查 ---")
    print(f"为 {len(df_pred)} 条序列生成了 {unique_generated} 个独特的 structure_features 值。")
    print(f"--- 检查结束 ---\n")

    df_pred.to_csv(CONFIG["output_csv"], index=False)
    print(f"处理完成！结果已保存至: {CONFIG['output_csv']}")

if __name__ == "__main__":
    main()
