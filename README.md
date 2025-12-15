# TransKP-CASKP-Project: Enzyme kcat Prediction

This project implements a comprehensive deep learning framework for predicting enzyme catalytic activity ($k_{cat}$). The framework consists of two main stages:

1.  **TransKP**: A multi-modal pre-training model that integrates protein sequences (via ESM-2) and substrate molecular graphs (via GNN).
2.  **CASKP**: A fine-tuning model that incorporates structural features (e.g., molecular docking scores) to enhance prediction accuracy.

## 📁 Project Structure

```text
.
├── final_predict/               # Data folder for the prediction/inference stage
│   ├── dock.csv                 # Reference data for feature generation
│   ├── input_for_prediction.csv # New sequences to be predicted
│   ├── prediction_with_features.csv # (Generated) Intermediate file with features
│   └── predicted_kcat_output.csv    # (Generated) Final kcat prediction results
│
├── local_models/                # (Download Required) Directory for ESM-2 model
│   └── esm2_t33_650M_UR50D/     # ESM-2 model files
│
├── models/                      # Directory for trained model checkpoints
│   ├── deep_fusion_kcat_pretrained.pt # Output of TransKP (Pre-training)
│   └── caskp_final_model.pt           # Output of CASKP (Fine-tuning)
│
├── scripts/                     # Source code directory
│   ├── model_transkp.py         # TransKP model definition
│   ├── train_transkp.py         # TransKP training script
│   ├── model_caskp.py           # CASKP model definition
│   ├── train_caskp.py           # CASKP training script
│   ├── predict_caskp_model.py   # Model definition for inference
│   ├── generate_features_by_similarity.py # Tool for generating structural features
│   └── caskp_predict.py         # Main inference script
│
├── train_caskp/                 # CASKP training data and logs
│   ├── caskp_train_data.csv     # Fine-tuning dataset
│   └── ... (logs)
│
├── train_transkp/               # TransKP training data and logs
│   ├── tanskp_train_data.csv    # Pre-training dataset
│   └── ... (logs)
│
└── requirements.txt             # Python dependencies