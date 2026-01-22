# TransKP-CASKP: A Deep Learning Framework for Enzyme kcat Prediction

This project implements a comprehensive deep learning framework for predicting enzyme catalytic activity ($k_{cat}$). The framework consists of two main stages:

1. **TransKP**: A multi-modal pre-training model that integrates protein sequences (via ESM-2) and substrate molecular graphs (via GNN).
2. **CASKP**: A fine-tuning model that incorporates structural features (e.g., molecular docking scores) to enhance prediction accuracy.

## Model Availability

Due to file size constraints, the pre-trained model weights are hosted on Zenodo:

https://doi.org/10.5281/zenodo.18342595

After downloading, place the model files in the `models/` directory.

## Web Server

An online prediction interface is available at:

https://huggingface.co/spaces/KangjieXu/TransKP

## Project Structure

```text
.
├── final_predict/                        # Data folder for the prediction/inference stage
│   ├── dock.csv                          # Reference data for feature generation
│   ├── input_for_prediction.csv          # New sequences to be predicted
│   ├── prediction_with_features.csv      # Intermediate file with generated features
│   └── predicted_kcat_output.csv         # Final kcat prediction results
│
├── local_models/                         # Directory for ESM-2 model (download required)
│   └── esm2_t33_650M_UR50D/              # ESM-2 model files
│
├── models/                               # Directory for trained model checkpoints (download from Zenodo)
│   ├── deep_fusion_kcat_pretrained.pt    # TransKP pre-trained model
│   └── caskp_final_model.pt              # CASKP fine-tuned model
│
├── scripts/                              # Source code directory
│   ├── model_transkp.py                  # TransKP model definition
│   ├── train_transkp.py                  # TransKP training script
│   ├── model_caskp.py                    # CASKP model definition
│   ├── train_caskp.py                    # CASKP training script
│   ├── predict_caskp_model.py            # Model definition for inference
│   ├── generate_features_by_similarity.py # Tool for generating structural features
│   └── caskp_predict.py                  # Main inference script
│
├── train_caskp/                          # CASKP training data and logs
│   ├── caskp_train_data.csv              # Fine-tuning dataset
│   ├── caskp_cv_log.csv                  # Cross-validation training log
│   └── caskp_cv_predictions.csv          # Cross-validation predictions
│
├── train_transkp/                        # TransKP training data and logs
│   ├── transkp_train_data.csv            # Pre-training dataset
│   ├── pretrain_log.csv                  # Pre-training log
│   └── best_epoch_predictions.csv        # Best epoch predictions
│
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation
