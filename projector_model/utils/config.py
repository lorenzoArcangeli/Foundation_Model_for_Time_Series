import os
import torch

# --- Paths ---
BASE_DIR = r"c:\Users\loren\Desktop\D vecchio\UNIVERSITA\MAGISTRALE\SecondYear\FoundationModel\FM_test\project_features"
# Note: The user mentioned this specific file path in previous context
RAW_DATA_PATH = r"c:\Users\loren\Desktop\D vecchio\UNIVERSITA\MAGISTRALE\SecondYear\FoundationModel\FM_test\project_features\skippd_train_aligned_v13_with_time_features.parquet"
DATA_PATH = r"c:\Users\loren\Desktop\D vecchio\UNIVERSITA\MAGISTRALE\SecondYear\FoundationModel\FM_test\project_features\datasets\skippd_train_embeddings.parquet"

# Output Directories
PROJECT_ROOT = r"c:\Users\loren\Desktop\D vecchio\UNIVERSITA\MAGISTRALE\SecondYear\FoundationModel\Foundation_Model_for_Time_Series\projector_model"
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "ckp")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# --- Model Configuration ---
VISION_MODEL = "facebook/dinov2-small"
CHRONOS_MODEL = "amazon/chronos-2"
COVARIATE_DIM = 16
HIDDEN_DIM = 128
CONTEXT_LENGTH = 2048
PREDICTION_LENGTH = 96

# --- Training Configuration ---
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 1.0
PEFT_TYPE = "lora"

# --- Hardware ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
