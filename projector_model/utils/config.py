import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Dynamic Path Resolution
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_UTILS_DIR)

# Paths
BASE_DIR = os.getenv("FM_DATA_DIR")
if not BASE_DIR:
    raise ValueError("FM_DATA_DIR not set. Please create a .env file with this variable. See .env.example.")

RAW_DATA_PATH = os.path.join(BASE_DIR, "skippd_train_aligned_v13_with_time_features.parquet")
DATA_PATH = os.path.join(BASE_DIR, "datasets", "skippd_train_embeddings.parquet")

# Output Directories
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Model Configuration
VISION_MODEL = "facebook/dinov2-small"
CHRONOS_MODEL = "amazon/chronos-2"
COVARIATE_DIM = 16
HIDDEN_DIM = 128
CONTEXT_LENGTH = 2048
PREDICTION_LENGTH = 96

# Training Configuration
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 1.0
PEFT_TYPE = "lora"

# Steps & Intervals
MAX_STEPS = 100
WARMUP_STEPS_RATIO = 0.1
PROJECTOR_WARMUP_STEPS = 150
VAL_CHECK_INTERVAL = 10
SAVE_INTERVAL = 10

# Regularization
DROPOUT = 0.2
NOISE_STD = 0.05

# Data specifics
SEASONALITY = 96

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
