# -------------------------
# Paths
# -------------------------
MODEL_PATH = "../model/loan_approval_classifier.pth"
FEATURE_TRANSFORMER_PATH = "../model/feature_transformer.pkl"
DATASET_CSV_PATH = "../data/loan_data.csv"

# -------------------------
# Training hyperparameters
# -------------------------
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
NUM_EPOCHS = 55
VAL_INTERVAL = 1

# -------------------------
# Data split ratios
# -------------------------
SPLIT_DATASET = True
SPLIT_RANDOMIZATION_SEED = 42
TRAIN_SPLIT_RATIO = 0.7
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15

# -------------------------
# Classification Threshold
# -------------------------
CLASSIFICATION_THRESHOLD = 0.5

