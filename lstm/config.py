import torch

# --- Data Parameters ---
# Update paths as needed
DATASET_PATH = "../SPARL3/data_f0_sparl3_with_gyro_baro_2.pkl"
SENSOR_COLS = ["Acc.x", "Acc.y", "Acc.z", "Gyro.x", "Gyro.y", "Gyro.z", "Baro.x"]
TEST_EXPERIMENT_ID = 1
VALIDATION_EXPERIMENT_ID = 2

# --- Preprocessing Parameters ---
ORIGINAL_FREQ = 2000
TARGET_FREQ = 100
SEQ_LEN = int(TARGET_FREQ*1.902381043660399)
DS_FACTOR = int(ORIGINAL_FREQ / TARGET_FREQ) if TARGET_FREQ > 0 else 1
FILTER_ORDER = 8
IS_WITHOUT_DS = False

# --- Model Parameters ---
MODEL_NAME = "CNN_1LSTM"
NUM_CNN_LAYERS = 1
IN_CHANNELS = len(SENSOR_COLS)

# --- Training Parameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 1e-5
VALIDATION_SPLIT = 0.5
RANDOM_STATE = 42
FC_DROPOUT = 0.6
LSTM_HIDDEN_SIZE = 128

USE_SUPERCLASSES = False
EARLY_STOPPING_ENABLED = False
EARLY_STOPPING_PATIENCE = 7
SCHEDULER_ENABLED = False
SCHEDULER_PATIENCE = 4
SCHEDULER_FACTOR = 0.1

AUGMENT_TRAINING_DATA = True
AUGMENTATION_PROBABILITY = 0.8
SMOTE_SAMPLING_RATIO = 0.5 

RESAMPLING_STRATEGY = "SMOTE"
MINORITY_INCREASE_PERCENTAGE = 100.0
USE_OVERSAMPLING = True

SUPERCLASS_MAPPING = {
    # Driving Activities
    "Driving(curve)": "Driving (curve)",
    "Driving(straight)": "Driving (straight)",
    "Lifting and driving": "Lifting and driving",
    # Lifting Activities
    "Lifting(lowering)": "Lifting (lowering)",
    "Lifting(raising)": "Lifting (raising)",
    # Wrapping Activities
    "Wrapping": "Turntable wrapping",
    # Idle or Miscellaneous States
    "Wrapping(preparation)": "Stationary processes",
    "Docking": "Stationary processes",
    "Forks(entering or leaving front)": "Stationary processes",
    "Forks(entering or leaving side)": "Stationary processes",
    "Standing": "Stationary processes"
}
