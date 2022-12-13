from os.path import join
import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
LR = 1e-5
EPOCHS = 20

SRC_PATH = "src"
DATA_BASE_PATH = "data"

MODEL_CHECKPOINT_PATH = "checkpoints"

RAW_DATA_BASE_PATH = join(DATA_BASE_PATH, "raw")
RAW_FULL_DATA_ZIP_PATH = join(RAW_DATA_BASE_PATH, "zip")
RAW_FULL_DATA_PATH = join(RAW_DATA_BASE_PATH, "TAU-urban-acoustic-scenes-2022-mobile-development")
RAW_SUBSET_DATA_PATH = join(RAW_DATA_BASE_PATH, "subset")

TRAIN_METADATA_FILE = join(RAW_FULL_DATA_PATH, "evaluation_setup/fold1_train.csv")
TRAIN_DATA_PATH = join(RAW_SUBSET_DATA_PATH, "train/audio")
TEST_METADATA_FILE = join(RAW_FULL_DATA_PATH, "evaluation_setup/fold1_test.csv")
TEST_DATA_PATH = join(RAW_SUBSET_DATA_PATH, "test/audio")

LABELS = [
    "airport",
    "shopping_mall",
    "metro_station",
    "street_pedestrian",
    "public_square",
    "street_traffic",
    "tram",
    "bus",
    "metro",
    "park"
]


featconf = {
    'dcRemoval': 'hpf',
    'samFreq': 44100,
    'lowFreq': 0,
    'highFreq': 22050,
    'stepSize_ms': 20,
    'frameSize_ms': 40,
    'melSize': 40
}