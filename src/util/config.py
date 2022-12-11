from os.path import join

PROJECT_BASE_PATH = '/workspaces/project_automated_sound_recognition'
SRC_PATH = join(PROJECT_BASE_PATH, "src")
DATA_BASE_PATH = join(PROJECT_BASE_PATH, "data")

RAW_DATA_BASE_PATH = join(DATA_BASE_PATH, "raw")
RAW_FULL_DATA_ZIP_PATH = join(RAW_DATA_BASE_PATH, "zip")
RAW_FULL_DATA_PATH = join(RAW_DATA_BASE_PATH, "TAU-urban-acoustic-scenes-2022-mobile-development")
RAW_SUBSET_DATA_PATH = join(RAW_DATA_BASE_PATH, "subset")

TRAIN_METADATA_FILE = join(RAW_FULL_DATA_PATH, "evaluation_setup/fold1_train.csv")
TRAIN_DATA_BASE_PATH = join(DATA_BASE_PATH, "train")
TEST_METADATA_FILE = join(RAW_FULL_DATA_PATH, "evaluation_setup/fold1_test.csv")
TEST_DATA_BASE_PATH = join(DATA_BASE_PATH, "test")

TRAIN_DATA_BASELINE = join(TRAIN_DATA_BASE_PATH, "baseline")
TRAIN_DATA_EX1 = join(TRAIN_DATA_BASE_PATH, "ex1")

N_COPIES =  5

featconf = {
    'dcRemoval': 'hpf',
    'samFreq': 44100,
    'lowFreq': 0,
    'highFreq': 22050,
    'stepSize_ms': 20,
    'frameSize_ms': 40,
    'melSize': 40
}

labels = [
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