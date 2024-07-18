BATCH_SIZE = 16
MAX_TRAIN_STEPS = 10
MAX_VAL_STEPS = 10
MAX_EPOCHS = 100

SAMPLES_PER_GROUP = 4000
LIMIT = int(1e2)
IMG_SCALING = (3, 3)

BASE_DIR = 'airbus-ship-detection/'
TRAIN_PATH = BASE_DIR + 'train_v2/'
TEST_PATH = BASE_DIR + 'test_v2/'
MASKS_PATH = BASE_DIR + 'train_ship_segmentations_v2.csv'
MODEL_PATH = 'models/model.h5'