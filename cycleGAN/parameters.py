import os

CODES = ["T1", "T2", "T1GD", "FLAIR", "T1-T2", "T1-T1GD", "T1-FLAIR", "T2-T1GD", "T2-FLAIR", "T1GD-FLAIR", "T1-T1GD-FLAIR"
        , "T1-T2-T1GD", "T1-T2-FLAIR", "T2-T1GD-FLAIR", "all"]
CHANNELS = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4]
MODALITIES = [["T1"], ["T2"], ["T1GD"], ["FLAIR"], ["T1", "T2"], ["T1", "T1GD"], ["T1", "FLAIR"], ["T2", "T1GD"], ["T2", "FLAIR"], ["T1GD", "FLAIR"], ["T1", "T1GD", "FLAIR"]
        , ["T1", "T2", "T1GD"], ["T1", "T2", "FLAIR"], ["T2", "T1GD", "FLAIR"], ["T1", "T2", "T1GD", "FLAIR"]]
LAMBDA = 10
IMG_HEIGHT = 256
IMG_WIDTH = 256
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 4
BATCH_SIZE = 1
ROOT_DIR = '/home/paperspace/Documents/'
ROOT_DIR_DATASET = '/home/paperspace/Documents/code/models/cycleGAN'
TRAIN_INPUT_PATH_DATASET = 'train_dataset_input'
TRAIN_TARGET_PATH_DATASET ='train_dataset_target'
TEST_INPUT_PATH_DATASET = 'test_dataset_input'
TEST_TARGET_PATH_DATASET = 'test_dataset_target'
RESULTS_DIR = os.path.join(ROOT_DIR, 'code/results/cycleGAN')
CHECKPOINT_DIR = './training_checkpoints'

if __name__ == "__main__":
    pass