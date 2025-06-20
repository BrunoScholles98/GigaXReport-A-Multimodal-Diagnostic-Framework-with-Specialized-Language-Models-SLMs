import torch
import os
import datetime


def create_run_dir_with_description():
    RUN_NAME =  str(datetime.datetime.now()) + "/"
    RUN_DIR = RUNS_DIR + RUN_NAME
    os.makedirs(RUN_DIR)
    IMGS_SAVED_DIR = RUN_DIR + "saved_images/"
    os.makedirs(IMGS_SAVED_DIR)
    os.makedirs(IMGS_SAVED_DIR + "val/")
    os.makedirs(IMGS_SAVED_DIR + "test/")
    
    f = open(RUN_DIR + "model_parameters.txt", "w")
    f.write("Dataset: " + DATASET_DIR + "\n")
    f.write("Batch size: " + str(BATCH_SIZE) + "\n")
    f.write("Number of epochs: " + str(NUM_EPOCHS) + "\n")
    f.write("Leaning Rate: " + str(LEARNING_RATE) + "->" + str(MIN_LEARNING_RATE) + "\n")
    f.write("Loss: " + LOSS_FUNCTION + "Gamma = " + str(GAMMA) + "\n")
    f.write("Notes: " + str(NOTES) + "\n")
    f.close()
    
    return RUN_DIR

# Hyperparameters etc.
BATCH_SIZE = 4
NUM_EPOCHS = 300

#LOSS_FUNCTION = "Focal Loss"
LOSS_FUNCTION = "IoU"

LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = LEARNING_RATE
GAMMA = 1

APPLY_TTA = False
THRESHOLD = 0.5
NUM_WORKERS = 0 #4
PIN_MEMORY = True
USE_DATA_PARALLEL = False
#DEVICE = 'cuda' if torch.cuda.is_available() else "cpu" #"cpu"
DEVICE = 'cuda:1' if torch.cuda.is_available() else "cpu"
#DEVICE = 'cpu'

TRAIN_SIZE = 512
IN_CHANNELS = 1

LOAD_RUN = True
TRAINING = True

#Folder paths
#DATASET_DIR = "/mnt/data/matheusvirgilio/gigasistemica/datasets/giga_new/dataset_corner_without_negatives_aug/"
#DATASET_DIR = '/mnt/data/matheusvirgilio/gigasistemica/datasets/giga_new/whole_dataset_revisited/new_dataset_validate_with_negatives/'
DATASET_DIR = '/mnt/nas/matheusvirgilio/gigasistemica/datasets/giga_new/final_dataset/folds/fold_1/'
TRAIN_IMG_DIR = DATASET_DIR + "images/train/"
TRAIN_MASK_DIR = DATASET_DIR + "masks/train/"
VAL_IMG_DIR = DATASET_DIR + "images/val/"
VAL_MASK_DIR = DATASET_DIR + "masks/val/"

DATASET_DICT = {"train_images"  : TRAIN_IMG_DIR, "train_masks" : TRAIN_MASK_DIR, 
                "val_images"    : VAL_IMG_DIR,   "val_masks"   : VAL_MASK_DIR,}
# Description
NOTES = 'Teste Focal Loss'
RUNS_DIR = "/mnt/nas/matheusvirgilio/gigasistemica/DC_UNET/runs/"
RUN_DIR = create_run_dir_with_description()
IMG_PATH = RUN_DIR + "saved_images/"
NAME_CHECKPOINT = RUN_DIR + "checkpoint.pth.tar"