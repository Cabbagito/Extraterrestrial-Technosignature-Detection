import sys
import os
from torch.optim import Adam, AdamW, RMSprop
from torch.nn import BCELoss


import time

sys.path.append(os.path.join(os.getcwd()))

from DataProcessing.dataLoader import DataLoader
from model import Model
from util import save_and_log, load, newest_model


###

DATADIR = "../Data/train"
MODELS_DIR = "ImageClassificationModel/models"

N_FULL = 60000
N_ANOMALY = 6000
N_NORMAL = 54000
IMG_SHAPE = (1, 819, 256)

###

USE_GPU = True
N_BATCHES = 500
N_VAL_BATCHES = 1
NORMAL_PCT = 0.7
SAVE_AFTER_N_BATCHES = 50
VALIDATE_AFTER_N_BATCHES = 50

###

LEARNING_RATE = 0.001
EPOCHS = 1
LOAD_NEWEST = False

###

AUGMENT_TRAINING_DATA = False
AUGMENT_TRAINING_DATA_PERCENTAGE = 0.25
AUGMENT_TESTING_DATA = False
AUGMENT_TESTING_DATA_PERCENTAGE = 0.25

###
newest = newest_model(MODELS_DIR)
if LOAD_NEWEST:
    model = load(newest)
else:
    model = Model()
loader = DataLoader(DATADIR)


n_params = sum(p.numel() for p in model.parameters())
n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(
    f"Number of parameters: {n_params}\nNumber of trainable parameters: {n_trainable_params}\n\n"
)


if USE_GPU:
    model = model.cuda()
    loader.use_gpu = True
    print("Using GPU\n\n")
else:
    print("Using CPU\n\n")


loss = BCELoss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)


loader.augment = AUGMENT_TRAINING_DATA
loader.augment_pct = AUGMENT_TRAINING_DATA_PERCENTAGE
loader.augment_validation = AUGMENT_TESTING_DATA
loader.augment_validation_pct = AUGMENT_TESTING_DATA_PERCENTAGE

n_per_batch, n_n_per_batch, n_a_per_batch, n_val = loader.start_loader(
    n_batches=N_BATCHES, normal_pct=NORMAL_PCT, n_val_batches=N_VAL_BATCHES
)

