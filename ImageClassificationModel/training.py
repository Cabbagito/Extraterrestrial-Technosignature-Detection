import sys
import os
from torch.optim import Adam, AdamW, RMSprop
from torch import save, load
from torch.nn import BCELoss


import time

sys.path.append(os.path.join(os.getcwd()))

from DataProcessing.dataLoader import DataLoader
from model import Model


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

AUGMENT_TRAINING_DATA = False
AUGMENT_TRAINING_DATA_PERCENTAGE = 0.25
AUGMENT_TESTING_DATA = False
AUGMENT_TESTING_DATA_PERCENTAGE = 0.25

###

LEARNING_RATE = 0.001


###

model = Model()
loader = DataLoader(DATADIR)
loss = BCELoss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

n_params = sum(p.numel() for p in model.parameters())
n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


if USE_GPU:
    model = model.cuda()
    loader.use_gpu = True
    print("Using GPU\n\n")
else:
    print("Using CPU\n\n")

print(
    f"Number of parameters: {n_params}\nNumber of trainable parameters: {n_trainable_params}\n\n"
)


loader.augment = AUGMENT_TRAINING_DATA
loader.augment_pct = AUGMENT_TRAINING_DATA_PERCENTAGE
loader.augment_validation = AUGMENT_TESTING_DATA
loader.augment_validation_pct = AUGMENT_TESTING_DATA_PERCENTAGE

n_per_batch, n_n_per_batch, n_a_per_batch, n_val = loader.start_loader(
    n_batches=N_BATCHES, normal_pct=NORMAL_PCT, n_val_batches=N_VAL_BATCHES
)

