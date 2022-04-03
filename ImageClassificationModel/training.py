from hashlib import new
import sys
import os
from torch.optim import Adam, AdamW, RMSprop
from torch.nn import BCELoss
from torch import no_grad


from time import time

sys.path.append(os.path.join(os.getcwd()))

from DataProcessing.dataLoader import DataLoader
from model import Model
from util import save_and_log, load, newest_model, validate


###

DATADIR = "../Data/train"
MODELS_DIR = "ImageClassificationModel/models"

N_FULL = 60000
N_ANOMALY = 6000
N_NORMAL = 54000
IMG_SHAPE = (1, 819, 256)

###

USE_GPU = True
N_BATCHES = 700
N_VAL_BATCHES = 10
NORMAL_PCT = 0.6
SAVE_AND_VALIDATE_AFTER_N_BATCHES = 10

###

LEARNING_RATE = 0.001
EPOCHS = 1
LOAD_NEWEST = False

###

AUGMENT_TRAINING_DATA = True
AUGMENT_TRAINING_DATA_PERCENTAGE = 0.8
AUGMENT_TESTING_DATA = False
AUGMENT_TESTING_DATA_PERCENTAGE = 0.25

###

newest = newest_model(MODELS_DIR)
if LOAD_NEWEST and newest != -1:
    model = load(newest)
else:
    newest = 0
    model = Model()
loader = DataLoader(DATADIR)


n_params = sum(p.numel() for p in model.parameters())


print(f"\nNumber of parameters: {n_params}\n\n")


if USE_GPU:
    model = model.cuda()
    loader.use_gpu = True
    print("Using GPU\n\n")
else:
    print("Using CPU\n\n")


loss_fn = BCELoss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)


loader.augment = AUGMENT_TRAINING_DATA
loader.augment_pct = AUGMENT_TRAINING_DATA_PERCENTAGE
loader.augment_validation = AUGMENT_TESTING_DATA
loader.augment_validation_pct = AUGMENT_TESTING_DATA_PERCENTAGE

n_per_batch, n_n_per_batch, n_a_per_batch, n_val = loader.start_loader(
    n_batches=N_BATCHES, normal_pct=NORMAL_PCT, n_val_batches=N_VAL_BATCHES
)


for epoch in range(EPOCHS):
    print("#" * 80)
    print(f"Epoch {epoch}")
    print("#" * 80, "\n")
    for batch in range(N_BATCHES - N_VAL_BATCHES):

        optimizer.zero_grad()

        x, yhat = loader.next_batch()
        start = time()
        y = model(x)
        end = time()

        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        val_string = ""
        if (batch % SAVE_AND_VALIDATE_AFTER_N_BATCHES == 0) and batch != 0:
            with no_grad():
                val_loss, val_accuracy, precision, recall, outcomes = validate(
                    model, loader, loss_fn
                )
                newest += 1
            save_and_log(
                model, newest, epoch, batch, (val_loss, val_accuracy, precision, recall)
            )
            val_string = f"\nValidation loss: {val_loss}, Validation accuracy: {val_accuracy}, Validation precision: {precision}, Validation recall: {recall}\n"

        print(f"Batch {batch} Loss: {loss}, Time: {end - start} {val_string}")

