import sys
import os
from torch import Tensor
import torch.optim as optim
import torch

import time

sys.path.append(os.path.join(os.getcwd()))

from DataProcessing.dataLoader import DataLoader
from model import Model

###
DATADIR = "../Data/train"

N_FULL = 60000
N_ANOMALY = 6000
N_NORMAL = 54000
IMG_SHAPE = (1, 819, 256)

USE_GPU = True
N_BATCHES = 500

###

model = Model()
loader = DataLoader(DATADIR)

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


dummy_size = 100
start = time.time()
# dummy_data = loader.load_n(dummy_size)
end = time.time()

print(f"Loading {dummy_size} observations took {end - start} seconds")


loader.start_loader(n_batches=420, normal_pct=0.5, n_val_batches=20)

batch = loader.next_batch()


start = time.time()
y = model.forward(batch)
end = time.time()

print(f"Forward Pass of size {dummy_size} took {end - start} seconds")

