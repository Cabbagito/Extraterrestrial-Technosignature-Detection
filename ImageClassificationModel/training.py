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

USE_GPU = False
N_BATCHES = 500

###

model = Model()
loader = DataLoader(DATADIR)


if USE_GPU:
    model = model.cuda()
    loader.use_gpu = True
    print("Using GPU")
else:
    print("Using CPU")

dummy_data = loader.load_n(5)


# loader.start_loader(n_batches=420, normal_pct=0.5, n_val_batches=20)


y = model.forward(dummy_data)

print(y.shape)

