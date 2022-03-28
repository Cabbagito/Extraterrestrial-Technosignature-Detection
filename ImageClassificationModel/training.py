from multiprocessing import dummy
import sys
import os
from torch import Tensor
import torch.optim as optim

sys.path.append(os.path.join(os.getcwd()))

from DataProcessing.dataLoader import DataLoader
from model import Model

###
DATADIR = "../Data/train"
N_FULL = 60000
N_ANOMALY = 6000
N_NORMAL = 54000
IMG_SHAPE = (1, 819, 256)
###

model = Model()
loader = DataLoader(DATADIR)


dummy_data = loader.load_n(20)

print(dummy_data.shape)

y = model.forward(dummy_data)

print(y.shape)
