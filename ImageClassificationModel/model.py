import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dropout_p=0.5):
        super(Model, self).__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5, stride=3, padding=3)
        self.pooling2x2 = nn.MaxPool2d(kernel_size=(2, 2), padding=(0, 0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling2x2(x)
        x = self.dropout(x)
        return x

