import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        CHANNELS_1=128,
        KERNEL_1=5,
        STRIDE_1=2,
        PADDING_1=20,
        CHANNELS_2=256,
        KERNEL_2=3,
        STRIDE_2=2,
        PADDING_2=20,
        CHANNELS_3=512,
        KERNEL_3=3,
        STRIDE_3=2,
        PADDING_3=10,
        DROPOUT=0.5,
    ):
        super(Model, self).__init__()

        self.dropout = nn.Dropout(DROPOUT)
        self.relu = nn.LeakyReLU()

        self.pooling3x3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pooling2x2 = nn.MaxPool2d(kernel_size=2)

        self.bn1 = nn.BatchNorm2d(CHANNELS_1)
        self.bn2 = nn.BatchNorm2d(CHANNELS_2)
        self.bn3 = nn.BatchNorm2d(CHANNELS_3)
        self.bn4 = nn.BatchNorm2d(CHANNELS_2)

        self.conv1 = nn.Conv2d(
            1, CHANNELS_1, kernel_size=KERNEL_1, stride=STRIDE_1, padding=PADDING_1
        )

        self.conv2 = nn.Conv2d(
            CHANNELS_1,
            CHANNELS_2,
            kernel_size=KERNEL_2,
            stride=STRIDE_2,
            padding=PADDING_2,
        )

        self.conv3 = nn.Conv2d(
            CHANNELS_2,
            CHANNELS_3,
            kernel_size=KERNEL_3,
            stride=STRIDE_3,
            padding=PADDING_3,
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling3x3(x)
        x = self.bn1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling2x2(x)
        x = self.bn2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pooling2x2(x)
        x = self.bn3(x)
        x = self.dropout(x)

        return x

