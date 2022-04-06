import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        INPUT_SHAPE=(1, 819, 256),
        CHANNELS_1=128,
        KERNEL_1=5,
        STRIDE_1=3,
        PADDING_1=20,
        CHANNELS_2=128,
        KERNEL_2=5,
        STRIDE_2=3,
        PADDING_2=15,
        CHANNELS_3=128,
        KERNEL_3=3,
        STRIDE_3=2,
        PADDING_3=8,
        CHANNELS_4=256,
        KERNEL_4=3,
        STRIDE_4=2,
        PADDING_4=3,
        LINEAR_1=512,
        LINEAR_2=256,
        DROPOUT_CONV=0.5,
        DROPOUT_FC=0.5,
    ):
        super(Model, self).__init__()

        self.dropout_conv = nn.Dropout(DROPOUT_CONV)
        self.dropout_fc = nn.Dropout(DROPOUT_FC)
        self.activation = nn.LeakyReLU()
        self.output_activation = nn.Sigmoid()
        self.flatten = nn.Flatten()

        self.pooling3x3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pooling2x2 = nn.MaxPool2d(kernel_size=2)

        self.bn1 = nn.BatchNorm2d(CHANNELS_1)
        self.bn2 = nn.BatchNorm2d(CHANNELS_2)
        self.bn3 = nn.BatchNorm2d(CHANNELS_3)
        self.bn4 = nn.BatchNorm2d(CHANNELS_4)
        self.bn5 = nn.BatchNorm1d(LINEAR_1)
        self.bn6 = nn.BatchNorm1d(LINEAR_2)

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

        self.conv4 = nn.Conv2d(
            CHANNELS_3,
            CHANNELS_4,
            kernel_size=KERNEL_4,
            stride=STRIDE_4,
            padding=PADDING_4,
        )

        linear_size = self.__get_linear_size(INPUT_SHAPE)

        self.linear1 = nn.Linear(linear_size, LINEAR_1)
        self.linear2 = nn.Linear(LINEAR_1, LINEAR_2)
        self.linear3 = nn.Linear(LINEAR_2, 1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pooling2x2(x)
        x = self.dropout_conv(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pooling2x2(x)
        x = self.dropout_conv(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.pooling2x2(x)
        x = self.dropout_conv(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.pooling2x2(x)
        x = self.dropout_conv(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = self.bn5(x)
        x = self.activation(x)
        x = self.dropout_fc(x)

        x = self.linear2(x)
        x = self.bn6(x)
        x = self.activation(x)
        x = self.dropout_fc(x)

        x = self.linear3(x)

        x = self.output_activation(x)

        return x

    def __get_linear_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self.conv1(x)
        x = self.pooling2x2(x)
        x = self.conv2(x)
        x = self.pooling2x2(x)
        x = self.conv3(x)
        x = self.pooling2x2(x)
        x = self.conv4(x)
        x = self.pooling2x2(x)
        x = self.flatten(x)
        return x.shape[1]
