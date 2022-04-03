import os
import pandas as pd
import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt
from scipy import ndimage
from random import uniform


class DataLoader:
    def __init__(self, path):
        self.path = path
        self.labels = pd.read_csv(self.path + "/../train_labels.csv")
        self.n = self.labels.shape[0]
        self.n_a = self.labels[self.labels["target"] == 1].shape[0]
        self.n_n = self.n - self.n_a
        self.__loader_initialized = False
        self.use_gpu = False
        self.augment = False
        self.augment_pct = 0.25
        self.augment_validation = False
        self.augment_validation_pct = 0.25

    def load_one(self, filename):
        for dir in os.listdir(self.path):
            for file in os.listdir(os.path.join(self.path, dir)):
                if file == filename + ".npy":
                    return np.expand_dims(
                        np.load(os.path.join(self.path, dir, file)), axis=0
                    )

    def load_n(self, n, label=None):
        subset = None
        if label is None:
            subset = self.labels.sample(n=n)
        else:
            subset = self.labels[self.labels["target"] == label].sample(n=n)

        retdata = []

        for index, row in subset.iterrows():
            retdata.append(self.load_one(row["id"]))

        if not self.use_gpu:
            return Tensor(np.array(retdata))
        else:
            return Tensor(np.array(retdata)).cuda()

    def get_files(self, target=None):
        if target is None:
            return self.labels["target"].tolist()
        else:
            return self.labels[self.labels["target"] == target]["target"].tolist()

    def start_loader(self, n_batches, n_val_batches=0, normal_pct=1):
        self.__loader_initialized = True

        n_n_per_batch = int(self.n_n * normal_pct / n_batches)
        n_a_per_batch = int(self.n_a / n_batches)

        n_per_batch = n_n_per_batch + n_a_per_batch

        n_val = n_val_batches * n_n_per_batch

        print(
            f"Per Batch: {n_per_batch} \nAnomalies Per Batch: {n_a_per_batch} \nNormal Per Batch: {n_n_per_batch} \nValidation Set size: {n_val}\n*Without augmentation*\n\n"
        )

        self.batches = []
        self.val_batches = []
        self.batch_idx = 0
        self.n_batches = n_batches
        self.n_val_batches = n_val_batches
        self.val_batch_idx = 0
        data = self.labels.copy()

        for batch in range(n_val_batches):
            batch_n_df = data[data["target"] == 0].sample(n=n_n_per_batch)
            batch_a_df = data[data["target"] == 1].sample(n=n_a_per_batch)

            to_add = batch_n_df["id"].tolist()
            to_add.extend(batch_a_df["id"].tolist())
            self.val_batches.append(to_add)

            data.drop(batch_n_df.index, inplace=True)
            data.drop(batch_a_df.index, inplace=True)

        for batch in range(n_batches - n_val_batches):
            batch_n_df = data[data["target"] == 0].sample(n=n_n_per_batch)
            batch_a_df = data[data["target"] == 1].sample(n=n_a_per_batch)

            to_add = batch_n_df["id"].tolist()
            to_add.extend(batch_a_df["id"].tolist())
            self.batches.append(to_add)

            data.drop(batch_n_df.index, inplace=True)
            data.drop(batch_a_df.index, inplace=True)

        return n_per_batch, n_n_per_batch, n_a_per_batch, n_val

    def next_batch(self):
        if not self.__loader_initialized:
            raise Exception("Loader not initialized")

        if self.batch_idx >= self.n_batches:
            self.batch_idx = 0

        self.batch_idx += 1
        data = self.__load_batch(
            self.batches[self.batch_idx - 1], self.augment, self.augment_pct
        )
        if self.use_gpu:
            return Tensor(data[0]).cuda(), Tensor(data[1]).unsqueeze(1).cuda()
        else:
            return Tensor(data[0]), Tensor(data[1]).unsqueeze(1)

    def next_val_batch(self):
        if not self.__loader_initialized:
            raise Exception("Loader not initialized")

        if self.val_batch_idx >= self.n_val_batches:
            self.val_batch_idx = 0

        self.val_batch_idx += 1
        data = self.__load_batch(
            self.val_batches[self.val_batch_idx - 1],
            self.augment_validation,
            self.augment_validation_pct,
        )
        if self.use_gpu:
            return Tensor(data[0]).cuda(), Tensor(data[1]).unsqueeze(1).cuda()
        else:
            return Tensor(data[0]), Tensor(data[1].unsqueeze(1))

    def __load_batch(self, batch, augment=False, aug_pct=0.0):
        retdata = []
        labels = []
        for dir in os.listdir(self.path):
            for file in os.listdir(os.path.join(self.path, dir)):
                if file[:-4] in batch:

                    label = self.labels[self.labels["id"] == file[:-4]][
                        "target"
                    ].values[0]

                    value = np.expand_dims(
                        np.load(os.path.join(self.path, dir, file)), axis=0
                    )
                    retdata.append(value)
                    labels.append(label)

                    if augment and label == 1:
                        augment_data = self.augment_img(value, aug_pct)
                        for aug_value in augment_data:
                            retdata.append(aug_value)
                            labels.append(label)

                    if len(retdata) == len(batch):
                        break

        return (np.array(retdata), labels)

    def augment_img(self, x, aug_pct=0.0):
        retdata = []

        if uniform(0, 1) < aug_pct:
            x1 = np.fliplr(x[0])
            x1 = np.expand_dims(x1, 0)
            retdata.append(x1)

        if uniform(0, 1) < aug_pct:
            x2 = np.flipud(x[0])
            x2 = np.expand_dims(x2, 0)
            retdata.append(x2)

        if uniform(0, 1) < aug_pct:
            x3 = ndimage.rotate(x[0], 5, reshape=False)
            x3 = np.expand_dims(x3, 0)
            retdata.append(x3)

        if uniform(0, 1) < aug_pct:
            x4 = ndimage.rotate(x[0], -5, reshape=False)
            x4 = np.expand_dims(x4, 0)
            retdata.append(x4)

        return retdata
