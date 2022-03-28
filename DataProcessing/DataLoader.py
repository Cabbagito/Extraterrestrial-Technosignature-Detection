import os
import pandas as pd
import numpy as np
from torch import Tensor


class DataLoader:
    def __init__(self, path):
        self.path = path
        self.labels = pd.read_csv(self.path + "/../train_labels.csv")
        self.n = self.labels.shape[0]
        self.n_a = self.labels[self.labels["target"] == 1].shape[0]
        self.n_n = self.n - self.n_a
        self.__loader_initialized = False
        self.use_gpu = False

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
            f"Per Batch: {n_per_batch} \nAnomalies Per Batch: {n_a_per_batch} \nNormal Per Batch: {n_n_per_batch} \nValidation Set size: {n_val}"
        )

    def next_batch(self):
        if not self.__loader_initialized:
            raise Exception("Loader not initialized")

