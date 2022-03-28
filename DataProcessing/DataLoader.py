import os
import pandas as pd
import numpy as np
from torch import Tensor


class DataLoader:
    def __init__(self, path):
        self.path = path
        self.labels = pd.read_csv(self.path + "/../train_labels.csv")
        self.n = self.labels.shape[0]
        self.a_pct = self.labels[self.labels["target"] == 1].shape[0] / self.n
        self.__loader_initialized = False

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

        return Tensor(np.array(retdata))

    def get_files(self, target=None):
        if target is None:
            return self.labels["target"].tolist()
        else:
            return self.labels[self.labels["target"] == target]["target"].tolist()

    def start_loader(self, n_batches, create_val=False):
        n_per_batch = int(self.n / n_batches)
        n_a_per_batch = int(n_per_batch * self.a_pct)
        n_n_per_batch = n_per_batch - n_a_per_batch

    def next_batch(self):
        if not self.__loader_initialized:
            raise Exception("Loader not initialized")

