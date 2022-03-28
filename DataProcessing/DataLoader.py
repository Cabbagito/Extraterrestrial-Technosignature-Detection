import os
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, path):
        self.path = path
        self.labels = pd.read_csv("../" + self.path + "/train_labels.csv")

    def load_one(self, filename):
        for dir in os.listdir(self.path):
            for file in os.listdir(os.path.join(self.path, dir)):
                if file == filename:
                    return np.load(os.path.join(self.path, dir, file))

    def load_n(self, n, label):
        subset = None

        if label is True:
            subset = self.labels[self.labels["label"] == "1"].sample(n=n)
        else:
            subset = self.labels[self.labels["label"] == "0"].sample(n=n)

        retdata = []

        for index, row in subset.iterrows():
            retdata.append(self.load_one(row["label"] + ".npy"))

        return np.array(retdata)

