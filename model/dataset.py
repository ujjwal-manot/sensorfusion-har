import os
import zipfile
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset


DOWNLOAD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"

SIGNAL_FILES = [
    "body_acc_x_{}.txt",
    "body_acc_y_{}.txt",
    "body_acc_z_{}.txt",
    "body_gyro_x_{}.txt",
    "body_gyro_y_{}.txt",
    "body_gyro_z_{}.txt",
]


class UCIHARDataset(Dataset):

    def __init__(self, root_dir, split="train"):
        assert split in ("train", "test")
        self.root_dir = root_dir
        self.split = split

        signals = []
        for fname_template in SIGNAL_FILES:
            fname = fname_template.format(split)
            fpath = os.path.join(root_dir, split, "Inertial Signals", fname)
            data = np.loadtxt(fpath)
            signals.append(data)

        self.X = torch.tensor(np.stack(signals, axis=-1), dtype=torch.float32)

        label_path = os.path.join(root_dir, split, "y_{}.txt".format(split))
        labels = np.loadtxt(label_path, dtype=int)
        self.y = torch.tensor(labels - 1, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @classmethod
    def get_normalization_stats(cls, root_dir):
        dataset = cls(root_dir, split="train")
        mean = dataset.X.mean(dim=(0, 1))
        std = dataset.X.std(dim=(0, 1))
        std[std < 1e-8] = 1.0
        return mean.tolist(), std.tolist()

    @staticmethod
    def download(dest_dir):
        zip_path = os.path.join(dest_dir, "uci_har.zip")
        os.makedirs(dest_dir, exist_ok=True)
        print("Downloading UCI HAR Dataset...")
        urllib.request.urlretrieve(DOWNLOAD_URL, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        os.remove(zip_path)
        extracted = os.path.join(dest_dir, "UCI HAR Dataset")
        if os.path.isdir(extracted):
            return extracted
        return dest_dir
