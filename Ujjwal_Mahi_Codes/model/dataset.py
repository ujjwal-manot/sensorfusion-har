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

    @classmethod
    def loso_split(cls, root_dir, test_subject):
        train_full = cls(root_dir, split="train")
        test_full = cls(root_dir, split="test")

        train_subjects = np.loadtxt(os.path.join(root_dir, "train", "subject_train.txt"), dtype=int)
        test_subjects = np.loadtxt(os.path.join(root_dir, "test", "subject_test.txt"), dtype=int)

        all_X = torch.cat([train_full.X, test_full.X], dim=0)
        all_y = torch.cat([train_full.y, test_full.y], dim=0)
        all_subjects = np.concatenate([train_subjects, test_subjects])

        test_mask = all_subjects == test_subject
        train_mask = ~test_mask

        train_ds = cls.__new__(cls)
        train_ds.X = all_X[train_mask]
        train_ds.y = all_y[train_mask]
        train_ds.root_dir = root_dir
        train_ds.split = "train"

        test_ds = cls.__new__(cls)
        test_ds.X = all_X[test_mask]
        test_ds.y = all_y[test_mask]
        test_ds.root_dir = root_dir
        test_ds.split = "test"

        return train_ds, test_ds

    @staticmethod
    def get_subjects(root_dir):
        train_subj = np.loadtxt(os.path.join(root_dir, "train", "subject_train.txt"), dtype=int)
        test_subj = np.loadtxt(os.path.join(root_dir, "test", "subject_test.txt"), dtype=int)
        return sorted(set(np.concatenate([train_subj, test_subj])))

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
