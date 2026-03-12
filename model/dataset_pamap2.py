import os
import zipfile
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset


DOWNLOAD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"

ACTIVITY_MAP = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    12: 7,
    13: 8,
    16: 9,
    17: 10,
    24: 11,
}

ACTIVITY_NAMES = [
    "Lying", "Sitting", "Standing", "Walking", "Running",
    "Cycling", "Nordic Walking", "Ascending Stairs",
    "Descending Stairs", "Vacuum Cleaning", "Ironing", "Rope Jumping",
]

HAND_ACC_COLS = [4, 5, 6]
HAND_GYRO_COLS = [10, 11, 12]
SENSOR_COLS = HAND_ACC_COLS + HAND_GYRO_COLS

TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6]
TEST_SUBJECTS = [7, 8, 9]

WINDOW_SIZE = 128
STEP_SIZE = 64
SAMPLE_RATE = 100


class PAMAP2Dataset(Dataset):

    def __init__(self, root_dir, split="train"):
        assert split in ("train", "test")
        self.root_dir = root_dir
        self.split = split

        subjects = TRAIN_SUBJECTS if split == "train" else TEST_SUBJECTS
        protocol_dir = os.path.join(root_dir, "Protocol")

        all_windows = []
        all_labels = []

        for subj in subjects:
            fpath = os.path.join(protocol_dir, "subject10{}.dat".format(subj))
            if not os.path.exists(fpath):
                continue

            raw = np.loadtxt(fpath)

            sensor_data = raw[:, SENSOR_COLS]
            activity_ids = raw[:, 1].astype(int)

            for col in range(sensor_data.shape[1]):
                mask = np.isnan(sensor_data[:, col])
                if mask.any():
                    valid = np.where(~mask)[0]
                    if len(valid) > 1:
                        sensor_data[mask, col] = np.interp(
                            np.where(mask)[0], valid, sensor_data[valid, col]
                        )
                    else:
                        sensor_data[mask, col] = 0.0

            for start in range(0, len(sensor_data) - WINDOW_SIZE + 1, STEP_SIZE):
                end = start + WINDOW_SIZE
                window_labels = activity_ids[start:end]
                unique, counts = np.unique(window_labels, return_counts=True)
                dominant = unique[np.argmax(counts)]

                if dominant not in ACTIVITY_MAP:
                    continue

                if counts.max() < WINDOW_SIZE * 0.8:
                    continue

                all_windows.append(sensor_data[start:end])
                all_labels.append(ACTIVITY_MAP[dominant])

        self.X = torch.tensor(np.array(all_windows), dtype=torch.float32)
        self.y = torch.tensor(np.array(all_labels), dtype=torch.long)

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
        all_subjects = list(range(1, 10))
        train_subjects = [s for s in all_subjects if s != test_subject]

        import model.dataset_pamap2 as mod
        orig_train = mod.TRAIN_SUBJECTS
        orig_test = mod.TEST_SUBJECTS

        mod.TRAIN_SUBJECTS = train_subjects
        mod.TEST_SUBJECTS = [test_subject]

        train_ds = cls(root_dir, split="train")
        test_ds = cls(root_dir, split="test")

        mod.TRAIN_SUBJECTS = orig_train
        mod.TEST_SUBJECTS = orig_test

        return train_ds, test_ds

    @staticmethod
    def get_subjects(root_dir):
        return list(range(1, 10))

    @staticmethod
    def download(dest_dir):
        zip_path = os.path.join(dest_dir, "pamap2.zip")
        os.makedirs(dest_dir, exist_ok=True)
        print("Downloading PAMAP2 Dataset...")
        urllib.request.urlretrieve(DOWNLOAD_URL, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        os.remove(zip_path)
        extracted = os.path.join(dest_dir, "PAMAP2_Dataset")
        if os.path.isdir(extracted):
            return extracted
        return dest_dir
