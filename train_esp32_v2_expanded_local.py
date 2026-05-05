"""Local training script for the ESP32 V2 expanded SensorFusion-HAR model.

Runs the same pipeline as sensorfusion_har_ESP32.ipynb but optimized for local
CPU execution. Produces:
  - checkpoints_v2/best_sensorfusion_esp32_v2_useful11_rw.pt
  - exports/esp32_v2/sensorfusion_esp32_v2_useful11_rw.onnx
  - outputs/esp32_v2/summary_v2_useful11_rw.json
  - outputs/esp32_v2/v2_useful11_rw_confusion_f1.png

TFLite INT8 conversion + ESP32 C header are produced separately (Colab) since
TensorFlow is unavailable on Python 3.14.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    ConfusionMatrixDisplay,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.dataset import UCIHARDataset
from model.dataset_pamap2 import PAMAP2Dataset

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
TARGET_TIME_STEPS = 50
INPUT_CHANNELS = 6
NUM_CLASSES = 7
ENTROPY_WEIGHT = 0.01
MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.30

ESP32_ACTIVITY_LABELS = [
    "Walking", "Lying Down", "Stairs Up", "Stairs Down",
    "Jogging", "Cycling", "Running",
]

CHECKPOINT_DIR = "checkpoints_v2"
EXPORT_DIR = "exports/esp32_v2"
OUTPUT_DIR = "outputs/esp32_v2"
MODEL_TAG = "sensorfusion_esp32_v2_useful11_rw"

UCI_TOTAL_SIGNAL_FILES = [
    "total_acc_x_{}.txt",
    "total_acc_y_{}.txt",
    "total_acc_z_{}.txt",
    "body_gyro_x_{}.txt",
    "body_gyro_y_{}.txt",
    "body_gyro_z_{}.txt",
]

MHEALTH_URLS = [
    "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip",
]

MHEALTH_TO_MERGED = {
    3: 1,
    4: 0,
    5: 2,
    9: 5,
    10: 4,
    11: 6,
}

MHEALTH_SENSOR_COLS = [14, 15, 16, 17, 18, 19]

REALWORLD_TO_MERGED = {
    "walking": 0,
    "lying": 1,
    "climbingup": 2,
    "climbingdown": 3,
    "running": 6,
}

# UCIHAR raw labels: 0:Walking, 1:Walking_Up, 2:Walking_Down, 3:Sitting(skip), 4:Standing(skip), 5:Laying
UCIHAR_TO_MERGED = {0: 0, 1: 2, 2: 3, 5: 1}

# PAMAP2 internal IDs (after dataset class re-indexes raw activity codes)
#  0 Lying, 1 Sitting(skip), 2 Standing(skip), 3 Walking, 4 Running, 5 Cycling,
#  6 Nordic Walking (skip), 7 Ascending Stairs, 8 Descending Stairs,
#  9 Vacuum Cleaning(skip), 10 Ironing(skip), 11 Rope Jumping(skip)
PAMAP2_TO_MERGED = {
    0: 1,
    3: 0,
    4: 6,
    5: 5,
    7: 2,
    8: 3,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class HARWindowDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X.float()
        self.y = y.long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


UCI_G_TO_MS2 = 9.81


class UCITotalHARDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train") -> None:
        assert split in ("train", "test")
        signals = []
        for template in UCI_TOTAL_SIGNAL_FILES:
            path = os.path.join(root_dir, split, "Inertial Signals", template.format(split))
            signals.append(np.loadtxt(path))
        raw = np.stack(signals, axis=-1).astype(np.float32)
        raw[:, :, :3] *= UCI_G_TO_MS2
        self.X = torch.tensor(raw, dtype=torch.float32)
        labels = np.loadtxt(os.path.join(root_dir, split, f"y_{split}.txt"), dtype=int)
        self.y = torch.tensor(labels - 1, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def downsample_windows(X: torch.Tensor, target_len: int = 50) -> torch.Tensor:
    if X.shape[1] == target_len:
        return X.float()
    X_t = X.permute(0, 2, 1).float()
    X_down = F.interpolate(X_t, size=target_len, mode="linear", align_corners=False)
    return X_down.permute(0, 2, 1).contiguous()


def downsample_np_windows(X: np.ndarray, target_len: int = 50) -> np.ndarray:
    if X.shape[1] == target_len:
        return X.astype(np.float32)
    x_t = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
    x_down = F.interpolate(x_t, size=target_len, mode="linear", align_corners=False)
    return x_down.permute(0, 2, 1).numpy().astype(np.float32)


def mhealth_windows_from_subject_array(
    raw: np.ndarray,
    mapping: dict[int, int],
    target_len: int = TARGET_TIME_STEPS,
    window_size: int = 100,
    step_size: int = 50,
    purity: float = 0.80,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] < 21:
        return np.empty((0, target_len, INPUT_CHANNELS), dtype=np.float32), np.empty((0,), dtype=np.int64)
    sensor = raw[:, MHEALTH_SENSOR_COLS].copy()
    labels = raw[:, -1].astype(np.int64)
    for col in range(sensor.shape[1]):
        mask = ~np.isfinite(sensor[:, col])
        if mask.any():
            valid = np.where(~mask)[0]
            sensor[mask, col] = np.interp(np.where(mask)[0], valid, sensor[valid, col]) if len(valid) > 1 else 0.0
    windows, y = [], []
    for start in range(0, len(sensor) - window_size + 1, step_size):
        end = start + window_size
        segment_labels = labels[start:end]
        unique, counts = np.unique(segment_labels, return_counts=True)
        dominant = int(unique[np.argmax(counts)])
        if dominant == 0 or dominant not in mapping:
            continue
        if counts.max() < window_size * purity:
            continue
        windows.append(sensor[start:end])
        y.append(mapping[dominant])
    if not windows:
        return np.empty((0, target_len, INPUT_CHANNELS), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return downsample_np_windows(np.stack(windows), target_len), np.asarray(y, dtype=np.int64)


class MHEALTHDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train") -> None:
        assert split in ("train", "test")
        subjects = range(1, 8) if split == "train" else range(8, 11)
        windows, labels = [], []
        for subject in subjects:
            path = self._subject_path(root_dir, subject)
            if not path:
                continue
            raw = np.loadtxt(path)
            X, y = mhealth_windows_from_subject_array(raw, MHEALTH_TO_MERGED)
            if len(y):
                windows.append(torch.tensor(X, dtype=torch.float32))
                labels.append(torch.tensor(y, dtype=torch.long))
        if windows:
            self.X = torch.cat(windows, dim=0)
            self.y = torch.cat(labels, dim=0)
        else:
            self.X = torch.empty((0, TARGET_TIME_STEPS, INPUT_CHANNELS), dtype=torch.float32)
            self.y = torch.empty((0,), dtype=torch.long)

    @staticmethod
    def _subject_path(root_dir: str, subject: int) -> str | None:
        candidates = [
            os.path.join(root_dir, f"mHealth_subject{subject}.log"),
            os.path.join(root_dir, "MHEALTHDATASET", f"mHealth_subject{subject}.log"),
            os.path.join(root_dir, "MHEALTHDATASET", f"mHealth_subject{subject}.txt"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        for current_root, _, files in os.walk(root_dir):
            for name in files:
                if name.lower() == f"mhealth_subject{subject}.log":
                    return os.path.join(current_root, name)
        return None

    @staticmethod
    def download(dest_dir: str) -> str:
        os.makedirs(dest_dir, exist_ok=True)
        target = os.path.join(dest_dir, "MHEALTHDATASET")
        if os.path.isdir(target):
            return target
        last_error = None
        for url in MHEALTH_URLS:
            zip_path = os.path.join(dest_dir, "mhealth_dataset.zip")
            try:
                urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(dest_dir)
                os.remove(zip_path)
                return target if os.path.isdir(target) else dest_dir
            except Exception as exc:
                last_error = exc
                if os.path.exists(zip_path):
                    os.remove(zip_path)
        raise RuntimeError(f"Unable to download MHEALTH from configured URLs: {last_error}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _interpolate_nonfinite(sensor: np.ndarray) -> np.ndarray:
    sensor = np.asarray(sensor, dtype=np.float32).copy()
    for col in range(sensor.shape[1]):
        mask = ~np.isfinite(sensor[:, col])
        if mask.any():
            valid = np.where(~mask)[0]
            sensor[mask, col] = np.interp(np.where(mask)[0], valid, sensor[valid, col]) if len(valid) > 1 else 0.0
    return sensor


def _load_realworld_zip_xyz(zip_path: str, position: str = "waist") -> np.ndarray:
    if not os.path.exists(zip_path):
        return np.empty((0, 3), dtype=np.float32)
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            return np.empty((0, 3), dtype=np.float32)
        preferred = [n for n in csv_names if position.lower() in n.lower()]
        name = preferred[0] if preferred else csv_names[0]
        with zf.open(name) as f:
            arr = np.genfromtxt(f, delimiter=",", names=True, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    arr = np.atleast_1d(arr)
    data = np.column_stack([arr["attr_x"], arr["attr_y"], arr["attr_z"]]).astype(np.float32)
    return _interpolate_nonfinite(data)


def realworld_zip_pair_to_windows(
    acc_zip_path: str,
    gyr_zip_path: str,
    label: int,
    target_len: int = TARGET_TIME_STEPS,
    window_size: int = 100,
    step_size: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    acc = _load_realworld_zip_xyz(acc_zip_path)
    gyr = _load_realworld_zip_xyz(gyr_zip_path)
    n = min(len(acc), len(gyr))
    if n < window_size:
        return np.empty((0, target_len, INPUT_CHANNELS), dtype=np.float32), np.empty((0,), dtype=np.int64)
    sensor = np.concatenate([acc[:n], gyr[:n]], axis=1)
    windows = []
    for start in range(0, n - window_size + 1, step_size):
        windows.append(sensor[start:start + window_size])
    if not windows:
        return np.empty((0, target_len, INPUT_CHANNELS), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X = downsample_np_windows(np.stack(windows), target_len)
    y = np.full((len(X),), int(label), dtype=np.int64)
    return X, y


class RealWorldLocalDataset(Dataset):
    def __init__(self, data_root: str) -> None:
        windows, labels = [], []
        roots = [
            Path(data_root) / "RealWorldHAR",
            Path(data_root) / "RealWorldHAR_sample",
        ]
        for root in roots:
            if not root.exists():
                continue
            for activity, label in REALWORLD_TO_MERGED.items():
                for acc_zip in root.rglob(f"acc_{activity}_csv.zip"):
                    gyr_zip = acc_zip.with_name(f"gyr_{activity}_csv.zip")
                    if not gyr_zip.exists():
                        continue
                    X, y = realworld_zip_pair_to_windows(str(acc_zip), str(gyr_zip), label)
                    if len(y):
                        windows.append(torch.tensor(X, dtype=torch.float32))
                        labels.append(torch.tensor(y, dtype=torch.long))
        if windows:
            self.X = torch.cat(windows, dim=0)
            self.y = torch.cat(labels, dim=0)
        else:
            self.X = torch.empty((0, TARGET_TIME_STEPS, INPUT_CHANNELS), dtype=torch.float32)
            self.y = torch.empty((0,), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def append_mapped_dataset(all_X, all_y, ds, mapping, target_len):
    X = downsample_windows(ds.X, target_len)
    for src_cls, dst_cls in mapping.items():
        mask = ds.y == src_cls
        count = int(mask.sum().item())
        if count > 0:
            all_X.append(X[mask])
            all_y.append(torch.full((count,), dst_cls, dtype=torch.long))


def build_merged_dataset(
    data_root: str = "data",
    include_mhealth: bool = True,
    include_realworld: bool = True,
):
    print("Loading UCI-HAR, PAMAP2, optional MHEALTH, and optional RealWorld HAR...")
    uci_dir = os.path.join(data_root, "UCI HAR Dataset")
    p2_dir = os.path.join(data_root, "PAMAP2_Dataset")
    mh_dir = os.path.join(data_root, "MHEALTHDATASET")

    ucihar_train = UCITotalHARDataset(uci_dir, split="train")
    ucihar_test = UCITotalHARDataset(uci_dir, split="test")
    pamap2_train = PAMAP2Dataset(p2_dir, split="train")
    pamap2_test = PAMAP2Dataset(p2_dir, split="test")
    mhealth_train = mhealth_test = None
    if include_mhealth:
        if not os.path.isdir(mh_dir):
            print("  MHEALTH not found; downloading from URL...")
            mh_dir = MHEALTHDataset.download(data_root)
        mhealth_train = MHEALTHDataset(mh_dir, split="train")
        mhealth_test = MHEALTHDataset(mh_dir, split="test")
    realworld = RealWorldLocalDataset(data_root) if include_realworld else None

    print(f"  UCI-HAR(total acc): {len(ucihar_train)} train + {len(ucihar_test)} test")
    print(f"  PAMAP2:  {len(pamap2_train)} train + {len(pamap2_test)} test")
    if include_mhealth:
        print(f"  MHEALTH: {len(mhealth_train)} train + {len(mhealth_test)} test")
    if include_realworld:
        print(f"  RealWorld HAR local: {len(realworld)} windows")

    all_X, all_y = [], []
    for ds in [ucihar_train, ucihar_test]:
        append_mapped_dataset(all_X, all_y, ds, UCIHAR_TO_MERGED, TARGET_TIME_STEPS)
    for ds in [pamap2_train, pamap2_test]:
        append_mapped_dataset(all_X, all_y, ds, PAMAP2_TO_MERGED, TARGET_TIME_STEPS)
    if include_mhealth:
        for ds in [mhealth_train, mhealth_test]:
            append_mapped_dataset(all_X, all_y, ds, {i: i for i in range(NUM_CLASSES)}, TARGET_TIME_STEPS)
    if include_realworld and len(realworld):
        append_mapped_dataset(all_X, all_y, realworld, {i: i for i in range(NUM_CLASSES)}, TARGET_TIME_STEPS)

    X_all = torch.cat(all_X, dim=0)
    y_all = torch.cat(all_y, dim=0)
    print(f"  merged windows: {len(X_all)}, shape: {X_all.shape}")

    indices = np.arange(len(y_all))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=SEED, stratify=y_all.numpy()
    )
    X_train_raw, y_train = X_all[train_idx], y_all[train_idx]
    X_test_raw, y_test = X_all[test_idx], y_all[test_idx]

    mean = X_train_raw.mean(dim=(0, 1))
    std = X_train_raw.std(dim=(0, 1))
    std[std < 1e-8] = 1.0

    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    train_ds = HARWindowDataset(X_train, y_train)
    test_ds = HARWindowDataset(X_test, y_test)

    norm_stats = {
        "mean": [float(v) for v in mean.tolist()],
        "std": [float(v) for v in std.tolist()],
        "target_time_steps": TARGET_TIME_STEPS,
        "input_channels": INPUT_CHANNELS,
        "num_classes": NUM_CLASSES,
        "labels": ESP32_ACTIVITY_LABELS,
    }
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(EXPORT_DIR, "normalization_stats.json"), "w") as f:
        json.dump(norm_stats, f, indent=2)
    manifest = {
        "datasets": ["UCI-HAR total_acc+gyro", "PAMAP2 hand acc+gyro"]
        + (["MHEALTH right-arm acc+gyro"] if include_mhealth else [])
        + (["RealWorld HAR waist acc+gyro local activity zips"] if include_realworld and len(realworld) else []),
        "mhealth_mapping": {str(k): ESP32_ACTIVITY_LABELS[v] for k, v in MHEALTH_TO_MERGED.items()},
        "realworld_mapping": {k: ESP32_ACTIVITY_LABELS[v] for k, v in REALWORLD_TO_MERGED.items()},
        "labels": ESP32_ACTIVITY_LABELS,
    }
    with open(os.path.join(OUTPUT_DIR, "dataset_manifest_v2.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nClass distribution (train / test):")
    for i, label in enumerate(ESP32_ACTIVITY_LABELS):
        tr = int((train_ds.y == i).sum().item())
        te = int((test_ds.y == i).sum().item())
        print(f"  {i:2d} {label:<18s} {tr:6d} / {te:5d}")
    return train_ds, test_ds, norm_stats, mean, std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        return weight.sign().masked_fill(weight == 0, 1.0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ScaledBinaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scale = nn.Parameter(torch.ones(out_features, 1))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        bw = BinarizeSTE.apply(self.weight) * self.scale.abs().clamp_min(1e-4)
        return F.linear(x, bw, self.bias)


class EchoStateNetworkEdge(nn.Module):
    def __init__(self, input_channels=6, reservoir_size=32,
                 spectral_radius=0.9, sparsity=0.80, dropout=0.10):
        super().__init__()
        self.reservoir_size = reservoir_size
        self.dropout = dropout
        W_in = torch.randn(input_channels, reservoir_size) * 0.10
        W_res = torch.randn(reservoir_size, reservoir_size)
        W_res = W_res * (torch.rand(reservoir_size, reservoir_size) > sparsity).float()
        radius = torch.linalg.eigvals(W_res).abs().max().item()
        if radius > 0:
            W_res = W_res * (spectral_radius / radius)
        self.register_buffer("W_in", W_in)
        self.register_buffer("W_res", W_res)
        self.register_buffer("_base_sr", torch.tensor(float(spectral_radius)))
        init_logit = math.log(spectral_radius / (1.0 - spectral_radius + 1e-7))
        self.sr_logit = nn.Parameter(torch.tensor(init_logit))

    @property
    def effective_spectral_radius(self):
        return torch.sigmoid(self.sr_logit)

    def _scaled_reservoir_weights(self):
        return self.W_res * (self.effective_spectral_radius / (self._base_sr + 1e-7))

    def forward(self, x):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.reservoir_size, device=x.device, dtype=x.dtype)
        prev = h
        W_r = self._scaled_reservoir_weights()
        x_proj = x @ self.W_in
        states = []
        diffs = []
        if self.training and self.dropout > 0:
            keep = (torch.rand(batch, 1, self.reservoir_size, device=x.device) > self.dropout).float()
            keep = keep / (1.0 - self.dropout)
        else:
            keep = None
        for t in range(seq_len):
            h = torch.tanh(x_proj[:, t] + h @ W_r)
            h_out = h * keep[:, 0, :] if keep is not None else h
            states.append(h_out.unsqueeze(1))
            diffs.append((h_out - prev).unsqueeze(1))
            prev = h_out
        return torch.cat([torch.cat(states, dim=1), torch.cat(diffs, dim=1)], dim=2)


class DSConvEncoderEdge(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 5, padding=2, groups=in_channels, bias=False),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),
            nn.Conv1d(out_channels, out_channels, 5, stride=2, padding=2, groups=out_channels, bias=False),
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleTemporalEdge(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.dw3 = nn.Conv1d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv1d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.dw9 = nn.Conv1d(channels, channels, 9, padding=4, groups=channels, bias=False)
        self.mix = nn.Sequential(
            nn.Conv1d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.mix(torch.cat([self.dw3(x), self.dw5(x), self.dw9(x)], dim=1)) + x


class EdgeSpectralGatedFusion(nn.Module):
    def __init__(self, reservoir_dim=64, channels=64, seq_len=25):
        super().__init__()
        self.channel_proj = nn.Conv1d(reservoir_dim, channels, 1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(seq_len)
        self.band_gate = nn.Linear(2, 1)

    def forward(self, reservoir_out, dsconv_out):
        res = self.temporal_pool(self.channel_proj(reservoir_out))
        low = F.avg_pool1d(dsconv_out, kernel_size=5, stride=1, padding=2)
        high = dsconv_out - low
        band_energy = torch.stack([low.abs().mean(dim=2), high.abs().mean(dim=2)], dim=-1)
        gate = torch.sigmoid(self.band_gate(band_energy))
        return dsconv_out + gate * res


class PatchMicroAttentionEdge(nn.Module):
    def __init__(self, in_channels=64, seq_len=25, patch_len=5, d_model=64, attn_drop=0.10):
        super().__init__()
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.patch_proj = nn.Linear(in_channels * patch_len, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(attn_drop)
        self.scale = d_model ** -0.5

    def forward(self, x, return_attention=False):
        b, c, t = x.shape
        usable_t = self.num_patches * self.patch_len
        x = x[:, :, :usable_t]
        patches = x.reshape(b, c, self.num_patches, self.patch_len).permute(0, 2, 1, 3)
        patches = patches.reshape(b, self.num_patches, c * self.patch_len)
        z = self.patch_proj(patches)
        q, k, v = self.q(z), self.k(z), self.v(z)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        z = self.norm(self.out(attn @ v) + z)
        pooled = z.mean(dim=1)
        entropy = -(attn * torch.log(attn.clamp_min(1e-8))).sum(dim=-1).mean()
        if return_attention:
            return pooled, attn, entropy
        return pooled


class OrientationMicroPath(nn.Module):
    def __init__(self, input_channels=6, d_model=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_channels * 3 + 9, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        acc = x[:, :, :3]
        stats = [
            x.mean(dim=1),
            x.std(dim=1),
            torch.sqrt((x ** 2).mean(dim=1).clamp_min(1e-8)),
            acc.mean(dim=1),
            acc.std(dim=1),
            torch.sqrt(((acc[:, 1:] - acc[:, :-1]) ** 2).mean(dim=1).clamp_min(1e-8)),
        ]
        return self.proj(torch.cat(stats, dim=1))


class SensorFusionESP32(nn.Module):
    def __init__(self, input_channels=6, reservoir_size=64, num_classes=11):
        super().__init__()
        self.reservoir_size = reservoir_size
        self.reservoir = EchoStateNetworkEdge(input_channels, reservoir_size)
        self.diff_gate = nn.Parameter(torch.zeros(reservoir_size))
        self.dsconv = DSConvEncoderEdge(reservoir_size, 64)
        self.multiscale = MultiScaleTemporalEdge(64)
        self.gate = EdgeSpectralGatedFusion(reservoir_size, 64, seq_len=TARGET_TIME_STEPS // 2)
        self.attention = PatchMicroAttentionEdge(64, seq_len=TARGET_TIME_STEPS // 2,
                                                  patch_len=5, d_model=64)
        self.orientation = OrientationMicroPath(input_channels, d_model=64)
        self.feature_fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.LayerNorm(64),
        )
        self.pre_cls_drop = nn.Dropout(0.25)
        self.classifier_bn = nn.BatchNorm1d(64)
        self.classifier = ScaledBinaryLinear(64, num_classes)

    def _merge_reservoir_states(self, h):
        rs = self.reservoir_size
        alpha = torch.sigmoid(self.diff_gate).view(1, 1, rs)
        return h[:, :, :rs] + alpha * h[:, :, rs:]

    def _features(self, h, x=None, return_aux=False):
        h = self._merge_reservoir_states(h)
        h_t = h.transpose(1, 2)
        ds = self.dsconv(h_t)
        ds = self.multiscale(ds)
        fused = self.gate(h_t, ds)
        if return_aux:
            feats, attn, entropy = self.attention(fused, return_attention=True)
            if x is not None:
                feats = self.feature_fusion(torch.cat([feats, self.orientation(x)], dim=1))
            return feats, {
                "attention_entropy": entropy,
                "attention_weights": attn,
                "features": feats,
                "spectral_radius": self.reservoir.effective_spectral_radius,
            }
        feats = self.attention(fused)
        if x is not None:
            feats = self.feature_fusion(torch.cat([feats, self.orientation(x)], dim=1))
        return self.pre_cls_drop(feats)

    def forward(self, x, return_aux=False):
        h = self.reservoir(x)
        if return_aux:
            feats, aux = self._features(h, x=x, return_aux=True)
            out = self.classifier(self.classifier_bn(feats))
            return out, aux
        feats = self._features(h, x=x)
        return self.classifier(self.classifier_bn(feats))

    def forward_eval(self, x):
        """Inference without dropout — sets eval mode internally."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
        if was_training:
            self.train()
        return out

    def reservoir_states(self, x):
        return self.reservoir(x)

    def forward_from_reservoir(self, h):
        feats = self._features(h)
        return self.classifier(self.classifier_bn(feats))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_kb(self):
        return self.count_parameters() * 4 / 1024


class MaskedSensorModelESP32(nn.Module):
    def __init__(self, input_channels=6, reservoir_size=64, mask_ratio=0.20):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, input_channels))
        self.backbone_reservoir = EchoStateNetworkEdge(input_channels, reservoir_size)
        self.backbone_dsconv = DSConvEncoderEdge(reservoir_size, 64)
        self.reconstruction_head = nn.Linear(64, input_channels)

    def forward(self, x, mask=None):
        b, t, c = x.shape
        if mask is None:
            n_mask = max(1, int(t * self.mask_ratio))
            mask = torch.zeros(b, t, device=x.device)
            for i in range(b):
                mask[i, torch.randperm(t, device=x.device)[:n_mask]] = 1.0
        x_masked = x * (1.0 - mask.unsqueeze(-1)) + self.mask_token * mask.unsqueeze(-1)
        h = self.backbone_reservoir(x_masked)
        rs = self.backbone_reservoir.reservoir_size
        h = h[:, :, :rs]
        ds = self.backbone_dsconv(h.transpose(1, 2))
        recon_features = F.interpolate(ds, size=t, mode="linear", align_corners=False)
        recon = self.reconstruction_head(recon_features.transpose(1, 2))
        return recon, mask


def transfer_msm_weights(pretrained_msm, target_model):
    target_model.reservoir.load_state_dict(
        pretrained_msm.backbone_reservoir.state_dict(), strict=False
    )
    target_model.dsconv.load_state_dict(
        pretrained_msm.backbone_dsconv.state_dict(), strict=False
    )
    return target_model


# ---------------------------------------------------------------------------
# Loss / aug / sampler
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.register_buffer(
            "alpha", alpha if alpha is not None else torch.ones(NUM_CLASSES)
        )
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce.detach())
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class PrototypeMarginLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, feature_dim=64, margin=1.0):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim) * 0.05)
        self.margin = margin

    def forward(self, features, targets):
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        distances = torch.cdist(features, prototypes, p=2)
        own = distances.gather(1, targets.view(-1, 1)).squeeze(1)
        masked = distances + F.one_hot(targets, distances.size(1)).float() * 1e6
        nearest_other = masked.min(dim=1).values
        return own.mean() + F.relu(self.margin + own - nearest_other).mean()


def _random_rotation_matrix():
    angles = [random.uniform(0, 2 * math.pi) for _ in range(3)]
    ca, sa = math.cos(angles[0]), math.sin(angles[0])
    cb, sb = math.cos(angles[1]), math.sin(angles[1])
    cg, sg = math.cos(angles[2]), math.sin(angles[2])
    Rx = torch.tensor([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=torch.float32)
    Ry = torch.tensor([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=torch.float32)
    Rz = torch.tensor([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]], dtype=torch.float32)
    return Rz @ Ry @ Rx


def augment_batch(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    noise = torch.randn_like(x) * 0.03
    scale = 1.0 + torch.randn(x.size(0), 1, x.size(2), device=x.device) * 0.08
    x_aug = (x + noise) * scale
    if random.random() < 0.20:
        ch = torch.randint(0, x.size(2), (1,), device=x.device).item()
        x_aug[:, :, ch] = 0.0
    if random.random() < 0.30:
        R = _random_rotation_matrix().to(x_aug.device)
        x_aug[:, :, :3] = x_aug[:, :, :3] @ R.T
        x_aug[:, :, 3:] = x_aug[:, :, 3:] @ R.T
    return x_aug


class AugmentedHARWindowDataset(Dataset):
    def __init__(self, base_ds, minority_classes, p_major=0.35, p_minor=0.65):
        self.base_ds = base_ds
        self.minority_classes = set(int(c) for c in minority_classes)
        self.p_major = p_major
        self.p_minor = p_minor
        self.X = base_ds.X
        self.y = base_ds.y

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        p = self.p_minor if int(y) in self.minority_classes else self.p_major
        if random.random() < p:
            x = augment_batch(x.unsqueeze(0)).squeeze(0)
        return x, y


def make_balanced_loader(dataset, batch_size=128):
    counts = torch.bincount(dataset.y, minlength=NUM_CLASSES).float()
    inv = 1.0 / counts.clamp_min(1.0)
    sample_weights = inv[dataset.y]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      drop_last=True, num_workers=0), counts


def reservoir_manifold_mixup_edge(model, x1, x2, y1, y2, criterion, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    h1 = model.reservoir_states(x1)
    h2 = model.reservoir_states(x2)
    h_mix = lam * h1 + (1.0 - lam) * h2
    logits = model.forward_from_reservoir(h_mix)
    return lam * criterion(logits, y1) + (1.0 - lam) * criterion(logits, y2)


def evaluate_model(model, loader, device, criterion=None):
    model.eval()
    preds, labels = [], []
    total_loss, batches = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            if criterion is not None:
                total_loss += float(criterion(logits, yb).item())
                batches += 1
            preds.append(logits.argmax(1).cpu())
            labels.append(yb.cpu())
    y_true = torch.cat(labels).numpy()
    y_pred = torch.cat(preds).numpy()
    acc = float((y_true == y_pred).mean())
    per_f1 = f1_score(y_true, y_pred, average=None,
                      labels=list(range(NUM_CLASSES)), zero_division=0)
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {
        "acc": acc,
        "macro_f1": macro,
        "min_f1": float(per_f1.min()),
        "per_class_f1": per_f1,
        "y_true": y_true,
        "y_pred": y_pred,
        "loss": total_loss / max(batches, 1),
    }


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msm-epochs", type=int, default=10)
    ap.add_argument("--head-epochs", type=int, default=50)
    ap.add_argument("--ft-epochs", type=int, default=0)
    ap.add_argument("--patience", type=int, default=10,
                    help="Early-stopping patience for head training.")
    ap.add_argument("--swa-start", type=int, default=30,
                    help="Epoch from which SWA averaging begins (head phase).")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--reservoir-size", type=int, default=64)
    ap.add_argument("--focal-gamma", type=float, default=1.5)
    ap.add_argument("--label-smoothing", type=float, default=0.02)
    ap.add_argument("--prototype-weight", type=float, default=0.01)
    ap.add_argument("--prototype-warmup", type=int, default=5)
    ap.add_argument("--no-mhealth", action="store_true")
    ap.add_argument("--no-realworld", action="store_true")
    ap.add_argument("--artifact-suffix", type=str, default="pocket_final")
    ap.add_argument("--dataset-only", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny budget to verify the pipeline.")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch.set_num_threads (0 = default)")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    artifact_suffix = args.artifact_suffix.strip()
    model_tag = f"sensorfusion_esp32_v2_{artifact_suffix}"

    if args.smoke:
        args.msm_epochs = 2
        args.head_epochs = 4
        args.ft_epochs = 0
        args.patience = 99
        args.swa_start = 99

    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, torch={torch.__version__}, threads={torch.get_num_threads()}")

    train_ds, test_ds, norm_stats, mean, std = build_merged_dataset(
        include_mhealth=not args.no_mhealth,
        include_realworld=not args.no_realworld,
    )
    if args.dataset_only:
        print("Dataset-only check complete.")
        return

    minority_threshold = 2000
    counts_train = torch.bincount(train_ds.y, minlength=NUM_CLASSES)
    minority = [i for i, c in enumerate(counts_train.tolist()) if c < minority_threshold]
    print("Minority classes:", [ESP32_ACTIVITY_LABELS[i] for i in minority])

    aug_train_ds = AugmentedHARWindowDataset(train_ds, minority_classes=minority)
    train_loader, counts = make_balanced_loader(aug_train_ds, args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    alpha = counts.sum() / (NUM_CLASSES * counts.clamp_min(1.0))
    alpha = (alpha / alpha.mean()).to(device)
    criterion = FocalLoss(
        alpha=alpha, gamma=args.focal_gamma, label_smoothing=args.label_smoothing
    ).to(device)
    prototype_loss = PrototypeMarginLoss(NUM_CLASSES, 64).to(device)

    # ---------- MSM pre-training ----------
    print("\n[MSM pre-training]")
    msm_model = MaskedSensorModelESP32(
        input_channels=INPUT_CHANNELS, reservoir_size=args.reservoir_size, mask_ratio=0.20
    ).to(device)
    msm_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             drop_last=True, num_workers=0)
    msm_opt = torch.optim.AdamW(msm_model.parameters(), lr=3e-4, weight_decay=1e-4)
    msm_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        msm_opt, T_max=max(args.msm_epochs, 1)
    )

    t0 = time.time()
    for epoch in range(1, args.msm_epochs + 1):
        msm_model.train()
        total, batches = 0.0, 0
        for xb, _ in msm_loader:
            xb = xb.to(device)
            recon, mask = msm_model(xb)
            mask_e = mask.unsqueeze(-1)
            loss = (((recon - xb) ** 2) * mask_e).sum() / (
                mask_e.sum().clamp_min(1.0) * xb.size(-1)
            )
            msm_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(msm_model.parameters(), 1.0)
            msm_opt.step()
            total += float(loss.item())
            batches += 1
        msm_sched.step()
        print(f"  MSM epoch {epoch:02d}/{args.msm_epochs} loss={total/max(batches,1):.5f}")
    print(f"  MSM pre-training: {(time.time() - t0)/60:.1f} min")
    torch.save(msm_model.state_dict(), os.path.join(CHECKPOINT_DIR, "esp32_v2_msm_pretrain.pt"))

    # ---------- Build supervised model ----------
    model = SensorFusionESP32(
        input_channels=INPUT_CHANNELS, reservoir_size=args.reservoir_size, num_classes=NUM_CLASSES
    ).to(device)
    model = transfer_msm_weights(msm_model, model)
    print(f"\nSupervised model parameters: {model.count_parameters():,}")
    print(f"FP32 size: {model.model_size_kb():.1f} KB")

    # ---------- Stage 1: head training with early stopping + SWA ----------
    print("\n[Stage 1: head training]")
    for name, p in model.named_parameters():
        p.requires_grad = not (name.startswith("reservoir") or name.startswith("dsconv"))
    head_opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4
    )
    head_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        head_opt, T_0=15, T_mult=1, eta_min=1e-5
    )
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_sched = torch.optim.swa_utils.SWALR(head_opt, swa_lr=2e-4, anneal_epochs=5)
    swa_started = False

    history = []
    best_score, best_metrics = -1.0, None
    patience_counter = 0

    for epoch in range(1, args.head_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            head_opt.zero_grad()
            logits, aux = model(xb, return_aux=True)
            loss = criterion(logits, yb) - ENTROPY_WEIGHT * aux["attention_entropy"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            head_opt.step()

        if epoch >= args.swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
            swa_started = True
        else:
            head_sched.step()

        m = evaluate_model(model, test_loader, device, criterion)
        score = m["min_f1"] * 4.0 + m["macro_f1"] + m["acc"]
        history.append({
            "stage": "head",
            "epoch": epoch,
            "acc": m["acc"],
            "macro_f1": m["macro_f1"],
            "min_f1": m["min_f1"],
            "loss": m["loss"],
        })
        if score > best_score:
            best_score = score
            best_metrics = copy.deepcopy(m)
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "prototype_state_dict": prototype_loss.state_dict(),
                "metrics": {"acc": m["acc"], "macro_f1": m["macro_f1"], "min_f1": m["min_f1"]},
                "labels": ESP32_ACTIVITY_LABELS,
                "normalization": norm_stats,
                "stage": "head",
                "epoch": epoch,
            }, os.path.join(CHECKPOINT_DIR, f"best_{model_tag}.pt"))
        else:
            patience_counter += 1
        print(f"  Head {epoch:02d}/{args.head_epochs} acc={m['acc']:.4f} macroF1={m['macro_f1']:.4f} minF1={m['min_f1']:.4f} patience={patience_counter}/{args.patience}")
        if patience_counter >= args.patience:
            print(f"  [Early stopping] No improvement for {args.patience} epochs.")
            break

    if swa_started:
        print("\n[SWA: updating BatchNorm statistics]")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_m = evaluate_model(swa_model, test_loader, device, criterion)
        swa_score = swa_m["min_f1"] * 4.0 + swa_m["macro_f1"] + swa_m["acc"]
        print(f"  SWA: acc={swa_m['acc']:.4f} macroF1={swa_m['macro_f1']:.4f} minF1={swa_m['min_f1']:.4f}")
        if swa_score > best_score:
            best_score = swa_score
            best_metrics = copy.deepcopy(swa_m)
            torch.save({
                "model_state_dict": swa_model.module.state_dict(),
                "prototype_state_dict": prototype_loss.state_dict(),
                "metrics": {"acc": swa_m["acc"], "macro_f1": swa_m["macro_f1"], "min_f1": swa_m["min_f1"]},
                "labels": ESP32_ACTIVITY_LABELS,
                "normalization": norm_stats,
                "stage": "swa",
                "epoch": epoch,
            }, os.path.join(CHECKPOINT_DIR, f"best_{model_tag}.pt"))
            print("  SWA checkpoint is the new best!")

    # ---------- Final eval ----------
    print("\n[Final evaluation]")
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, f"best_{model_tag}.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final = evaluate_model(model, test_loader, device, criterion)
    print(f"Accuracy: {final['acc']:.4f}")
    print(f"Macro F1: {final['macro_f1']:.4f}")
    print(f"Min F1:   {final['min_f1']:.4f}")
    print("\nPer-class F1:")
    for label, val in zip(ESP32_ACTIVITY_LABELS, final["per_class_f1"]):
        flag = "OK" if val >= 0.87 else "CHECK"
        print(f"  {label:<18s} {val:.4f}  {flag}")
    print("\n" + classification_report(
        final["y_true"], final["y_pred"], target_names=ESP32_ACTIVITY_LABELS,
        digits=4, zero_division=0
    ))

    cm = confusion_matrix(final["y_true"], final["y_pred"], labels=list(range(NUM_CLASSES)))
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    ConfusionMatrixDisplay(cm, display_labels=ESP32_ACTIVITY_LABELS).plot(
        ax=axes[0], cmap="inferno", colorbar=True
    )
    axes[0].set_title(f"V2 {artifact_suffix} Confusion Matrix")
    axes[0].set_xticklabels(ESP32_ACTIVITY_LABELS, rotation=45, ha="right")
    bars = axes[1].bar(ESP32_ACTIVITY_LABELS, final["per_class_f1"],
                        color="#2196F3", edgecolor="black", alpha=0.85)
    axes[1].axhline(0.87, color="red", linestyle="--", label="Target 0.87")
    for bar, val in zip(bars, final["per_class_f1"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                      min(1.03, val + 0.015), f"{val:.2f}",
                      ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Per-Class F1")
    axes[1].set_xticklabels(ESP32_ACTIVITY_LABELS, rotation=45, ha="right")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"v2_{artifact_suffix}_confusion_f1.png"), dpi=160, bbox_inches="tight")
    plt.close()

    summary = {
        "accuracy": final["acc"],
        "macro_f1": final["macro_f1"],
        "min_f1": final["min_f1"],
        "per_class_f1": {l: float(v) for l, v in zip(ESP32_ACTIVITY_LABELS, final["per_class_f1"])},
        "parameter_count": model.count_parameters(),
        "fp32_size_kb": model.model_size_kb(),
        "history": history,
        "config": {
            "msm_epochs": args.msm_epochs,
            "head_epochs": args.head_epochs,
            "ft_epochs": args.ft_epochs,
            "batch_size": args.batch_size,
            "include_mhealth": not args.no_mhealth,
            "include_realworld": not args.no_realworld,
            "artifact_suffix": artifact_suffix,
            "prototype_weight": args.prototype_weight,
            "prototype_warmup": args.prototype_warmup,
        },
    }
    with open(os.path.join(OUTPUT_DIR, f"summary_v2_{artifact_suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---------- ONNX export ----------
    print("\n[ONNX export]")
    try:
        import onnx
        model_cpu = model.cpu().eval()
        onnx_path = os.path.join(EXPORT_DIR, f"{model_tag}.onnx")
        dummy = torch.randn(1, TARGET_TIME_STEPS, INPUT_CHANNELS, dtype=torch.float32)
        torch.onnx.export(
            model_cpu, dummy, onnx_path, opset_version=18,
            input_names=["input"], output_names=["logits"], dynamic_axes=None,
        )
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX OK: {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)", flush=True)
    except Exception as onnx_err:
        print(f"[ONNX export skipped] {onnx_err}")
        model_cpu = model.cpu().eval()

    # ---------- Calibration data for Colab TFLite step ----------
    n_calib = min(256, len(train_ds))
    calib = train_ds.X[:n_calib].numpy().astype(np.float32)
    calib_path = os.path.join(EXPORT_DIR, "calib_data_v2.npy")
    np.save(calib_path, calib)
    print(f"Calibration data saved: {calib_path} {calib.shape}")

    print(f"\nDone. Best checkpoint: {os.path.join(CHECKPOINT_DIR, f'best_{model_tag}.pt')}")


if __name__ == "__main__":
    main()
