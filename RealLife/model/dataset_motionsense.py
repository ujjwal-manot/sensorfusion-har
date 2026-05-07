"""MotionSense dataset loader for pocket-phone HAR fine-tuning.

Source: https://github.com/mmalekzadeh/motion-sense
- iPhone 6 stored in participants' trousers' FRONT POCKET (identical placement
  to the target deployment scenario).
- 24 subjects, 15 trials per activity, 50 Hz.
- Folder A (DeviceMotion): attitude, gravity, userAcceleration, rotationRate
  We reconstruct total_acc = gravity + userAcceleration and use rotationRate as
  gyro — this exactly matches the [ax,ay,az,gx,gy,gz] convention used by the
  model and the phone WebSocket streamer.

Activity mapping to the 11-class label space:
  dws -> 5 (Stairs Down)
  ups -> 4 (Stairs Up)
  sit -> 1 (Sitting)
  std -> 2 (Standing)   <-- the class missing from pseudo-label fine-tuning
  wlk -> 0 (Walking)
  jog -> 6 (Jogging)
"""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

MOTIONSENSE_ZIP_URL = (
    "https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip"
)

MOTIONSENSE_TO_MERGED = {
    "dws": 5,  # Stairs Down
    "ups": 4,  # Stairs Up
    "sit": 1,  # Sitting
    "std": 2,  # Standing
    "wlk": 0,  # Walking
    "jog": 6,  # Jogging
}

_GRAVITY = 9.81  # MotionSense uses normalised g units; convert to m/s²

_ACCEL_COLS = [
    "userAcceleration.x", "userAcceleration.y", "userAcceleration.z",
    "gravity.x", "gravity.y", "gravity.z",
]
_GYRO_COLS = [
    "rotationRate.x", "rotationRate.y", "rotationRate.z",
]


def _parse_subject_csv(data: bytes) -> Optional[np.ndarray]:
    """Parse a MotionSense subject CSV and return (N,6) float32 [ax,ay,az,gx,gy,gz]."""
    try:
        import csv
        reader = csv.DictReader(io.TextIOWrapper(io.BytesIO(data)))
        rows = list(reader)
        if not rows:
            return None
        acc_x, acc_y, acc_z = [], [], []
        gx, gy, gz = [], [], []
        for r in rows:
            try:
                ua_x = float(r.get("userAcceleration.x", r.get("userAccelerationx", 0)))
                ua_y = float(r.get("userAcceleration.y", r.get("userAccelerationy", 0)))
                ua_z = float(r.get("userAcceleration.z", r.get("userAccelerationz", 0)))
                gv_x = float(r.get("gravity.x", r.get("gravityx", 0)))
                gv_y = float(r.get("gravity.y", r.get("gravityy", 0)))
                gv_z = float(r.get("gravity.z", r.get("gravityz", 0)))
                rx = float(r.get("rotationRate.x", r.get("rotationRatex", 0)))
                ry = float(r.get("rotationRate.y", r.get("rotationRatey", 0)))
                rz = float(r.get("rotationRate.z", r.get("rotationRatez", 0)))
                acc_x.append((ua_x + gv_x) * _GRAVITY)
                acc_y.append((ua_y + gv_y) * _GRAVITY)
                acc_z.append((ua_z + gv_z) * _GRAVITY)
                gx.append(rx)
                gy.append(ry)
                gz.append(rz)
            except (ValueError, TypeError):
                continue
        if len(acc_x) < 20:
            return None
        return np.column_stack([acc_x, acc_y, acc_z, gx, gy, gz]).astype(np.float32)
    except Exception:
        return None


def _windows_from_signal(
    signal: np.ndarray,
    label: int,
    window_size: int = 100,
    stride: int = 50,
    target_len: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide window over (N,6) signal, downsample to target_len."""
    if len(signal) < window_size:
        return np.empty((0, target_len, 6), dtype=np.float32), np.empty((0,), dtype=np.int64)
    wins = []
    for start in range(0, len(signal) - window_size + 1, stride):
        wins.append(signal[start: start + window_size])
    if not wins:
        return np.empty((0, target_len, 6), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X = np.stack(wins)  # (W, window_size, 6)
    if window_size != target_len:
        import torch.nn.functional as F
        t = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        t = F.interpolate(t, size=target_len, mode="linear", align_corners=False)
        X = t.permute(0, 2, 1).numpy().astype(np.float32)
    y = np.full((len(X),), label, dtype=np.int64)
    return X, y


def load_motionsense(
    data_root: str,
    target_len: int = 50,
    train_subjects: Optional[list[int]] = None,
    test_subjects: Optional[list[int]] = None,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Load MotionSense from extracted ZIP directory.

    Args:
        data_root: Path where A_DeviceMotion_data.zip was extracted.
                   Expects subdirectory A_DeviceMotion_data/ inside.
        split: 'train' or 'test'
        train_subjects: subject IDs (1-24) for training; defaults to 1-18
        test_subjects:  subject IDs for testing; defaults to 19-24
    """
    if train_subjects is None:
        train_subjects = list(range(1, 19))
    if test_subjects is None:
        test_subjects = list(range(19, 25))
    subjects = train_subjects if split == "train" else test_subjects

    base = Path(data_root)
    data_dir = base / "A_DeviceMotion_data"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"MotionSense data directory not found at {data_dir}. "
            "Call MotionSenseDataset.download(data_root) first."
        )

    all_X, all_y = [], []
    for act_name, label in MOTIONSENSE_TO_MERGED.items():
        for trial in range(1, 16):
            act_dir = data_dir / f"{act_name}_{trial}"
            if not act_dir.exists():
                continue
            for sub_id in subjects:
                csv_path = act_dir / f"sub_{sub_id}.csv"
                if not csv_path.exists():
                    continue
                try:
                    with open(csv_path, "rb") as f:
                        raw_bytes = f.read()
                    signal = _parse_subject_csv(raw_bytes)
                    if signal is None or len(signal) < 50:
                        continue
                    X, y = _windows_from_signal(signal, label, target_len=target_len)
                    if len(y):
                        all_X.append(X)
                        all_y.append(y)
                except Exception:
                    continue

    if not all_X:
        return np.empty((0, target_len, 6), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


class MotionSenseDataset(Dataset):
    """PyTorch Dataset wrapping MotionSense pocket-phone sensor data."""

    def __init__(self, data_root: str, split: str = "train", target_len: int = 50):
        X, y = load_motionsense(data_root, target_len=target_len, split=split)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    @staticmethod
    def download(dest_dir: str) -> str:
        """Download and extract A_DeviceMotion_data.zip from GitHub."""
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        extracted = dest / "A_DeviceMotion_data"
        if extracted.exists() and any(extracted.iterdir()):
            print(f"[MotionSense] Already extracted at {extracted}")
            return str(dest)
        zip_path = dest / "A_DeviceMotion_data.zip"
        if not zip_path.exists():
            print(f"[MotionSense] Downloading from GitHub...")
            urllib.request.urlretrieve(MOTIONSENSE_ZIP_URL, str(zip_path))
            print(f"[MotionSense] Downloaded ({zip_path.stat().st_size // 1024} KB)")
        print(f"[MotionSense] Extracting...")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(dest))
        print(f"[MotionSense] Extracted to {dest}")
        return str(dest)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
