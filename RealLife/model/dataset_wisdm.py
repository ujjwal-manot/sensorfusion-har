"""WISDM v1 dataset loader for fine-tuning.

Source: https://github.com/tdavchev/WISDM  (raw CSV, ~2 MB)
- Smartphone in pants pocket/waist, 6 activities, 20 Hz

Activities mapped to our 11-class label space:
  Walking   -> 0
  Jogging   -> 6
  Sitting   -> 1
  Standing  -> 2
  Upstairs  -> 4 (Stairs Up)
  Downstairs-> 5 (Stairs Down)

Note: No Cycling/Lying/Jumping/Running/WaistBending — those come from PAMAP2+MHEALTH.
"""
from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

WISDM_CSV_URL = (
    "https://raw.githubusercontent.com/mario-bermonti/wisdm-dataset/main/WISDM_ar_v1.1_raw.txt"
)

WISDM_TO_MERGED = {
    "Walking":    0,
    "Jogging":    6,
    "Sitting":    1,
    "Standing":   2,
    "Upstairs":   4,
    "Downstairs": 5,
}

_WISDM_HZ = 20  # approximate; timestamps vary but nominal is 20 Hz


def _parse_wisdm_csv(text: str) -> list[tuple[int, str, float, float, float]]:
    """Return list of (user_id, activity, ax, ay, az)."""
    records = []
    for line in text.splitlines():
        line = line.strip().rstrip(";").strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            user = int(parts[0].strip())
            activity = parts[1].strip()
            ax = float(parts[3].strip())
            ay = float(parts[4].strip())
            az = float(parts[5].strip())
            records.append((user, activity, ax, ay, az))
        except (ValueError, IndexError):
            continue
    return records


def _build_windows(
    records: list,
    window_size: int = 80,   # 80 samples @ 20 Hz = 4 s window
    stride: int = 40,
    target_len: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment per-activity runs into windows, downsample to target_len."""
    import torch.nn.functional as F

    by_user_act: dict[tuple, list] = {}
    for user, act, ax, ay, az in records:
        if act not in WISDM_TO_MERGED:
            continue
        key = (user, act)
        if key not in by_user_act:
            by_user_act[key] = []
        by_user_act[key].append([ax, ay, az])

    all_X, all_y = [], []
    for (user, act), samples in by_user_act.items():
        label = WISDM_TO_MERGED[act]
        arr = np.array(samples, dtype=np.float32)
        gyro = np.zeros_like(arr)  # WISDM has no gyro; pad with zeros
        sig = np.concatenate([arr, gyro], axis=1)  # (N,6)
        for start in range(0, len(sig) - window_size + 1, stride):
            w = sig[start: start + window_size]
            all_X.append(w)
            all_y.append(label)

    if not all_X:
        return np.empty((0, target_len, 6), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.stack(all_X)  # (N, window_size, 6)
    if window_size != target_len:
        t = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        t = F.interpolate(t, size=target_len, mode="linear", align_corners=False)
        X = t.permute(0, 2, 1).numpy().astype(np.float32)
    y = np.array(all_y, dtype=np.int64)
    return X, y


def load_wisdm(
    data_root: str,
    split: str = "train",
    train_users: Optional[list[int]] = None,
    test_users: Optional[list[int]] = None,
    target_len: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Load WISDM data from extracted CSV file."""
    if train_users is None:
        train_users = list(range(1, 28))
    if test_users is None:
        test_users = list(range(28, 37))
    allowed = set(train_users if split == "train" else test_users)

    csv_path = Path(data_root) / "WISDM_ar_v1.1_raw.txt"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"WISDM CSV not found at {csv_path}. Call WISDMDataset.download() first."
        )
    with open(str(csv_path), "r", errors="replace") as f:
        text = f.read()
    records = _parse_wisdm_csv(text)
    records = [(u, a, ax, ay, az) for (u, a, ax, ay, az) in records if u in allowed]
    return _build_windows(records, target_len=target_len)


class WISDMDataset(Dataset):
    """PyTorch Dataset wrapping WISDM v1 smartphone pocket sensor data."""

    def __init__(self, data_root: str, split: str = "train", target_len: int = 50):
        X, y = load_wisdm(data_root, split=split, target_len=target_len)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    @staticmethod
    def download(dest_dir: str) -> str:
        """Download WISDM raw CSV from GitHub."""
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        csv_path = dest / "WISDM_ar_v1.1_raw.txt"
        if csv_path.exists() and csv_path.stat().st_size > 100_000:
            print(f"[WISDM] Already downloaded at {csv_path}")
            return str(dest)
        print(f"[WISDM] Downloading from GitHub...")
        urllib.request.urlretrieve(WISDM_CSV_URL, str(csv_path))
        print(f"[WISDM] Downloaded ({csv_path.stat().st_size // 1024} KB)")
        return str(dest)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
