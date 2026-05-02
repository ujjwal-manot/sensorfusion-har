"""
Merged 10-class HAR dataset combining UCI HAR + PAMAP2 + synthetic falls.

Classes:
  0: Walking        (UCI HAR class 0 + PAMAP2 Walking idx 3)
  1: Sitting        (UCI HAR class 3 + PAMAP2 Sitting idx 1)
  2: Standing       (UCI HAR class 4 + PAMAP2 Standing idx 2)
  3: Lying Down     (UCI HAR class 5 + PAMAP2 Lying idx 0)
  4: Stairs Up      (UCI HAR class 1 + PAMAP2 Ascending idx 7)
  5: Stairs Down    (UCI HAR class 2 + PAMAP2 Descending idx 8)
  6: Jogging        (PAMAP2 Running idx 4)
  7: Jumping        (PAMAP2 Rope Jumping idx 11)
  8: Soft Fall      (Synthetic)
  9: Hard Collapse  (Synthetic)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

__all__ = ["MergedHARDataset", "MERGED_ACTIVITY_LABELS"]

MERGED_ACTIVITY_LABELS = [
    "Walking", "Sitting", "Standing", "Lying Down",
    "Stairs Up", "Stairs Down", "Jogging", "Jumping",
    "Soft Fall", "Hard Collapse",
]

# UCI HAR original class index -> merged class index
UCIHAR_CLASS_MAP = {
    0: 0,  # Walking -> Walking
    1: 4,  # Walking Upstairs -> Stairs Up
    2: 5,  # Walking Downstairs -> Stairs Down
    3: 1,  # Sitting -> Sitting
    4: 2,  # Standing -> Standing
    5: 3,  # Laying -> Lying Down
}

# PAMAP2 class index -> merged class index (only classes we use)
PAMAP2_CLASS_MAP = {
    0: 3,   # Lying -> Lying Down
    1: 1,   # Sitting -> Sitting
    2: 2,   # Standing -> Standing
    3: 0,   # Walking -> Walking
    4: 6,   # Running -> Jogging
    7: 4,   # Ascending Stairs -> Stairs Up
    8: 5,   # Descending Stairs -> Stairs Down
    11: 6,  # Rope Jumping -> Jumping  (mapped to 7 below, see note)
}
# Fix: Rope Jumping should map to class 7 (Jumping)
PAMAP2_CLASS_MAP[11] = 7


def _downsample_windows(X, target_len=50):
    """Downsample temporal dimension from current length to target_len via linear interpolation."""
    if X.shape[1] == target_len:
        return X
    # X shape: (N, T, C) -> transpose to (N, C, T) for interpolation
    X_t = X.permute(0, 2, 1).float()
    X_down = torch.nn.functional.interpolate(X_t, size=target_len, mode='linear', align_corners=False)
    return X_down.permute(0, 2, 1)


def _generate_synthetic_falls(n_samples=800, time_steps=50, seed=42):
    """Generate synthetic 6-axis IMU data for Soft Fall and Hard Collapse.

    Returns:
        X_soft: (n_samples, time_steps, 6) - Soft Fall patterns
        X_hard: (n_samples, time_steps, 6) - Hard Collapse patterns
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, time_steps)

    X_soft = np.zeros((n_samples, time_steps, 6), dtype=np.float32)
    X_hard = np.zeros((n_samples, time_steps, 6), dtype=np.float32)

    for i in range(n_samples):
        # --- Soft Fall ---
        # Gradual deceleration followed by moderate impact, then stillness
        impact_t = rng.uniform(0.3, 0.5)  # impact occurs 30-50% through window
        impact_idx = int(impact_t * time_steps)
        impact_mag = rng.uniform(2.0, 4.0)  # moderate g-force

        for ch in range(3):  # accelerometer channels
            # Pre-fall: slight movement
            pre = rng.normal(0, 0.3, impact_idx)
            # Impact: moderate spike
            spike_len = max(2, int(rng.uniform(3, 6)))
            spike = np.zeros(time_steps - impact_idx)
            spike[:spike_len] = impact_mag * np.exp(-np.arange(spike_len) * 0.8) * rng.uniform(0.5, 1.5)
            # Post-impact: near zero (lying still)
            post_start = spike_len
            spike[post_start:] = rng.normal(0, 0.1, len(spike) - post_start)
            # Gravity shift on one axis
            if ch == 1:  # y-axis gets gravity offset post-fall
                spike[post_start:] += rng.uniform(0.8, 1.1)

            X_soft[i, :impact_idx, ch] = pre
            X_soft[i, impact_idx:, ch] = spike

        for ch in range(3, 6):  # gyroscope channels
            # Rotation during fall, then stillness
            rot_len = int(rng.uniform(0.2, 0.4) * time_steps)
            rot_start = max(0, impact_idx - rot_len // 2)
            rot = rng.uniform(1.0, 3.0) * np.sin(np.linspace(0, np.pi, rot_len))
            X_soft[i, rot_start:rot_start + rot_len, ch] = rot * rng.choice([-1, 1])
            # Add noise
            X_soft[i, :, ch] += rng.normal(0, 0.15, time_steps)

        # --- Hard Collapse ---
        # Sharp, high-g impact spike with abrupt onset, sustained ground contact
        impact_t = rng.uniform(0.2, 0.4)  # earlier impact
        impact_idx = int(impact_t * time_steps)
        impact_mag = rng.uniform(5.0, 10.0)  # high g-force

        for ch in range(3):  # accelerometer channels
            # Pre-fall: normal activity or standing
            pre = rng.normal(0, 0.2, impact_idx)
            # Sharp impact spike
            spike_len = max(2, int(rng.uniform(2, 4)))
            spike = np.zeros(time_steps - impact_idx)
            spike[:spike_len] = impact_mag * np.exp(-np.arange(spike_len) * 1.5) * rng.uniform(0.7, 1.3)
            # Possible secondary bounce
            if rng.random() > 0.5:
                bounce_idx = spike_len + int(rng.uniform(2, 5))
                if bounce_idx < len(spike) - 2:
                    spike[bounce_idx:bounce_idx + 2] = impact_mag * 0.3 * rng.uniform(0.5, 1.0)
            # Post-impact: stillness with gravity
            post_start = spike_len + 3
            if post_start < len(spike):
                spike[post_start:] = rng.normal(0, 0.05, len(spike) - post_start)
                if ch == 2:  # z-axis gravity shift
                    spike[post_start:] += rng.uniform(0.9, 1.2)

            X_hard[i, :impact_idx, ch] = pre
            X_hard[i, impact_idx:, ch] = spike

        for ch in range(3, 6):  # gyroscope channels
            # Abrupt, high-amplitude rotation
            rot_len = int(rng.uniform(0.1, 0.25) * time_steps)
            rot_start = max(0, impact_idx - 2)
            rot = rng.uniform(3.0, 6.0) * np.sin(np.linspace(0, np.pi, rot_len))
            end_idx = min(rot_start + rot_len, time_steps)
            actual_len = end_idx - rot_start
            X_hard[i, rot_start:end_idx, ch] = rot[:actual_len] * rng.choice([-1, 1])
            X_hard[i, :, ch] += rng.normal(0, 0.1, time_steps)

    return torch.tensor(X_soft), torch.tensor(X_hard)


def build_merged_dataset(ucihar_train, ucihar_test, pamap2_train, pamap2_test,
                         target_time_steps=50, n_fall_samples=800, seed=42):
    """Build merged 10-class dataset from pre-loaded UCI HAR and PAMAP2 datasets.

    Args:
        ucihar_train, ucihar_test: UCIHARDataset instances (already normalized or raw)
        pamap2_train, pamap2_test: PAMAP2Dataset instances (already normalized or raw)
        target_time_steps: target temporal length after downsampling
        n_fall_samples: number of synthetic samples per fall class
        seed: random seed for synthetic data and splitting

    Returns:
        train_ds, test_ds: MergedHARDataset instances
        norm_mean, norm_std: per-channel normalization stats (from training set)
    """
    all_X = []
    all_y = []

    # --- UCI HAR ---
    for ds in [ucihar_train, ucihar_test]:
        X = _downsample_windows(ds.X, target_time_steps)
        for orig_cls, merged_cls in UCIHAR_CLASS_MAP.items():
            mask = ds.y == orig_cls
            if mask.sum() > 0:
                all_X.append(X[mask])
                all_y.append(torch.full((mask.sum().item(),), merged_cls, dtype=torch.long))

    # --- PAMAP2 ---
    for ds in [pamap2_train, pamap2_test]:
        X = _downsample_windows(ds.X, target_time_steps)
        for orig_cls, merged_cls in PAMAP2_CLASS_MAP.items():
            mask = ds.y == orig_cls
            if mask.sum() > 0:
                all_X.append(X[mask])
                all_y.append(torch.full((mask.sum().item(),), merged_cls, dtype=torch.long))

    # --- Synthetic Falls ---
    X_soft, X_hard = _generate_synthetic_falls(n_fall_samples, target_time_steps, seed)
    all_X.append(X_soft)
    all_y.append(torch.full((n_fall_samples,), 8, dtype=torch.long))  # Soft Fall
    all_X.append(X_hard)
    all_y.append(torch.full((n_fall_samples,), 9, dtype=torch.long))  # Hard Collapse

    # Concatenate
    X_all = torch.cat(all_X, dim=0)
    y_all = torch.cat(all_y, dim=0)

    # Stratified train/test split (80/20)
    indices = np.arange(len(y_all))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=y_all.numpy()
    )

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    # Normalize per-channel using training stats
    mean = X_train.mean(dim=(0, 1))
    std = X_train.std(dim=(0, 1))
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    train_ds = MergedHARDataset(X_train, y_train)
    test_ds = MergedHARDataset(X_test, y_test)

    return train_ds, test_ds, mean, std


class MergedHARDataset(Dataset):
    """Simple dataset wrapper for merged 10-class HAR data."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
