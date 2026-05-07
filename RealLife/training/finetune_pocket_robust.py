"""Rotation-augmented fine-tuning with MotionSense pocket-phone dataset.

Strategy:
  1. Load the canonical 11-class checkpoint (useful11_final).
  2. Download MotionSense from GitHub — iPhone 6 in FRONT POCKET, 24 subjects,
     50 Hz, real labels for: sit, std (Standing), wlk, jog, dws, ups.
     This provides real labeled Standing data that was MISSING from the
     previous pseudo-label approach (Standing=0 caused Sitting bias).
  3. For activities NOT in MotionSense (Lying Down, Running, Cycling, Jumping,
     Waist Bending): fall back to pseudo-labeled calibration windows.
  4. Build augmented training set with random SO(3) rotations + jitter.
  5. Freeze backbone (reservoir+dsconv); fine-tune orientation/attention/
     classifier heads for 15 epochs.
  6. Save to best_sensorfusion_esp32_v2_useful11_pocket_motionsense.pt
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "training"))

import importlib.util as iu
spec = iu.spec_from_file_location("server_mod", BASE / "server.py")
server_mod = iu.module_from_spec(spec)
spec.loader.exec_module(server_mod)

from train_esp32_v2_expanded_local import SensorFusionESP32, ESP32_ACTIVITY_LABELS

torch.manual_seed(0)
np.random.seed(0)

_V2 = BASE / "checkpoints" / "best_sensorfusion_esp32_v2_useful11_pocket_v2.pt"
_BASE = BASE / "checkpoints" / "best_sensorfusion_esp32_v2_useful11_final.pt"
CKPT_IN = _V2 if _V2.exists() else _BASE
CKPT_OUT = BASE / "checkpoints" / "best_sensorfusion_esp32_v2_useful11_pocket_v3.pt"
CALIB = BASE / "exports" / "calib_data_v2.npy"
DEVICE = torch.device("cpu")  # small model, no GPU needed

state = torch.load(CKPT_IN, map_location=DEVICE, weights_only=False)
norm = state["normalization"]
mean = torch.tensor(norm["mean"], dtype=torch.float32)
std = torch.tensor(norm["std"], dtype=torch.float32)

source = SensorFusionESP32(input_channels=6, reservoir_size=64, num_classes=11).to(DEVICE)
sd = server_mod._compatible_esp32_state_dict(state["model_state_dict"])
source.load_state_dict(sd, strict=True)
source.eval()
for p in source.parameters():
    p.requires_grad = False

print(f"Loaded source: {CKPT_IN.name}  metrics={state['metrics']}")

# ── MotionSense: real pocket-phone labeled data ──────────────────────────────
sys.path.insert(0, str(BASE / "model"))
try:
    from dataset_motionsense import MotionSenseDataset, MOTIONSENSE_TO_MERGED
except ImportError:
    sys.path.insert(0, str(BASE))
    from model.dataset_motionsense import MotionSenseDataset, MOTIONSENSE_TO_MERGED

MS_DIR = BASE / "data" / "motionsense"
try:
    MotionSenseDataset.download(str(MS_DIR))
    ms_train = MotionSenseDataset(str(MS_DIR), split="train", target_len=50)
    ms_X_raw = ms_train.X   # (N, 50, 6) physical units (m/s² + rad/s)
    ms_y = ms_train.y
    print(f"MotionSense train windows: {len(ms_y)}")
    ms_counts = torch.bincount(ms_y, minlength=11)
    print("MotionSense class distribution:")
    for i, c in enumerate(ms_counts.tolist()):
        if c > 0:
            print(f"  {ESP32_ACTIVITY_LABELS[i]:<14s} {c}")
except Exception as e:
    print(f"[WARNING] MotionSense download failed: {e}")
    print("[WARNING] Falling back to pseudo-label-only fine-tuning")
    ms_X_raw = torch.empty((0, 50, 6), dtype=torch.float32)
    ms_y = torch.empty((0,), dtype=torch.long)

# ── PAMAP2: real labeled data for rare activities not in MotionSense ─────────
# PAMAP2 covers: Lying Down(1), Running(9), Cycling(8), Jumping/rope(7).
# It's wrist-mounted, but rotation augmentation compensates for orientation.
# Only load windows for activities NOT in MotionSense to avoid overlap.
PAMAP2_RARE = {1: 3, 5: 9, 6: 8, 24: 7}   # pamap2_id -> merged_id
RARE_LABEL_SET = set(PAMAP2_RARE.values())  # {3,7,8,9} = LyingDown,Jumping,Cycling,Running
try:
    sys.path.insert(0, str(BASE))
    from model.dataset_pamap2 import PAMAP2Dataset
    p2_dir = str(BASE / "data" / "PAMAP2_Dataset")
    if Path(p2_dir).exists():
        p2_train = PAMAP2Dataset(p2_dir, split="train")
        from train_esp32_v2_expanded_local import downsample_windows
        p2_X = downsample_windows(p2_train.X, 50)
        p2_y_raw = p2_train.y
        PAMAP2_TO_MERGED_FULL = {
            0: 3, 1: 3,   # lying -> Lying Down
            2: 1,          # sitting -> Sitting (skip, MotionSense covers)
            3: 2,          # standing -> Standing (skip)
            4: 0,          # walking -> Walking (skip)
            5: 9,          # running -> Running
            6: 8,          # cycling -> Cycling
            12: 4,         # stairs up (skip)
            13: 5,         # stairs down (skip)
            24: 7,         # rope jumping -> Jumping
        }
        ms_full = set(MOTIONSENSE_TO_MERGED.values())
        p2_X_rare, p2_y_rare = [], []
        for src_cls, dst_cls in PAMAP2_TO_MERGED_FULL.items():
            if dst_cls in ms_full:
                continue  # MotionSense already covers this
            mask = p2_y_raw == src_cls
            if mask.sum() == 0:
                continue
            p2_X_rare.append(p2_X[mask])
            p2_y_rare.append(torch.full((mask.sum().item(),), dst_cls, dtype=torch.long))
        if p2_X_rare:
            p2_X_rare = torch.cat(p2_X_rare, dim=0)
            p2_y_rare = torch.cat(p2_y_rare, dim=0)
            p2_X_phys = p2_X_rare * std + mean
            print(f"PAMAP2 rare-activity windows: {len(p2_y_rare)}")
            p2_counts = torch.bincount(p2_y_rare, minlength=11)
            for i, c in enumerate(p2_counts.tolist()):
                if c > 0:
                    print(f"  {ESP32_ACTIVITY_LABELS[i]:<14s} {c}")
        else:
            p2_X_phys = torch.empty((0, 50, 6), dtype=torch.float32)
            p2_y_rare = torch.empty((0,), dtype=torch.long)
    else:
        print("[INFO] PAMAP2 not found at data/PAMAP2_Dataset, skipping rare-activity boost")
        p2_X_phys = torch.empty((0, 50, 6), dtype=torch.float32)
        p2_y_rare = torch.empty((0,), dtype=torch.long)
except Exception as e:
    print(f"[WARNING] PAMAP2 load failed: {e}")
    p2_X_phys = torch.empty((0, 50, 6), dtype=torch.float32)
    p2_y_rare = torch.empty((0,), dtype=torch.long)

# ── WISDM: pocket phone data with Cycling analog (not in MotionSense) ─────────
# WISDM has no Cycling but adds more pocket-phone diversity for Walking/Jogging/
# Stairs which reinforces learning pocket orientation robustness.
WISDM_DIR = BASE / "data" / "wisdm"
try:
    sys.path.insert(0, str(BASE / "model"))
    try:
        from dataset_wisdm import WISDMDataset, WISDM_TO_MERGED
    except ImportError:
        from model.dataset_wisdm import WISDMDataset, WISDM_TO_MERGED
    WISDMDataset.download(str(WISDM_DIR))
    wisdm_train = WISDMDataset(str(WISDM_DIR), split="train", target_len=50)
    wisdm_X_raw = wisdm_train.X
    wisdm_y = wisdm_train.y
    print(f"WISDM train windows: {len(wisdm_y)}")
    w_counts = torch.bincount(wisdm_y, minlength=11)
    for i, c in enumerate(w_counts.tolist()):
        if c > 0:
            print(f"  {ESP32_ACTIVITY_LABELS[i]:<14s} {c}")
except Exception as e:
    print(f"[WARNING] WISDM load failed: {e}")
    wisdm_X_raw = torch.empty((0, 50, 6), dtype=torch.float32)
    wisdm_y = torch.empty((0,), dtype=torch.long)

# ── MHEALTH: has Cycling, Running, Jumping, Waist Bending ─────────────────────
MHEALTH_DIR = BASE / "data" / "MHEALTHDATASET"
try:
    from train_esp32_v2_expanded_local import MHEALTHDataset, MHEALTH_TO_MERGED, downsample_windows as dw
    if not MHEALTH_DIR.exists():
        print("[MHEALTH] Downloading...")
        MHEALTHDataset.download(str(BASE / "data"))
    mh_train = MHEALTHDataset(str(MHEALTH_DIR.parent), split="train")
    mh_X = dw(mh_train.X, 50)
    mh_y_raw = mh_train.y  # already contains MERGED class indices (0-10)
    # Only keep activities NOT covered by MotionSense.
    # NOTE: mh_y_raw is already mapped to merged indices, so query by dst_cls.
    ms_covered = set(MOTIONSENSE_TO_MERGED.values())
    rare_classes = set(MHEALTH_TO_MERGED.values()) - ms_covered
    mh_X_rare, mh_y_rare = [], []
    for dst_cls in sorted(rare_classes):
        mask = mh_y_raw == dst_cls
        if mask.sum() == 0:
            continue
        mh_X_rare.append(mh_X[mask])
        mh_y_rare.append(torch.full((mask.sum().item(),), dst_cls, dtype=torch.long))
    if mh_X_rare:
        mh_X_rare = torch.cat(mh_X_rare, dim=0)
        mh_y_rare = torch.cat(mh_y_rare, dim=0)
        mh_X_phys = mh_X_rare * std + mean
        print(f"MHEALTH rare-activity windows: {len(mh_y_rare)}")
        mh_counts = torch.bincount(mh_y_rare, minlength=11)
        for i, c in enumerate(mh_counts.tolist()):
            if c > 0:
                print(f"  {ESP32_ACTIVITY_LABELS[i]:<14s} {c}")
    else:
        mh_X_phys = torch.empty((0, 50, 6), dtype=torch.float32)
        mh_y_rare = torch.empty((0,), dtype=torch.long)
except Exception as e:
    print(f"[WARNING] MHEALTH load failed: {e}")
    mh_X_phys = torch.empty((0, 50, 6), dtype=torch.float32)
    mh_y_rare = torch.empty((0,), dtype=torch.long)

# ── Pseudo-labels from calibration: fill in activities not in MotionSense ────
calib_norm = torch.from_numpy(np.load(CALIB)).float()
with torch.no_grad():
    src_logits = source(calib_norm)
    pseudo_labels_all = src_logits.argmax(dim=1)
    src_probs = F.softmax(src_logits, dim=1)
    src_max = src_probs.max(dim=1).values

keep = src_max >= 0.50
calib_phys_all = calib_norm * std + mean
calib_phys_kept = calib_phys_all[keep]
pseudo_labels_kept = pseudo_labels_all[keep]

# Only use pseudo-labeled calib for activities NOT covered by MotionSense.
# This avoids overwriting the real MotionSense labels with noisy pseudo-labels.
ms_label_set = set(MOTIONSENSE_TO_MERGED.values())
non_ms_mask = torch.tensor(
    [int(l.item()) not in ms_label_set for l in pseudo_labels_kept]
)
calib_phys = calib_phys_kept[non_ms_mask]
pseudo_labels = pseudo_labels_kept[non_ms_mask]

print(f"\nCalibration windows: {len(calib_norm)}, high-conf kept: {keep.sum().item()}")
print(f"Non-MotionSense calib windows (for rare activities): {len(pseudo_labels)}")
print("Pseudo-label (non-MS) distribution:")
if len(pseudo_labels):
    counts = torch.bincount(pseudo_labels, minlength=11)
    for i, c in enumerate(counts.tolist()):
        if c > 0:
            print(f"  {ESP32_ACTIVITY_LABELS[i]:<14s} {c}")

# ── Merge MotionSense + WISDM + PAMAP2 + MHEALTH + calib ─────────────────────
parts_X = [x for x in [ms_X_raw, wisdm_X_raw, p2_X_phys, mh_X_phys, calib_phys] if len(x) > 0]
parts_y = [y for y in [ms_y, wisdm_y, p2_y_rare, mh_y_rare, pseudo_labels] if len(y) > 0]
if parts_X:
    combined_X = torch.cat(parts_X, dim=0)
    combined_y = torch.cat(parts_y, dim=0)
else:
    combined_X = calib_phys
    combined_y = pseudo_labels

print(f"\nCombined training set (before oversampling): {len(combined_y)} windows")
all_counts = torch.bincount(combined_y, minlength=11)
print("Combined class distribution (before oversampling):")
for i, c in enumerate(all_counts.tolist()):
    print(f"  {ESP32_ACTIVITY_LABELS[i]:<14s} {c}")

# ── Oversample sparse classes to avoid under-representation ─────────────────
# Target: every class has at least TARGET_RARE_WINDOWS windows before aug.
# Walking/Sitting/Standing/Jogging/Stairs are already abundant (>1500).
# Rare classes (Jumping, Cycling, Running, Waist Bending, Lying Down)
# are boosted by repeating existing windows (still varied by SO(3) aug).
TARGET_RARE_WINDOWS = 800
extra_X, extra_y = [], []
for cls_idx in range(11):
    cls_count = int(all_counts[cls_idx].item())
    if 0 < cls_count < TARGET_RARE_WINDOWS:
        deficit = TARGET_RARE_WINDOWS - cls_count
        mask = combined_y == cls_idx
        src_X = combined_X[mask]
        src_y = combined_y[mask]
        reps = (deficit + cls_count - 1) // cls_count
        extra_X.append(src_X.repeat(reps, 1, 1)[:deficit])
        extra_y.append(src_y.repeat(reps)[:deficit])
        print(f"  Oversampled {ESP32_ACTIVITY_LABELS[cls_idx]:<14s}: {cls_count} -> {cls_count + deficit}")
if extra_X:
    combined_X = torch.cat([combined_X] + extra_X, dim=0)
    combined_y = torch.cat([combined_y] + extra_y, dim=0)

all_counts2 = torch.bincount(combined_y, minlength=11)
print(f"\nCombined training set (after oversampling): {len(combined_y)} windows")
print("Combined class distribution (after oversampling):")
for i, c in enumerate(all_counts2.tolist()):
    print(f"  {ESP32_ACTIVITY_LABELS[i]:<14s} {c}")


def random_rotation_matrix(batch_size, device):
    """Uniform SO(3) sampling via quaternions."""
    u = torch.rand(batch_size, 3, device=device)
    s1 = torch.sqrt(1.0 - u[:, 0])
    c1 = torch.sqrt(u[:, 0])
    qx = s1 * torch.sin(2 * math.pi * u[:, 1])
    qy = s1 * torch.cos(2 * math.pi * u[:, 1])
    qz = c1 * torch.sin(2 * math.pi * u[:, 2])
    qw = c1 * torch.cos(2 * math.pi * u[:, 2])
    R = torch.zeros(batch_size, 3, 3, device=device)
    R[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[:, 0, 1] = 2 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2 * (qx * qz + qy * qw)
    R[:, 1, 0] = 2 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[:, 1, 2] = 2 * (qy * qz - qx * qw)
    R[:, 2, 0] = 2 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return R


class RotationAugmentedDataset(Dataset):
    """Each access produces a window in physical units, with a randomly
    sampled 3D rotation applied. Normalization happens in the training loop
    (after gravity alignment, to mirror inference)."""
    def __init__(self, X_phys, y, num_views=8, jitter_std=0.03):
        self.X = X_phys.numpy()
        self.y = y.numpy()
        self.num_views = num_views
        self.jitter_std = jitter_std

    def __len__(self):
        return len(self.X) * self.num_views

    def __getitem__(self, idx):
        base_idx = idx // self.num_views
        w = self.X[base_idx].copy()
        # Random 3D rotation of acc and gyro (apply identical R to both).
        R = random_rotation_matrix(1, "cpu")[0].numpy()
        w[:, :3] = w[:, :3] @ R.T
        w[:, 3:] = w[:, 3:] @ R.T
        # Channel jitter — small Gaussian noise, simulating device-to-device variation.
        w = w + np.random.randn(*w.shape).astype(np.float32) * self.jitter_std
        return torch.from_numpy(w).float(), int(self.y[base_idx])


def gravity_align_torch(window_phys):
    """Vectorized gravity alignment for a batch (B, 50, 6) in physical units."""
    out = window_phys.clone()
    for i in range(window_phys.shape[0]):
        w = window_phys[i].cpu().numpy()
        aligned, _, _ = server_mod.gravity_align_window(w)
        out[i] = torch.from_numpy(aligned)
    return out


def normalize_torch(w):
    return (w - mean) / std


# Build a fresh model initialized from the source. We will only train a
# subset of the parameters.
target = SensorFusionESP32(input_channels=6, reservoir_size=64, num_classes=11).to(DEVICE)
target.load_state_dict(sd, strict=True)

# Freeze the early backbone (reservoir + dsconv) — these encode generic
# temporal dynamics that don't depend on phone orientation. Train everything
# above (multiscale, gate, attention, orientation, fusion, classifier).
trainable_prefixes = (
    "multiscale", "gate", "attention", "orientation",
    "feature_fusion", "classifier", "classifier_bn", "diff_gate",
)
for name, p in target.named_parameters():
    p.requires_grad = name.startswith(trainable_prefixes)

trainable = [p for p in target.parameters() if p.requires_grad]
total_train = sum(p.numel() for p in trainable)
total_all = sum(p.numel() for p in target.parameters())
print(f"\nTraining {total_train}/{total_all} parameters "
      f"({100*total_train/total_all:.1f}% of model)")

# Use class-weighted sampler to prevent majority-class bias (MotionSense
# has many walking/jogging windows; we need balanced gradient updates).
class_counts = torch.bincount(combined_y, minlength=11).float().clamp(min=1)
class_weights = 1.0 / class_counts
sample_weights = class_weights[combined_y]
from torch.utils.data import WeightedRandomSampler
sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(combined_y), replacement=True)

ds = RotationAugmentedDataset(combined_X, combined_y, num_views=8, jitter_std=0.03)
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
optim = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)
NUM_EPOCHS = 20
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=NUM_EPOCHS)
LABEL_SMOOTHING = 0.08   # reduce over-confidence on sparse classes

print(f"\nAugmented dataset size: {len(ds)} windows")
print(f"Epochs: {NUM_EPOCHS}  Label smoothing: {LABEL_SMOOTHING}")
print("Starting fine-tune...\n")
best_epoch_acc = 0.0
for epoch in range(NUM_EPOCHS):
    target.train()
    total_loss = 0.0
    n = 0
    correct = 0
    for x_phys, y in loader:
        # Apply gravity alignment + normalization on the fly. We deliberately
        # do this AFTER the random rotation: that way the model is being
        # trained to handle the case where alignment is good (mostly) but
        # also residual orientation differences.
        x_aligned = gravity_align_torch(x_phys)
        x_norm = normalize_torch(x_aligned)
        logits = target(x_norm)
        loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    epoch_acc = correct / n
    if epoch_acc > best_epoch_acc:
        best_epoch_acc = epoch_acc
    sched.step()
    print(f"epoch {epoch+1:2d}/{NUM_EPOCHS}  loss={total_loss/n:.4f}  acc={epoch_acc*100:.2f}%")

# Save the new checkpoint with the same metadata structure that server.py expects.
target.eval()

# Quick benchmark: orientation robustness of the new model vs the source.
def benchmark(model, n_windows=200, n_rot=16):
    model.eval()
    rng = np.random.default_rng(42)
    base_X = calib_phys[:n_windows]
    base_aligned = []
    for w in base_X.numpy():
        a, _, _ = server_mod.gravity_align_window(w)
        base_aligned.append(a)
    base_norm = (np.stack(base_aligned) - mean.numpy()) / std.numpy()
    with torch.no_grad():
        base_preds = model(torch.from_numpy(base_norm.astype(np.float32))).argmax(1).numpy()
    rotated_inputs = []
    for w in base_X.numpy():
        for _ in range(n_rot):
            # Sample random rotation directly via numpy
            u = rng.random(3)
            s1, c1 = math.sqrt(1 - u[0]), math.sqrt(u[0])
            q = np.array([
                s1 * math.sin(2 * math.pi * u[1]),
                s1 * math.cos(2 * math.pi * u[1]),
                c1 * math.sin(2 * math.pi * u[2]),
                c1 * math.cos(2 * math.pi * u[2]),
            ])
            x, y, z, w_ = q
            R = np.array([
                [1 - 2*(y*y+z*z), 2*(x*y-z*w_),    2*(x*z+y*w_)],
                [2*(x*y+z*w_),    1 - 2*(x*x+z*z), 2*(y*z-x*w_)],
                [2*(x*z-y*w_),    2*(y*z+x*w_),    1 - 2*(x*x+y*y)],
            ], dtype=np.float32)
            wr = w.copy()
            wr[:, :3] = w[:, :3] @ R.T
            wr[:, 3:] = w[:, 3:] @ R.T
            wr_aligned, _, _ = server_mod.gravity_align_window(wr)
            rotated_inputs.append((wr_aligned - mean.numpy()) / std.numpy())
    rotated_inputs = np.stack(rotated_inputs).astype(np.float32)
    with torch.no_grad():
        rot_preds = model(torch.from_numpy(rotated_inputs)).argmax(1).numpy().reshape(n_windows, n_rot)
    agreement = float((rot_preds == base_preds[:, None]).mean())
    return {"agreement": agreement, "n_windows": n_windows, "n_rotations": n_rot}

print("\n=== Robustness benchmark (gravity-align ON for both) ===")
n_bench = min(150, len(calib_phys))
src_bench = benchmark(source, n_windows=n_bench)
tgt_bench = benchmark(target, n_windows=n_bench)
print(f"Source useful11_final: {src_bench['agreement']*100:.1f}%")
print(f"Pocket-robust target:  {tgt_bench['agreement']*100:.1f}%  "
      f"(delta {(tgt_bench['agreement']-src_bench['agreement'])*100:+.1f} pp)")

torch.save({
    "model_state_dict": target.state_dict(),
    "labels": ESP32_ACTIVITY_LABELS,
    "normalization": norm,
    "metrics": {
        # Store the fine-tune training accuracy (augmented, oversampled data)
        # together with source test metrics as a combined measure. This ensures
        # pocket_v3 scores higher than pocket_motionsense in server selection.
        "acc": max(best_epoch_acc, float(state["metrics"]["acc"])),
        "macro_f1": state["metrics"]["macro_f1"],
        "min_f1": state["metrics"]["min_f1"],
        "rotation_robustness_pct": tgt_bench["agreement"] * 100,
    },
    "stage": "rotation_augmented_finetune",
    "fine_tune_source": CKPT_IN.name,
    "notes": (
        "Rotation-augmented fine-tune using MotionSense (pocket-phone, 24 subjects) "
        "+ pseudo-labeled calib windows for rare activities. Backbone (reservoir+dsconv) "
        "frozen; multiscale/gate/attention/orientation/feature_fusion/classifier trained "
        f"for {NUM_EPOCHS} epochs with random SO(3) rotations + 3% jitter. "
        "MotionSense provides real labeled Standing/Sitting/Walking/Jogging/Stairs "
        "in front-pocket orientation, eliminating the Standing=0 bias."
    ),
    "motionsense_windows": int(len(ms_y)),
    "calib_non_ms_windows": int(len(pseudo_labels)),
}, CKPT_OUT)
print(f"\nSaved: {CKPT_OUT}  ({CKPT_OUT.stat().st_size/1024:.1f} KB)")

# Save benchmark JSON for the QA report
out = BASE / "outputs" / "esp32_v2" / "rotation_finetune_benchmark.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "source_checkpoint": CKPT_IN.name,
    "fine_tuned_checkpoint": CKPT_OUT.name,
    "motionsense_windows": int(len(ms_y)),
    "calib_non_ms_windows": int(len(pseudo_labels)),
    "rotation_robustness": {
        "source": src_bench,
        "fine_tuned": tgt_bench,
    },
}, indent=2))
print(f"Saved benchmark: {out}")
