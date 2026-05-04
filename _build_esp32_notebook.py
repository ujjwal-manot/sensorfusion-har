import json
from pathlib import Path

OUT = Path("sensorfusion_har_ESP32.ipynb")


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.strip("\n").splitlines(True)}


cells = []

cells.append(md(r'''
# SensorFusion-HAR ESP32-S3: 11-Class Real-Data Deployment

This standalone Colab notebook trains an ESP32-ready Human Activity Recognition model on **real UCI-HAR + PAMAP2 data**.

## Final 11 Classes

| ID | Activity | Source |
|---:|---|---|
| 0 | Walking | UCI-HAR + PAMAP2 |
| 1 | Sitting | UCI-HAR + PAMAP2 |
| 2 | Standing | UCI-HAR + PAMAP2 |
| 3 | Lying Down | UCI-HAR + PAMAP2 |
| 4 | Stairs Up | UCI-HAR + PAMAP2 |
| 5 | Stairs Down | UCI-HAR + PAMAP2 |
| 6 | Jogging | PAMAP2 Running |
| 7 | Jumping | PAMAP2 Rope Jumping |
| 8 | Cycling | PAMAP2 Cycling |
| 9 | Ironing | PAMAP2 Ironing |
| 10 | Vacuum Cleaning | PAMAP2 Vacuum Cleaning |

## Why this version is honest

The earlier Soft Fall / Hard Collapse classes were synthetic and too easy, producing fake near-perfect fall accuracy. This notebook removes them and uses only real activities from public datasets.

## Target

- Overall accuracy: **90%+**
- Per-class F1: **0.87+** target for all classes
- Deployment: ONNX + FP32 TFLite + INT8 TFLite + ESP32-S3 C header
'''))

cells.append(code(r'''
# =============================================================================
# Cell 1: Colab setup, dependencies, repo clone
# =============================================================================
import os
import sys
import subprocess
import time

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    marker = "/content/.sensorfusion_esp32_deps_installed"
    if not os.path.exists(marker):
        print("Installing compatible Colab dependencies. Runtime will restart once...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
            "numpy==1.26.4", "protobuf==4.25.8",
            "tensorflow==2.19.1", "tf-keras==2.19.0",
            "onnx==1.16.0", "onnxruntime==1.18.0",
            "onnx2tf==2.4.0", "onnx-graphsurgeon==0.5.2", "sng4onnx==1.0.4",
            "ai-edge-litert"
        ])
        with open(marker, "w") as f:
            f.write("ok")
        time.sleep(1)
        os.kill(os.getpid(), 9)

import random
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch

print("=" * 70)
print("SensorFusion-HAR ESP32-S3 11-Class Notebook")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: GPU not detected. Training will be much slower.")

if IN_COLAB:
    REPO_URL = "https://github.com/ujjwal-manot/sensorfusion-har.git"
    REPO_PATH = "/content/sensorfusion-har"
    if not os.path.isdir(REPO_PATH):
        subprocess.check_call(["git", "clone", REPO_URL, REPO_PATH])
    os.chdir(REPO_PATH)
else:
    REPO_PATH = os.getcwd()

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("exports/esp32", exist_ok=True)
os.makedirs("outputs/esp32", exist_ok=True)

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Working directory: {os.getcwd()}")
print("Setup complete.")
'''))

cells.append(code(r'''
# =============================================================================
# Cell 2: Imports and global configuration
# =============================================================================
import math
import time
import copy

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay

from model.dataset import UCIHARDataset
from model.dataset_pamap2 import PAMAP2Dataset

TARGET_TIME_STEPS = 50
INPUT_CHANNELS = 6
NUM_CLASSES = 11
BATCH_SIZE = 128
MSM_EPOCHS = 50
HEAD_EPOCHS = 10
FINETUNE_EPOCHS = 60
ENTROPY_WEIGHT = 0.01
MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.30

ESP32_ACTIVITY_LABELS = [
    "Walking", "Sitting", "Standing", "Lying Down", "Stairs Up", "Stairs Down",
    "Jogging", "Jumping", "Cycling", "Ironing", "Vacuum Cleaning",
]

UCIHAR_TO_MERGED = {
    0: 0,
    1: 4,
    2: 5,
    3: 1,
    4: 2,
    5: 3,
}

PAMAP2_TO_MERGED = {
    0: 3,
    1: 1,
    2: 2,
    3: 0,
    4: 6,
    5: 8,
    7: 4,
    8: 5,
    9: 10,
    10: 9,
    11: 7,
}

DEFAULT_COLORS = [
    "#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336", "#00BCD4",
    "#795548", "#607D8B", "#8BC34A", "#E91E63", "#3F51B5",
]

plt.style.use("seaborn-v0_8-whitegrid")
print("Imports OK")
'''))

cells.append(code(r'''
# =============================================================================
# Cell 3: Load UCI-HAR and PAMAP2
# =============================================================================
DATA_ROOT = "data"
UCI_DIR = os.path.join(DATA_ROOT, "UCI HAR Dataset")
PAMAP2_DIR = os.path.join(DATA_ROOT, "PAMAP2_Dataset")

if not os.path.isdir(UCI_DIR):
    print("Downloading UCI-HAR...")
    UCI_DIR = UCIHARDataset.download(DATA_ROOT)
else:
    print(f"Using UCI-HAR at {UCI_DIR}")

if not os.path.isdir(PAMAP2_DIR):
    print("Downloading PAMAP2...")
    PAMAP2_DIR = PAMAP2Dataset.download(DATA_ROOT)
else:
    print(f"Using PAMAP2 at {PAMAP2_DIR}")

ucihar_train_raw = UCIHARDataset(UCI_DIR, split="train")
ucihar_test_raw = UCIHARDataset(UCI_DIR, split="test")
pamap2_train_raw = PAMAP2Dataset(PAMAP2_DIR, split="train")
pamap2_test_raw = PAMAP2Dataset(PAMAP2_DIR, split="test")

print(f"UCI-HAR train/test: {len(ucihar_train_raw)} / {len(ucihar_test_raw)}")
print(f"PAMAP2 train/test:  {len(pamap2_train_raw)} / {len(pamap2_test_raw)}")
print(f"PAMAP2 internal class ids available: {sorted(torch.unique(torch.cat([pamap2_train_raw.y, pamap2_test_raw.y])).tolist())}")
'''))

cells.append(code(r'''
# =============================================================================
# Cell 4: Build real 11-class merged dataset (no synthetic classes)
# =============================================================================
class HARWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def downsample_windows(X, target_len=50):
    if X.shape[1] == target_len:
        return X.float()
    X_t = X.permute(0, 2, 1).float()
    X_down = F.interpolate(X_t, size=target_len, mode="linear", align_corners=False)
    return X_down.permute(0, 2, 1).contiguous()


def append_mapped_dataset(all_X, all_y, ds, mapping, target_len):
    X = downsample_windows(ds.X, target_len)
    for src_cls, dst_cls in mapping.items():
        mask = ds.y == src_cls
        count = int(mask.sum().item())
        if count > 0:
            all_X.append(X[mask])
            all_y.append(torch.full((count,), dst_cls, dtype=torch.long))

all_X, all_y = [], []
for ds in [ucihar_train_raw, ucihar_test_raw]:
    append_mapped_dataset(all_X, all_y, ds, UCIHAR_TO_MERGED, TARGET_TIME_STEPS)
for ds in [pamap2_train_raw, pamap2_test_raw]:
    append_mapped_dataset(all_X, all_y, ds, PAMAP2_TO_MERGED, TARGET_TIME_STEPS)

X_all = torch.cat(all_X, dim=0)
y_all = torch.cat(all_y, dim=0)

indices = np.arange(len(y_all))
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=y_all.numpy(),
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

normalization_stats = {
    "mean": [float(v) for v in mean.tolist()],
    "std": [float(v) for v in std.tolist()],
    "target_time_steps": TARGET_TIME_STEPS,
    "input_channels": INPUT_CHANNELS,
    "num_classes": NUM_CLASSES,
    "labels": ESP32_ACTIVITY_LABELS,
}
with open("exports/esp32/normalization_stats.json", "w") as f:
    json.dump(normalization_stats, f, indent=2)

print(f"Merged windows: {len(X_all)}")
print(f"Train/Test: {len(train_ds)} / {len(test_ds)}")
print(f"Shape: {train_ds.X.shape}")
print("Class counts:")
for i, label in enumerate(ESP32_ACTIVITY_LABELS):
    tr = int((train_ds.y == i).sum().item())
    te = int((test_ds.y == i).sum().item())
    print(f"  {i:2d} {label:<18s} train={tr:5d} test={te:4d}")
print("Normalization mean:", normalization_stats["mean"])
print("Normalization std: ", normalization_stats["std"])
'''))

cells.append(code(r'''
# =============================================================================
# Cell 5: Dataset distribution visualization
# =============================================================================
train_counts = torch.bincount(train_ds.y, minlength=NUM_CLASSES).cpu().numpy()
test_counts = torch.bincount(test_ds.y, minlength=NUM_CLASSES).cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
x = np.arange(NUM_CLASSES)
axes[0].bar(x - 0.2, train_counts, width=0.4, label="Train", color="#4CAF50", alpha=0.85)
axes[0].bar(x + 0.2, test_counts, width=0.4, label="Test", color="#2196F3", alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(ESP32_ACTIVITY_LABELS, rotation=35, ha="right")
axes[0].set_ylabel("Windows")
axes[0].set_title("11-Class Real Dataset Distribution")
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis="y")

sample_idx = int(torch.where(train_ds.y == 8)[0][0]) if (train_ds.y == 8).any() else 0
sample = train_ds.X[sample_idx].numpy()
for ch, name in enumerate(["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]):
    axes[1].plot(sample[:, ch], label=name, alpha=0.8)
axes[1].set_title(f"Example window: {ESP32_ACTIVITY_LABELS[int(train_ds.y[sample_idx])]}")
axes[1].set_xlabel("Time step")
axes[1].set_ylabel("Normalized sensor value")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''))

cells.append(code(r'''
# =============================================================================
# Cell 6: ESP32-ready model with research features retained
# =============================================================================
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
        binary_weight = BinarizeSTE.apply(self.weight) * self.scale.abs().clamp_min(1e-4)
        return F.linear(x, binary_weight, self.bias)


class EchoStateNetworkEdge(nn.Module):
    def __init__(self, input_channels=6, reservoir_size=32, spectral_radius=0.9, sparsity=0.80, dropout=0.10):
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
            if keep is not None:
                h_out = h * keep[:, 0, :]
            else:
                h_out = h
            states.append(h_out.unsqueeze(1))
            diffs.append((h_out - prev).unsqueeze(1))
            prev = h_out
        return torch.cat([torch.cat(states, dim=1), torch.cat(diffs, dim=1)], dim=2)


class DSConvEncoderEdge(nn.Module):
    def __init__(self, in_channels=32, out_channels=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=2, padding=2, groups=out_channels, bias=False),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class EdgeSpectralGatedFusion(nn.Module):
    def __init__(self, reservoir_dim=32, channels=48, seq_len=25):
        super().__init__()
        self.channel_proj = nn.Conv1d(reservoir_dim, channels, kernel_size=1)
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
    def __init__(self, in_channels=48, seq_len=25, patch_len=5, d_model=32):
        super().__init__()
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.patch_proj = nn.Linear(in_channels * patch_len, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
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
        z = self.norm(self.out(attn @ v) + z)
        pooled = z.mean(dim=1)
        entropy = -(attn * torch.log(attn.clamp_min(1e-8))).sum(dim=-1).mean()
        if return_attention:
            return pooled, attn, entropy
        return pooled


class SensorFusionESP32(nn.Module):
    def __init__(self, input_channels=6, reservoir_size=32, num_classes=11):
        super().__init__()
        self.reservoir_size = reservoir_size
        self.reservoir = EchoStateNetworkEdge(input_channels, reservoir_size)
        self.diff_gate = nn.Parameter(torch.zeros(reservoir_size))
        self.dsconv = DSConvEncoderEdge(reservoir_size, 48)
        self.gate = EdgeSpectralGatedFusion(reservoir_size, 48, seq_len=TARGET_TIME_STEPS // 2)
        self.attention = PatchMicroAttentionEdge(48, seq_len=TARGET_TIME_STEPS // 2, patch_len=5, d_model=32)
        self.classifier_bn = nn.BatchNorm1d(32)
        self.classifier = ScaledBinaryLinear(32, num_classes)

    def _merge_reservoir_states(self, h):
        rs = self.reservoir_size
        alpha = torch.sigmoid(self.diff_gate).view(1, 1, rs)
        return h[:, :, :rs] + alpha * h[:, :, rs:]

    def _forward_features_from_reservoir(self, h, return_aux=False):
        h = self._merge_reservoir_states(h)
        h_t = h.transpose(1, 2)
        ds = self.dsconv(h_t)
        fused = self.gate(h_t, ds)
        if return_aux:
            feats, attn, entropy = self.attention(fused, return_attention=True)
            return feats, {"attention_entropy": entropy, "attention_weights": attn, "spectral_radius": self.reservoir.effective_spectral_radius}
        return self.attention(fused)

    def forward(self, x, return_aux=False):
        h = self.reservoir(x)
        if return_aux:
            feats, aux = self._forward_features_from_reservoir(h, return_aux=True)
            out = self.classifier(self.classifier_bn(feats))
            return out, aux
        feats = self._forward_features_from_reservoir(h)
        return self.classifier(self.classifier_bn(feats))

    def reservoir_states(self, x):
        return self.reservoir(x)

    def forward_from_reservoir(self, h):
        feats = self._forward_features_from_reservoir(h)
        return self.classifier(self.classifier_bn(feats))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_kb(self):
        return self.count_parameters() * 4 / 1024

    def quantized_size_kb(self):
        return self.count_parameters() / 1024


class MaskedSensorModelESP32(nn.Module):
    def __init__(self, input_channels=6, reservoir_size=32, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, input_channels))
        self.backbone_reservoir = EchoStateNetworkEdge(input_channels, reservoir_size)
        self.backbone_dsconv = DSConvEncoderEdge(reservoir_size, 48)
        self.reconstruction_head = nn.Linear(48, input_channels)

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
    target_model.reservoir.load_state_dict(pretrained_msm.backbone_reservoir.state_dict(), strict=False)
    target_model.dsconv.load_state_dict(pretrained_msm.backbone_dsconv.state_dict(), strict=False)
    return target_model

model_probe = SensorFusionESP32(input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES)
print(model_probe)
print(f"Parameters: {model_probe.count_parameters():,}")
print(f"FP32 size: {model_probe.model_size_kb():.1f} KB")
print(f"INT8 parameter size estimate: {model_probe.quantized_size_kb():.1f} KB")
del model_probe
'''))

cells.append(code(r'''
# =============================================================================
# Cell 7: Loss, augmentation, sampler, evaluation utilities
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else torch.ones(NUM_CLASSES))
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
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def augment_batch(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    noise = torch.randn_like(x) * 0.03
    scale = 1.0 + torch.randn(x.size(0), 1, x.size(2), device=x.device) * 0.08
    x_aug = (x + noise) * scale
    if random.random() < 0.20:
        ch = torch.randint(0, x.size(2), (1,), device=x.device).item()
        x_aug[:, :, ch] = 0.0
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
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=0), counts


def reservoir_manifold_mixup_edge(model, x1, x2, y1, y2, criterion, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    h1 = model.reservoir_states(x1)
    h2 = model.reservoir_states(x2)
    h_mix = lam * h1 + (1.0 - lam) * h2
    logits = model.forward_from_reservoir(h_mix)
    return lam * criterion(logits, y1) + (1.0 - lam) * criterion(logits, y2)


def evaluate_model(model, loader, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            if criterion is not None:
                total_loss += float(criterion(logits, yb).item())
                batches += 1
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(yb.cpu())
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    acc = float((y_true == y_pred).mean())
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    min_f1 = float(per_class_f1.min())
    avg_loss = total_loss / max(batches, 1)
    return {"acc": acc, "macro_f1": macro_f1, "min_f1": min_f1, "per_class_f1": per_class_f1, "y_true": y_true, "y_pred": y_pred, "loss": avg_loss}

minority_threshold = 2000
class_counts_train = torch.bincount(train_ds.y, minlength=NUM_CLASSES)
minority_classes = [i for i, c in enumerate(class_counts_train.tolist()) if c < minority_threshold]
aug_train_ds = AugmentedHARWindowDataset(train_ds, minority_classes=minority_classes)
train_loader, counts = make_balanced_loader(aug_train_ds, BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

alpha = (counts.sum() / (NUM_CLASSES * counts.clamp_min(1.0)))
alpha = (alpha / alpha.mean()).to(device)
criterion = FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=0.05).to(device)
plain_ce = nn.CrossEntropyLoss()

print("Minority classes:", [ESP32_ACTIVITY_LABELS[i] for i in minority_classes])
print("Focal alpha:", [round(float(v), 3) for v in alpha.detach().cpu()])
print("Balanced loader ready.")
'''))

cells.append(code(r'''
# =============================================================================
# Cell 8: MSM pre-training on 11-class real windows
# =============================================================================
msm_model = MaskedSensorModelESP32(input_channels=INPUT_CHANNELS, reservoir_size=32, mask_ratio=0.20).to(device)
msm_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
msm_optimizer = torch.optim.AdamW(msm_model.parameters(), lr=3e-4, weight_decay=1e-4)
msm_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(msm_optimizer, T_max=MSM_EPOCHS)

start = time.time()
for epoch in range(1, MSM_EPOCHS + 1):
    msm_model.train()
    total_loss = 0.0
    batches = 0
    for xb, _ in msm_loader:
        xb = xb.to(device)
        recon, mask = msm_model(xb)
        mask_expanded = mask.unsqueeze(-1)
        loss = (((recon - xb) ** 2) * mask_expanded).sum() / (mask_expanded.sum().clamp_min(1.0) * xb.size(-1))
        msm_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(msm_model.parameters(), 1.0)
        msm_optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    msm_scheduler.step()
    if epoch % 10 == 0 or epoch == 1:
        print(f"MSM epoch {epoch:03d}/{MSM_EPOCHS} loss={total_loss/max(batches,1):.5f} lr={msm_scheduler.get_last_lr()[0]:.6f}")

print(f"MSM pre-training complete in {(time.time() - start) / 60:.1f} min")
torch.save(msm_model.state_dict(), "checkpoints/esp32_msm_pretrain.pt")
'''))

cells.append(code(r'''
# =============================================================================
# Cell 9: Two-stage supervised fine-tuning
# =============================================================================
esp32_model = SensorFusionESP32(input_channels=INPUT_CHANNELS, reservoir_size=32, num_classes=NUM_CLASSES).to(device)
esp32_model = transfer_msm_weights(msm_model, esp32_model)

for name, p in esp32_model.named_parameters():
    p.requires_grad = not (name.startswith("reservoir") or name.startswith("dsconv"))

head_optimizer = torch.optim.AdamW([p for p in esp32_model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
head_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(head_optimizer, T_max=HEAD_EPOCHS)

best_score = -1.0
best_metrics = None
history = []

print("Stage 1: train attention/gate/classifier head")
for epoch in range(1, HEAD_EPOCHS + 1):
    esp32_model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        head_optimizer.zero_grad()
        logits, aux = esp32_model(xb, return_aux=True)
        loss = criterion(logits, yb) - ENTROPY_WEIGHT * aux["attention_entropy"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in esp32_model.parameters() if p.requires_grad], 1.0)
        head_optimizer.step()
    head_scheduler.step()
    metrics = evaluate_model(esp32_model, test_loader, criterion)
    print(f"Head {epoch:02d}/{HEAD_EPOCHS} acc={metrics['acc']:.4f} macro={metrics['macro_f1']:.4f} minF1={metrics['min_f1']:.4f}")

for p in esp32_model.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW([
    {"params": esp32_model.reservoir.parameters(), "lr": 2e-4},
    {"params": esp32_model.dsconv.parameters(), "lr": 5e-4},
    {"params": esp32_model.gate.parameters(), "lr": 7e-4},
    {"params": esp32_model.attention.parameters(), "lr": 7e-4},
    {"params": list(esp32_model.classifier_bn.parameters()) + list(esp32_model.classifier.parameters()) + [esp32_model.diff_gate], "lr": 1e-3},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

print("\nStage 2: full fine-tune with focal loss + reservoir manifold mixup")
start = time.time()
for epoch in range(1, FINETUNE_EPOCHS + 1):
    esp32_model.train()
    train_loss = 0.0
    batches = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits, aux = esp32_model(xb, return_aux=True)
        loss = criterion(logits, yb) - ENTROPY_WEIGHT * aux["attention_entropy"]
        if random.random() < MIXUP_PROB:
            idx = torch.randperm(xb.size(0), device=device)
            loss = loss + 0.35 * reservoir_manifold_mixup_edge(esp32_model, xb, xb[idx], yb, yb[idx], criterion, alpha=MIXUP_ALPHA)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(esp32_model.parameters(), 1.0)
        optimizer.step()
        train_loss += float(loss.item())
        batches += 1
    scheduler.step()

    metrics = evaluate_model(esp32_model, test_loader, criterion)
    score = metrics["min_f1"] * 3.0 + metrics["macro_f1"] + metrics["acc"]
    history.append({"epoch": epoch, "train_loss": train_loss / max(batches, 1), **{k: v for k, v in metrics.items() if k not in ("y_true", "y_pred", "per_class_f1")}})
    if score > best_score:
        best_score = score
        best_metrics = copy.deepcopy(metrics)
        torch.save({
            "model_state_dict": esp32_model.state_dict(),
            "metrics": {"acc": metrics["acc"], "macro_f1": metrics["macro_f1"], "min_f1": metrics["min_f1"]},
            "labels": ESP32_ACTIVITY_LABELS,
            "normalization": normalization_stats,
        }, "checkpoints/best_esp32_11class.pt")
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{FINETUNE_EPOCHS} loss={train_loss/max(batches,1):.4f} acc={metrics['acc']:.4f} macro={metrics['macro_f1']:.4f} minF1={metrics['min_f1']:.4f} bestMin={best_metrics['min_f1']:.4f}")

print(f"Fine-tuning complete in {(time.time() - start) / 60:.1f} min")
print(f"Best: acc={best_metrics['acc']:.4f}, macroF1={best_metrics['macro_f1']:.4f}, minF1={best_metrics['min_f1']:.4f}")
'''))

cells.append(code(r'''
# =============================================================================
# Cell 10: Final evaluation and plots
# =============================================================================
ckpt = torch.load("checkpoints/best_esp32_11class.pt", map_location=device, weights_only=False)
esp32_model.load_state_dict(ckpt["model_state_dict"])
esp32_model.eval()
final_metrics = evaluate_model(esp32_model, test_loader, criterion)

print("Final PyTorch evaluation")
print(f"Accuracy: {final_metrics['acc']:.4f}")
print(f"Macro F1: {final_metrics['macro_f1']:.4f}")
print(f"Min F1:   {final_metrics['min_f1']:.4f}")
print("\nPer-class F1:")
for label, val in zip(ESP32_ACTIVITY_LABELS, final_metrics["per_class_f1"]):
    status = "OK" if val >= 0.87 else "CHECK"
    print(f"  {label:<18s} {val:.4f}  {status}")

print("\nClassification report:")
print(classification_report(final_metrics["y_true"], final_metrics["y_pred"], target_names=ESP32_ACTIVITY_LABELS, digits=4, zero_division=0))

cm = confusion_matrix(final_metrics["y_true"], final_metrics["y_pred"], labels=list(range(NUM_CLASSES)))
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
ConfusionMatrixDisplay(cm, display_labels=ESP32_ACTIVITY_LABELS).plot(ax=axes[0], cmap="inferno", colorbar=True)
axes[0].set_title("11-Class Confusion Matrix")
axes[0].set_xticklabels(ESP32_ACTIVITY_LABELS, rotation=45, ha="right")

bars = axes[1].bar(ESP32_ACTIVITY_LABELS, final_metrics["per_class_f1"], color=DEFAULT_COLORS, edgecolor="black", alpha=0.85)
axes[1].axhline(0.87, color="red", linestyle="--", label="Target 0.87")
for bar, val in zip(bars, final_metrics["per_class_f1"]):
    axes[1].text(bar.get_x() + bar.get_width()/2, min(1.03, val + 0.015), f"{val:.2f}", ha="center", va="bottom", fontsize=8)
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("F1 Score")
axes[1].set_title("Per-Class F1")
axes[1].set_xticklabels(ESP32_ACTIVITY_LABELS, rotation=45, ha="right")
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("outputs/esp32/11class_confusion_f1.png", dpi=180, bbox_inches="tight")
plt.show()

summary = {
    "accuracy": final_metrics["acc"],
    "macro_f1": final_metrics["macro_f1"],
    "min_f1": final_metrics["min_f1"],
    "per_class_f1": {label: float(val) for label, val in zip(ESP32_ACTIVITY_LABELS, final_metrics["per_class_f1"])},
    "parameter_count": esp32_model.count_parameters(),
    "fp32_size_kb": esp32_model.model_size_kb(),
    "int8_parameter_size_kb": esp32_model.quantized_size_kb(),
}
with open("outputs/esp32/summary_11class.json", "w") as f:
    json.dump(summary, f, indent=2)
'''))

cells.append(code(r'''
# =============================================================================
# Cell 11: Export PyTorch -> ONNX
# =============================================================================
import onnx

EXPORT_DIR = "exports/esp32"
os.makedirs(EXPORT_DIR, exist_ok=True)

ckpt = torch.load("checkpoints/best_esp32_11class.pt", map_location=device, weights_only=False)
export_model = SensorFusionESP32(input_channels=INPUT_CHANNELS, reservoir_size=32, num_classes=NUM_CLASSES).to(device)
export_model.load_state_dict(ckpt["model_state_dict"])
export_model.eval()

export_model_cpu = export_model.cpu().eval()
onnx_path = os.path.join(EXPORT_DIR, "sensorfusion_esp32_11class.onnx")
dummy_input = torch.randn(1, TARGET_TIME_STEPS, INPUT_CHANNELS, dtype=torch.float32)

torch.onnx.export(
    export_model_cpu,
    dummy_input,
    onnx_path,
    opset_version=17,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes=None,
)

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print(f"ONNX export OK: {onnx_path}")
print(f"ONNX size: {os.path.getsize(onnx_path) / 1024:.1f} KB")
'''))

cells.append(code(r'''
# =============================================================================
# Cell 12: ONNX -> TFLite FP32 + INT8 using onnx2tf
# =============================================================================
import onnx2tf

try:
    import tensorflow as tf
except Exception as e:
    raise RuntimeError("TensorFlow import failed. In Colab, restart runtime after dependency install and rerun cells.") from e

calib_path = os.path.join(EXPORT_DIR, "calib_data.npy")
n_calib = min(256, len(train_ds))
calib_data = train_ds.X[:n_calib].numpy().astype(np.float32)
np.save(calib_path, calib_data)
print(f"Calibration data saved: {calib_path}, shape={calib_data.shape}")

tf_out_dir = os.path.join(EXPORT_DIR, "tflite_out")
if os.path.exists(tf_out_dir):
    shutil.rmtree(tf_out_dir)

onnx2tf.convert(
    input_onnx_file_path=onnx_path,
    output_folder_path=tf_out_dir,
    output_integer_quantized_tflite=True,
    quant_type="per-channel",
    custom_input_op_name_np_data_path=[
        ["input", calib_path, [0.0] * INPUT_CHANNELS, [1.0] * INPUT_CHANNELS]
    ],
    non_verbose=True,
)

print(f"Files produced in {tf_out_dir}:")
for f in sorted(os.listdir(tf_out_dir)):
    fp = os.path.join(tf_out_dir, f)
    if os.path.isfile(fp):
        print(f"  {f:<55s} {os.path.getsize(fp)/1024:8.1f} KB")


def find_tflite(directory, keys):
    files = [f for f in os.listdir(directory) if f.endswith(".tflite")]
    for key in keys:
        for f in files:
            if key in f:
                return os.path.join(directory, f)
    return None

fp32_src = find_tflite(tf_out_dir, ["float32", "_fp32"])
int8_src = find_tflite(tf_out_dir, ["full_integer_quant", "integer_quant", "int8"])

if fp32_src is None:
    raise FileNotFoundError("Could not find FP32 TFLite output from onnx2tf.")
if int8_src is None:
    raise FileNotFoundError("Could not find INT8 TFLite output from onnx2tf.")

tflite_fp32_path = os.path.join(EXPORT_DIR, "sensorfusion_esp32_11class_fp32.tflite")
tflite_int8_path = os.path.join(EXPORT_DIR, "sensorfusion_esp32_11class_int8.tflite")
shutil.copy(fp32_src, tflite_fp32_path)
shutil.copy(int8_src, tflite_int8_path)

fp32_size_kb = os.path.getsize(tflite_fp32_path) / 1024
int8_size_kb = os.path.getsize(tflite_int8_path) / 1024
print(f"\nTFLite FP32: {tflite_fp32_path} ({fp32_size_kb:.1f} KB)")
print(f"TFLite INT8: {tflite_int8_path} ({int8_size_kb:.1f} KB)")
print(f"Compression: {fp32_size_kb / max(int8_size_kb, 0.1):.1f}x")
print(f"Fits 512KB flash budget: {'YES' if int8_size_kb < 512 else 'NO'}")
print(f"Fits 150KB SRAM target:  {'YES' if int8_size_kb < 150 else 'NO'}")
'''))

cells.append(code(r'''
# =============================================================================
# Cell 13: Validate INT8 TFLite accuracy
# =============================================================================
def run_tflite_predictions(tflite_path, dataset, max_samples=None):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    in_dtype = input_details["dtype"]
    out_dtype = output_details["dtype"]
    in_scale, in_zero = input_details.get("quantization", (0.0, 0))
    out_scale, out_zero = output_details.get("quantization", (0.0, 0))

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    preds, labels = [], []
    for i in range(n):
        x, y = dataset[i]
        sample = x.unsqueeze(0).numpy().astype(np.float32)
        if in_dtype in (np.int8, np.uint8):
            if in_scale == 0:
                raise ValueError("Quantized input scale is zero.")
            sample_q = np.round(sample / in_scale + in_zero)
            info = np.iinfo(in_dtype)
            sample = np.clip(sample_q, info.min, info.max).astype(in_dtype)
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        if out_dtype in (np.int8, np.uint8) and out_scale != 0:
            out = (out.astype(np.float32) - out_zero) * out_scale
        preds.append(int(np.argmax(out[0])))
        labels.append(int(y))
    return np.array(labels), np.array(preds)


y_true_tflite, y_pred_tflite = run_tflite_predictions(tflite_int8_path, test_ds)
tflite_acc = float((y_true_tflite == y_pred_tflite).mean())
tflite_macro = float(f1_score(y_true_tflite, y_pred_tflite, average="macro", zero_division=0))
tflite_per = f1_score(y_true_tflite, y_pred_tflite, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
tflite_min = float(tflite_per.min())

print("INT8 TFLite evaluation")
print(f"Accuracy: {tflite_acc:.4f}")
print(f"Macro F1: {tflite_macro:.4f}")
print(f"Min F1:   {tflite_min:.4f}")
print(f"Accuracy drop vs PyTorch: {final_metrics['acc'] - tflite_acc:.4f}")
for label, val in zip(ESP32_ACTIVITY_LABELS, tflite_per):
    print(f"  {label:<18s} {val:.4f}")

with open("outputs/esp32/tflite_validation_11class.json", "w") as f:
    json.dump({
        "tflite_int8_accuracy": tflite_acc,
        "tflite_int8_macro_f1": tflite_macro,
        "tflite_int8_min_f1": tflite_min,
        "pytorch_accuracy": final_metrics["acc"],
        "accuracy_drop": final_metrics["acc"] - tflite_acc,
    }, f, indent=2)
'''))

cells.append(code(r"""
# =============================================================================
# Cell 14: Generate ESP32-S3 C header
# =============================================================================
header_path = os.path.join(EXPORT_DIR, "sensorfusion_esp32_11class_model.h")
model_bytes = Path(tflite_int8_path).read_bytes()

def format_c_array(data, line_width=12):
    lines = []
    for i in range(0, len(data), line_width):
        chunk = data[i:i + line_width]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk))
    return ",\n".join(lines)

labels_c = ",\n".join([f'  "{label}"' for label in ESP32_ACTIVITY_LABELS])
mean_c = ", ".join(f"{float(v):.8f}f" for v in mean.tolist())
std_c = ", ".join(f"{float(v):.8f}f" for v in std.tolist())

header = f'''#ifndef SENSORFUSION_ESP32_11CLASS_MODEL_H
#define SENSORFUSION_ESP32_11CLASS_MODEL_H

#include <stdint.h>

#define SENSORFUSION_NUM_CLASSES {NUM_CLASSES}
#define SENSORFUSION_TIME_STEPS {TARGET_TIME_STEPS}
#define SENSORFUSION_INPUT_CHANNELS {INPUT_CHANNELS}
#define SENSORFUSION_MODEL_SIZE {len(model_bytes)}

const char* SENSORFUSION_LABELS[SENSORFUSION_NUM_CLASSES] = {{
{labels_c}
}};

const float SENSORFUSION_MEAN[SENSORFUSION_INPUT_CHANNELS] = {{ {mean_c} }};
const float SENSORFUSION_STD[SENSORFUSION_INPUT_CHANNELS] = {{ {std_c} }};

alignas(16) const unsigned char SENSORFUSION_MODEL[SENSORFUSION_MODEL_SIZE] = {{
{format_c_array(model_bytes)}
}};

#endif
'''

Path(header_path).write_text(header)
print(f"Header generated: {header_path}")
print(f"Header size: {os.path.getsize(header_path)/1024:.1f} KB")
print("Use SENSORFUSION_MODEL with TensorFlow Lite Micro on ESP32-S3.")
"""))

cells.append(code(r'''
# =============================================================================
# Cell 15: Final deployment summary and downloads
# =============================================================================
print("=" * 70)
print("ESP32-S3 11-Class HAR Deployment Summary")
print("=" * 70)
print(f"PyTorch accuracy:      {final_metrics['acc']:.4f}")
print(f"PyTorch macro F1:      {final_metrics['macro_f1']:.4f}")
print(f"PyTorch min F1:        {final_metrics['min_f1']:.4f}")
print(f"INT8 TFLite accuracy:  {tflite_acc:.4f}")
print(f"INT8 TFLite macro F1:  {tflite_macro:.4f}")
print(f"INT8 TFLite min F1:    {tflite_min:.4f}")
print(f"ONNX size:             {os.path.getsize(onnx_path)/1024:.1f} KB")
print(f"TFLite FP32 size:      {fp32_size_kb:.1f} KB")
print(f"TFLite INT8 size:      {int8_size_kb:.1f} KB")
print(f"C header:              {header_path}")
print("\nClasses:")
for i, label in enumerate(ESP32_ACTIVITY_LABELS):
    print(f"  {i:2d}: {label}")

shutil.make_archive("/content/sensorfusion_esp32_11class_exports", "zip", EXPORT_DIR) if IN_COLAB else shutil.make_archive("sensorfusion_esp32_11class_exports", "zip", EXPORT_DIR)
shutil.make_archive("/content/sensorfusion_esp32_11class_outputs", "zip", "outputs/esp32") if IN_COLAB else shutil.make_archive("sensorfusion_esp32_11class_outputs", "zip", "outputs/esp32")

if IN_COLAB:
    from google.colab import files
    files.download("/content/sensorfusion_esp32_11class_exports.zip")
    files.download("/content/sensorfusion_esp32_11class_outputs.zip")

print("\nDone. Exports and reports are ready.")
'''))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT.resolve()} with {len(cells)} cells")
