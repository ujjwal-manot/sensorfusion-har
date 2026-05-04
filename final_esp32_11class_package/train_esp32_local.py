"""Local training script for the ESP32 11-class SensorFusion-HAR model.

Runs the same pipeline as sensorfusion_har_ESP32.ipynb but optimized for local
CPU execution. Produces:
  - checkpoints/best_esp32_11class.pt
  - exports/esp32/sensorfusion_esp32_11class.onnx
  - outputs/esp32/summary_11class.json
  - outputs/esp32/11class_confusion_f1.png

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
import warnings
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
NUM_CLASSES = 11
ENTROPY_WEIGHT = 0.01
MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.30

ESP32_ACTIVITY_LABELS = [
    "Walking", "Sitting", "Standing", "Lying Down", "Stairs Up", "Stairs Down",
    "Jogging", "Jumping", "Cycling", "Ironing", "Vacuum Cleaning",
]

# UCIHAR raw labels: 0:Walking, 1:Walking_Up, 2:Walking_Down, 3:Sitting, 4:Standing, 5:Laying
UCIHAR_TO_MERGED = {0: 0, 1: 4, 2: 5, 3: 1, 4: 2, 5: 3}

# PAMAP2 internal IDs (after dataset class re-indexes raw activity codes)
#  0 Lying, 1 Sitting, 2 Standing, 3 Walking, 4 Running, 5 Cycling,
#  6 Nordic Walking (skip), 7 Ascending Stairs, 8 Descending Stairs,
#  9 Vacuum Cleaning, 10 Ironing, 11 Rope Jumping
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


def downsample_windows(X: torch.Tensor, target_len: int = 50) -> torch.Tensor:
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


def build_merged_dataset(data_root: str = "data"):
    print("Loading UCI-HAR and PAMAP2...")
    uci_dir = os.path.join(data_root, "UCI HAR Dataset")
    p2_dir = os.path.join(data_root, "PAMAP2_Dataset")

    ucihar_train = UCIHARDataset(uci_dir, split="train")
    ucihar_test = UCIHARDataset(uci_dir, split="test")
    pamap2_train = PAMAP2Dataset(p2_dir, split="train")
    pamap2_test = PAMAP2Dataset(p2_dir, split="test")

    print(f"  UCI-HAR: {len(ucihar_train)} train + {len(ucihar_test)} test")
    print(f"  PAMAP2:  {len(pamap2_train)} train + {len(pamap2_test)} test")

    all_X, all_y = [], []
    for ds in [ucihar_train, ucihar_test]:
        append_mapped_dataset(all_X, all_y, ds, UCIHAR_TO_MERGED, TARGET_TIME_STEPS)
    for ds in [pamap2_train, pamap2_test]:
        append_mapped_dataset(all_X, all_y, ds, PAMAP2_TO_MERGED, TARGET_TIME_STEPS)

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
    os.makedirs("exports/esp32", exist_ok=True)
    with open("exports/esp32/normalization_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

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
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 5, padding=2, groups=in_channels, bias=False),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 5, stride=2, padding=2, groups=out_channels, bias=False),
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


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
    def __init__(self, in_channels=64, seq_len=25, patch_len=5, d_model=64):
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
    def __init__(self, input_channels=6, reservoir_size=64, num_classes=11):
        super().__init__()
        self.reservoir_size = reservoir_size
        self.reservoir = EchoStateNetworkEdge(input_channels, reservoir_size)
        self.diff_gate = nn.Parameter(torch.zeros(reservoir_size))
        self.dsconv = DSConvEncoderEdge(reservoir_size, 64)
        self.gate = EdgeSpectralGatedFusion(reservoir_size, 64, seq_len=TARGET_TIME_STEPS // 2)
        self.attention = PatchMicroAttentionEdge(64, seq_len=TARGET_TIME_STEPS // 2,
                                                  patch_len=5, d_model=64)
        self.classifier_bn = nn.BatchNorm1d(64)
        self.classifier = ScaledBinaryLinear(64, num_classes)

    def _merge_reservoir_states(self, h):
        rs = self.reservoir_size
        alpha = torch.sigmoid(self.diff_gate).view(1, 1, rs)
        return h[:, :, :rs] + alpha * h[:, :, rs:]

    def _features(self, h, return_aux=False):
        h = self._merge_reservoir_states(h)
        h_t = h.transpose(1, 2)
        ds = self.dsconv(h_t)
        fused = self.gate(h_t, ds)
        if return_aux:
            feats, attn, entropy = self.attention(fused, return_attention=True)
            return feats, {
                "attention_entropy": entropy,
                "attention_weights": attn,
                "spectral_radius": self.reservoir.effective_spectral_radius,
            }
        return self.attention(fused)

    def forward(self, x, return_aux=False):
        h = self.reservoir(x)
        if return_aux:
            feats, aux = self._features(h, return_aux=True)
            out = self.classifier(self.classifier_bn(feats))
            return out, aux
        feats = self._features(h)
        return self.classifier(self.classifier_bn(feats))

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
    ap.add_argument("--msm-epochs", type=int, default=8)
    ap.add_argument("--head-epochs", type=int, default=4)
    ap.add_argument("--ft-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--reservoir-size", type=int, default=64)
    ap.add_argument("--focal-gamma", type=float, default=1.5)
    ap.add_argument("--label-smoothing", type=float, default=0.02)
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny budget to verify the pipeline.")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch.set_num_threads (0 = default)")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.smoke:
        args.msm_epochs = 1
        args.head_epochs = 1
        args.ft_epochs = 2

    set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("exports/esp32", exist_ok=True)
    os.makedirs("outputs/esp32", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, torch={torch.__version__}, threads={torch.get_num_threads()}")

    train_ds, test_ds, norm_stats, mean, std = build_merged_dataset()

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
    torch.save(msm_model.state_dict(), "checkpoints/esp32_msm_pretrain.pt")

    # ---------- Build supervised model ----------
    model = SensorFusionESP32(
        input_channels=INPUT_CHANNELS, reservoir_size=args.reservoir_size, num_classes=NUM_CLASSES
    ).to(device)
    model = transfer_msm_weights(msm_model, model)
    print(f"\nSupervised model parameters: {model.count_parameters():,}")
    print(f"FP32 size: {model.model_size_kb():.1f} KB")

    # ---------- Stage 1: head-only ----------
    print("\n[Stage 1: head training]")
    for name, p in model.named_parameters():
        p.requires_grad = not (name.startswith("reservoir") or name.startswith("dsconv"))
    head_opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4
    )
    head_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        head_opt, T_max=max(args.head_epochs, 1)
    )
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
        head_sched.step()
        m = evaluate_model(model, test_loader, device, criterion)
        print(f"  Head {epoch:02d}/{args.head_epochs} acc={m['acc']:.4f} macroF1={m['macro_f1']:.4f} minF1={m['min_f1']:.4f}")

    # ---------- Stage 2: full fine-tuning ----------
    print("\n[Stage 2: full fine-tune]")
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW([
        {"params": model.reservoir.parameters(), "lr": 2e-4},
        {"params": model.dsconv.parameters(), "lr": 5e-4},
        {"params": model.gate.parameters(), "lr": 7e-4},
        {"params": model.attention.parameters(), "lr": 7e-4},
        {"params": list(model.classifier_bn.parameters())
                   + list(model.classifier.parameters())
                   + [model.diff_gate], "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.ft_epochs, 1)
    )

    best_score, best_metrics = -1.0, None
    history = []
    t0 = time.time()
    for epoch in range(1, args.ft_epochs + 1):
        model.train()
        train_loss, batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, aux = model(xb, return_aux=True)
            loss = criterion(logits, yb) - ENTROPY_WEIGHT * aux["attention_entropy"]
            if random.random() < MIXUP_PROB:
                idx = torch.randperm(xb.size(0), device=device)
                loss = loss + 0.35 * reservoir_manifold_mixup_edge(
                    model, xb, xb[idx], yb, yb[idx], criterion, alpha=MIXUP_ALPHA
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item())
            batches += 1
        scheduler.step()

        m = evaluate_model(model, test_loader, device, criterion)
        score = m["min_f1"] * 3.0 + m["macro_f1"] + m["acc"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / max(batches, 1),
            "acc": m["acc"], "macro_f1": m["macro_f1"], "min_f1": m["min_f1"],
            "loss": m["loss"],
        })
        if score > best_score:
            best_score = score
            best_metrics = copy.deepcopy(m)
            torch.save({
                "model_state_dict": model.state_dict(),
                "metrics": {"acc": m["acc"], "macro_f1": m["macro_f1"], "min_f1": m["min_f1"]},
                "labels": ESP32_ACTIVITY_LABELS,
                "normalization": norm_stats,
            }, "checkpoints/best_esp32_11class.pt")
        print(f"  FT {epoch:03d}/{args.ft_epochs} loss={train_loss/max(batches,1):.4f} "
              f"acc={m['acc']:.4f} macro={m['macro_f1']:.4f} minF1={m['min_f1']:.4f} "
              f"bestMin={best_metrics['min_f1']:.4f}")
    print(f"  Fine-tuning: {(time.time() - t0)/60:.1f} min")

    # ---------- Final eval ----------
    print("\n[Final evaluation]")
    ckpt = torch.load("checkpoints/best_esp32_11class.pt", map_location=device, weights_only=False)
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
    axes[0].set_title("11-Class Confusion Matrix")
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
    plt.savefig("outputs/esp32/11class_confusion_f1.png", dpi=160, bbox_inches="tight")
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
        },
    }
    with open("outputs/esp32/summary_11class.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---------- ONNX export ----------
    print("\n[ONNX export]")
    import onnx
    model_cpu = model.cpu().eval()
    onnx_path = "exports/esp32/sensorfusion_esp32_11class.onnx"
    dummy = torch.randn(1, TARGET_TIME_STEPS, INPUT_CHANNELS, dtype=torch.float32)
    torch.onnx.export(
        model_cpu, dummy, onnx_path, opset_version=17,
        input_names=["input"], output_names=["logits"], dynamic_axes=None,
    )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX OK: {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")

    # ---------- Calibration data for Colab TFLite step ----------
    n_calib = min(256, len(train_ds))
    calib = train_ds.X[:n_calib].numpy().astype(np.float32)
    np.save("exports/esp32/calib_data.npy", calib)
    print(f"Calibration data saved: exports/esp32/calib_data.npy {calib.shape}")

    print("\nDone. Best checkpoint: checkpoints/best_esp32_11class.pt")


if __name__ == "__main__":
    main()
