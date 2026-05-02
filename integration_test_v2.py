"""Integration test v2: simulate Phase A baseline + Phase B 10-class extension."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))

from model import SensorFusionHAR
from model.augmentation import SensorAugmentor
from model.masked_pretrain import MaskedSensorModel, transfer_masked_weights, create_mask
from model.contrastive import SensorSimCLR, nt_xent_loss, transfer_weights
from model.mixup import reservoir_manifold_mixup
from sklearn.metrics import f1_score, accuracy_score


REPORT = []


def run(name, fn):
    print(f"[..] {name}", flush=True)
    try:
        fn()
        print(f"[OK] {name}\n", flush=True)
        REPORT.append((name, True))
    except Exception:
        print(traceback.format_exc())
        print(f"[FAIL] {name}\n", flush=True)
        REPORT.append((name, False))


DEVICE = torch.device("cpu")


# ---- Phase A: UCI-HAR-only-style 6-class data ----
N_TRAIN_A = 200
N_TEST_A = 60
T = 128
C = 6

X_a_tr = np.random.randn(N_TRAIN_A, T, C).astype(np.float32)
y_a_tr = np.random.randint(0, 6, N_TRAIN_A).astype(np.int64)
X_a_te = np.random.randn(N_TEST_A, T, C).astype(np.float32)
y_a_te = np.random.randint(0, 6, N_TEST_A).astype(np.int64)


def eval_simple(m, X, y):
    m.eval()
    with torch.no_grad():
        out = m(torch.from_numpy(X.astype(np.float32)))
    pred = out.argmax(-1).cpu().numpy()
    return accuracy_score(y, pred), f1_score(y, pred, average="macro", zero_division=0)


model_a = None


def t_phase_a():
    global model_a
    model_a = SensorFusionHAR(input_channels=6, reservoir_size=32, num_classes=6).to(DEVICE)
    opt = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    EASY = [3, 4, 5]; MEDIUM = [0]; HARD = [1, 2]
    PHASES = [EASY, MEDIUM, HARD]
    for ep in range(3):
        phase = ep
        active = sorted({c for p in PHASES[: phase + 1] for c in p})
        mask = np.isin(y_a_tr, active)
        if mask.sum() < 4:
            continue
        bx = torch.from_numpy(X_a_tr[mask][:16])
        by = torch.from_numpy(y_a_tr[mask][:16])
        out, aux = model_a(bx, return_aux=True)
        loss = ce(out, by) - 0.01 * aux["attention_entropy"]
        idx = torch.randperm(bx.shape[0])
        mix = reservoir_manifold_mixup(model_a, bx, bx[idx], by, by[idx], ce, alpha=0.2)
        loss = loss + 0.5 * mix
        opt.zero_grad(); loss.backward(); opt.step()
    acc, f1 = eval_simple(model_a, X_a_te, y_a_te)
    print(f"   Phase A toy run: acc={acc:.3f} f1={f1:.3f}")


# ---- Phase B: 10-class with backbone init from Phase A ----
N_TRAIN_B = 300
N_TEST_B = 80
X_b_tr = np.random.randn(N_TRAIN_B, T, C).astype(np.float32)
y_b_tr = np.random.randint(0, 10, N_TRAIN_B).astype(np.int64)
X_b_te = np.random.randn(N_TEST_B, T, C).astype(np.float32)
y_b_te = np.random.randint(0, 10, N_TEST_B).astype(np.int64)

model_b = None


def t_phase_b_init_from_a():
    global model_b
    model_b = SensorFusionHAR(input_channels=6, reservoir_size=32, num_classes=10).to(DEVICE)
    src = model_a.state_dict()
    dst = model_b.state_dict()
    moved = 0; skipped = []
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k] = v; moved += 1
        else:
            skipped.append((k, tuple(v.shape), tuple(dst[k].shape) if k in dst else "missing"))
    model_b.load_state_dict(dst)
    print(f"   Phase B init: moved {moved}, skipped {len(skipped)}")
    assert moved > 5, "should transfer most backbone tensors"
    assert any("classifier" in k for k, _, _ in skipped), "classifier head should be skipped (6->10)"


class MPU6050NoiseNormalized:
    def __init__(self, sigma_acc=0.05, sigma_gyro=0.05, bias_acc=0.03, bias_gyro=0.03, lsb=0.005):
        self.sigma_acc = sigma_acc; self.sigma_gyro = sigma_gyro
        self.bias_acc = bias_acc; self.bias_gyro = bias_gyro; self.lsb = lsb

    def __call__(self, x):
        is_torch = torch.is_tensor(x)
        a = x.detach().cpu().numpy().copy() if is_torch else np.asarray(x).copy()
        T_ = a.shape[0]
        a[:, :3] += np.random.normal(0, self.sigma_acc, (T_, 3))
        a[:, 3:] += np.random.normal(0, self.sigma_gyro, (T_, 3))
        a[:, :3] += np.random.normal(0, self.bias_acc, (1, 3))
        a[:, 3:] += np.random.normal(0, self.bias_gyro, (1, 3))
        a = np.round(a / self.lsb) * self.lsb
        return torch.from_numpy(a.astype(np.float32)) if is_torch else a.astype(np.float32)


class CombinedAugmentor:
    def __init__(self, p_sensor=0.3, p_mpu=0.3):
        self.sensor = SensorAugmentor(p=p_sensor)
        self.mpu = MPU6050NoiseNormalized()
        self.p_mpu = p_mpu

    def __call__(self, x):
        is_torch = torch.is_tensor(x)
        x_np = x.detach().cpu().numpy().copy() if is_torch else np.asarray(x).copy()
        x_np = self.sensor(x_np)
        if not isinstance(x_np, np.ndarray):
            x_np = np.asarray(x_np)
        x_np = x_np.astype(np.float32)
        if np.random.rand() < self.p_mpu:
            x_np = self.mpu(x_np)
        x_np = np.asarray(x_np, dtype=np.float32)
        return torch.from_numpy(x_np) if is_torch else x_np


augmentor = CombinedAugmentor(p_sensor=0.3, p_mpu=0.3)


class TensorDS(torch.utils.data.Dataset):
    def __init__(self, X, y, augment=None):
        self.X = X.astype(np.float32) if isinstance(X, np.ndarray) else X
        self.y = y.astype(np.int64) if isinstance(y, np.ndarray) else y
        self.aug = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.aug is not None:
            x = self.aug(x)
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32, copy=False)
            x = torch.from_numpy(x)
        elif x.dtype != torch.float32:
            x = x.float()
        return x, int(self.y[i])


train_loader_b = torch.utils.data.DataLoader(TensorDS(X_b_tr, y_b_tr, augmentor), batch_size=16, shuffle=True, drop_last=False)
test_loader_b = torch.utils.data.DataLoader(TensorDS(X_b_te, y_b_te), batch_size=32, shuffle=False)


def t_phase_b_msm():
    msm = MaskedSensorModel(input_channels=6, reservoir_size=32, mask_ratio=0.15).to(DEVICE)
    opt = torch.optim.AdamW(msm.parameters(), lr=3e-4)
    msm.train()
    for x, _ in train_loader_b:
        x = x.to(DEVICE)
        mask = create_mask(x.shape[0], 128, 0.15, x.device)
        recon, ret_mask = msm(x, mask=mask)
        m_e = ret_mask.unsqueeze(-1).float()
        loss = ((recon - x) ** 2 * m_e).sum() / (m_e.sum() * 6 + 1e-6)
        opt.zero_grad(); loss.backward(); opt.step()
        break
    transfer_masked_weights(msm, model_b)


def t_phase_b_simclr():
    s = SensorSimCLR(input_channels=6, reservoir_size=32).to(DEVICE)
    opt = torch.optim.AdamW(s.parameters(), lr=3e-4)
    sa = CombinedAugmentor(p_sensor=0.5, p_mpu=0.3)
    for x, _ in train_loader_b:
        x_np = x.numpy()
        v1 = np.stack([np.asarray(sa(xi), dtype=np.float32) for xi in x_np])
        v2 = np.stack([np.asarray(sa(xi), dtype=np.float32) for xi in x_np])
        v1t = torch.from_numpy(v1).to(DEVICE)
        v2t = torch.from_numpy(v2).to(DEVICE)
        z1 = s(v1t); z2 = s(v2t)
        loss = nt_xent_loss(z1, z2, 0.1)
        opt.zero_grad(); loss.backward(); opt.step()
        break
    transfer_weights(s, model_b)


def t_phase_b_curriculum():
    elderly_phases = [[3, 4, 5], [0, 1, 2, 9], [6, 7, 8]]
    opt = torch.optim.AdamW(model_b.parameters(), lr=5e-4)
    ce = nn.CrossEntropyLoss()
    for phase, classes in enumerate(elderly_phases):
        active = sorted({c for p in elderly_phases[: phase + 1] for c in p})
        mask_tr = np.isin(y_b_tr, active)
        if mask_tr.sum() < 4:
            continue
        sub_loader = torch.utils.data.DataLoader(
            TensorDS(X_b_tr[mask_tr], y_b_tr[mask_tr], augment=augmentor),
            batch_size=8, shuffle=True, drop_last=False
        )
        model_b.train()
        for x, y in sub_loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            out, aux = model_b(x, return_aux=True)
            loss = ce(out, y) - 0.01 * aux["attention_entropy"]
            if x.shape[0] >= 2:
                idx = torch.randperm(x.shape[0], device=DEVICE)
                mix = reservoir_manifold_mixup(model_b, x, x[idx], y, y[idx], ce, alpha=0.2)
                loss = loss + 0.5 * mix
            opt.zero_grad(); loss.backward(); opt.step()
            break


def t_eval():
    model_b.eval()
    with torch.no_grad():
        for x, y in test_loader_b:
            _ = model_b(x.to(DEVICE))
            break


run("Phase A: 6-class curriculum (toy)", t_phase_a)
run("Phase B: init backbone from Phase A", t_phase_b_init_from_a)
run("Phase B: MSM warm-start (6 ep equiv)", t_phase_b_msm)
run("Phase B: SimCLR warm-start (4 ep equiv)", t_phase_b_simclr)
run("Phase B: curriculum on 10 classes", t_phase_b_curriculum)
run("Phase B: evaluate", t_eval)

print("=" * 60)
failed = [n for n, ok in REPORT if not ok]
if failed:
    print(f"FAILED: {len(failed)}")
    for n in failed:
        print(f"  - {n}")
    sys.exit(1)
print(f"ALL {len(REPORT)} V2 INTEGRATION TESTS PASSED")
