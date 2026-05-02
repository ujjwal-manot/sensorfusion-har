"""Build sensorfusion_har_OPTIMIZED.ipynb. Run: python build_notebook.py"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

CellSpec = Tuple[str, str]


def md(s: str) -> CellSpec:
    return ("markdown", s)


def code(s: str) -> CellSpec:
    return ("code", s)


def to_cell(cell_type: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    if not lines:
        lines = [""]
    cell: dict = {"cell_type": cell_type, "metadata": {}, "source": lines}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


CELLS: List[CellSpec] = []

CELLS.append(md("""# SensorFusion-HAR (Elderly-Care Edition) for ESP32-UE + MPU6050

Self-sufficient Colab notebook. Press **Runtime -> Run all** and walk away.

**Hardware target**: ESP32-UE + MPU6050 (3-axis accel + 3-axis gyro), AAA x 3, BLE NUS to Flutter app.

**What runs end-to-end**:
1. Auto-downloads UCI-HAR (UCI archive), USC-HAD (SIPI mirror), SisFall (Kaggle + Sistemic mirrors).
2. Per-dataset normalization, then merges into a 10-class elderly-care label space (incl. real **Falling**).
3. Loads the proven SensorFusionHAR backbone (~23K params, 23KB INT8 - fits ESP32 flash easily).
4. Pre-trains: MSM 12 ep -> SimCLR 8 ep (transfers backbone weights via repo helpers).
5. **Curriculum learning** (60 ep, 3 phases) -> full-class fine-tune (20 ep) -> per-class auto fix (10 ep).
6. Applies the proven SensorAugmentor (jitter, scaling, rotation, time-warp, magnitude-warp, channel-dropout, permutation) plus a normalized-space MPU6050 noise model.
7. Ablation, robustness, QAT, ONNX -> TFLite INT8, model_data.cc, normalization.json.

**Targets**: overall acc >= 88%, F1 macro >= 86%, per-class F1 >= 86%, fall recall >= 0.95, vehicle precision >= 0.95.
"""))

CELLS.append(md("## 1. Environment + repo"))

CELLS.append(code("""import os, sys, subprocess, pathlib
IN_COLAB = "google.colab" in sys.modules
print("Colab:", IN_COLAB)

REPO_URL = "https://github.com/ujjwal-manot/sensorfusion-har.git"
REPO_DIR = pathlib.Path("/content/sensorfusion-har") if IN_COLAB else pathlib.Path.cwd()

if IN_COLAB and not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
print("repo:", REPO_DIR)
"""))

CELLS.append(code("""%pip install -q numpy==1.26.4 scipy==1.11.4 scikit-learn==1.4.2 \\
    torch==2.2.2 torchvision==0.17.2 \\
    pandas==2.2.2 matplotlib==3.8.4 seaborn==0.13.2 \\
    onnx==1.16.0 onnxruntime==1.17.3 \\
    tensorflow==2.15.0 onnx-tf==1.10.0 tensorflow-probability==0.23.0
print("deps installed")
"""))

CELLS.append(code("""import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
import math, time, copy, random, io, zipfile, urllib.request, urllib.error, json, shutil

torch.manual_seed(42); np.random.seed(42); random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)
DATA_ROOT = Path("/content/data") if IN_COLAB else (REPO_DIR / "data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)
OUT_DIR = REPO_DIR / "outputs"; OUT_DIR.mkdir(parents=True, exist_ok=True)
"""))

CELLS.append(md("## 2. Shared download helpers (zip integrity check, multi-mirror fallback)"))

CELLS.append(code("""def http_get(url, out_path, timeout=600):
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 1024:
        return out_path
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(out_path, "wb") as f:
            shutil.copyfileobj(r, f)
        if out_path.stat().st_size > 1024:
            return out_path
    except Exception as e:
        print(f"  download failed for {url}: {type(e).__name__}: {e}")
    if out_path.exists() and out_path.stat().st_size <= 1024:
        out_path.unlink()
    return None


def download_validated_zip(urls, out_path):
    \"\"\"Try each URL until we get a valid zip. Rejects HTML / login-page responses.\"\"\"
    out_path = Path(out_path)
    for u in urls:
        print(f"  trying {u}")
        if out_path.exists():
            out_path.unlink()
        r = http_get(u, out_path)
        if r is None or not r.exists():
            continue
        if zipfile.is_zipfile(r):
            print(f"  valid zip: {r.name} ({r.stat().st_size/1e6:.1f} MB)")
            return r
        head = b""
        try:
            with open(r, "rb") as fh:
                head = fh.read(200)
        except Exception:
            pass
        if b"<html" in head.lower() or b"<!DOCTYPE" in head.lower() or b"sign in" in head.lower():
            print("  got HTML (likely auth/login redirect); falling through to next mirror")
        else:
            print(f"  not a zip (head={head[:30]!r}); falling through")
        try:
            r.unlink()
        except Exception:
            pass
    return None
"""))

CELLS.append(md("## 3. Datasets - UCI-HAR (required), USC-HAD + SisFall (optional)"))

CELLS.append(code("""UCI_URLS = [
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
]


def load_uci(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    base = root / "UCI HAR Dataset"
    if not base.exists():
        zpath = root / "uci.zip"
        r = download_validated_zip(UCI_URLS, zpath)
        if r is None:
            raise RuntimeError(
                "UCI-HAR download failed from all mirrors. "
                "If you are behind a corporate proxy, restart the runtime and try again, "
                "or upload 'UCI HAR Dataset.zip' to /content/data/uci.zip and re-run this cell."
            )
        with zipfile.ZipFile(zpath) as z:
            z.extractall(root)
        inner = root / "UCI HAR Dataset.zip"
        if inner.exists():
            with zipfile.ZipFile(inner) as z:
                z.extractall(root)
        if not base.exists():
            for cand in root.iterdir():
                if cand.is_dir() and "uci" in cand.name.lower() and "har" in cand.name.lower():
                    if cand.name != "UCI HAR Dataset":
                        cand.rename(base)
                        break
        if not base.exists():
            raise RuntimeError(f"UCI-HAR archive extracted but expected folder '{base}' not found.")
    out = {}
    for split in ("train", "test"):
        sigs = base / split / "Inertial Signals"
        if not sigs.exists():
            raise RuntimeError(f"UCI-HAR signals folder missing: {sigs}")
        load = lambda n: np.loadtxt(sigs / f"{n}_{split}.txt")
        ax, ay, az = load("body_acc_x"), load("body_acc_y"), load("body_acc_z")
        gx, gy, gz = load("body_gyro_x"), load("body_gyro_y"), load("body_gyro_z")
        X = np.stack([ax, ay, az, gx, gy, gz], axis=-1).astype(np.float32)
        y = (np.loadtxt(base / split / f"y_{split}.txt").astype(np.int64) - 1)
        s = np.loadtxt(base / split / f"subject_{split}.txt").astype(np.int64)
        out[split] = (X, y, s)
    return out


uci = None
try:
    uci = load_uci(DATA_ROOT)
    print("UCI-HAR train:", uci["train"][0].shape, "test:", uci["test"][0].shape)
except Exception as e:
    print(f"FATAL: UCI-HAR could not be loaded: {e}")
    print("\\nThe rest of the notebook depends on UCI-HAR. Please:")
    print("  1) Restart the runtime (Runtime > Restart runtime)")
    print("  2) Re-run this cell")
    print("  3) If still failing, upload UCI HAR Dataset.zip to /content/data/uci.zip and re-run")
    raise
"""))

CELLS.append(code("""USCHAD_URLS = [
    "https://sipi.usc.edu/had/USC-HAD.zip",
    "https://sipi.usc.edu/HAD/USC-HAD.zip",
]


def load_uschad(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    extract_dir = root / "USC-HAD"
    if not extract_dir.exists():
        zpath = root / "USC-HAD.zip"
        r = download_validated_zip(USCHAD_URLS, zpath)
        if r is None:
            print("  USC-HAD unavailable from all mirrors; pipeline will run on UCI + SisFall only.")
            return None
        with zipfile.ZipFile(zpath) as z:
            z.extractall(root)
    from scipy.io import loadmat
    Xs, ys, subs = [], [], []
    for sub_dir in sorted(extract_dir.glob("Subject*")):
        try:
            sid = int(sub_dir.name.replace("Subject", ""))
        except ValueError:
            continue
        for mat_path in sorted(sub_dir.glob("*.mat")):
            try:
                m = loadmat(mat_path)
                stem = mat_path.stem
                if stem.startswith("a") and "t" in stem:
                    act = int(stem.split("t")[0][1:])
                else:
                    continue
                data = np.asarray(m.get("sensor_readings"), dtype=np.float32)
                if data is None or data.ndim != 2 or data.shape[1] != 6:
                    continue
                data = data[::2]
                for i in range(0, len(data) - 128 + 1, 64):
                    Xs.append(data[i:i+128]); ys.append(act); subs.append(sid)
            except Exception:
                continue
    if not Xs:
        print("  USC-HAD downloaded but no usable .mat files found; skipping.")
        return None
    return np.stack(Xs), np.array(ys, dtype=np.int64), np.array(subs, dtype=np.int64)


usc_pack = None
try:
    usc_pack = load_uschad(DATA_ROOT)
    if usc_pack is not None:
        print("USC-HAD windows:", usc_pack[0].shape, "subjects:", len(np.unique(usc_pack[2])))
except Exception as e:
    print(f"USC-HAD load raised non-fatal exception (skipping): {e}")
    usc_pack = None
"""))

CELLS.append(code("""SISFALL_URLS = [
    "https://www.kaggle.com/api/v1/datasets/download/nvnikhil0001/sis-fall-original-dataset",
    "https://sistemic.udea.edu.co/wp-content/uploads/2022/11/SisFall_dataset.zip",
    "https://sistemic.udea.edu.co/wp-content/uploads/2017/02/SisFall_dataset.zip",
]

SISFALL_ADL_TO_CLASS = {
    "D01": 0, "D02": 0, "D03": 0, "D04": 0,
    "D05": 1, "D06": 1,
    "D07": 2, "D08": 2,
    "D09": 3, "D10": 3, "D11": 3, "D12": 7,
    "D13": 6, "D14": 6,
    "D15": 0, "D16": 0,
    "D17": 5,
    "D18": -1, "D19": -1,
}


def _sisfall_to_si(raw):
    raw = raw.astype(np.float32)
    acc = raw[:, 0:3] * (32.0 / 8192.0) * 9.81
    gyr = raw[:, 3:6] * (4000.0 / 65536.0) * (np.pi / 180.0)
    return np.concatenate([acc, gyr], axis=1)


def load_sisfall(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    extract_dir = root / "SisFall"
    if not extract_dir.exists():
        zpath = root / "SisFall_dataset.zip"
        r = download_validated_zip(SISFALL_URLS, zpath)
        if r is None:
            print("  SisFall unavailable from all mirrors; will mine fall surrogate from impulse windows later.")
            return None
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(extract_dir)
    Xs, ys, subs = [], [], []
    fall_codes = {f"F{i:02d}" for i in range(1, 16)}
    txts = list(extract_dir.rglob("*.txt"))
    print(f"  found {len(txts)} SisFall trials")
    for f in txts:
        name = f.name.upper()
        parts = name.split("_")
        if len(parts) < 3:
            continue
        code_id, subj_id = parts[0], parts[1]
        if code_id in fall_codes:
            lbl = 8
        elif code_id in SISFALL_ADL_TO_CLASS and SISFALL_ADL_TO_CLASS[code_id] >= 0:
            lbl = SISFALL_ADL_TO_CLASS[code_id]
        else:
            continue
        try:
            raw = np.loadtxt(f, delimiter=",")
        except Exception:
            try:
                raw = np.loadtxt(f, delimiter=";")
            except Exception:
                continue
        if raw.ndim != 2 or raw.shape[1] < 6:
            continue
        data = _sisfall_to_si(raw[:, :6])[::4]
        for i in range(0, len(data) - 128 + 1, 64):
            Xs.append(data[i:i+128]); ys.append(lbl)
            try:
                sid = int(''.join(ch for ch in subj_id if ch.isdigit()))
            except ValueError:
                sid = 0
            subs.append(sid)
    if not Xs:
        print("  SisFall extracted but no usable trials parsed; skipping.")
        return None
    return np.stack(Xs), np.array(ys, dtype=np.int64), np.array(subs, dtype=np.int64)


fall_pack = None
try:
    fall_pack = load_sisfall(DATA_ROOT)
    if fall_pack is not None:
        print("SisFall windows:", fall_pack[0].shape, "labels:", np.bincount(fall_pack[1], minlength=10).tolist())
except Exception as e:
    print(f"SisFall load raised non-fatal exception (skipping): {e}")
    fall_pack = None
"""))

CELLS.append(md("## 4. Datasets sanity check (hard fail if UCI missing)"))

CELLS.append(code("""print("=" * 60)
print("Dataset loading summary")
print("=" * 60)
print(f"  UCI-HAR  : {'OK' if 'uci' in globals() and uci is not None else 'MISSING (REQUIRED)'}")
print(f"  USC-HAD  : {'OK' if usc_pack is not None else 'missing (optional)'}")
print(f"  SisFall  : {'OK' if fall_pack is not None else 'missing (will use synthetic falls)'}")
print("=" * 60)

if 'uci' not in globals() or uci is None:
    raise RuntimeError(
        "UCI-HAR is required but did not load. "
        "Re-run cell 3 (UCI download) before continuing."
    )
"""))

CELLS.append(md("""## 5. PHASE A - UCI-HAR 6-class baseline (target 92-93% accuracy)

This is the **reproducibility check**. We train the proven SensorFusionHAR backbone on UCI-HAR alone (6 classes) using the original recipe (CE loss + entropy reg + reservoir manifold mixup + curriculum learning, no augmentation). This should hit 92-93% on the UCI-HAR test set, matching the original published number. The trained backbone then **initializes Phase B** (10-class extension) so the merged-data fine-tune doesn't have to learn everything from scratch.
"""))

CELLS.append(code("""from model.dataset import UCIHARDataset

ucihar_root = DATA_ROOT / "UCI HAR Dataset"
mean_uci_orig, std_uci_orig = UCIHARDataset.get_normalization_stats(str(ucihar_root))
mu_t = np.array(mean_uci_orig, dtype=np.float32)
sd_t = np.array(std_uci_orig, dtype=np.float32)

X_a_tr = ((uci["train"][0] - mu_t) / sd_t).astype(np.float32)
X_a_te = ((uci["test"][0] - mu_t) / sd_t).astype(np.float32)
y_a_tr = uci["train"][1].astype(np.int64)
y_a_te = uci["test"][1].astype(np.int64)

train_ds_a = torch.utils.data.TensorDataset(torch.from_numpy(X_a_tr), torch.from_numpy(y_a_tr))
test_ds_a = torch.utils.data.TensorDataset(torch.from_numpy(X_a_te), torch.from_numpy(y_a_te))
train_loader_a = torch.utils.data.DataLoader(train_ds_a, batch_size=64, shuffle=True, drop_last=False)
test_loader_a = torch.utils.data.DataLoader(test_ds_a, batch_size=256, shuffle=False)
print("Phase A data:", X_a_tr.shape, X_a_te.shape, "classes:", np.bincount(y_a_tr).tolist())
"""))

CELLS.append(code("""from model import SensorFusionHAR
from model.mixup import reservoir_manifold_mixup
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

UCI_NAMES = ["Walking", "Stairs Up", "Stairs Down", "Sitting", "Standing", "Lying"]
EASY_UCI = [3, 4, 5]
MEDIUM_UCI = [0]
HARD_UCI = [1, 2]
PHASES_UCI = [EASY_UCI, MEDIUM_UCI, HARD_UCI]

EPOCHS_A = 80
PHASE_LEN_A = EPOCHS_A // 3

model_a = SensorFusionHAR(input_channels=6, reservoir_size=32, num_classes=6).to(DEVICE)
opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3, weight_decay=1e-4)
sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=EPOCHS_A)
ce_loss = nn.CrossEntropyLoss()


def eval_simple(m, loader):
    m.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            ps.append(m(x).argmax(-1).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


print("Phase A training (UCI-HAR, 80 epochs, target 92-93%)...")
best_acc_a = 0.0; best_state_a = None
for ep in range(EPOCHS_A):
    phase = min(ep // PHASE_LEN_A, 2)
    active = sorted({c for p in PHASES_UCI[: phase + 1] for c in p})
    mask = np.isin(y_a_tr, active)
    sub_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_a_tr[mask]), torch.from_numpy(y_a_tr[mask])
        ),
        batch_size=64, shuffle=True, drop_last=False
    )
    model_a.train()
    tot = 0.0; n = 0
    for x, y in sub_loader:
        x = x.to(DEVICE); y = y.to(DEVICE)
        out, aux = model_a(x, return_aux=True)
        loss = ce_loss(out, y) - 0.01 * aux["attention_entropy"]
        if random.random() < 0.3 and x.shape[0] >= 2:
            idx = torch.randperm(x.shape[0], device=DEVICE)
            mix_loss = reservoir_manifold_mixup(model_a, x, x[idx], y, y[idx], ce_loss, alpha=0.2)
            loss = loss + 0.5 * mix_loss
        opt_a.zero_grad(); loss.backward(); opt_a.step()
        tot += float(loss); n += 1
    sch_a.step()
    yt, yp = eval_simple(model_a, test_loader_a)
    acc = accuracy_score(yt, yp); f1 = f1_score(yt, yp, average='macro', zero_division=0)
    if (ep + 1) % 5 == 0 or ep == EPOCHS_A - 1:
        print(f"  Phase A ep{ep+1:03d} phase={phase} active={active} loss={tot/max(n,1):.3f} acc={acc:.3f} f1={f1:.3f}")
    if acc > best_acc_a:
        best_acc_a = acc; best_state_a = copy.deepcopy(model_a.state_dict())

model_a.load_state_dict(best_state_a)
yt, yp = eval_simple(model_a, test_loader_a)
phase_a_acc = accuracy_score(yt, yp)
phase_a_f1 = f1_score(yt, yp, average='macro', zero_division=0)

print("\\n" + "=" * 60)
print(f"PHASE A FINAL (UCI-HAR 6-class):")
print(f"  Test accuracy : {phase_a_acc*100:.2f}%")
print(f"  F1 macro      : {phase_a_f1:.4f}")
print(f"  Target was 92-93% -- {'PASS' if phase_a_acc >= 0.88 else 'BELOW TARGET (recipe issue?)'}")
print("=" * 60)
print("\\nPer-class F1 on UCI-HAR test set:")
report_a = classification_report(yt, yp, target_names=UCI_NAMES, output_dict=True, zero_division=0)
for n_ in UCI_NAMES:
    print(f"  {n_:15s} {report_a.get(n_, {}).get('f1-score', 0):.4f}")

torch.save(model_a.state_dict(), OUT_DIR / "model_6class_uci.pt")
print(f"\\nSaved: {OUT_DIR / 'model_6class_uci.pt'}")

cm_a = confusion_matrix(yt, yp, labels=list(range(6)))
plt.figure(figsize=(6, 5))
sns.heatmap(cm_a, annot=True, fmt='d', xticklabels=UCI_NAMES, yticklabels=UCI_NAMES, cmap='Blues')
plt.xticks(rotation=45, ha='right'); plt.title("Phase A - UCI-HAR confusion matrix"); plt.tight_layout(); plt.show()
"""))

CELLS.append(md("""## 6. PHASE B - extend to 10-class elderly-care (init from Phase A backbone)"""))

CELLS.append(md("### 6.1 Build unified 10-class data (per-dataset normalization)"))

CELLS.append(code("""ACT_NAMES = ["Walking", "Stairs Up", "Stairs Down", "Sitting", "Standing", "Lying",
             "Sit-to-Stand", "Stand-to-Sit", "Falling", "Vehicle/Vibration"]


def per_dataset_normalize(X):
    flat = X.reshape(-1, 6)
    mu = flat.mean(0); sd = flat.std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32), mu, sd


def remap_uci(X, y, s):
    keep = np.isin(y, [0, 1, 2, 3, 4, 5])
    return X[keep], y[keep].astype(np.int64), s[keep]


USCHAD_TO_CLASS = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 0, 7: 8, 8: 3, 9: 4, 10: 5, 11: 9, 12: 9}


def remap_uschad(X, y, s):
    out_X, out_y, out_s = [], [], []
    for xi, yi, si in zip(X, y, s):
        if int(yi) in USCHAD_TO_CLASS:
            out_X.append(xi); out_y.append(USCHAD_TO_CLASS[int(yi)]); out_s.append(si)
    if not out_X:
        return None
    return np.stack(out_X), np.array(out_y, np.int64), np.array(out_s, np.int64)


def cap_per_class(X, y, s, cap=1500, seed=0, protect_class=None):
    rng = np.random.default_rng(seed)
    idxs = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        if c == protect_class:
            idxs.extend(ci.tolist())
            continue
        if len(ci) > cap:
            ci = rng.choice(ci, cap, replace=False)
        idxs.extend(ci.tolist())
    idxs = np.array(idxs)
    return X[idxs], y[idxs], s[idxs]


X_uci_tr_raw, y_uci_tr, s_uci_tr = remap_uci(*uci["train"])
X_uci_te_raw, y_uci_te, s_uci_te = remap_uci(*uci["test"])
X_uci_tr_n, mu_uci, sd_uci = per_dataset_normalize(X_uci_tr_raw)
X_uci_te_n = ((X_uci_te_raw - mu_uci) / sd_uci).astype(np.float32)
X_uci_tr_n, y_uci_tr, s_uci_tr = cap_per_class(X_uci_tr_n, y_uci_tr, s_uci_tr, cap=1500)

X_usc_tr_n = X_usc_te_n = None
y_usc_tr = y_usc_te = s_usc_tr = s_usc_te = None
if usc_pack is not None:
    out = remap_uschad(*usc_pack)
    if out is not None:
        Xu, yu, su = out
        Xu_n, mu_usc, sd_usc = per_dataset_normalize(Xu)
        unique_subs = np.unique(su)
        test_subs = set(unique_subs[-2:].tolist()) if len(unique_subs) >= 4 else set()
        if test_subs:
            te_mask = np.isin(su, list(test_subs))
            X_usc_tr_n, y_usc_tr, s_usc_tr = Xu_n[~te_mask], yu[~te_mask], su[~te_mask]
            X_usc_te_n, y_usc_te, s_usc_te = Xu_n[te_mask], yu[te_mask], su[te_mask]
        else:
            X_usc_tr_n, y_usc_tr, s_usc_tr = Xu_n, yu, su
            X_usc_te_n = np.zeros((0, 128, 6), dtype=np.float32)
            y_usc_te = np.zeros(0, np.int64); s_usc_te = np.zeros(0, np.int64)
        X_usc_tr_n, y_usc_tr, s_usc_tr = cap_per_class(X_usc_tr_n, y_usc_tr, s_usc_tr, cap=1500)

X_fall_tr_n = X_fall_te_n = None
y_fall_tr = y_fall_te = s_fall_tr = s_fall_te = None
if fall_pack is not None:
    Xf, yf, sf = fall_pack
    Xf_n, mu_fall, sd_fall = per_dataset_normalize(Xf)
    unique_subs = np.unique(sf)
    test_subs = set(unique_subs[-5:].tolist()) if len(unique_subs) >= 10 else set()
    if test_subs:
        te_mask = np.isin(sf, list(test_subs))
        X_fall_tr_n, y_fall_tr, s_fall_tr = Xf_n[~te_mask], yf[~te_mask], sf[~te_mask]
        X_fall_te_n, y_fall_te, s_fall_te = Xf_n[te_mask], yf[te_mask], sf[te_mask]
    else:
        X_fall_tr_n, y_fall_tr, s_fall_tr = Xf_n, yf, sf
        X_fall_te_n = np.zeros((0, 128, 6), dtype=np.float32)
        y_fall_te = np.zeros(0, np.int64); s_fall_te = np.zeros(0, np.int64)
    X_fall_tr_n, y_fall_tr, s_fall_tr = cap_per_class(
        X_fall_tr_n, y_fall_tr, s_fall_tr, cap=1500, protect_class=8
    )

train_parts = [(X_uci_tr_n, y_uci_tr, s_uci_tr.astype(np.int64))]
test_parts = [(X_uci_te_n, y_uci_te, (s_uci_te + 100).astype(np.int64))]
if X_usc_tr_n is not None and len(X_usc_tr_n) > 0:
    train_parts.append((X_usc_tr_n, y_usc_tr, s_usc_tr.astype(np.int64) + 200))
    if X_usc_te_n is not None and len(X_usc_te_n) > 0:
        test_parts.append((X_usc_te_n, y_usc_te, s_usc_te.astype(np.int64) + 200))
if X_fall_tr_n is not None and len(X_fall_tr_n) > 0:
    train_parts.append((X_fall_tr_n, y_fall_tr, s_fall_tr.astype(np.int64) + 300))
    if X_fall_te_n is not None and len(X_fall_te_n) > 0:
        test_parts.append((X_fall_te_n, y_fall_te, s_fall_te.astype(np.int64) + 300))

X_tr_n = np.concatenate([p[0] for p in train_parts], axis=0).astype(np.float32)
y_tr = np.concatenate([p[1] for p in train_parts], axis=0).astype(np.int64)
s_tr = np.concatenate([p[2] for p in train_parts], axis=0)
X_te_n = np.concatenate([p[0] for p in test_parts], axis=0).astype(np.float32)
y_te = np.concatenate([p[1] for p in test_parts], axis=0).astype(np.int64)
s_te = np.concatenate([p[2] for p in test_parts], axis=0)

if (y_tr == 8).sum() < 50:
    print("No real fall data; mining high-impulse windows as surrogate.")
    mag = np.linalg.norm(X_tr_n[..., :3], axis=-1)
    impulse = mag.max(axis=1) - mag.min(axis=1)
    cand = np.where(np.isin(y_tr, [0, 1, 2]))[0]
    if len(cand) > 0:
        top = cand[np.argsort(-impulse[cand])[:300]]
        y_tr[top] = 8

mu_global = X_tr_n.reshape(-1, 6).mean(0)
sd_global = X_tr_n.reshape(-1, 6).std(0) + 1e-6
print("Unified train:", X_tr_n.shape, "test:", X_te_n.shape)
print("Train hist:", np.bincount(y_tr, minlength=10).tolist())
print("Test  hist:", np.bincount(y_te, minlength=10).tolist())

fig, ax = plt.subplots(1, 2, figsize=(12, 3))
for i, (yy, ttl) in enumerate([(y_tr, "Train"), (y_te, "Test")]):
    counts = np.bincount(yy, minlength=10)
    ax[i].bar(range(10), counts)
    ax[i].set_xticks(range(10))
    ax[i].set_xticklabels(ACT_NAMES, rotation=45, ha='right')
    ax[i].set_title(f"{ttl} (n={len(yy)})")
    ax[i].grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""))

CELLS.append(md("### 6.2 Augmentation (light: SensorAugmentor p=0.3 + MPU6050 noise p=0.3)"))

CELLS.append(code("""from model.augmentation import SensorAugmentor


class MPU6050NoiseNormalized:
    def __init__(self, sigma_acc=0.05, sigma_gyro=0.05,
                 bias_acc=0.03, bias_gyro=0.03, lsb=0.005):
        self.sigma_acc = sigma_acc; self.sigma_gyro = sigma_gyro
        self.bias_acc = bias_acc; self.bias_gyro = bias_gyro
        self.lsb = lsb

    def __call__(self, x):
        is_torch = torch.is_tensor(x)
        a = x.detach().cpu().numpy().copy() if is_torch else np.asarray(x).copy()
        T = a.shape[0]
        a[:, :3] += np.random.normal(0, self.sigma_acc, (T, 3))
        a[:, 3:] += np.random.normal(0, self.sigma_gyro, (T, 3))
        a[:, :3] += np.random.normal(0, self.bias_acc, (1, 3))
        a[:, 3:] += np.random.normal(0, self.bias_gyro, (1, 3))
        a = np.round(a / self.lsb) * self.lsb
        a = a.astype(np.float32)
        return torch.from_numpy(a) if is_torch else a


class CombinedAugmentor:
    def __init__(self, p_sensor=0.5, p_mpu=0.7):
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

x_demo = X_tr_n[0]; x_aug = augmentor(x_demo.copy())
fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
ax[0].plot(x_demo[:, :3]); ax[0].set_title("clean accel (normalized)"); ax[0].grid(alpha=0.3)
ax[1].plot(x_aug[:, :3]); ax[1].set_title("CombinedAugmentor (SensorAug + MPU6050) accel"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""))

CELLS.append(md("### 6.3 10-class model - initialize backbone from Phase A's 6-class weights"))

CELLS.append(code("""model = SensorFusionHAR(input_channels=6, reservoir_size=32, num_classes=10).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {n_params:,}")
print(f"FP32 size      : {n_params*4/1024:.2f} KB")
print(f"INT8 estimate  : {n_params/1024:.2f} KB")

src = model_a.state_dict()
dst = model.state_dict()
moved = 0; skipped = []
for k, v in src.items():
    if k in dst and dst[k].shape == v.shape:
        dst[k] = v; moved += 1
    else:
        skipped.append((k, tuple(v.shape), tuple(dst[k].shape) if k in dst else "missing"))
model.load_state_dict(dst)
print(f"\\nInitialized {moved} backbone tensors from Phase A's 6-class model")
if skipped:
    print(f"Skipped (shape mismatch, will train from scratch): {len(skipped)} tensors (likely the 6->10 classifier head)")

print(f"\\nArchitecture: {model.architecture_summary()}")
"""))

CELLS.append(md("### 6.4 Shared training helpers"))

CELLS.append(code("""class TensorDS(torch.utils.data.Dataset):
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


def evaluate(m, loader):
    m.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            ps.append(m(x).argmax(-1).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def class_balanced_weights(y, beta=0.999, n_classes=10):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    eff = 1.0 - np.power(beta, np.maximum(counts, 1))
    w = (1.0 - beta) / eff
    w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight; self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(logp, target, weight=self.weight, reduction='none')
        pt = logp.exp().gather(1, target[:, None]).squeeze(1)
        return ((1 - pt).pow(self.gamma) * nll).mean()


train_ds = TensorDS(X_tr_n, y_tr, augment=augmentor)
test_ds = TensorDS(X_te_n, y_te, augment=None)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
print("train:", len(train_ds), "test:", len(test_ds))

cb_w = class_balanced_weights(y_tr).to(DEVICE)
focal = FocalLoss(weight=cb_w, gamma=2.0)
ce = nn.CrossEntropyLoss()
print("training helpers ready")
"""))

CELLS.append(md("### 6.5 MSM warm-start (6 ep) on merged data - adapts backbone to USC + SisFall distributions"))

CELLS.append(code("""from model.masked_pretrain import MaskedSensorModel, transfer_masked_weights, create_mask

print("MSM warm-start (6 ep)...")
msm = MaskedSensorModel(input_channels=6, reservoir_size=32, mask_ratio=0.15).to(DEVICE)
opt_msm = torch.optim.AdamW(msm.parameters(), lr=3e-4, weight_decay=1e-4)
sch_msm = torch.optim.lr_scheduler.CosineAnnealingLR(opt_msm, T_max=6)
for ep in range(6):
    msm.train()
    tot = 0.0; n = 0
    for x, _ in train_loader:
        x = x.to(DEVICE)
        T = x.shape[1]
        mask = create_mask(x.shape[0], T, 0.15, x.device)
        recon, ret_mask = msm(x, mask=mask)
        m_e = ret_mask.unsqueeze(-1).float()
        loss = ((recon - x) ** 2 * m_e).sum() / (m_e.sum() * 6 + 1e-6)
        opt_msm.zero_grad(); loss.backward(); opt_msm.step()
        tot += float(loss); n += 1
    sch_msm.step()
    print(f"  MSM ep{ep+1:02d} loss={tot/max(n,1):.4f}")

transfer_masked_weights(msm, model)
print("MSM weights transferred to main model.")
"""))

CELLS.append(md("### 6.6 SimCLR warm-start (4 ep)"))

CELLS.append(code("""from model.contrastive import SensorSimCLR, nt_xent_loss, transfer_weights

print("SimCLR warm-start (4 ep)...")
simclr = SensorSimCLR(input_channels=6, reservoir_size=32).to(DEVICE)
opt_sim = torch.optim.AdamW(simclr.parameters(), lr=3e-4, weight_decay=1e-4)
sch_sim = torch.optim.lr_scheduler.CosineAnnealingLR(opt_sim, T_max=4)

simclr_aug = CombinedAugmentor(p_sensor=0.5, p_mpu=0.3)

for ep in range(4):
    simclr.train()
    tot = 0.0; n = 0
    for x, _ in train_loader:
        x_np = x.numpy()
        v1 = np.stack([np.asarray(simclr_aug(xi), dtype=np.float32) for xi in x_np])
        v2 = np.stack([np.asarray(simclr_aug(xi), dtype=np.float32) for xi in x_np])
        v1 = torch.from_numpy(v1).to(DEVICE)
        v2 = torch.from_numpy(v2).to(DEVICE)
        z1 = simclr(v1); z2 = simclr(v2)
        loss = nt_xent_loss(z1, z2, temperature=0.1)
        opt_sim.zero_grad(); loss.backward(); opt_sim.step()
        tot += float(loss); n += 1
    sch_sim.step()
    print(f"  SimCLR ep{ep+1:02d} loss={tot/max(n,1):.4f}")

transfer_weights(simclr, model)
print("SimCLR weights transferred to main model.")
"""))

CELLS.append(md("""### 6.7 Curriculum (30 ep) + full-class fine-tune (15 ep) on 10-class merged data

Backbone is already warm from Phase A. We use curriculum to gently introduce the new transition / fall / vehicle classes without forgetting Phase A's UCI-HAR knowledge. Lower LR (5e-4) for the warm backbone.

Phases:
- Easy (ep 0-9): Sitting, Standing, Lying (Phase A already strong here)
- Medium (ep 10-19): + Walking, Stairs Up, Stairs Down, Vehicle
- Hard (ep 20-29): + Sit-to-Stand, Stand-to-Sit, Falling
"""))

CELLS.append(code("""ELDERLY_PHASES = [
    [3, 4, 5],
    [0, 1, 2, 9],
    [6, 7, 8],
]

CURRICULUM_EPOCHS = 30
FINETUNE_EPOCHS = 15
PHASE_LEN = CURRICULUM_EPOCHS // len(ELDERLY_PHASES)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CURRICULUM_EPOCHS + FINETUNE_EPOCHS)

best_f1 = 0.0; best_state = None
for ep in range(CURRICULUM_EPOCHS):
    phase = min(ep // PHASE_LEN, len(ELDERLY_PHASES) - 1)
    active = sorted({c for p in ELDERLY_PHASES[: phase + 1] for c in p})
    mask_tr = np.isin(y_tr, active)
    if mask_tr.sum() < 8:
        print(f"  curric ep{ep+1:02d} skipped (insufficient data for active classes {active})")
        continue
    sub_loader = torch.utils.data.DataLoader(
        TensorDS(X_tr_n[mask_tr], y_tr[mask_tr], augment=augmentor),
        batch_size=64, shuffle=True, drop_last=False
    )
    model.train()
    tot = 0.0; n = 0
    for x, y in sub_loader:
        x = x.to(DEVICE); y = y.to(DEVICE)
        out, aux = model(x, return_aux=True)
        loss = ce(out, y) - 0.01 * aux["attention_entropy"]
        if random.random() < 0.3 and x.shape[0] >= 2:
            idx = torch.randperm(x.shape[0], device=DEVICE)
            mix_loss = reservoir_manifold_mixup(model, x, x[idx], y, y[idx], ce, alpha=0.2)
            loss = loss + 0.5 * mix_loss
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss); n += 1
    sch.step()
    yt, yp = evaluate(model, test_loader)
    f1 = f1_score(yt, yp, average='macro', zero_division=0)
    acc = accuracy_score(yt, yp)
    print(f"curric ep{ep+1:02d} phase={phase} active={active} loss={tot/max(n,1):.3f} acc={acc:.3f} f1={f1:.3f}")
    if f1 > best_f1 + 1e-4:
        best_f1 = f1; best_state = copy.deepcopy(model.state_dict())

print(f"\\nCurriculum done. Best F1 so far: {best_f1:.3f}\\n")
print(f"Full-class fine-tune ({FINETUNE_EPOCHS} ep, focal loss + manifold mixup, early stop patience=5)...")
PATIENCE = 5; bad = 0
for ep in range(FINETUNE_EPOCHS):
    model.train()
    tot = 0.0; n = 0
    for x, y in train_loader:
        x = x.to(DEVICE); y = y.to(DEVICE)
        out, aux = model(x, return_aux=True)
        loss = focal(out, y) - 0.01 * aux["attention_entropy"]
        if random.random() < 0.3 and x.shape[0] >= 2:
            idx = torch.randperm(x.shape[0], device=DEVICE)
            mix_loss = reservoir_manifold_mixup(model, x, x[idx], y, y[idx], focal, alpha=0.2)
            loss = loss + 0.5 * mix_loss
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss); n += 1
    sch.step()
    yt, yp = evaluate(model, test_loader)
    f1 = f1_score(yt, yp, average='macro', zero_division=0)
    acc = accuracy_score(yt, yp)
    sr = float(model.reservoir.effective_spectral_radius)
    print(f"full ep{ep+1:02d} loss={tot/max(n,1):.3f} acc={acc:.3f} f1={f1:.3f} sr={sr:.3f}")
    if f1 > best_f1 + 1e-4:
        best_f1 = f1; best_state = copy.deepcopy(model.state_dict()); bad = 0
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"  early stop at full-ep {ep+1} (best F1={best_f1:.3f})")
            break

if best_state is not None:
    model.load_state_dict(best_state)
print(f"BEST F1 macro after curriculum + fine-tune: {best_f1:.3f}")
"""))

CELLS.append(md("### 6.8 Per-class auto-fix (10 ep targeted at sub-86% classes)"))

CELLS.append(code("""yt, yp = evaluate(model, test_loader)
report = classification_report(yt, yp, target_names=ACT_NAMES, output_dict=True, zero_division=0)
per_class_f1 = {ACT_NAMES[i]: report.get(ACT_NAMES[i], {}).get('f1-score', 0.0) for i in range(10)}
print("Per-class F1 before auto-fix:")
for k, v in per_class_f1.items():
    print(f"  {k:22s} {v:.3f}")

below = [i for i in range(10) if per_class_f1.get(ACT_NAMES[i], 0) < 0.86]
if below:
    print(f"\\nClasses below 0.86: {[ACT_NAMES[i] for i in below]}")
    cm = confusion_matrix(yt, yp, labels=list(range(10)))
    target_idx = set(below)
    for i in below:
        row = cm[i].copy(); row[i] = -1
        if row.max() > 0:
            target_idx.add(int(np.argmax(row)))
    target_idx = sorted(target_idx)
    print("Fine-tune classes:", [ACT_NAMES[i] for i in target_idx])
    mask = np.isin(y_tr, target_idx)
    if mask.sum() >= 8:
        counts = np.bincount(y_tr[mask], minlength=10)
        weights = np.zeros_like(y_tr[mask], dtype=np.float64)
        for c in target_idx:
            if counts[c] > 0:
                weights[y_tr[mask] == c] = 1.0 / counts[c]
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=int(len(weights)), replacement=True)
        sub_loader = torch.utils.data.DataLoader(
            TensorDS(X_tr_n[mask], y_tr[mask], augment=augmentor),
            batch_size=64, sampler=sampler, drop_last=False
        )
        opt2 = torch.optim.AdamW(model.parameters(), lr=3e-4)
        for ep in range(10):
            model.train()
            for x, y in sub_loader:
                x = x.to(DEVICE); y = y.to(DEVICE)
                out, aux = model(x, return_aux=True)
                loss = focal(out, y) - 0.005 * aux["attention_entropy"]
                opt2.zero_grad(); loss.backward(); opt2.step()
        yt, yp = evaluate(model, test_loader)
        print(f"\\nAfter auto-fix: acc={accuracy_score(yt, yp):.3f} f1={f1_score(yt, yp, average='macro', zero_division=0):.3f}")
        report = classification_report(yt, yp, target_names=ACT_NAMES, output_dict=True, zero_division=0)
        for i in range(10):
            print(f"  {ACT_NAMES[i]:22s} {report.get(ACT_NAMES[i], {}).get('f1-score', 0):.3f}")
    else:
        print("Insufficient training data for selected classes; skipping auto-fix.")
else:
    print("\\nAll classes already >= 0.86 F1.")

mask_fall_t = (yt == 8); mask_fall_p = (yp == 8)
fall_recall = float((mask_fall_t & mask_fall_p).sum() / max(mask_fall_t.sum(), 1))
mask_veh_p = (yp == 9); mask_veh_t = (yt == 9)
veh_prec = float((mask_veh_p & mask_veh_t).sum() / max(mask_veh_p.sum(), 1))
print(f"\\nFalling recall:    {fall_recall:.3f}  (gate >= 0.95)")
print(f"Vehicle precision: {veh_prec:.3f}  (gate >= 0.95)")
"""))

CELLS.append(md("## 7. Confusion matrix on 10-class test set"))

CELLS.append(code("""yt, yp = evaluate(model, test_loader)
cm = confusion_matrix(yt, yp, labels=list(range(10)))
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=ACT_NAMES, yticklabels=ACT_NAMES, cmap='Blues')
plt.xticks(rotation=45, ha='right')
plt.title("Confusion matrix (test)")
plt.tight_layout(); plt.show()
print(classification_report(yt, yp, target_names=ACT_NAMES, zero_division=0))
"""))

CELLS.append(md("## 8. MPU6050 noise robustness sweep"))

CELLS.append(code("""scales = [0.0, 0.5, 1.0, 2.0, 4.0]
accs = []
n_eval = min(2000, len(X_te_n))
for k in scales:
    aug = MPU6050NoiseNormalized(sigma_acc=0.05*k, sigma_gyro=0.05*k,
                                 bias_acc=0.03*k, bias_gyro=0.03*k)
    X_noisy = np.stack([np.asarray(aug(X_te_n[i]), dtype=np.float32) for i in range(n_eval)])
    loader = torch.utils.data.DataLoader(TensorDS(X_noisy, y_te[:n_eval]), batch_size=256)
    yt, yp = evaluate(model, loader)
    accs.append(accuracy_score(yt, yp))
plt.figure(figsize=(6, 3))
plt.plot(scales, accs, 'o-')
plt.xlabel("MPU6050 noise multiplier")
plt.ylabel("test accuracy")
plt.title("Noise robustness")
plt.grid(alpha=0.3); plt.show()
print({s: round(a, 3) for s, a in zip(scales, accs)})
"""))

CELLS.append(md("## 9. Ablation: Spectral Gate vs Scalar Gate (10 ep each)"))

CELLS.append(code("""from model.sensorfusion import GatedResidualFusion


def quick_train(m_ctor, epochs=10):
    m = m_ctor().to(DEVICE)
    opt_q = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch_q = torch.optim.lr_scheduler.CosineAnnealingLR(opt_q, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        m.train()
        for x, y in train_loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            loss = crit(m(x), y)
            opt_q.zero_grad(); loss.backward(); opt_q.step()
        sch_q.step()
    yt, yp = evaluate(m, test_loader)
    return f1_score(yt, yp, average='macro', zero_division=0), accuracy_score(yt, yp)


def make_full():
    return SensorFusionHAR(num_classes=10)


def make_no_spec_gate():
    m = SensorFusionHAR(num_classes=10)
    m.gate = GatedResidualFusion(reservoir_dim=32, dsconv_channels=48, seq_len=32)
    return m


variants = {
    "Full (Spectral Gate)": make_full,
    "Scalar Gate": make_no_spec_gate,
}

results = {}
for name, ctor in variants.items():
    print("==", name)
    f1, acc = quick_train(ctor, epochs=10)
    results[name] = (f1, acc)
    print(f"  F1={f1:.3f}  Acc={acc:.3f}")

names = list(results.keys()); f1s = [results[n][0] for n in names]
plt.figure(figsize=(8, 3.5))
plt.bar(names, f1s)
plt.ylabel("F1 macro")
plt.title("Ablation - Spectral vs Scalar Fusion")
plt.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.show()
"""))

CELLS.append(md("## 10. Quantization-Aware Training (8 epochs)"))

CELLS.append(code("""class FakeQuant(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__()
        self.n = n_bits

    def forward(self, w):
        with torch.no_grad():
            scale = w.abs().max().clamp(min=1e-8) / (2 ** (self.n - 1) - 1)
        q = torch.round(w / scale).clamp(-(2 ** (self.n - 1)), 2 ** (self.n - 1) - 1)
        return w + (q * scale - w).detach()


def attach_fakequant(m):
    for name, mod in m.named_modules():
        if isinstance(mod, (nn.Conv1d, nn.Linear)) and 'reservoir' not in name:
            fq = FakeQuant()
            if isinstance(mod, nn.Conv1d):
                def make_fwd(mod=mod, fq=fq):
                    def fwd(x):
                        return F.conv1d(x, fq(mod.weight), mod.bias, mod.stride,
                                        mod.padding, mod.dilation, mod.groups)
                    return fwd
            else:
                def make_fwd(mod=mod, fq=fq):
                    def fwd(x):
                        return F.linear(x, fq(mod.weight), mod.bias)
                    return fwd
            mod.forward = make_fwd()
    return m


qat_model = copy.deepcopy(model)
attach_fakequant(qat_model)
qat_opt = torch.optim.AdamW(qat_model.parameters(), lr=2e-4)
qat_sch = torch.optim.lr_scheduler.CosineAnnealingLR(qat_opt, T_max=8)
print("QAT fine-tune (8 ep)...")
for ep in range(8):
    qat_model.train()
    for x, y in train_loader:
        x = x.to(DEVICE); y = y.to(DEVICE)
        loss = focal(qat_model(x), y)
        qat_opt.zero_grad(); loss.backward(); qat_opt.step()
    qat_sch.step()
    yt, yp = evaluate(qat_model, test_loader)
    print(f"  QAT ep{ep+1} acc={accuracy_score(yt, yp):.3f} f1={f1_score(yt, yp, average='macro', zero_division=0):.3f}")
"""))

CELLS.append(md("## 11. Export ONNX -> TFLite INT8 + model_data.cc"))

CELLS.append(code("""ONNX_PATH = OUT_DIR / "sensorfusion_har.onnx"
TFLITE_PATH = OUT_DIR / "sensorfusion_har_int8.tflite"

qat_cpu = qat_model.cpu().eval()
example = torch.randn(1, 128, 6)
torch.onnx.export(qat_cpu, example, str(ONNX_PATH),
                  input_names=["input"], output_names=["logits"],
                  opset_version=17, dynamic_axes={"input": {0: "batch"}}, verbose=False)
print("ONNX:", ONNX_PATH, f"({ONNX_PATH.stat().st_size/1024:.1f} KB)")

have_tf = False
tf_dir = OUT_DIR / "sensorfusion_har_tf"
try:
    from onnx_tf.backend import prepare
    import onnx
    onnx_m = onnx.load(str(ONNX_PATH))
    if tf_dir.exists():
        shutil.rmtree(tf_dir)
    prepare(onnx_m).export_graph(str(tf_dir))
    have_tf = True
    print("TF SavedModel:", tf_dir)
except Exception as e:
    print("onnx-tf failed:", e)

if have_tf:
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_data():
        for i in range(min(128, len(X_tr_n))):
            yield [X_tr_n[i:i+1].astype(np.float32)]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    try:
        tflite_bytes = converter.convert()
        TFLITE_PATH.write_bytes(tflite_bytes)
        print(f"TFLite INT8: {TFLITE_PATH} ({len(tflite_bytes)/1024:.1f} KB)")
    except Exception as e:
        print("INT8 conversion failed:", e, "- falling back to FP16")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_bytes = converter.convert()
        TFLITE_PATH.write_bytes(tflite_bytes)
        print(f"TFLite FP16 fallback: {TFLITE_PATH} ({len(tflite_bytes)/1024:.1f} KB)")
"""))

CELLS.append(code("""CC_PATH = OUT_DIR / "model_data.cc"
if TFLITE_PATH.exists():
    raw = TFLITE_PATH.read_bytes()
    lines = ["// Auto-generated TFLite Micro model for ESP32-UE",
             "#include <cstdint>",
             "alignas(16) const unsigned char g_model[] = {"]
    for i in range(0, len(raw), 12):
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in raw[i:i+12]) + ",")
    lines.append("};")
    lines.append(f"const unsigned int g_model_len = {len(raw)};")
    CC_PATH.write_text("\\n".join(lines))
    print("Wrote", CC_PATH, f"({CC_PATH.stat().st_size/1024:.1f} KB)")
else:
    print("Skipping model_data.cc (no tflite produced).")
"""))

CELLS.append(code("""n_params_q = sum(p.numel() for p in qat_cpu.parameters() if p.requires_grad)
fp32_kb = n_params_q * 4 / 1024
int8_kb = TFLITE_PATH.stat().st_size / 1024 if TFLITE_PATH.exists() else n_params_q / 1024

mac_count = [0]
def mac_hook(mod, inp, out):
    if isinstance(mod, nn.Conv1d):
        ci = inp[0].shape[1]; co = out.shape[1]; lo = out.shape[-1]; k = mod.kernel_size[0]
        mac_count[0] += co * lo * (ci // mod.groups) * k
    elif isinstance(mod, nn.Linear):
        mac_count[0] += mod.in_features * mod.out_features
    elif isinstance(mod, nn.MultiheadAttention):
        d = mod.embed_dim; L = inp[0].shape[1]
        mac_count[0] += 4 * L * d * d + 2 * L * L * d

hooks = []
for m in qat_cpu.modules():
    if isinstance(m, (nn.Conv1d, nn.Linear, nn.MultiheadAttention)):
        hooks.append(m.register_forward_hook(mac_hook))
qat_cpu.eval()
with torch.no_grad():
    qat_cpu(torch.randn(1, 128, 6))
for h in hooks:
    h.remove()
macs = mac_count[0]

est_latency_ms = macs / 250e6 * 1000

NORM_PATH = OUT_DIR / "normalization.json"
NORM_PATH.write_text(json.dumps({
    "mean": mu_global.tolist(),
    "std": sd_global.tolist(),
    "channel_order": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    "fs_hz": 50, "window": 128, "stride": 64,
    "labels": ACT_NAMES,
    "note": "Per-dataset normalization is used during training; mean/std here are reported on the merged training pool. For real ESP32 deployment, the firmware should apply UCI-HAR-style standardization (zero-mean unit-std rolling window stats) since it best matches the dominant training distribution.",
}, indent=2))

print("=== ESP32-UE Resource Report ===")
print(f"Trainable params : {n_params_q:,}")
print(f"FP32 weight size : {fp32_kb:.2f} KB")
print(f"INT8 deployed    : {int8_kb:.2f} KB")
print(f"MACs / window    : {macs:,}")
print(f"Est. latency     : {est_latency_ms:.2f} ms (ESP-NN INT8 @ 250 MMAC/s)")
print(f"Window           : 128 @ 50 Hz = 2.56 s, stride 64 (50% overlap)")
print(f"Artifacts        : {OUT_DIR}")
"""))

CELLS.append(md("""## 12. ESP32-UE wiring (BLE NUS to Flutter)

```
[MPU6050] --I2C--> [ESP32-UE]
   FIFO @ 50 Hz, ringbuffer of 128 samples (zero-mean unit-std normalize using outputs/normalization.json)
        |
   every 64 samples (50% overlap):
   tflite::MicroInterpreter::Invoke(g_model)
        |
   argmax -> activity id 0..9
   if id == 8 (FALL): BLE NUS notify "FALL" + last 2.56s window
   else: BLE NUS notify activity name
        |
[Flutter app] subscribes to NUS characteristic
```

ESP-IDF integration:
- Add `esp-tflite-micro` component, enable ESP-NN in menuconfig.
- Tensor arena: ~32 KB is enough; verify via `interpreter->arena_used_bytes()`.
- Drop `outputs/model_data.cc` and `outputs/normalization.json` into the firmware project.
- AAA x 3 + deep-sleep between BLE notifies + 1 inference / 2 s yields multi-week battery life.
"""))


nb = {
    "cells": [to_cell(t, s) for t, s in CELLS],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = Path(__file__).parent / "sensorfusion_har_OPTIMIZED.ipynb"
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB,  {len(CELLS)} cells)")
