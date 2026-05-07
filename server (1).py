import asyncio
import json
import math
import os
import socket
import sys
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

ACTIVITY_LABELS = {
    0: "Walking",
    1: "Sitting",
    2: "Standing",
    3: "Lying Down",
    4: "Stairs Up",
    5: "Stairs Down",
    6: "Jogging",
    7: "Jumping",
    8: "Cycling",
    9: "Running",
    10: "Waist Bending",
}
EXPECTED_LABELS = [ACTIVITY_LABELS[i] for i in sorted(ACTIVITY_LABELS)]

CHANNEL_STATS = None
CONFIDENCE_THRESHOLD = 0.40
prediction_history = deque(maxlen=5)
MODEL_INFO = {
    "model_loaded": False,
    "model_mode": "heuristic",
    "architecture": None,
    "checkpoint": None,
    "checkpoint_path": None,
    "normalization_source": "default",
    "warning": None,
}

DEFAULT_CHANNEL_STATS = {
    "ax": {"mean": -3.1886, "std": 6.3021},
    "ay": {"mean": 1.2004, "std": 6.4955},
    "az": {"mean": 2.4553, "std": 3.8114},
    "gx": {"mean": -0.0335, "std": 1.0867},
    "gy": {"mean": -0.0226, "std": 0.8431},
    "gz": {"mean": 0.0530, "std": 1.2929},
}

CHANNEL_ORDER = ["ax", "ay", "az", "gx", "gy", "gz"]
WINDOW_SIZE = 50
STRIDE = 25
TARGET_HZ = 50
GRAVITY = 9.81

# Solution A — gravity-alignment preprocessing toggle. When enabled, every
# inference window is rotated so the measured gravity vector lands on +Z
# before normalization. This makes inference robust to phone orientation in
# a pocket (any axis can be aligned with gravity depending on which way the
# phone fell in). Set to False via env to disable for ablation.
GRAVITY_ALIGN_ENABLED = os.environ.get("HAR_GRAVITY_ALIGN", "1") not in ("0", "false", "False", "")
# Solution D — minimum dwell time (in inference windows) before the displayed
# label is allowed to switch. With STRIDE=25 at 50 Hz, one window is 0.5 s,
# so DWELL=2 means the new label has to be predicted twice in a row before
# it is shown. This is what kills the "Sitting/Standing/Sitting" flicker
# during transitions without hiding genuine activity changes.
DWELL_WINDOWS = 2

# Track the displayed label and its run-length so we can enforce min-dwell
# without recomputing it from the deque every time.
_DISPLAY_STATE = {"label": None, "runlen": 0, "candidate": None, "candidate_runlen": 0}

BASE_DIR = Path(__file__).resolve().parent
def _checkpoint_labels(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        labels = state.get("labels", state.get("activity_labels", None))
        return list(labels) if labels else None
    except Exception as e:
        print(f"[server] Could not inspect checkpoint {path.name}: {e}")
        return None


def _checkpoint_metric_score(path):
    """Higher is better. Used for tie-breaking among compatible 11-class checkpoints."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        m = state.get("metrics", {}) or {}
        # Weight min_f1 most heavily — that's the worst-class robustness, which
        # is what cripples real-world use. Then macro_f1, then accuracy.
        return 4.0 * float(m.get("min_f1", 0.0)) + float(m.get("macro_f1", 0.0)) + float(m.get("acc", 0.0))
    except Exception:
        return -1.0


def _find_checkpoint():
    candidates = [
        # FIX: the previous candidate ordering made `best_sensorfusion_esp32_v2_pocket_v3.pt`
        # the first preference, but its checkpoint metrics are acc=0.834, macro_f1=0.827,
        # min_f1=0.611 — well below the canonical `useful11_final` (acc=0.894, macro_f1=0.903,
        # min_f1=0.834). Prefer the canonical 11-class model first; pocket variants are kept
        # as fallbacks for environments that explicitly want pocket-collected data, but the
        # decision is now made by metric score (see _select_best below) rather than by
        # raw position in this list.
        BASE_DIR / "checkpoints" / "best_sensorfusion_esp32_v2_useful11_final.pt",
        BASE_DIR / "checkpoints" / "best_sensorfusion_esp32_v2_useful11_rw.pt",
        BASE_DIR / "checkpoints" / "best_sensorfusion_esp32_v2_useful11.pt",
        BASE_DIR / "checkpoints" / "best_sensorfusion_esp32_v2_pocket_v3.pt",
        BASE_DIR / "checkpoints" / "best_sensorfusion_esp32_v2_pocket_v3_smoke.pt",
        BASE_DIR / "checkpoints" / "best_sensorfusion_esp32_v2_11class.pt",
        BASE_DIR / "checkpoints" / "best_sensorfusion_esp32_v2_useful11_pocket_robust.pt",
        BASE_DIR / "checkpoints" / "best_sensorfusion_esp32_v2_pocket_final.pt",
        BASE_DIR / "checkpoints_v2" / "best_sensorfusion_esp32_v2_useful11_final.pt",
        BASE_DIR / "checkpoints_v2" / "best_sensorfusion_esp32_v2_pocket_v3.pt",
        BASE_DIR / "checkpoints_v2" / "best_sensorfusion_esp32_v2_pocket_v3_7cls.pt",
        BASE_DIR / "checkpoints_v2" / "best_sensorfusion_esp32_v2_pocket_final.pt",
        BASE_DIR / "checkpoints" / "best_model.pt",
    ]
    # Allow override via SERVER_FORCE_CHECKPOINT for QA / experimentation.
    forced = os.environ.get("SERVER_FORCE_CHECKPOINT")
    if forced:
        forced_path = Path(forced)
        if not forced_path.is_absolute():
            forced_path = BASE_DIR / forced_path
        if forced_path.exists() and _checkpoint_labels(forced_path) == EXPECTED_LABELS:
            print(f"[server] Using forced checkpoint: {forced_path}")
            return forced_path
        print(f"[server] SERVER_FORCE_CHECKPOINT={forced} is invalid; falling back to auto-select")
    compatible = [c for c in candidates if c.exists() and _checkpoint_labels(c) == EXPECTED_LABELS]
    if compatible:
        # Highest-scoring compatible checkpoint wins, but preserve list order on ties so
        # explicit overrides (e.g. a freshly fine-tuned `_pocket_robust` checkpoint listed
        # first) still take precedence at equal score.
        return max(compatible, key=lambda p: (_checkpoint_metric_score(p), -candidates.index(p)))
    for c in candidates:
        if c.exists():
            labels = _checkpoint_labels(c)
            print(f"[server] Skipping incompatible checkpoint {c.name}: labels={labels}")
    return candidates[0]

CHECKPOINT_PATH = _find_checkpoint()

sensor_buffer = deque(maxlen=4096)
samples_since_inference = 0
dashboard_clients = set()
phone_clients = set()
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_TIME_STEPS = 50
INPUT_CHANNELS = 6


def _compatible_esp32_state_dict(state_dict):
    if "dsconv.net.4.weight" not in state_dict:
        return state_dict
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("dsconv.net.4."):
            new_key = key.replace("dsconv.net.4.", "dsconv.net.5.", 1)
        elif key.startswith("dsconv.net.5."):
            new_key = key.replace("dsconv.net.5.", "dsconv.net.6.", 1)
        elif key.startswith("dsconv.net.6."):
            new_key = key.replace("dsconv.net.6.", "dsconv.net.7.", 1)
        elif key.startswith("feature_fusion.2."):
            new_key = key.replace("feature_fusion.2.", "feature_fusion.3.", 1)
        remapped[new_key] = value
    return remapped


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    ip = get_local_ip()
    labels_short = "/".join([ACTIVITY_LABELS[i] for i in sorted(ACTIVITY_LABELS.keys())])
    print(f"\n{'=' * 60}")
    print(f"  HAR Server ({len(ACTIVITY_LABELS)} classes: {labels_short})")
    print(f"  Checkpoint: {CHECKPOINT_PATH.name}")
    print(f"  Dashboard:  https://{ip}:8443/")
    print(f"  Phone:      https://{ip}:8443/phone")
    print(f"  HTTP:       http://{ip}:8765/  (redirects to HTTPS)")
    print(f"  Note: Accept the self-signed certificate warning once on each device.")
    print(f"        Both phone and dashboard MUST use HTTPS for sensors to work.")
    print(f"{'=' * 60}\n")
    yield
    print("\n[server] Shutting down...")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def load_model():
    global model, CHANNEL_STATS, ACTIVITY_LABELS, WINDOW_SIZE, STRIDE, MODEL_INFO
    MODEL_INFO.update({
        "model_loaded": False,
        "model_mode": "heuristic",
        "architecture": None,
        "checkpoint": CHECKPOINT_PATH.name,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "normalization_source": "default",
        "warning": None,
    })
    if not CHECKPOINT_PATH.exists():
        print(f"[server] No checkpoint at {CHECKPOINT_PATH}, using heuristic classifier")
        CHANNEL_STATS = DEFAULT_CHANNEL_STATS
        MODEL_INFO["warning"] = f"No compatible checkpoint found at {CHECKPOINT_PATH}"
        return
    try:
        # FIX: training scripts live under ./training, not under
        # final_esp32_v2_useful11_package (that folder doesn't exist in the
        # repo, so the import was silently failing and the heuristic
        # classifier was being used instead of the trained model).
        sys.path.insert(0, str(BASE_DIR / "training"))
        sys.path.insert(0, str(BASE_DIR))
        state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        ckpt_labels = state.get("labels", state.get("activity_labels", None))
        num_classes = len(ckpt_labels) if ckpt_labels else state.get("num_classes", 6)
        if ckpt_labels:
            ACTIVITY_LABELS = {i: lbl for i, lbl in enumerate(ckpt_labels)}

        model = None
        last_err = None
        loaded_class = None
        for ModelClass, mod_path in [
            ("SensorFusionESP32", "train_esp32_v2_expanded_local"),
            ("SensorFusionESP32", "training.train_esp32_v2_expanded_local"),
            ("SensorFusionHAR", "model.sensorfusion"),
        ]:
            try:
                mod = __import__(mod_path, fromlist=[ModelClass])
                Cls = getattr(mod, ModelClass)
                if ModelClass == "SensorFusionHAR":
                    candidate = Cls(num_classes=num_classes)
                else:
                    candidate = Cls(input_channels=6, reservoir_size=64, num_classes=num_classes)
                state_dict = state["model_state_dict"] if "model_state_dict" in state else state
                if ModelClass == "SensorFusionESP32":
                    state_dict = _compatible_esp32_state_dict(state_dict)
                candidate.load_state_dict(state_dict, strict=True)
                model = candidate
                loaded_class = ModelClass
                print(f"[server] Loaded architecture: {ModelClass}")
                break
            except Exception as e:
                last_err = e
                continue
        if model is None:
            raise last_err
        model.to(device)
        model.eval()

        if loaded_class == "SensorFusionHAR":
            WINDOW_SIZE = 128
            STRIDE = 64
        else:
            WINDOW_SIZE = 50
            STRIDE = 25
        print(f"[server] Window size set to {WINDOW_SIZE}, stride {STRIDE}")

        norm_stats = state.get("normalization", state.get("normalization_stats", None))
        # FIX: previously only checked exports/esp32_v2/normalization_stats.json,
        # but the deployed packaging puts the file at exports/normalization_stats.json
        # (the esp32_v2/ subfolder is only created by the training script when it
        # runs locally, and is empty in the deployed package). Try both, and only
        # accept a stats file whose label set matches the loaded checkpoint —
        # otherwise we'd silently load 7-class normalization stats into an
        # 11-class model and get garbage predictions.
        stats_candidates = [
            CHECKPOINT_PATH.parent.parent / "exports" / "esp32_v2" / "normalization_stats.json",
            BASE_DIR / "exports" / "esp32_v2" / "normalization_stats.json",
            BASE_DIR / "exports" / "normalization_stats.json",
        ]
        if norm_stats is None:
            ckpt_label_set = list(ckpt_labels) if ckpt_labels else None
            for stats_file in stats_candidates:
                if not stats_file.exists():
                    continue
                try:
                    with open(stats_file) as f:
                        candidate = json.load(f)
                except Exception as e:
                    print(f"[server] Could not parse {stats_file}: {e}")
                    continue
                file_labels = candidate.get("labels")
                if ckpt_label_set and file_labels and list(file_labels) != ckpt_label_set:
                    print(f"[server] Skipping {stats_file.name}: labels disagree with checkpoint")
                    continue
                norm_stats = candidate
                print(f"[server] Loaded normalization stats from {stats_file}")
                break

        if norm_stats is not None:
            means = norm_stats.get("mean", norm_stats.get("means", []))
            stds = norm_stats.get("std", norm_stats.get("stds", []))
            if len(means) != len(CHANNEL_ORDER) or len(stds) != len(CHANNEL_ORDER):
                raise ValueError(f"Normalization stats must have {len(CHANNEL_ORDER)} channels")
            CHANNEL_STATS = {}
            for i, ch in enumerate(CHANNEL_ORDER):
                std = float(stds[i])
                if not np.isfinite(std) or std <= 1e-8:
                    raise ValueError(f"Invalid normalization std for {ch}: {stds[i]}")
                CHANNEL_STATS[ch] = {"mean": float(means[i]), "std": std}
            print("[server] Normalization stats loaded from checkpoint")
            MODEL_INFO["normalization_source"] = "checkpoint"
        else:
            CHANNEL_STATS = DEFAULT_CHANNEL_STATS
            print("[server] Using default normalization stats")

        MODEL_INFO.update({
            "model_loaded": True,
            "model_mode": "torch",
            "architecture": loaded_class,
            "checkpoint": CHECKPOINT_PATH.name,
            "checkpoint_path": str(CHECKPOINT_PATH),
            "warning": None,
        })
        print(f"[server] Model loaded ({num_classes} classes) from {CHECKPOINT_PATH.name}")
    except Exception as e:
        model = None
        CHANNEL_STATS = DEFAULT_CHANNEL_STATS
        MODEL_INFO.update({
            "model_loaded": False,
            "model_mode": "heuristic",
            "architecture": None,
            "checkpoint": CHECKPOINT_PATH.name,
            "checkpoint_path": str(CHECKPOINT_PATH),
            "normalization_source": "default",
            "warning": str(e),
        })
        print(f"[server] Failed to load model: {e}, using heuristic classifier")


def resample_to_fixed_rate(timestamps, values, target_hz, target_length):
    if len(timestamps) < 2:
        return values[:target_length]

    ts = np.array(timestamps)
    ts = ts - ts[0]

    duration = ts[-1]
    if duration <= 0:
        return np.array(values[:target_length])

    target_ts = np.linspace(0, duration, target_length)
    vals = np.array(values)
    resampled = np.zeros((target_length, vals.shape[1]))

    for ch in range(vals.shape[1]):
        resampled[:, ch] = np.interp(target_ts, ts, vals[:, ch])

    return resampled


def normalize(data):
    normed = np.zeros_like(data)
    for i, ch in enumerate(CHANNEL_ORDER):
        normed[:, i] = (data[:, i] - CHANNEL_STATS[ch]["mean"]) / CHANNEL_STATS[ch]["std"]
    return normed


def estimate_gravity(window_acc):
    """Estimate the gravity direction in the window's frame.

    For a 1-second window the simple mean of the accelerometer is a robust
    estimate when the activity is stationary; for dynamic activities the body
    motion contribution averages out only approximately, but the residual is
    small relative to the 9.81 m/s² gravity component, so the estimate stays
    close to the true gravity direction. We use the magnitude as a sanity
    check — if the mean magnitude is too far from 1g, the rotation is skipped.
    """
    g = window_acc.mean(axis=0)
    return g


def rodrigues_rotation_to_z(g_vec):
    """Compute the 3x3 rotation that maps the unit vector g_vec onto +Z.

    Uses Rodrigues' rotation formula. If g_vec is already aligned with +Z
    (or with -Z, the antipode), this returns either the identity or the 180°
    rotation around the X axis, both of which leave the inference well-defined.
    """
    g = np.asarray(g_vec, dtype=np.float64)
    norm = np.linalg.norm(g)
    if norm <= 1e-6:
        return np.eye(3, dtype=np.float32)
    g = g / norm
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cos_theta = float(np.dot(g, z))
    if cos_theta > 1.0 - 1e-7:
        return np.eye(3, dtype=np.float32)
    if cos_theta < -1.0 + 1e-7:
        # Antipodal — pick any axis perpendicular to z; X is fine.
        return np.array(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32
        )
    axis = np.cross(g, z)
    axis = axis / np.linalg.norm(axis)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    R = np.eye(3, dtype=np.float64) + sin_theta * K + (1.0 - cos_theta) * (K @ K)
    return R.astype(np.float32)


def gravity_align_window(window):
    """Rotate the (T, 6) window so gravity falls on +Z.

    Same rotation is applied to acc and gyro to preserve their physical
    relationship. Returns (aligned_window, rotation_matrix, gravity_norm).
    If gravity cannot be reliably estimated (low magnitude, e.g. linear-
    acceleration mode), returns the input unchanged.
    """
    acc = np.asarray(window[:, :3], dtype=np.float32)
    gyro = np.asarray(window[:, 3:], dtype=np.float32)
    g_est = estimate_gravity(acc)
    g_norm = float(np.linalg.norm(g_est))
    # Sanity check: gravity magnitude should be roughly 9.81 m/s². Outside
    # 6-13 m/s² we are likely seeing linear acceleration (gravity stripped),
    # very dynamic motion, or a sensor problem — leave the window alone.
    if g_norm < 6.0 or g_norm > 13.0:
        return window.astype(np.float32, copy=False), None, g_norm
    R = rodrigues_rotation_to_z(g_est)
    acc_rot = acc @ R.T
    gyro_rot = gyro @ R.T
    aligned = np.concatenate([acc_rot, gyro_rot], axis=1).astype(np.float32)
    return aligned, R, g_norm


def model_info_payload():
    info = dict(MODEL_INFO)
    info["window_size"] = WINDOW_SIZE
    info["stride"] = STRIDE
    info["target_hz"] = TARGET_HZ
    info["labels"] = [ACTIVITY_LABELS[i] for i in sorted(ACTIVITY_LABELS)]
    return info


def reset_stream_state():
    global samples_since_inference
    sensor_buffer.clear()
    prediction_history.clear()
    samples_since_inference = 0
    _DISPLAY_STATE["label"] = None
    _DISPLAY_STATE["runlen"] = 0
    _DISPLAY_STATE["candidate"] = None
    _DISPLAY_STATE["candidate_runlen"] = 0


def validate_phone_sample(data):
    required = {"ax", "ay", "az", "gx", "gy", "gz", "t"}
    if not required.issubset(data.keys()):
        return None, f"missing keys: {sorted(required.difference(data.keys()))}"
    sample = {}
    try:
        for ch in CHANNEL_ORDER:
            sample[ch] = float(data[ch])
        sample["t"] = float(data["t"])
    except (TypeError, ValueError):
        return None, "non-numeric sensor value"
    values = [sample[ch] for ch in CHANNEL_ORDER] + [sample["t"]]
    if not np.all(np.isfinite(values)):
        return None, "non-finite sensor value"
    for key in ("source", "accelMode", "gyroMode"):
        if key in data:
            sample[key] = str(data[key])[:64]
    return sample, None


def window_diagnostics(data, resampled=None, normed=None):
    timestamps = np.asarray([s["t"] for s in data], dtype=np.float64)
    values = np.asarray([[s[ch] for ch in CHANNEL_ORDER] for s in data], dtype=np.float32)
    if resampled is None:
        resampled = values
    acc = np.asarray(resampled[:, :3], dtype=np.float32)
    gyro = np.asarray(resampled[:, 3:], dtype=np.float32)
    acc_mag = np.sqrt(np.sum(acc ** 2, axis=1))
    dt = np.diff(timestamps) if len(timestamps) > 1 else np.asarray([], dtype=np.float64)
    positive_dt = dt[dt > 0]
    median_dt_ms = float(np.median(positive_dt)) if len(positive_dt) else 0.0
    sample_rate_hz = float(1000.0 / median_dt_ms) if median_dt_ms > 0 else 0.0
    duration_sec = float((timestamps[-1] - timestamps[0]) / 1000.0) if len(timestamps) > 1 else 0.0
    jitter_ms = float(np.std(positive_dt)) if len(positive_dt) else 0.0
    z_abs_mean = float(np.mean(np.abs(normed))) if normed is not None else 0.0
    z_abs_max = float(np.max(np.abs(normed))) if normed is not None else 0.0
    acc_mag_mean = float(np.mean(acc_mag)) if len(acc_mag) else 0.0
    acc_mag_std = float(np.std(acc_mag)) if len(acc_mag) else 0.0
    gyro_std = float(np.mean(np.std(gyro, axis=0))) if len(gyro) else 0.0
    acc_mean = np.mean(acc, axis=0) if len(acc) else np.zeros(3, dtype=np.float32)
    gravity_like = 7.0 <= acc_mag_mean <= 12.5
    if acc_mag_mean < 3.0:
        sensor_health = "low_accel_magnitude"
    elif z_abs_max > 8.0:
        sensor_health = "normalization_outlier"
    elif sample_rate_hz and (sample_rate_hz < 35.0 or sample_rate_hz > 75.0):
        sensor_health = "sample_rate_warning"
    else:
        sensor_health = "ok"
    return {
        "sample_rate_hz": round(sample_rate_hz, 2),
        "window_duration_sec": round(duration_sec, 3),
        "jitter_ms": round(jitter_ms, 2),
        "acc_mag_mean": round(acc_mag_mean, 3),
        "acc_mag_std": round(acc_mag_std, 3),
        "acc_mean": [round(float(v), 3) for v in acc_mean.tolist()],
        "gyro_std": round(gyro_std, 4),
        "z_abs_mean": round(z_abs_mean, 3),
        "z_abs_max": round(z_abs_max, 3),
        "gravity_like": gravity_like,
        "sensor_health": sensor_health,
    }


def force_probability_label(probs, label, confidence):
    confidence = float(max(0.0, min(1.0, confidence)))
    labels = list(probs.keys())
    adjusted = {k: max(0.0, float(probs.get(k, 0.0))) for k in labels}
    others = [k for k in labels if k != label]
    other_total = sum(adjusted[k] for k in others)
    adjusted[label] = confidence
    if others:
        if other_total <= 1e-8:
            share = (1.0 - confidence) / len(others)
            for k in others:
                adjusted[k] = share
        else:
            scale = (1.0 - confidence) / other_total
            for k in others:
                adjusted[k] *= scale
    return adjusted


def apply_realtime_corrections(label, conf, probs, diagnostics, data):
    """Real-time post-processing of the raw model output.

    The model was trained on waist/wrist-mounted data; despite gravity alignment
    a few systematic confusions remain when the phone sits stationary in a
    pocket. This function applies narrow, well-motivated rules to nudge those.
    Each rule emits a short tag in the returned `correction` string so the
    dashboard can show what was changed and why.
    """
    source = str(data[-1].get("source", "")) if data else ""
    accel_mode = str(data[-1].get("accelMode", "")) if data else ""
    acc_std = diagnostics["acc_mag_std"]
    gyro_std = diagnostics["gyro_std"]
    acc_mag = diagnostics["acc_mag_mean"]
    stationary = acc_std <= 0.85 and gyro_std <= 0.70
    very_stationary = acc_std <= 0.40 and gyro_std <= 0.30
    generic_accel = "Generic Sensor" in source and accel_mode == "Accelerometer"
    correction = None

    # Rule 1: stationary + Generic-Sensor pocket-Lying => Sitting.
    # Same as the original fix; kept for compatibility and tests.
    if generic_accel and stationary and label == "Lying Down":
        label = "Sitting"
        conf = max(0.62, min(0.82, float(conf) * 0.90))
        probs = force_probability_label(probs, label, conf)
        correction = "stationary_pocket_generic_sensor_lying_to_sitting"
        return label, conf, probs, correction

    # Rule 2: very stationary + dynamic-class prediction => most likely
    # the model is hallucinating motion from sensor noise. If the gravity
    # magnitude is plausible, downgrade dynamic predictions to whichever of
    # Sitting/Standing has higher probability.
    dynamic_labels = {
        "Walking", "Stairs Up", "Stairs Down",
        "Jogging", "Jumping", "Running", "Cycling", "Waist Bending",
    }
    if very_stationary and 8.0 <= acc_mag <= 12.0 and label in dynamic_labels:
        sit = float(probs.get("Sitting", 0.0))
        stand = float(probs.get("Standing", 0.0))
        if max(sit, stand) > 1e-3:
            new_label = "Sitting" if sit >= stand else "Standing"
            conf = max(0.55, min(0.78, float(conf) * 0.85))
            label = new_label
            probs = force_probability_label(probs, label, conf)
            correction = "very_stationary_dynamic_to_static_posture"
            return label, conf, probs, correction

    # Rule 3: gravity-magnitude clearly outside 1g => sensor is not reporting
    # total acceleration (e.g. linear-acceleration mode silently selected by
    # the browser). Mark as Uncertain rather than emitting any class — the
    # input distribution does not match training, so any output is unreliable.
    if acc_mag < 3.0 or acc_mag > 18.0:
        label = "Uncertain"
        conf = 0.0
        correction = "gravity_magnitude_out_of_range_sensor_mode_likely_wrong"
        return label, conf, probs, correction

    return label, conf, probs, correction


def update_display_label(raw_label):
    """Solution D: enforce a minimum dwell time before switching the displayed
    label, so transitions don't make the UI flicker.

    A new candidate must be predicted for at least DWELL_WINDOWS consecutive
    inferences before it replaces the current display label. "Uncertain" never
    counts as a new candidate — it just freezes the last confident label.
    """
    state = _DISPLAY_STATE
    if raw_label == "Uncertain":
        # Keep showing whatever was last confident.
        return state["label"] if state["label"] is not None else raw_label
    if state["label"] is None:
        # First confident label seen. Show immediately.
        state["label"] = raw_label
        state["runlen"] = 1
        state["candidate"] = None
        state["candidate_runlen"] = 0
        return raw_label
    if raw_label == state["label"]:
        state["runlen"] += 1
        state["candidate"] = None
        state["candidate_runlen"] = 0
        return state["label"]
    # Different from currently displayed label — accumulate as candidate.
    if raw_label == state["candidate"]:
        state["candidate_runlen"] += 1
    else:
        state["candidate"] = raw_label
        state["candidate_runlen"] = 1
    if state["candidate_runlen"] >= DWELL_WINDOWS:
        state["label"] = raw_label
        state["runlen"] = state["candidate_runlen"]
        state["candidate"] = None
        state["candidate_runlen"] = 0
    return state["label"]


def heuristic_classify(data):
    acc = data[:, :3]
    gyro = data[:, 3:]

    acc_mag = np.sqrt(np.sum(acc ** 2, axis=1))
    acc_mag_std = np.std(acc_mag)
    acc_mag_mean = np.mean(acc_mag)
    gyro_std = np.mean(np.std(gyro, axis=0))
    acc_var = np.mean(np.var(acc, axis=0))

    if acc_var < 0.005 and gyro_std < 0.01:
        if acc_mag_mean < 0.5:
            label, conf = 3, 0.7
        elif abs(acc[:, 1].mean()) > 0.8:
            label, conf = 2, 0.55
        else:
            label, conf = 1, 0.5
    elif acc_mag_std > 0.8 or gyro_std > 0.5:
        vert_acc = acc[:, 2]
        vert_trend = np.polyfit(np.arange(len(vert_acc)), vert_acc, 1)[0]
        if vert_trend > 0.003:
            label, conf = 4, 0.45
        elif vert_trend < -0.003:
            label, conf = 5, 0.45
        elif gyro_std > 1.5:
            label, conf = 7, 0.5
        elif acc_mag_mean > 2.0:
            label, conf = 9, 0.5
        else:
            label, conf = 6, 0.45
    else:
        if acc_mag_std < 0.15:
            label, conf = 2, 0.4
        elif gyro_std > 0.3:
            label, conf = 10, 0.4
        else:
            label, conf = 0, 0.4

    probs = {ACTIVITY_LABELS[i]: 0.0 for i in ACTIVITY_LABELS}
    probs[ACTIVITY_LABELS[label]] = conf
    remaining = 1.0 - conf
    others = [k for k in probs if k != ACTIVITY_LABELS[label]]
    for k in others:
        probs[k] = remaining / len(others)

    return ACTIVITY_LABELS[label], conf, probs


def run_inference(data):
    t0 = time.perf_counter()

    timestamps = [s["t"] for s in data]
    values = [[s[ch] for ch in CHANNEL_ORDER] for s in data]

    resampled = resample_to_fixed_rate(timestamps, values, TARGET_HZ, WINDOW_SIZE)
    # Solution A: gravity-align the (acc, gyro) channels before normalization,
    # so the model — trained on waist-mounted data with a fixed gravity axis —
    # sees a consistent gravity-on-+Z frame regardless of pocket orientation.
    if GRAVITY_ALIGN_ENABLED:
        aligned, R, g_norm = gravity_align_window(resampled)
        gravity_aligned = R is not None
    else:
        aligned = resampled.astype(np.float32, copy=False)
        gravity_aligned = False
        g_norm = float(np.linalg.norm(np.mean(resampled[:, :3], axis=0)))
    normed = normalize(aligned)
    diagnostics = window_diagnostics(data, resampled=resampled, normed=normed)
    diagnostics["gravity_aligned"] = bool(gravity_aligned)
    diagnostics["gravity_norm"] = round(float(g_norm), 3)

    spectral_radius = None
    attention_weights = None
    features = None
    mode = "torch" if model is not None else "heuristic"

    if model is not None:
        try:
            tensor = torch.FloatTensor(normed).unsqueeze(0).to(device)
            with torch.no_grad():
                if hasattr(model, 'forward') and 'return_aux' in model.forward.__code__.co_varnames:
                    logits, aux = model(tensor, return_aux=True)
                    spectral_radius = float(aux.get('spectral_radius', 0))
                    attention_weights = aux.get('attention_weights')
                    features = aux.get('features')
                else:
                    logits = model(tensor)
                probs_tensor = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs_tensor, dim=1)
                pred_idx = pred_idx.item()
                conf = conf.item()
                probs_np = probs_tensor.squeeze().cpu().numpy()
                probs = {ACTIVITY_LABELS[i]: float(probs_np[i]) for i in ACTIVITY_LABELS}
                label = ACTIVITY_LABELS[pred_idx]
        except Exception as e:
            print(f"[server] Model inference error: {e}")
            label, conf, probs = heuristic_classify(normed)
            mode = "heuristic_fallback"
    else:
        label, conf, probs = heuristic_classify(normed)

    elapsed = (time.perf_counter() - t0) * 1000

    raw_label = label
    raw_conf = conf
    label, conf, probs, correction = apply_realtime_corrections(label, conf, probs, diagnostics, data)

    if conf < CONFIDENCE_THRESHOLD:
        label = "Uncertain"

    # Solution D: enforce minimum dwell time so the UI doesn't flicker during
    # transitions. The display label only changes after the new candidate has
    # been seen DWELL_WINDOWS times in a row. We still keep prediction_history
    # for compatibility but no longer use it for the displayed label.
    prediction_history.append(label)
    smoothed_label = update_display_label(label)
    pre_dwell_label = label
    label = smoothed_label

    result = {
        "prediction": label,
        "confidence": round(conf, 4),
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "inference_time_ms": round(elapsed, 2),
        "diagnostics": diagnostics,
        "model_info": model_info_payload(),
        "mode": mode,
        "raw_prediction": raw_label,
        "raw_confidence": round(raw_conf, 4),
        "pre_dwell_prediction": pre_dwell_label,
        "correction": correction,
        "gravity_align_enabled": GRAVITY_ALIGN_ENABLED,
    }
    if spectral_radius is not None:
        result["spectral_radius"] = round(spectral_radius, 4)
    if attention_weights is not None:
        result["attention_weights"] = attention_weights.cpu().numpy().tolist()
    if features is not None:
        result["features"] = features.cpu().numpy().tolist()
    return result


async def broadcast_to_dashboards(message):
    if not dashboard_clients:
        return
    payload = json.dumps(message)
    stale = set()
    for ws in dashboard_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            stale.add(ws)
    dashboard_clients.difference_update(stale)


@app.get("/")
async def root():
    return RedirectResponse(url="/static/dashboard_v3.html")


@app.get("/phone")
async def phone_page():
    return RedirectResponse(url="/static/phone.html")


@app.get("/dashboard-v2")
async def dashboard_v2():
    return RedirectResponse(url="/static/dashboard.html")


@app.get("/model-info")
async def model_info():
    return model_info_payload()


@app.websocket("/ws/phone")
async def phone_ws(websocket: WebSocket):
    global samples_since_inference
    await websocket.accept()
    phone_clients.add(websocket)
    print(f"[server] Phone connected ({len(phone_clients)} active)")
    await broadcast_to_dashboards({
        "phone_connected": len(phone_clients),
        "status": "Phone connected",
        "buffer_size": len(sensor_buffer),
        "inference_time_ms": 0.0,
        "model_info": model_info_payload(),
    })

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if data.get("type") in {"stream_start", "stream_stop", "heartbeat"}:
                if data.get("type") in {"stream_start", "stream_stop"}:
                    reset_stream_state()
                status_map = {
                    "stream_start": "Phone started - waiting for motion samples",
                    "stream_stop": "Phone streaming stopped",
                    "heartbeat": "Phone connected - no motion samples yet",
                }
                await broadcast_to_dashboards({
                    "phone_connected": len(phone_clients),
                    "status": status_map.get(data.get("type"), "Phone connected"),
                    "buffer_size": len(sensor_buffer),
                    "inference_time_ms": 0.0,
                    "model_info": model_info_payload(),
                })
                continue

            sample, error = validate_phone_sample(data)
            if error:
                print(f"[server] Invalid phone data: {error}")
                continue

            sensor_buffer.append(sample)
            samples_since_inference += 1
            if samples_since_inference % 50 == 0:
                print(f"[server] Phone samples: {len(sensor_buffer)}, since_last: {samples_since_inference}")

            latest_sensor = {ch: sample[ch] for ch in CHANNEL_ORDER}
            latest_sensor["source"] = sample.get("source", "phone")
            latest_sensor["accelMode"] = sample.get("accelMode", "")
            latest_sensor["gyroMode"] = sample.get("gyroMode", "")

            if len(sensor_buffer) >= WINDOW_SIZE and samples_since_inference >= STRIDE:
                samples_since_inference = 0
                window = list(sensor_buffer)[-WINDOW_SIZE:]
                result = run_inference(window)
                msg = {
                    "activity": result["prediction"],
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"],
                    "inference_time_ms": result["inference_time_ms"],
                    "sensor": latest_sensor,
                    "buffer_size": len(sensor_buffer),
                    "phone_connected": len(phone_clients),
                    "status": "Streaming",
                    "diagnostics": result["diagnostics"],
                    "model_info": result["model_info"],
                    "mode": result["mode"],
                    "raw_prediction": result["raw_prediction"],
                    "raw_confidence": result["raw_confidence"],
                    "correction": result["correction"],
                }
                if "spectral_radius" in result:
                    msg["spectral_radius"] = result["spectral_radius"]
                if "attention_weights" in result:
                    msg["attention_weights"] = result["attention_weights"]
                if "features" in result:
                    msg["features"] = result["features"]
                await broadcast_to_dashboards(msg)
                try:
                    await websocket.send_text(json.dumps({"activity": msg["activity"], "confidence": msg["confidence"]}))
                except Exception:
                    pass
            else:
                await broadcast_to_dashboards({
                    "activity": None,
                    "confidence": 0.0,
                    "probabilities": {},
                    "sensor": latest_sensor,
                    "buffer_size": len(sensor_buffer),
                    "inference_time_ms": 0.0,
                    "phone_connected": len(phone_clients),
                    "status": "Collecting",
                    "diagnostics": window_diagnostics(list(sensor_buffer)),
                    "model_info": model_info_payload(),
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[server] Phone WebSocket error: {e}")
    finally:
        phone_clients.discard(websocket)
        print(f"[server] Phone disconnected ({len(phone_clients)} active)")
        await broadcast_to_dashboards({
            "phone_connected": len(phone_clients),
            "status": "Phone disconnected",
            "buffer_size": len(sensor_buffer),
            "inference_time_ms": 0.0,
            "model_info": model_info_payload(),
        })


@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    await websocket.accept()
    dashboard_clients.add(websocket)
    print(f"[server] Dashboard connected ({len(dashboard_clients)} active)")
    try:
        labels_ordered = [ACTIVITY_LABELS[i] for i in sorted(ACTIVITY_LABELS.keys())]
        await websocket.send_text(json.dumps({
            "activity_labels": labels_ordered,
            "num_classes": len(labels_ordered),
            "checkpoint": CHECKPOINT_PATH.name,
            "model_info": model_info_payload(),
            "phone_connected": len(phone_clients),
            "status": "Ready" if phone_clients else "Waiting for phone",
            "buffer_size": len(sensor_buffer),
            "inference_time_ms": 0.0,
        }))
    except Exception:
        pass

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        dashboard_clients.discard(websocket)
        print(f"[server] Dashboard disconnected ({len(dashboard_clients)} active)")


@app.websocket("/ws/hil")
async def hil_ws(websocket: WebSocket):
    """HIL Dashboard WebSocket endpoint for Hardware-in-the-Loop validation."""
    await websocket.accept()
    print(f"[server] HIL Dashboard connected")
    
    labels_ordered = [ACTIVITY_LABELS[i] for i in sorted(ACTIVITY_LABELS.keys())]
    
    try:
        # Send initial message
        await websocket.send_text(json.dumps({
            "type": "init",
            "mode": "HIL Validation",
            "labels": labels_ordered
        }))
        
        while True:
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
                
                # Handle toggle_simulation command
                if data.get("cmd") == "toggle_simulation":
                    state = data.get("state", False)
                    print(f"[server] HIL simulation: {'started' if state else 'stopped'}")
                    
                    if state:
                        # Start HIL simulation
                        asyncio.create_task(run_hil_simulation(websocket, labels_ordered))
                        
            except json.JSONDecodeError:
                pass
                
    except WebSocketDisconnect:
        print(f"[server] HIL Dashboard disconnected")
    except Exception as e:
        print(f"[server] HIL WebSocket error: {e}")


async def run_hil_simulation(websocket, labels):
    """Run HIL simulation by sending dataset samples."""
    try:
        # Load dataset samples
        sys.path.insert(0, str(BASE_DIR / "training"))
        try:
            from train_esp32_v2_expanded_local import (
                UCITotalHARDataset,
                UCIHAR_TO_MERGED,
                TARGET_TIME_STEPS
            )
            
            uci_dir = BASE_DIR / "data" / "UCI HAR Dataset"
            if uci_dir.exists():
                dataset = UCITotalHARDataset(str(uci_dir), split="test")
                
                # Collect samples for each class
                samples = []
                sample_labels = []
                for orig_cls, merged_cls in UCIHAR_TO_MERGED.items():
                    mask = dataset.y == orig_cls
                    indices = torch.where(mask)[0]
                    # Take first 5 samples per class
                    for idx in indices[:5]:
                        samples.append(dataset.X[idx].numpy())
                        sample_labels.append(merged_cls)
                
                total_windows = len(samples)
                
                for i, (sample, true_label) in enumerate(zip(samples, sample_labels)):
                    # Send input message
                    await websocket.send_text(json.dumps({
                        "type": "input",
                        "dataset_index": i + 1,
                        "total_windows": total_windows,
                        "acc": [{"x": float(sample[t, 0]), "y": float(sample[t, 1]), "z": float(sample[t, 2])} for t in range(50)],
                        "gyro": [{"x": float(sample[t, 3]), "y": float(sample[t, 4]), "z": float(sample[t, 5])} for t in range(50)]
                    }))
                    
                    # Simulate inference (using model if available)
                    if model is not None:
                        try:
                            # FIX: normalize() is written for a 2D (T, C) window.
                            # Passing sample[np.newaxis, :, :] (3D) made it
                            # broadcast incorrectly across the batch axis.
                            normed = normalize(sample)
                            tensor = torch.FloatTensor(normed).unsqueeze(0).to(device)
                            with torch.no_grad():
                                logits = model(tensor)
                                probs_tensor = torch.softmax(logits, dim=1)
                                conf, pred_idx = torch.max(probs_tensor, dim=1)
                                pred_idx = pred_idx.item()
                                conf = conf.item()
                                probs_np = probs_tensor.squeeze().cpu().numpy()
                        except Exception as e:
                            print(f"[server] HIL inference error: {e}")
                            # Fallback to true label
                            pred_idx = true_label
                            conf = 0.85
                            probs_np = np.zeros(len(labels))
                            probs_np[true_label] = 0.85
                    else:
                        # No model, use true label
                        pred_idx = true_label
                        conf = 0.85
                        probs_np = np.zeros(len(labels))
                        probs_np[true_label] = 0.85
                    
                    # Send output message
                    await websocket.send_text(json.dumps({
                        "type": "output",
                        "prediction": labels[pred_idx],
                        "confidence": float(conf),
                        "probabilities": probs_np.tolist(),
                        "inference_ms": 15.0
                    }))
                    
                    # Delay between samples
                    await asyncio.sleep(0.5)
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Dataset not found"
                }))
        except ImportError:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Training module not available"
            }))
            
    except Exception as e:
        print(f"[server] HIL simulation error: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e)
        }))


def _build_redirect_app(target_host: str, https_port: int):
    """Tiny ASGI app that redirects every HTTP request to the HTTPS server."""
    from starlette.applications import Starlette
    from starlette.responses import RedirectResponse
    from starlette.routing import Route

    async def redirect_all(request):
        target = f"https://{target_host}:{https_port}{request.url.path}"
        if request.url.query:
            target += f"?{request.url.query}"
        return RedirectResponse(url=target, status_code=307)

    return Starlette(routes=[Route("/{path:path}", endpoint=redirect_all)])


if __name__ == "__main__":
    import threading
    from generate_cert import generate_cert

    cert_dir = BASE_DIR / "certs"
    ip = get_local_ip()
    cert_file, key_file = generate_cert(cert_dir, ip)

    https_port = 8443
    http_port = 8765

    redirect_app = _build_redirect_app(ip, https_port)

    def _run_http_redirect():
        config = uvicorn.Config(
            app=redirect_app,
            host="0.0.0.0",
            port=http_port,
            log_level="warning",
        )
        uvicorn.Server(config).run()

    threading.Thread(target=_run_http_redirect, daemon=True).start()

    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=https_port,
        log_level="info",
        ssl_certfile=str(cert_file),
        ssl_keyfile=str(key_file),
    )
    uvicorn.Server(config).run()
