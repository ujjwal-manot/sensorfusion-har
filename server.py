import asyncio
import json
import socket
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

ACTIVITY_LABELS = {
    0: "Walking",
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Sitting",
    4: "Standing",
    5: "Laying",
}

CHANNEL_STATS = None

DEFAULT_CHANNEL_STATS = {
    "ax": {"mean": -0.5016, "std": 0.5592},
    "ay": {"mean": 0.7806, "std": 0.5129},
    "az": {"mean": -0.0455, "std": 0.4037},
    "gx": {"mean": -0.0249, "std": 0.4524},
    "gy": {"mean": 0.0786, "std": 0.3516},
    "gz": {"mean": 0.0049, "std": 0.3291},
}

CHANNEL_ORDER = ["ax", "ay", "az", "gx", "gy", "gz"]
WINDOW_SIZE = 128
STRIDE = 64
TARGET_HZ = 50

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_model.pt"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

sensor_buffer = deque(maxlen=4096)
samples_since_inference = 0
dashboard_clients = set()
phone_clients = set()
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    global model, CHANNEL_STATS
    if not CHECKPOINT_PATH.exists():
        print(f"[server] No checkpoint at {CHECKPOINT_PATH}, using heuristic classifier")
        CHANNEL_STATS = DEFAULT_CHANNEL_STATS
        return
    try:
        from model.sensorfusion import SensorFusionHAR
        state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        num_classes = state.get("num_classes", 6)
        model = SensorFusionHAR(num_classes=num_classes)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()

        norm_stats = state.get("normalization_stats", None)
        stats_file = CHECKPOINT_PATH.parent / "normalization_stats.json"
        if norm_stats is None and stats_file.exists():
            with open(stats_file) as f:
                norm_stats = json.load(f)

        if norm_stats is not None:
            means = norm_stats["means"]
            stds = norm_stats["stds"]
            CHANNEL_STATS = {}
            for i, ch in enumerate(CHANNEL_ORDER):
                CHANNEL_STATS[ch] = {"mean": means[i], "std": stds[i]}
            print("[server] Normalization stats loaded from checkpoint")
        else:
            CHANNEL_STATS = DEFAULT_CHANNEL_STATS
            print("[server] Using default normalization stats")

        if "activity_labels" in state:
            global ACTIVITY_LABELS
            labels = state["activity_labels"]
            ACTIVITY_LABELS = {i: labels[i] for i in range(len(labels))}

        print(f"[server] Model loaded ({num_classes} classes)")
    except Exception as e:
        model = None
        CHANNEL_STATS = DEFAULT_CHANNEL_STATS
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
            label, conf = 5, 0.7
        elif abs(acc[:, 1].mean()) > 0.8:
            label, conf = 4, 0.55
        else:
            label, conf = 3, 0.5
    elif acc_mag_std > 0.8 or gyro_std > 0.5:
        vert_acc = acc[:, 2]
        vert_trend = np.polyfit(np.arange(len(vert_acc)), vert_acc, 1)[0]
        if vert_trend > 0.003:
            label, conf = 1, 0.45
        elif vert_trend < -0.003:
            label, conf = 2, 0.45
        else:
            label, conf = 0, 0.6
    else:
        if acc_mag_std < 0.15:
            label, conf = 4, 0.4
        else:
            label, conf = 0, 0.4

    probs = {ACTIVITY_LABELS[i]: 0.0 for i in range(6)}
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
    normed = normalize(resampled)

    if model is not None:
        try:
            tensor = torch.FloatTensor(normed).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs_tensor = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs_tensor, dim=1)
                pred_idx = pred_idx.item()
                conf = conf.item()
                probs_np = probs_tensor.squeeze().cpu().numpy()
                probs = {ACTIVITY_LABELS[i]: float(probs_np[i]) for i in range(6)}
                label = ACTIVITY_LABELS[pred_idx]
        except Exception:
            label, conf, probs = heuristic_classify(normed)
    else:
        label, conf, probs = heuristic_classify(normed)

    elapsed = (time.perf_counter() - t0) * 1000

    return {
        "prediction": label,
        "confidence": round(conf, 4),
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "inference_time_ms": round(elapsed, 2),
    }


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
    return RedirectResponse(url="/static/dashboard.html")


@app.get("/phone")
async def phone_page():
    return RedirectResponse(url="/static/phone.html")


@app.websocket("/ws/phone")
async def phone_ws(websocket: WebSocket):
    global samples_since_inference
    await websocket.accept()
    phone_clients.add(websocket)
    print(f"[server] Phone connected ({len(phone_clients)} active)")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            required = {"ax", "ay", "az", "gx", "gy", "gz", "t"}
            if not required.issubset(data.keys()):
                continue

            sensor_buffer.append(data)
            samples_since_inference += 1

            latest_sensor = {ch: data[ch] for ch in CHANNEL_ORDER}

            if len(sensor_buffer) >= WINDOW_SIZE and samples_since_inference >= STRIDE:
                samples_since_inference = 0
                window = list(sensor_buffer)[-WINDOW_SIZE:]
                result = run_inference(window)
                result["sensor_data"] = latest_sensor
                result["buffer_size"] = len(sensor_buffer)
                await broadcast_to_dashboards(result)
            else:
                await broadcast_to_dashboards({
                    "prediction": None,
                    "confidence": 0.0,
                    "probabilities": {},
                    "sensor_data": latest_sensor,
                    "buffer_size": len(sensor_buffer),
                    "inference_time_ms": 0.0,
                })

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        phone_clients.discard(websocket)
        print(f"[server] Phone disconnected ({len(phone_clients)} active)")


@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    await websocket.accept()
    dashboard_clients.add(websocket)
    print(f"[server] Dashboard connected ({len(dashboard_clients)} active)")

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


@app.on_event("startup")
async def startup():
    load_model()
    ip = get_local_ip()
    port = 8765
    print(f"\n{'=' * 50}")
    print(f"  SensorFusion-HAR Server")
    print(f"  Dashboard:  http://{ip}:{port}/")
    print(f"  Phone:      http://{ip}:{port}/phone")
    print(f"  QR target:  http://{ip}:{port}/phone")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8765, reload=False, log_level="info")
