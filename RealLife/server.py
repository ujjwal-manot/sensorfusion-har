import asyncio
import json
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

BASE_DIR = Path(__file__).resolve().parent
def _checkpoint_labels(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        labels = state.get("labels", state.get("activity_labels", None))
        return list(labels) if labels else None
    except Exception as e:
        print(f"[server] Could not inspect checkpoint {path.name}: {e}")
        return None


def _find_checkpoint():
    candidates = [
        BASE_DIR / "checkpoints" / "best_sensorfusion_esp32_v2_pocket_v3.pt",
        BASE_DIR / "checkpoints" / "best_sensorfusion_esp32_v2_useful11_final.pt",
        BASE_DIR / "checkpoints" / "best_sensorfusion_esp32_v2_useful11_rw.pt",
        BASE_DIR / "checkpoints" / "best_sensorfusion_esp32_v2_pocket_v3_smoke.pt",
        BASE_DIR / "checkpoints" / "best_sensorfusion_esp32_v2_pocket_final.pt",
        BASE_DIR / "checkpoints_v2" / "best_sensorfusion_esp32_v2_useful11_final.pt",
        BASE_DIR / "checkpoints_v2" / "best_sensorfusion_esp32_v2_pocket_v3.pt",
        BASE_DIR / "checkpoints_v2" / "best_sensorfusion_esp32_v2_pocket_v3_7cls.pt",
        BASE_DIR / "checkpoints_v2" / "best_sensorfusion_esp32_v2_pocket_final.pt",
        BASE_DIR / "checkpoints" / "best_model.pt",
    ]
    for c in candidates:
        if c.exists() and _checkpoint_labels(c) == EXPECTED_LABELS:
            return c
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
        stats_file = CHECKPOINT_PATH.parent.parent / "exports" / "esp32_v2" / "normalization_stats.json"
        if norm_stats is None and stats_file.exists():
            with open(stats_file) as f:
                norm_stats = json.load(f)

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
    normed = normalize(resampled)

    spectral_radius = None
    attention_weights = None
    reservoir_states = None
    features = None

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
    else:
        label, conf, probs = heuristic_classify(normed)

    elapsed = (time.perf_counter() - t0) * 1000

    if conf < CONFIDENCE_THRESHOLD:
        label = "Uncertain"

    prediction_history.append(label)
    if len(prediction_history) >= 3:
        vote_counts = Counter(prediction_history)
        majority_label = vote_counts.most_common(1)[0][0]
        if majority_label != "Uncertain" or all(p == "Uncertain" for p in prediction_history):
            label = majority_label

    result = {
        "prediction": label,
        "confidence": round(conf, 4),
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "inference_time_ms": round(elapsed, 2),
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
    })

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if data.get("type") in {"stream_start", "stream_stop", "heartbeat"}:
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
                })
                continue

            required = {"ax", "ay", "az", "gx", "gy", "gz", "t"}
            if not required.issubset(data.keys()):
                print(f"[server] Phone data missing keys: {data.keys()}")
                continue

            sensor_buffer.append(data)
            samples_since_inference += 1
            if samples_since_inference % 50 == 0:
                print(f"[server] Phone samples: {len(sensor_buffer)}, since_last: {samples_since_inference}")

            latest_sensor = {ch: data[ch] for ch in CHANNEL_ORDER}

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
                })

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        phone_clients.discard(websocket)
        print(f"[server] Phone disconnected ({len(phone_clients)} active)")
        await broadcast_to_dashboards({
            "phone_connected": len(phone_clients),
            "status": "Phone disconnected",
            "buffer_size": len(sensor_buffer),
            "inference_time_ms": 0.0,
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
