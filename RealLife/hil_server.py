"""
HIL Server - Hardware-in-the-Loop Simulator for ESP32 Validation.

Loads dataset samples, normalises them, optionally streams them to an ESP32
over UART, and broadcasts everything to a HIL dashboard over a WebSocket.

Usage:
    python hil_server.py --port COM3 --dataset uci --samples 10
    python hil_server.py --no-serial --dataset uci --samples 5     # PC-only

What was wrong with the previous version (and why this rewrite was needed):

  1.  The argparse flag was --baud but the code read args.baud_rate, which
      raised AttributeError on every run.
  2.  load_normalization_stats() looked up data['normalization']['mean'],
      but the JSON file written by the training pipeline stores 'mean' and
      'std' at the top level. This always fell back to defaults silently.
  3.  Path("exports/...") is relative to the CWD, so the script only worked
      if you happened to run it from the right folder.
  4.  The class instantiated `websockets.serve(...)` (the websockets lib)
      but its handler called `websocket.accept()`, `.receive_text()` and
      `.send_text()` -- those are FastAPI/Starlette methods. The two APIs
      are not interchangeable, so the dashboard never received any frames.

This file uses the websockets library consistently (recv / send / no accept).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import websockets
from sklearn.metrics import classification_report, confusion_matrix

try:
    import serial  # pyserial
except ImportError:
    serial = None

HERE = Path(__file__).resolve().parent
LABELS_11 = [
    "Walking", "Sitting", "Standing", "Lying Down", "Stairs Up",
    "Stairs Down", "Jogging", "Jumping", "Cycling", "Running", "Waist Bending",
]


class HILServer:
    def __init__(self, serial_port, baud_rate=921600,
                 dataset_name="uci", samples_per_class=10, use_serial=True):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.dataset_name = dataset_name
        self.samples_per_class = samples_per_class
        self.use_serial = use_serial and serial is not None

        self.norm_mean, self.norm_std = self.load_normalization_stats()
        self.samples, self.labels = self.load_dataset()

        self.dashboard_clients = set()
        self.is_simulating = False
        self.current_index = 0
        self.esp32_ready = False

    # ------------------------------------------------------------------ data

    def load_normalization_stats(self):
        """
        Read normalization stats. The training script writes a flat JSON of
        the form {"mean": [...], "std": [...]}. The previous code expected
        a nested {"normalization": {"mean": [...], ...}} layout that doesn't
        exist on disk, so it always silently fell back to defaults.
        """
        candidates = [
            HERE / "exports" / "normalization_stats.json",
            HERE / "exports" / "esp32_v2" / "normalization_stats.json",
        ]
        for path in candidates:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                # Accept both flat and nested layouts to stay compatible
                # with older checkpoints.
                if "normalization" in data:
                    data = data["normalization"]
                mean = data.get("mean")
                std = data.get("std")
                if mean is not None and std is not None:
                    print(f"[hil] loaded normalization from {path}")
                    return mean, std

        print("[hil] normalization_stats.json not found; using config defaults")
        return (
            [-3.1886296272, 1.2004078627, 2.4552721977,
             -0.0335379131, -0.0226027351, 0.0529510267],
            [6.3020558357, 6.4955196381, 3.8114185333,
             1.0866706371, 0.8430932164, 1.2929021120],
        )

    def load_dataset(self):
        """Load test windows from UCI-HAR or MHEALTH; fall back to mock data."""
        sys.path.insert(0, str(HERE))
        sys.path.insert(0, str(HERE / "training"))
        try:
            from train_esp32_v2_expanded_local import (
                UCITotalHARDataset,
                MHEALTHDataset,
                UCIHAR_TO_MERGED,
                MHEALTH_TO_MERGED,
            )
        except ImportError as exc:
            print(f"[hil] training module unavailable ({exc}); using mock data")
            return self.generate_mock_data()

        if self.dataset_name == "uci":
            uci_dir = HERE / "data" / "UCI HAR Dataset"
            if not uci_dir.exists():
                print(f"[hil] UCI dataset not found at {uci_dir}; using mock data")
                return self.generate_mock_data()
            ds = UCITotalHARDataset(str(uci_dir), split="test")
            mapping = UCIHAR_TO_MERGED
        elif self.dataset_name == "mhealth":
            mh_dir = HERE / "data" / "MHEALTHDATASET"
            if not mh_dir.exists():
                print(f"[hil] MHEALTH dataset not found at {mh_dir}; using mock data")
                return self.generate_mock_data()
            ds = MHEALTHDataset(str(mh_dir), split="test")
            mapping = MHEALTH_TO_MERGED
        else:
            print(f"[hil] unknown dataset {self.dataset_name!r}; using mock data")
            return self.generate_mock_data()

        samples, labels = [], []
        for orig_cls, merged_cls in mapping.items():
            mask = ds.y == orig_cls
            indices = torch.where(mask)[0]
            count = min(self.samples_per_class, len(indices))
            for idx in indices[:count]:
                samples.append(ds.X[idx].numpy())
                labels.append(merged_cls)
        if not samples:
            return self.generate_mock_data()
        return np.array(samples), np.array(labels)

    def generate_mock_data(self):
        print("[hil] generating mock data (synthetic Gaussian windows)")
        samples, labels = [], []
        for cls in range(11):
            for _ in range(self.samples_per_class):
                samples.append(np.random.randn(50, 6).astype(np.float32) * 2.0)
                labels.append(cls)
        return np.array(samples), np.array(labels)

    # --------------------------------------------------------------- helpers

    def normalize_sample(self, sample):
        return (sample - np.asarray(self.norm_mean)) / np.asarray(self.norm_std)

    @staticmethod
    def serialize_sample(sample):
        flat = sample.astype(np.float32).flatten()
        return struct.pack(f"{len(flat)}f", *flat)

    # --------------------------------------------------------------- serial

    async def connect_serial(self):
        if not self.use_serial:
            return None
        if serial is None:
            print("[hil] pyserial not installed; running in PC-only mode")
            return None
        try:
            ser = serial.Serial(self.serial_port, self.baud_rate, timeout=2.0)
            await asyncio.sleep(2.0)  # ESP32 boot
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if ser.in_waiting > 0:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if line == "HIL_READY":
                        print("[hil] ESP32 reports HIL_READY")
                        self.esp32_ready = True
                        return ser
                await asyncio.sleep(0.1)
            print("[hil] no HIL_READY received; proceeding anyway")
            return ser
        except serial.SerialException as exc:
            print(f"[hil] could not open serial port {self.serial_port}: {exc}")
            return None

    async def send_to_esp32(self, ser, sample):
        ser.write(b"HIL_SYNC\n")
        await asyncio.sleep(0.01)
        ser.write(self.serialize_sample(sample))

        deadline = time.time() + 2.0
        buf = b""
        while time.time() < deadline:
            if ser.in_waiting > 0:
                buf += ser.read(ser.in_waiting)
                try:
                    line = buf.decode("utf-8", errors="ignore").strip()
                except UnicodeDecodeError:
                    line = ""
                if line.startswith("HIL_RES:"):
                    parts = line.split(":")
                    if len(parts) == 4:
                        try:
                            return {
                                "class": int(parts[1]),
                                "confidence": float(parts[2]),
                                "inference_ms": float(parts[3]),
                            }
                        except ValueError:
                            pass
            await asyncio.sleep(0.01)
        return None

    # --------------------------------------------------------- websocket I/O

    async def broadcast(self, message):
        if not self.dashboard_clients:
            return
        payload = json.dumps(message)
        stale = set()
        for ws in self.dashboard_clients:
            try:
                await ws.send(payload)             # websockets lib API
            except Exception:
                stale.add(ws)
        self.dashboard_clients.difference_update(stale)

    async def run_simulation(self, ser):
        await self.broadcast({
            "type": "init",
            "mode": "HIL Simulation",
            "labels": LABELS_11,
        })

        total = len(self.samples)
        y_true, y_pred = [], []

        while self.is_simulating and self.current_index < total:
            sample = self.samples[self.current_index]
            true_label = int(self.labels[self.current_index])

            normed = self.normalize_sample(sample)

            await self.broadcast({
                "type": "input",
                "dataset_index": self.current_index + 1,
                "total_windows": total,
                "true_label": LABELS_11[true_label],
                "acc": [{"x": float(sample[i, 0]),
                         "y": float(sample[i, 1]),
                         "z": float(sample[i, 2])} for i in range(50)],
                "gyro": [{"x": float(sample[i, 3]),
                          "y": float(sample[i, 4]),
                          "z": float(sample[i, 5])} for i in range(50)],
            })

            if ser is not None:
                result = await self.send_to_esp32(ser, normed)
                if result is None:
                    print(f"[hil] no ESP32 response on sample {self.current_index}")
                    self.current_index += 1
                    await asyncio.sleep(0.2)
                    continue
                pred_idx = result["class"]
                conf = result["confidence"]
                inference_ms = result["inference_ms"]
            else:
                # PC-only path: spoof the prediction with the true label
                # so the dashboard pipeline can be exercised end-to-end.
                await asyncio.sleep(0.1)
                pred_idx = true_label
                conf = 0.85
                inference_ms = 15.0

            y_true.append(true_label)
            y_pred.append(pred_idx)

            probs = np.zeros(11, dtype=np.float32)
            probs[pred_idx] = conf
            if probs.sum() > 0:
                probs = probs / probs.sum()

            await self.broadcast({
                "type": "output",
                "prediction": LABELS_11[pred_idx],
                "confidence": float(conf),
                "probabilities": probs.tolist(),
                "inference_ms": float(inference_ms),
            })

            self.current_index += 1
            await asyncio.sleep(0.5)

        if y_true:
            print("\n[hil] classification report")
            print(classification_report(
                y_true, y_pred,
                labels=list(range(11)),
                target_names=LABELS_11,
                zero_division=0,
                digits=4,
            ))
            print("[hil] confusion matrix")
            print(confusion_matrix(y_true, y_pred, labels=list(range(11))))

        self.is_simulating = False

    async def handle_dashboard(self, websocket):
        """websockets-library handler. No accept(); recv()/send() only."""
        self.dashboard_clients.add(websocket)
        print(f"[hil] dashboard connected ({len(self.dashboard_clients)} active)")
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                if data.get("cmd") == "toggle_simulation":
                    self.is_simulating = bool(data.get("state", False))
                    if self.is_simulating:
                        self.current_index = 0
                        ser = await self.connect_serial()
                        asyncio.create_task(self.run_simulation(ser))
        finally:
            self.dashboard_clients.discard(websocket)
            print(f"[hil] dashboard disconnected ({len(self.dashboard_clients)} active)")


async def amain():
    parser = argparse.ArgumentParser(description="HIL Server for ESP32 Validation")
    parser.add_argument("--port", type=str, default="COM3",
                        help="Serial port (e.g. COM3, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=921600, help="UART baud rate")
    parser.add_argument("--dataset", type=str, default="uci",
                        choices=["uci", "mhealth"], help="Source dataset")
    parser.add_argument("--samples", type=int, default=10,
                        help="Samples per class")
    parser.add_argument("--ws-port", type=int, default=8444,
                        help="WebSocket port for the HIL dashboard")
    parser.add_argument("--no-serial", action="store_true",
                        help="Do not open a serial port; run entirely on PC")
    args = parser.parse_args()

    print("=" * 60)
    print("HIL Server  (FIX: --baud now read correctly; no FastAPI mismatch)")
    print("=" * 60)
    print(f"  serial port  : {args.port if not args.no_serial else '(disabled)'}")
    print(f"  baud rate    : {args.baud}")
    print(f"  dataset      : {args.dataset}")
    print(f"  samples/cls  : {args.samples}")
    print(f"  websocket    : ws://0.0.0.0:{args.ws_port}")
    print("=" * 60)

    server = HILServer(
        serial_port=args.port,
        baud_rate=args.baud,
        dataset_name=args.dataset,
        samples_per_class=args.samples,
        use_serial=not args.no_serial,
    )

    async with websockets.serve(server.handle_dashboard, "0.0.0.0", args.ws_port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n[hil] interrupted, shutting down")
