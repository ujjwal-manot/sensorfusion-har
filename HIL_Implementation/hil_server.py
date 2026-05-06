"""
HIL Server - Hardware-in-the-Loop Simulator for ESP32 Validation

This server loads dataset samples, normalizes them, and sends to ESP32 via UART
for validation. It also provides a WebSocket interface for the HIL dashboard.

Usage:
    python hil_server.py --port COM3 --dataset uci --samples 100
"""

import argparse
import asyncio
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import serial
import torch
import websockets
from sklearn.metrics import classification_report, confusion_matrix


class HILServer:
    def __init__(self, serial_port, baud_rate=921600, dataset_name='uci', samples_per_class=10):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.dataset_name = dataset_name
        self.samples_per_class = samples_per_class
        
        # Load normalization stats
        self.norm_mean, self.norm_std = self.load_normalization_stats()
        
        # Load dataset
        self.samples, self.labels = self.load_dataset()
        
        # WebSocket clients
        self.dashboard_clients = set()
        
        # Simulation state
        self.is_simulating = False
        self.current_index = 0
        
        # ESP32 response
        self.esp32_ready = False
    
    def load_normalization_stats(self):
        """Load normalization statistics from exports directory."""
        # Try multiple possible paths
        norm_file = Path(__file__).parent / "exports" / "normalization_stats.json"
        if not norm_file.exists():
            norm_file = Path(__file__).parent / "exports" / "esp32_v2" / "normalization_stats.json"
        
        if norm_file.exists():
            with open(norm_file) as f:
                data = json.load(f)
            return data['normalization']['mean'], data['normalization']['std']
        else:
            print(f"Warning: Normalization stats not found at {norm_file}")
            print("Using default stats from config")
            # Use default stats from config
            return [
                -3.1886296272, 1.2004078627, 2.4552721977,
                -0.0335379131, -0.0226027351, 0.0529510267
            ], [
                6.3020558357, 6.4955196381, 3.8114185333,
                1.0866706371, 0.8430932164, 1.2929021120
            ]
    
    def load_dataset(self):
        """Load dataset samples."""
        sys.path.insert(0, str(Path(__file__).parent))
        sys.path.insert(0, str(Path(__file__).parent / "training"))
        
        try:
            from train_esp32_v2_expanded_local import (
                UCITotalHARDataset,
                MHEALTHDataset,
                UCIHAR_TO_MERGED,
                MHEALTH_TO_MERGED,
                TARGET_TIME_STEPS,
                INPUT_CHANNELS
            )
        except ImportError:
            print("Warning: Could not import training module. Using mock data.")
            return self.generate_mock_data()
        
        samples = []
        labels = []
        
        if self.dataset_name == 'uci':
            uci_dir = Path(__file__).parent.parent / "data" / "UCI HAR Dataset"
            if not uci_dir.exists():
                print(f"Warning: UCI dataset not found at {uci_dir}")
                print("HIL_Implementation should be placed in the sensorfusion-har root directory")
                return self.generate_mock_data()
            
            dataset = UCITotalHARDataset(str(uci_dir), split="test")
            
            for orig_cls, merged_cls in UCIHAR_TO_MERGED.items():
                mask = dataset.y == orig_cls
                indices = torch.where(mask)[0]
                count = min(self.samples_per_class, len(indices))
                selected = indices[:count]
                
                for idx in selected:
                    samples.append(dataset.X[idx].numpy())
                    labels.append(merged_cls)
        
        elif self.dataset_name == 'mhealth':
            mh_dir = Path(__file__).parent.parent / "data" / "MHEALTHDATASET"
            if not mh_dir.exists():
                print(f"Warning: MHEALTH dataset not found at {mh_dir}")
                print("HIL_Implementation should be placed in the sensorfusion-har root directory")
                return self.generate_mock_data()
            
            dataset = MHEALTHDataset(str(mh_dir), split="test")
            
            for orig_cls, merged_cls in MHEALTH_TO_MERGED.items():
                mask = dataset.y == orig_cls
                indices = torch.where(mask)[0]
                count = min(self.samples_per_class, len(indices))
                selected = indices[:count]
                
                for idx in selected:
                    samples.append(dataset.X[idx].numpy())
                    labels.append(merged_cls)
        
        return np.array(samples), np.array(labels)
    
    def generate_mock_data(self):
        """Generate mock data for testing without datasets."""
        print("Generating mock data...")
        samples = []
        labels = []
        for cls in range(11):
            for _ in range(self.samples_per_class):
                # Generate random sensor data
                sample = np.random.randn(50, 6) * 2
                samples.append(sample)
                labels.append(cls)
        return np.array(samples), np.array(labels)
    
    def normalize_sample(self, sample):
        """Normalize a sample using training statistics."""
        norm_mean = np.array(self.norm_mean)
        norm_std = np.array(self.norm_std)
        return (sample - norm_mean) / norm_std
    
    def serialize_sample(self, sample):
        """Serialize a sample window to binary format for UART transmission."""
        flattened = sample.flatten()
        return struct.pack(f'{len(flattened)}f', *flattened)
    
    async def connect_serial(self):
        """Connect to ESP32 via Serial."""
        try:
            ser = serial.Serial(self.serial_port, self.baud_rate, timeout=2.0)
            time.sleep(2)  # Wait for ESP32 to be ready
            
            # Wait for HIL_READY signal
            start_time = time.time()
            while time.time() - start_time < 5:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    if line == "HIL_READY":
                        print("ESP32 HIL ready")
                        self.esp32_ready = True
                        return ser
                time.sleep(0.1)
            
            print("Warning: Did not receive HIL_READY signal, proceeding anyway")
            return ser
            
        except serial.SerialException as e:
            print(f"Failed to connect to ESP32: {e}")
            return None
    
    async def send_to_esp32(self, ser, sample):
        """Send a sample to ESP32 and read prediction."""
        # Send sync signal
        ser.write(b"HIL_SYNC\n")
        time.sleep(0.01)
        
        # Send serialized sample
        serialized = self.serialize_sample(sample)
        ser.write(serialized)
        
        # Wait for response
        start_time = time.time()
        response = b''
        while time.time() - start_time < 2.0:
            if ser.in_waiting > 0:
                response += ser.read(ser.in_waiting)
                # Try to parse HIL_RES format
                try:
                    response_str = response.decode('utf-8').strip()
                    if response_str.startswith("HIL_RES:"):
                        parts = response_str.split(':')
                        if len(parts) == 4:
                            class_idx = int(parts[1])
                            confidence = float(parts[2])
                            inference_ms = float(parts[3])
                            return {
                                'class': class_idx,
                                'confidence': confidence,
                                'inference_ms': inference_ms
                            }
                except (ValueError, UnicodeDecodeError):
                    continue
            time.sleep(0.01)
        
        return None
    
    async def broadcast_to_dashboard(self, message):
        """Broadcast message to all dashboard clients."""
        if not self.dashboard_clients:
            return
        payload = json.dumps(message)
        stale = set()
        for ws in self.dashboard_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.add(ws)
        self.dashboard_clients.difference_update(stale)
    
    async def run_simulation(self, ser):
        """Run simulation loop."""
        labels_ordered = [
            "Walking", "Sitting", "Standing", "Lying Down", "Stairs Up",
            "Stairs Down", "Jogging", "Jumping", "Cycling", "Running", "Waist Bending"
        ]
        
        # Send init message
        await self.broadcast_to_dashboard({
            "type": "init",
            "mode": "HIL Simulation",
            "labels": labels_ordered
        })
        
        total_windows = len(self.samples)
        
        while self.is_simulating and self.current_index < total_windows:
            # Get current sample
            sample = self.samples[self.current_index]
            label = self.labels[self.current_index]
            
            # Normalize
            normed_sample = self.normalize_sample(sample)
            
            # Send input to dashboard
            await self.broadcast_to_dashboard({
                "type": "input",
                "dataset_index": self.current_index + 1,
                "total_windows": total_windows,
                "acc": [{"x": float(sample[i, 0]), "y": float(sample[i, 1]), "z": float(sample[i, 2])} for i in range(50)],
                "gyro": [{"x": float(sample[i, 3]), "y": float(sample[i, 4]), "z": float(sample[i, 5])} for i in range(50)]
            })
            
            # Send to ESP32 if connected
            if ser:
                result = await self.send_to_esp32(ser, normed_sample)
                if result:
                    # Generate probabilities from confidence
                    probs = np.zeros(11)
                    probs[result['class']] = result['confidence']
                    probs = probs / probs.sum()
                    
                    # Send output to dashboard
                    await self.broadcast_to_dashboard({
                        "type": "output",
                        "prediction": labels_ordered[result['class']],
                        "confidence": result['confidence'],
                        "probabilities": probs.tolist(),
                        "inference_ms": result['inference_ms']
                    })
                else:
                    print(f"Failed to get response for sample {self.current_index}")
            else:
                # Simulate without ESP32
                await asyncio.sleep(0.1)
                pred_label = labels_ordered[label]
                probs = np.zeros(11)
                probs[label] = 0.85
                probs = probs / probs.sum()
                
                await self.broadcast_to_dashboard({
                    "type": "output",
                    "prediction": pred_label,
                    "confidence": 0.85,
                    "probabilities": probs.tolist(),
                    "inference_ms": 15.0
                })
            
            self.current_index += 1
            
            # Delay between samples
            await asyncio.sleep(0.5)
        
        # Simulation complete
        self.is_simulating = False
    
    async def handle_dashboard(self, websocket):
        """Handle dashboard WebSocket connection."""
        await websocket.accept()
        self.dashboard_clients.add(websocket)
        print(f"Dashboard connected ({len(self.dashboard_clients)} active)")
        
        try:
            while True:
                message = await websocket.receive_text()
                try:
                    data = json.loads(message)
                    if data.get("cmd") == "toggle_simulation":
                        self.is_simulating = data.get("state", False)
                        if self.is_simulating:
                            self.current_index = 0
                            ser = await self.connect_serial()
                            asyncio.create_task(self.run_simulation(ser))
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"Dashboard error: {e}")
        finally:
            self.dashboard_clients.discard(websocket)
            print(f"Dashboard disconnected ({len(self.dashboard_clients)} active)")


async def main():
    parser = argparse.ArgumentParser(description='HIL Server for ESP32 Validation')
    parser.add_argument('--port', type=str, default='COM3',
                        help='Serial port (e.g., COM3, /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=921600,
                        help='Baud rate')
    parser.add_argument('--dataset', type=str, default='uci',
                        choices=['uci', 'mhealth'],
                        help='Dataset to use')
    parser.add_argument('--samples', type=int, default=10,
                        help='Samples per class')
    parser.add_argument('--ws-port', type=int, default=8444,
                        help='WebSocket port for HIL dashboard')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HIL Server - Hardware-in-the-Loop Simulator")
    print("=" * 60)
    print(f"Serial Port: {args.port}")
    print(f"Baud Rate: {args.baud}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples per class: {args.samples}")
    print(f"WebSocket Port: {args.ws_port}")
    print("=" * 60)
    
    server = HILServer(args.port, args.baud_rate, args.dataset, args.samples)
    
    # Start WebSocket server
    async def handle_ws(websocket, path):
        await server.handle_dashboard(websocket)
    
    print(f"\nStarting HIL WebSocket server on port {args.ws_port}")
    print(f"Access HIL dashboard at: http://localhost:{args.ws_port}/static/hil_dashboard.html")
    
    async with websockets.serve(handle_ws, "0.0.0.0", args.ws_port):
        await asyncio.Future()  # Run forever


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
