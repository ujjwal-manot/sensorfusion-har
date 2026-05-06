"""
Dataset testing script for ESP32 deployment.
Sends dataset samples to ESP32 via Serial for validation.

Usage:
    python esp32_dataset_test.py --port COM3 --dataset uci --samples_per_class 10
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import serial
import torch
from sklearn.metrics import classification_report, confusion_matrix


def load_normalization_stats(norm_json_path):
    """Load normalization statistics from JSON file.

    FIX: training pipeline writes a flat {"mean": [...], "std": [...]} layout;
    the previous code looked up data['normalization']['mean'] which never
    matched. Accept both layouts so old files keep working too.
    """
    with open(norm_json_path) as f:
        data = json.load(f)
    if "normalization" in data:
        data = data["normalization"]
    return data["mean"], data["std"]


def load_dataset_samples(dataset_name, data_root='data', samples_per_class=10):
    """Load sample windows from dataset."""
    # FIX: previous code added `final_esp32_v2_useful11_package` to sys.path,
    # but that folder doesn't exist anywhere in the repo — the actual training
    # script lives under ./training/.
    here = Path(__file__).parent
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(here / "training"))

    from train_esp32_v2_expanded_local import (
        UCITotalHARDataset,
        MHEALTHDataset,
        RealWorldLocalDataset,
        UCIHAR_TO_MERGED,
        MHEALTH_TO_MERGED,
        REALWORLD_TO_MERGED,
        TARGET_TIME_STEPS,
        INPUT_CHANNELS
    )

    samples = []
    labels = []

    if dataset_name == 'uci':
        uci_dir = Path(data_root) / "UCI HAR Dataset"
        dataset = UCITotalHARDataset(str(uci_dir), split="test")

        # Get samples for each mapped class
        for orig_cls, merged_cls in UCIHAR_TO_MERGED.items():
            mask = dataset.y == orig_cls
            indices = torch.where(mask)[0]
            count = min(samples_per_class, len(indices))
            selected = indices[:count]

            for idx in selected:
                samples.append(dataset.X[idx].numpy())
                labels.append(merged_cls)

    elif dataset_name == 'mhealth':
        mh_dir = Path(data_root) / "MHEALTHDATASET"
        dataset = MHEALTHDataset(str(mh_dir), split="test")

        for orig_cls, merged_cls in MHEALTH_TO_MERGED.items():
            mask = dataset.y == orig_cls
            indices = torch.where(mask)[0]
            count = min(samples_per_class, len(indices))
            selected = indices[:count]

            for idx in selected:
                samples.append(dataset.X[idx].numpy())
                labels.append(merged_cls)

    elif dataset_name == 'realworld':
        dataset = RealWorldLocalDataset(str(data_root))

        for activity, label in REALWORLD_TO_MERGED.items():
            mask = dataset.y == label
            indices = torch.where(mask)[0]
            count = min(samples_per_class, len(indices))
            selected = indices[:count]

            for idx in selected:
                samples.append(dataset.X[idx].numpy())
                labels.append(label)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return np.array(samples), np.array(labels)


def normalize_samples(samples, norm_mean, norm_std):
    """Normalize samples using training statistics."""
    norm_mean = np.array(norm_mean)
    norm_std = np.array(norm_std)
    return (samples - norm_mean) / norm_std


def serialize_sample(sample):
    """Serialize a sample window to binary format for Serial transmission."""
    # Shape: (50, 6) -> flatten to (300,)
    flattened = sample.flatten()
    # Pack as 300 float32 values
    return struct.pack(f'{len(flattened)}f', *flattened)


def send_to_esp32(serial_port, sample, timeout=2.0):
    """Send a sample to ESP32 and read prediction.

    FIX: the ESP32 HIL firmware (esp32_hil.ino) waits for a literal
    `HIL_SYNC\\n` line before reading the 1200-byte float payload, and
    replies with `HIL_RES:<class>:<confidence>:<ms>` (NOT JSON). The previous
    version sent the binary payload immediately and tried to parse JSON, so
    the firmware never recognised the start-of-frame and the host hung.
    """
    # 1) sync line so the firmware knows a payload is coming
    serial_port.write(b"HIL_SYNC\n")
    time.sleep(0.01)
    # 2) binary payload
    serial_port.write(serialize_sample(sample))

    start_time = time.time()
    response = b''
    while time.time() - start_time < timeout:
        if serial_port.in_waiting > 0:
            response += serial_port.read(serial_port.in_waiting)
            try:
                line = response.decode('utf-8', errors='ignore').strip()
            except UnicodeDecodeError:
                line = ''
            # Try the firmware's HIL_RES format first
            if line.startswith("HIL_RES:"):
                parts = line.split(':')
                if len(parts) == 4:
                    try:
                        return {
                            'activity': 'Unknown',
                            'class': int(parts[1]),
                            'confidence': float(parts[2]),
                            'inference_time_ms': float(parts[3]),
                        }
                    except ValueError:
                        pass
            # Fall back to JSON if someone is using the non-HIL firmware
            try:
                data = json.loads(line)
                return data
            except (json.JSONDecodeError, ValueError):
                pass
        time.sleep(0.01)

    return None


def simulate_esp32_inference(sample, model, device='cpu'):
    """Simulate ESP32 inference using PyTorch model for testing without hardware."""
    sample_tensor = torch.FloatTensor(sample).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(sample_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        pred_idx = pred_idx.item()
        conf = conf.item()

    return {
        'activity': 'Unknown',  # Would need label mapping
        'confidence': conf,
        'class': pred_idx,
        'inference_time_ms': 0.0
    }


def main():
    parser = argparse.ArgumentParser(description='Test ESP32 with dataset samples')
    parser.add_argument('--port', type=str, default='COM3',
                        help='Serial port (e.g., COM3, /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=115200,
                        help='Baud rate')
    parser.add_argument('--dataset', type=str, default='uci',
                        choices=['uci', 'mhealth', 'realworld'],
                        help='Dataset to test with')
    parser.add_argument('--data-root', type=str, default='data',
                        help='Data root directory')
    parser.add_argument('--samples-per-class', type=int, default=10,
                        help='Number of samples per class')
    parser.add_argument('--norm-json', type=str,
                        default='exports/esp32_v2/normalization_stats.json',
                        help='Normalization JSON path (FIX: default updated to actual file location)')
    parser.add_argument('--simulate', action='store_true',
                        help='Simulate ESP32 inference without hardware')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints_v2/best_sensorfusion_esp32_v2_pocket_v3.pt',
                        help='PyTorch checkpoint for simulation')

    args = parser.parse_args()

    print("=" * 60)
    print("ESP32 Dataset Testing")
    print("=" * 60)

    # Load normalization stats
    norm_mean, norm_std = load_normalization_stats(args.norm_json)
    print(f"Normalization stats loaded from {args.norm_json}")

    # Load dataset samples
    print(f"\nLoading {args.dataset} dataset...")
    samples, labels = load_dataset_samples(
        args.dataset, args.data_root, args.samples_per_class
    )
    print(f"Loaded {len(samples)} samples")

    # Normalize samples
    samples = normalize_samples(samples, norm_mean, norm_std)

    if args.simulate:
        # Simulation mode - use PyTorch model
        print("\nRunning in simulation mode (no ESP32 hardware)")
        # FIX: training script lives in ./training/, not directly under cwd
        sys.path.insert(0, str(Path(__file__).parent))
        sys.path.insert(0, str(Path(__file__).parent / "training"))
        from train_esp32_v2_expanded_local import SensorFusionESP32

        device = 'cpu'
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        num_classes = len(state.get("labels", state.get("activity_labels", [])))
        model = SensorFusionESP32(input_channels=6, reservoir_size=64, num_classes=num_classes)
        model.load_state_dict(state["model_state_dict"], strict=False)
        model.eval()
        model.to(device)

        predictions = []
        confidences = []

        for i, sample in enumerate(samples):
            result = simulate_esp32_inference(sample, model, device)
            predictions.append(result['class'])
            confidences.append(result['confidence'])

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(samples)} samples")

    else:
        # Real ESP32 mode
        print(f"\nConnecting to ESP32 on {args.port}...")
        try:
            ser = serial.Serial(args.port, args.baud, timeout=2.0)
            time.sleep(2)  # Wait for ESP32 to be ready
            print("Connected to ESP32")
        except serial.SerialException as e:
            print(f"Failed to connect to ESP32: {e}")
            print("Use --simulate flag to test without hardware")
            return 1

        predictions = []
        confidences = []

        for i, sample in enumerate(samples):
            result = send_to_esp32(ser, sample)
            if result:
                predictions.append(result['class'])
                confidences.append(result['confidence'])
            else:
                print(f"Failed to get response for sample {i}")
                predictions.append(-1)
                confidences.append(0.0)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(samples)} samples")

        ser.close()

    # Evaluate results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    # Filter out failed predictions
    valid_mask = np.array(predictions) != -1
    valid_predictions = np.array(predictions)[valid_mask]
    valid_labels = np.array(labels)[valid_mask]

    if len(valid_predictions) > 0:
        accuracy = (valid_predictions == valid_labels).mean()
        print(f"Accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(valid_labels, valid_predictions, zero_division=0))

        print("\nConfusion Matrix:")
        print(confusion_matrix(valid_labels, valid_predictions))
    else:
        print("No valid predictions received")

    print(f"\nAverage confidence: {np.mean(confidences):.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
