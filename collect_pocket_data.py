"""
Data collection script for pocket-specific HAR training.
Records sensor data from Android phone kept in pants pocket.

Usage:
    python collect_pocket_data.py --output data/pocket_collected.npz --duration 30
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


class PocketDataCollector:
    """Collects sensor data from phone WebSocket for pocket-specific training."""

    def __init__(self, output_path, activity_name, duration_sec=30, sample_rate=50):
        self.output_path = Path(output_path)
        self.activity_name = activity_name
        self.duration_sec = duration_sec
        self.sample_rate = sample_rate
        self.target_samples = duration_sec * sample_rate

        self.data = {
            'ax': [],
            'ay': [],
            'az': [],
            'gx': [],
            'gy': [],
            'gz': [],
            'timestamp': []
        }

        self.start_time = None
        self.collected = 0

    def add_sample(self, sample):
        """Add a sensor sample to the buffer."""
        if self.start_time is None:
            self.start_time = sample.get('t', time.time())

        self.data['ax'].append(sample.get('ax', 0))
        self.data['ay'].append(sample.get('ay', 0))
        self.data['az'].append(sample.get('az', 0))
        self.data['gx'].append(sample.get('gx', 0))
        self.data['gy'].append(sample.get('gy', 0))
        self.data['gz'].append(sample.get('gz', 0))
        self.data['timestamp'].append(sample.get('t', time.time()))

        self.collected += 1

    def is_complete(self):
        """Check if collection is complete."""
        return self.collected >= self.target_samples

    def save(self):
        """Save collected data to NPZ file."""
        if self.collected == 0:
            print("No data collected!")
            return False

        # Convert to numpy arrays
        sensor_data = np.column_stack([
            self.data['ax'],
            self.data['ay'],
            self.data['az'],
            self.data['gx'],
            self.data['gy'],
            self.data['gz']
        ])

        timestamps = np.array(self.data['timestamp'])

        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with metadata
        metadata = {
            'activity': self.activity_name,
            'duration_sec': self.duration_sec,
            'sample_rate': self.sample_rate,
            'samples_collected': self.collected,
            'channels': ['ax', 'ay', 'az', 'gx', 'gy', 'gz'],
            'start_time': self.start_time,
            'end_time': timestamps[-1] if len(timestamps) > 0 else None
        }

        np.savez(
            self.output_path,
            sensor_data=sensor_data,
            timestamps=timestamps,
            metadata=metadata
        )

        print(f"Saved {self.collected} samples to {self.output_path}")
        print(f"Activity: {self.activity_name}")
        print(f"Duration: {self.duration_sec}s (target), actual: {len(timestamps)/self.sample_rate:.2f}s")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Data shape: {sensor_data.shape}")

        return True


def manual_collection_mode():
    """Interactive mode for manual data collection."""
    print("=" * 60)
    print("Pocket Data Collection - Manual Mode")
    print("=" * 60)
    print("\nThis script collects sensor data from your Android phone")
    print("kept in a pants pocket for pocket-specific HAR training.")
    print("\nInstructions:")
    print("1. Start the server: python server.py")
    print("2. Open https://<your-ip>:8443/phone on your phone")
    print("3. Place phone in pants pocket")
    print("4. For each activity, collect data for specified duration")
    print("\nAvailable activities:")
    activities = [
        "Walking", "Sitting", "Standing", "Lying Down",
        "Stairs Up", "Stairs Down", "Jogging", "Jumping",
        "Cycling", "Running", "Waist Bending"
    ]
    for i, act in enumerate(activities):
        print(f"  {i+1}. {act}")

    print("\nNote: This is a template script. For actual data collection,")
    print("you would need to:")
    print("- Connect to the phone WebSocket to receive sensor data")
    print("- Implement real-time data recording")
    print("- Save data in the format expected by the training script")
    print("\nFor now, this script demonstrates the structure.")


def main():
    parser = argparse.ArgumentParser(description='Collect pocket-specific HAR data')
    parser.add_argument('--output', type=str, required=True,
                        help='Output NPZ file path')
    parser.add_argument('--activity', type=str, required=True,
                        help='Activity name (e.g., Walking, Sitting, etc.)')
    parser.add_argument('--duration', type=int, default=30,
                        help='Collection duration in seconds')
    parser.add_argument('--sample-rate', type=int, default=50,
                        help='Target sample rate (Hz)')
    parser.add_argument('--manual', action='store_true',
                        help='Show manual collection instructions')

    args = parser.parse_args()

    if args.manual:
        manual_collection_mode()
        return 0

    print("=" * 60)
    print("Pocket Data Collection")
    print("=" * 60)
    print(f"Activity: {args.activity}")
    print(f"Duration: {args.duration}s")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Target samples: {args.duration * args.sample_rate}")
    print(f"Output: {args.output}")
    print("\nNote: This is a template for data collection.")
    print("To collect actual data, you would need to:")
    print("1. Connect to the phone WebSocket from server.py")
    print("2. Stream sensor data in real-time")
    print("3. Record and save the data")
    print("\nFor immediate use, consider using the existing datasets")
    print("(UCI-HAR, PAMAP2, MHEALTH, RealWorld) which already include")
    print("chest-mounted sensor data that approximates pocket placement.")

    # Create a dummy collector to demonstrate the structure
    collector = PocketDataCollector(args.output, args.activity, args.duration, args.sample_rate)

    # In a real implementation, you would:
    # 1. Connect to WebSocket: ws://<server-ip>:8443/ws/phone
    # 2. Receive sensor samples
    # 3. Add samples using collector.add_sample(sample)
    # 4. Check collector.is_complete()
    # 5. Save using collector.save()

    print("\n" + "=" * 60)
    print("Template complete. Implement WebSocket connection for actual collection.")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
