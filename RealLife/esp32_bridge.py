"""
Serial-to-WebSocket bridge for ESP32 deployment.
Reads JSON predictions from ESP32 Serial and forwards to WebSocket server.

Usage:
    python esp32_bridge.py --port COM3 --ws-url ws://localhost:8443/ws/dashboard
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import serial
import websockets


async def serial_to_websocket(serial_port, ws_url, baud_rate=115200):
    """Bridge Serial data to WebSocket."""
    print(f"Connecting to Serial port {serial_port}...")
    ser = serial.Serial(serial_port, baud_rate, timeout=1.0)
    print(f"Connected to Serial port {serial_port}")

    print(f"Connecting to WebSocket {ws_url}...")
    async with websockets.connect(ws_url) as websocket:
        print(f"Connected to WebSocket {ws_url}")
        print("Bridge running. Press Ctrl+C to stop.")

        buffer = ""
        while True:
            try:
                # Read from Serial
                if ser.in_waiting > 0:
                    data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                    buffer += data

                    # Try to parse complete JSON messages
                    while '{' in buffer and '}' in buffer:
                        start = buffer.find('{')
                        end = buffer.find('}', start) + 1
                        json_str = buffer[start:end]

                        try:
                            # Parse JSON
                            prediction = json.loads(json_str)

                            # Add source identifier
                            prediction['source'] = 'ESP32'

                            # Forward to WebSocket
                            await websocket.send(json.dumps(prediction))

                            # Print to console
                            print(f"ESP32: {prediction['activity']} (conf: {prediction['confidence']:.3f})")

                            # Remove processed message from buffer
                            buffer = buffer[end:]

                        except json.JSONDecodeError:
                            # Incomplete JSON, wait for more data
                            buffer = buffer[start:]
                            break

                await asyncio.sleep(0.01)

            except serial.SerialException as e:
                print(f"Serial error: {e}")
                print("Attempting to reconnect...")
                await asyncio.sleep(2)
                try:
                    ser = serial.Serial(serial_port, baud_rate, timeout=1.0)
                except:
                    continue

            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed. Reconnecting...")
                await asyncio.sleep(2)
                continue

            except KeyboardInterrupt:
                print("\nStopping bridge...")
                break

    ser.close()
    print("Bridge stopped.")


async def simulate_esp32(ws_url):
    """Simulate ESP32 predictions for testing without hardware."""
    import random
    activities = ["Walking", "Sitting", "Standing", "Lying Down", "Stairs Up",
                  "Stairs Down", "Jogging", "Jumping", "Cycling", "Running", "Waist Bending"]

    print(f"Simulating ESP32 predictions to {ws_url}")
    print("Press Ctrl+C to stop.")

    async with websockets.connect(ws_url) as websocket:
        while True:
            prediction = {
                'source': 'ESP32',
                'activity': random.choice(activities),
                'confidence': random.uniform(0.5, 0.99),
                'class': random.randint(0, 10),
                'inference_time_ms': random.uniform(10, 30)
            }

            await websocket.send(json.dumps(prediction))
            print(f"Simulated ESP32: {prediction['activity']} (conf: {prediction['confidence']:.3f})")

            await asyncio.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description='Serial-to-WebSocket bridge for ESP32')
    parser.add_argument('--port', type=str, default='COM3',
                        help='Serial port (e.g., COM3, /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=115200,
                        help='Baud rate')
    parser.add_argument('--ws-url', type=str,
                        default='ws://localhost:8443/ws/dashboard',
                        help='WebSocket URL')
    parser.add_argument('--simulate', action='store_true',
                        help='Simulate ESP32 predictions without hardware')

    args = parser.parse_args()

    print("=" * 60)
    print("ESP32 Serial-to-WebSocket Bridge")
    print("=" * 60)

    if args.simulate:
        asyncio.run(simulate_esp32(args.ws_url))
    else:
        asyncio.run(serial_to_websocket(args.port, args.ws_url, args.baud))

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
