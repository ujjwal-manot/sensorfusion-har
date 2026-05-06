# HIL Implementation - Hardware-in-the-Loop Validation

Complete Hardware-in-the-Loop (HIL) validation system for ESP32 Human Activity Recognition deployment.

## Overview

This HIL implementation allows you to validate the ESP32 TFLite model using controlled dataset samples instead of physical sensor data. This isolates model inference issues from sensor data quality problems.

## Directory Structure

```
HIL_Implementation/
├── esp32/
│   ├── esp32_hil.ino              # ESP32 HIL firmware (receives UART data)
│   └── esp32_v2_useful11_config.h # Configuration and normalization stats
├── static/
│   └── hil_dashboard.html          # HIL validation dashboard UI
├── training/
│   └── train_esp32_v2_expanded_local.py  # Training script (for dataset loading)
├── exports/
│   └── esp32_v2/                  # Normalization stats and ONNX exports
├── server.py                       # WebSocket server with HIL endpoint
├── hil_server.py                   # Standalone HIL UART simulator
└── generate_cert.py                # SSL certificate generation
```

## Components

### 1. ESP32 HIL Firmware (`esp32/esp32_hil.ino`)
- Receives 50×6 float array (1200 bytes) via UART from PC
- Runs TFLite Micro inference
- Sends back prediction via UART in format: `HIL_RES:<class>:<confidence>:<ms>`
- Does NOT read from physical MPU6050 sensor

### 2. HIL Dashboard (`static/hil_dashboard.html`)
- Visualizes dataset input waveforms (accelerometer + gyroscope)
- Shows ESP32 pipeline animation
- Displays predictions with confidence and probability distribution
- Connects to server via WebSocket at `/ws/hil`

### 3. HIL Server (`hil_server.py`)
- Loads dataset samples (UCI-HAR, MHEALTH)
- Normalizes data using training statistics
- Sends to ESP32 via UART at 921600 baud
- Parses ESP32 response and forwards to dashboard

### 4. Main Server (`server.py`)
- Provides `/ws/hil` WebSocket endpoint for dashboard
- Can run HIL simulation without physical ESP32 (uses PyTorch model)
- Supports both phone sensor streaming and HIL validation

## Setup Instructions

### Prerequisites

Place `HIL_Implementation` folder in the sensorfusion-har root directory:
```
sensorfusion-har/
├── HIL_Implementation/
├── data/                          # UCI-HAR, MHEALTH datasets
├── checkpoints_v2/                # Model checkpoints
└── ...
```

### Option 1: Using Main Server with HIL Endpoint

1. Install dependencies:
```bash
pip install -r ../requirements_v2_final.txt
```

2. Start server:
```bash
python server.py
```

3. Access HIL dashboard:
```
https://<your-ip>:8443/static/hil_dashboard.html
```

4. Click "Start Simulation" to begin HIL validation

### Option 2: Using Standalone HIL Server with Physical ESP32

1. Flash `esp32/esp32_hil.ino` to ESP32
2. Connect ESP32 via USB (note COM port)
3. Generate TFLite model (see ESP32 tuning guide)
4. Run HIL server:
```bash
python hil_server.py --port COM3 --dataset uci --samples 10
```

## TFLite Model Generation

The ESP32 firmware requires a TFLite model in `model_data.h`. This must be generated via Colab:

1. Open the Colab notebook: `../final_esp32_v2_useful11_package/sensorfusion_har_ESP32_v2_useful11_final.ipynb`
2. Install dependencies: `!pip install -q numpy scipy scikit-learn matplotlib onnx onnx2tf tensorflow ai-edge-litert`
3. Run TFLite conversion cells
4. Download generated `model_data.h`
5. Place in `HIL_Implementation/esp32/` (rename to `model_data.h`)

**Note**: Python 3.14 (local) cannot install TensorFlow cleanly. Use Colab with Python 3.10-3.11.

## Normalization Statistics

The ESP32 firmware uses normalization stats from `esp32_v2_useful11_config.h`:

```c
static const float SENSORFUSION_V2_MEAN[6] = {
    -3.1886296272f, 1.2004078627f, 2.4552721977f,
    -0.0335379131f, -0.0226027351f, 0.0529510267f
};
static const float SENSORFUSION_V2_STD[6] = {
    6.3020558357f, 6.4955196381f, 3.8114185333f,
    1.0866706371f, 0.8430932164f, 1.2929021120f
};
```

These match the pocket_v3 checkpoint with corrected units (all m/s²).

## Communication Protocol

### PC → ESP32
1. Send sync: `HIL_SYNC\n`
2. Send 1200 bytes binary float data (50×6 array)

### ESP32 → PC
Format: `HIL_RES:<class_index>:<confidence>:<inference_ms>\n`

Example: `HIL_RES:0:0.9234:15.23`

## WebSocket Message Types

### Server → Dashboard
- `init`: Initialization with mode and labels
- `input`: Dataset window data with waveforms
- `output`: Inference result with prediction and confidence
- `error`: Error message

### Dashboard → Server
- `toggle_simulation`: Start/stop HIL simulation

## Troubleshooting

### Dashboard shows "Disconnected"
- Check server is running
- Verify WebSocket endpoint is `/ws/hil`
- Check browser console for WebSocket errors

### ESP32 not responding
- Verify TFLite model is generated (not placeholder)
- Check baud rate matches (921600)
- Verify UART connection
- Check Serial Monitor for "HIL_READY" signal

### Dataset not found
- Ensure HIL_Implementation is in sensorfusion-har root
- Check data/UCI HAR Dataset exists
- Fallback: System will generate mock data

### Normalization stats not found
- Check exports/esp32_v2/normalization_stats.json exists
- System will use default stats as fallback

## Bugs Fixed

### 1. WebSocket Endpoint
- **Issue**: Dashboard connected to `/ws/dashboard` instead of `/ws/hil`
- **Fixed**: Changed to correct endpoint in `hil_dashboard.html`

### 2. Normalization Stats Path
- **Issue**: Used relative paths that didn't work from subdirectory
- **Fixed**: Use `Path(__file__).parent` for absolute paths

### 3. Dataset Path
- **Issue**: Dataset paths assumed running from root directory
- **Fixed**: Use `Path(__file__).parent.parent / "data"` to find datasets

### 4. Server HIL Dataset Path
- **Issue**: Server HIL endpoint used BASE_DIR for dataset
- **Fixed**: Use `BASE_DIR.parent / "data"` to access parent directory

## Performance Targets

- **Inference time**: < 50 ms on ESP32
- **UART transmission**: < 10 ms for 1200 bytes at 921600 baud
- **WebSocket latency**: < 5 ms
- **Total window processing**: < 100 ms

## Next Steps

1. Generate TFLite model via Colab
2. Flash ESP32 with HIL firmware
3. Test with physical ESP32 and UART
4. Validate predictions match training accuracy (~89%)
5. Compare with real sensor deployment

## Related Files

- `../esp32_deploy/ESP32_TUNING_GUIDE.md` - ESP32 deployment tuning
- `../final_esp32_v2_useful11_package/README_V2_FINAL.md` - Model details
- `../server.py` - Main server (original)
