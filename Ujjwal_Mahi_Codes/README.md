# SensorFusion-HAR Complete Package

Complete Human Activity Recognition implementation with ESP32 deployment support.

## Package Structure

```
HAR_Complete_Package/
├── notebooks/
│   └── sensorfusion_har_ESP32_v2_useful11_final.ipynb  # Colab training notebook
├── training/
│   └── train_esp32_v2_expanded_local.py                # Local training script
├── model/                                              # Model architecture files
├── checkpoints/                                        # Trained model checkpoints
├── exports/                                            # ONNX and normalization exports
├── esp32/
│   ├── esp32_hil.ino                                   # HIL validation firmware
│   ├── esp32_har.ino                                   # Real sensor firmware
│   └── model_data.h                                    # TFLite model header (placeholder)
├── static/
│   ├── dashboard_v3.html                               # Main dashboard
│   ├── phone.html                                      # Phone sensor interface
│   └── hil_dashboard.html                              # HIL validation dashboard
├── server.py                                           # WebSocket server
├── esp32_v2_useful11_config.h                         # ESP32 configuration
├── requirements_v2_final.txt                           # Python dependencies
├── export_tflite.py                                    # TFLite conversion
├── esp32_dataset_test.py                               # Dataset validation
├── esp32_bridge.py                                     # Serial-to-WebSocket bridge
├── generate_cert.py                                    # SSL certificate generation
└── certs/                                              # SSL certificates
```

## Model Performance

- **Accuracy**: 89.36%
- **Macro F1**: 90.28%
- **Classes**: 11 activities
- **Activities**: Walking, Sitting, Standing, Lying Down, Stairs Up, Stairs Down, Jogging, Jumping, Cycling, Running, Waist Bending

## Quick Start

### 1. Training (Local)

```bash
pip install -r requirements_v2_final.txt
python training/train_esp32_v2_expanded_local.py --msm-epochs 10 --head-epochs 18 --ft-epochs 0
```

### 2. Training (Colab)

Open `notebooks/sensorfusion_har_ESP32_v2_useful11_final.ipynb` in Google Colab and run all cells.

### 3. Start Server

```bash
python server.py
```

Access dashboard at: `https://<your-ip>:8443/`
Access phone interface at: `https://<your-ip>:8443/phone`

### 4. ESP32 Deployment

1. Convert model to TFLite:
```bash
python export_tflite.py --checkpoint checkpoints/best_sensorfusion_esp32_v2_useful11_final.pt --output esp32/model_data.h
```

2. Upload firmware to ESP32:
- For real sensor: `esp32/esp32_har.ino`
- For HIL validation: `esp32/esp32_hil.ino`

## HIL Validation Approach

The Hardware-in-the-Loop (HIL) approach validates the ESP32 implementation using known-good dataset samples before real-world sensor deployment.

### Components

- **hil_dashboard.html**: Visualizes dataset input, ESP32 pipeline, and predictions
- **esp32_hil.ino**: ESP32 firmware that receives data via UART instead of physical sensor
- **hil_server.py**: PC simulator that sends dataset samples to ESP32 (needs to be created)
- **server.py**: WebSocket server with HIL support (needs modification)

### HIL Workflow

1. Load dataset samples (UCI-HAR test set)
2. Send via UART to ESP32 at 921600 baud
3. ESP32 runs TFLite inference
4. Results displayed in HIL dashboard

## Datasets Used

- UCI-HAR: Total acceleration + body gyroscope
- PAMAP2: Hand accelerometer + gyroscope
- MHEALTH: Right-arm accelerometer + gyroscope

## Normalization Stats

```python
mean = [-3.1886, 1.2004, 2.4553, -0.0335, -0.0226, 0.0530]
std  = [ 6.3021, 6.4955, 3.8114,  1.0867,  0.8431, 1.2929]
```

## Input Format

- Shape: (50, 6)
- Channels: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
- Sampling rate: 50 Hz
- Window size: 1 second

## Dependencies

See `requirements_v2_final.txt` for complete list.

## License

This is a research project for Human Activity Recognition.
