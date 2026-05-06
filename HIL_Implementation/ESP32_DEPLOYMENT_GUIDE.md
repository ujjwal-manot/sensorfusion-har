# ESP32 HIL Deployment - Step-by-Step Guide

Complete step-by-step instructions to deploy the HIL validation firmware to ESP32 hardware.

## Prerequisites

### Hardware
- ESP32 development board (ESP32 Dev Module, ESP32-WROOM, etc.)
- USB data cable (not just charging cable)
- Computer with USB port

### Software
- Arduino IDE 2.x or PlatformIO
- Google Colab account (for TFLite conversion)
- Python 3.10-3.11 (if running HIL server locally)

### Files Required
- `HIL_Implementation/esp32/esp32_hil.ino`
- `HIL_Implementation/esp32/esp32_v2_useful11_config.h`
- `HIL_Implementation/esp32/model_data.h` (to be generated)
- Model checkpoint: `checkpoints_v2/best_sensorfusion_esp32_v2_pocket_v3.pt`

---

## Step 1: Generate TFLite Model (Required)

The ESP32 firmware requires a TFLite model in C header format. This must be done in Google Colab because Python 3.14 cannot install TensorFlow.

### 1.1 Open Colab Notebook

1. Go to Google Colab: https://colab.research.google.com/
2. Click "File" → "Open notebook"
3. Navigate to your project and open: `final_esp32_v2_useful11_package/sensorfusion_har_ESP32_v2_useful11_final.ipynb`

### 1.2 Install Dependencies

Run this cell in Colab:
```python
!pip install -q numpy scipy scikit-learn matplotlib onnx onnx2tf tensorflow ai-edge-litert
```

**Important**: After installation, Colab may ask to restart the runtime. Click "Restart Runtime" and continue.

### 1.3 Upload Checkpoint

1. In Colab, click the folder icon on the left sidebar
2. Click "Upload" icon
3. Upload: `checkpoints_v2/best_sensorfusion_esp32_v2_pocket_v3.pt`

### 1.4 Run TFLite Conversion

Run the TFLite conversion cells in the notebook. Look for cells that:
- Load the PyTorch checkpoint
- Export to ONNX
- Convert ONNX to TFLite
- Generate C header file

The output should include a downloadable `model_data.h` file.

### 1.5 Download model_data.h

1. When conversion completes, the notebook will show a download link for `model_data.h`
2. Download this file
3. Save it to: `HIL_Implementation/esp32/model_data.h`

**Verify**: The file should be ~50-100 KB (not 22 bytes placeholder)

---

## Step 2: Prepare ESP32 Firmware

### 2.1 Copy model_data.h to ESP32 Folder

Ensure the file structure is:
```
HIL_Implementation/esp32/
├── esp32_hil.ino
├── esp32_v2_useful11_config.h
└── model_data.h  ← Generated from Step 1
```

### 2.2 Verify Configuration

Open `esp32_v2_useful11_config.h` and verify:
- `SENSORFUSION_V2_NUM_CLASSES = 11`
- `SENSORFUSION_V2_TIME_STEPS = 50`
- `SENSORFUSION_V2_INPUT_CHANNELS = 6`
- Normalization stats match pocket_v3 checkpoint

---

## Step 3: Install Arduino IDE

### 3.1 Download Arduino IDE

1. Download Arduino IDE 2.x from: https://www.arduino.cc/en/software
2. Install and launch Arduino IDE

### 3.2 Install ESP32 Board Support

1. In Arduino IDE, go to: File → Preferences
2. In "Additional Board Manager URLs", add:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Click OK
4. Go to: Tools → Board → Boards Manager
5. Search for "esp32"
6. Install "esp32 by Espressif Systems"

### 3.3 Install TensorFlow Lite for Microcontrollers

1. Go to: Tools → Manage Libraries
2. Search for "TensorFlow Lite"
3. Install "TensorFlow Lite for Microcontrollers" by TensorFlow

---

## Step 4: Open and Configure Project

### 4.1 Open Sketch

1. In Arduino IDE, go to: File → Open
2. Navigate to: `HIL_Implementation/esp32/esp32_hil.ino`
3. Open the file

### 4.2 Select Board

1. Go to: Tools → Board → esp32
2. Select your board (e.g., "ESP32 Dev Module")

### 4.3 Select Port

1. Connect ESP32 to computer via USB
2. Go to: Tools → Port
3. Select the COM port (Windows) or /dev/ttyUSB0 (Linux/Mac)

**Note**: If port not listed, install ESP32 USB drivers:
- Windows: Download from Silicon Labs or CH340 driver
- Linux/Mac: Usually auto-detected

### 4.4 Configure Board Settings

Go to: Tools and set:
- **CPU Frequency**: 240 MHz (default)
- **Flash Frequency**: 80 MHz (default)
- **Flash Mode**: QIO (default)
- **Partition Scheme**: Default 4MB with spiffs (default)
- **Upload Speed**: 921600 (for faster upload)

---

## Step 5: Compile Firmware

### 5.1 Verify Code

1. Click the checkmark icon (Verify) in Arduino IDE
2. Wait for compilation to complete

**Expected Output**:
```
Sketch uses XXXXX bytes (X%) of program storage space.
Global variables use XXXX bytes (X%) of dynamic memory.
```

**Common Errors**:
- `model_data.h not found`: Ensure file is in same folder as .ino
- `TensorFlow library not found`: Install TFLM library via Library Manager
- `Compilation error`: Check board selection and library versions

### 5.2 Troubleshooting Compilation

If compilation fails:
1. Ensure all three files are in the same folder
2. Verify TFLM library is installed
3. Check board settings match your hardware
4. Try a different board variant (e.g., "ESP32 Wrover Module")

---

## Step 6: Upload to ESP32

### 6.1 Upload Firmware

1. Click the right arrow icon (Upload) in Arduino IDE
2. Wait for upload to complete (may take 30-60 seconds)

**Expected Output**:
```
Writing at 0x00001000... (100%)
Wrote XXXXX bytes (XXX sec) at 1000000 bps
```

### 6.2 Troubleshooting Upload

**If upload fails**:
1. Press and hold BOOT button on ESP32
2. Click Upload in Arduino IDE
3. Release BOOT button when "Connecting..." appears
4. Try different upload speed (115200 if 921600 fails)

**If port not found**:
1. Reconnect USB cable
2. Try different USB port
3. Install USB drivers
4. Check Device Manager (Windows) for COM port

---

## Step 7: Verify ESP32 is Running

### 7.1 Open Serial Monitor

1. In Arduino IDE, click magnifying glass icon (Serial Monitor)
2. Set baud rate to: 921600
3. Press ESP32 RESET button

### 7.2 Expected Output

You should see:
```
HIL_READY
```

**If you see errors**:
- `HIL_ERR:TFLMInitFailed`: TFLite model is invalid or corrupted
- `ERROR: Model schema version mismatch`: TFLM library version mismatch
- Nothing: Check baud rate (must be 921600)

### 7.3 Troubleshooting ESP32

**No serial output**:
- Verify baud rate is 921600
- Check TX/RX pins are not used by other peripherals
- Try a different USB cable (data cable, not charging only)

**TFLM init failed**:
- Regenerate `model_data.h` from Colab
- Verify TFLM library version matches model
- Check tensor arena size (60 KB should be sufficient)

---

## Step 8: Run HIL Validation

### Option A: Using Standalone HIL Server (Recommended for ESP32 Testing)

#### 8.1 Install Python Dependencies

```bash
pip install pyserial websockets numpy torch scikit-learn
```

#### 8.2 Run HIL Server

```bash
cd HIL_Implementation
python hil_server.py --port COM3 --dataset uci --samples 10
```

Replace `COM3` with your ESP32's COM port.

#### 8.3 Expected Output

```
============================================================
HIL Server - Hardware-in-the-Loop Simulator
============================================================
Serial Port: COM3
Baud Rate: 921600
Dataset: uci
Samples per class: 10
WebSocket Port: 8444
============================================================
ESP32 HIL ready
```

#### 8.4 Access Dashboard

Open browser and navigate to:
```
http://localhost:8444/static/hil_dashboard.html
```

Click "Start Simulation" to begin HIL validation.

### Option B: Using Main Server (No ESP32 Hardware Required)

#### 8.1 Start Main Server

```bash
cd HIL_Implementation
python server.py
```

#### 8.2 Access HIL Dashboard

```
https://<your-ip>:8443/static/hil_dashboard.html
```

This mode uses the PyTorch model for simulation instead of physical ESP32.

---

## Step 9: Validate Results

### 9.1 Check Dashboard

The dashboard should show:
- Input waveforms (accelerometer + gyroscope)
- ESP32 pipeline animation
- Predicted activity with confidence
- Probability distribution bars
- Inference time in milliseconds

### 9.2 Expected Performance

- **Inference time**: < 50 ms on ESP32
- **Confidence**: > 0.40 for correct predictions
- **Accuracy**: ~89% (matches training)

### 9.3 Common Issues

**Low confidence predictions**:
- Check normalization stats match training
- Verify TFLite model was generated from correct checkpoint
- Ensure data is normalized before sending to ESP32

**Wrong predictions**:
- Verify dataset samples are in correct format (50×6)
- Check channel order: [ax, ay, az, gx, gy, gz]
- Ensure normalization is applied correctly

**Slow inference (> 100 ms)**:
- Check CPU frequency is 240 MHz
- Consider INT8 quantization (requires calibration)
- Reduce model complexity

---

## Step 10: Troubleshooting Reference

### ESP32 Won't Boot

**Symptoms**: No serial output, ESP32 doesn't respond

**Solutions**:
1. Check power supply (use USB data cable)
2. Press BOOT button during upload
3. Try lower upload speed (115200)
4. Flash a simple blink sketch to test hardware

### UART Communication Fails

**Symptoms**: HIL server can't connect to ESP32

**Solutions**:
1. Verify COM port is correct
2. Check baud rate is 921600 on both ends
3. Ensure ESP32 is sending "HIL_READY"
4. Check TX/RX pins are not shorted

### TFLite Model Issues

**Symptoms**: `HIL_ERR:TFLMInitFailed` or wrong predictions

**Solutions**:
1. Regenerate `model_data.h` from Colab
2. Verify TFLM library version
3. Check tensor arena size (increase to 80 KB if needed)
4. Validate ONNX export before TFLite conversion

### Dashboard Issues

**Symptoms**: Dashboard shows "Disconnected" or no data

**Solutions**:
1. Check WebSocket endpoint is `/ws/hil`
2. Verify server is running
3. Check browser console for errors
4. Ensure CORS is not blocking connection

---

## Quick Reference

### File Locations
```
HIL_Implementation/
├── esp32/
│   ├── esp32_hil.ino              # Firmware
│   ├── esp32_v2_useful11_config.h # Config
│   └── model_data.h               # TFLite model (generated)
├── static/
│   └── hil_dashboard.html          # Dashboard
├── hil_server.py                  # UART simulator
└── server.py                      # Main server
```

### Key Settings
- **Baud Rate**: 921600
- **Tensor Arena**: 60 KB
- **Window Size**: 50 samples
- **Channels**: 6 (ax, ay, az, gx, gy, gz)
- **Classes**: 11

### Commands
```bash
# Generate TFLite (Colab)
!pip install -q numpy scipy scikit-learn matplotlib onnx onnx2tf tensorflow ai-edge-litert

# Upload firmware (Arduino IDE)
Tools → Upload

# Run HIL server
python hil_server.py --port COM3 --dataset uci --samples 10

# Run main server
python server.py
```

---

## Next Steps

After successful HIL validation:

1. **Compare with Real Sensor Deployment**: Test with physical MPU6050 sensor
2. **Optimize Model**: Try INT8 quantization for faster inference
3. **Add More Datasets**: Test with MHEALTH or custom data
4. **Production Deployment**: Consider OTA updates and power optimization

---

## Support

For issues:
1. Check `ESP32_TUNING_GUIDE.md` for ESP32-specific tuning
2. Check `README.md` for HIL implementation details
3. Review Arduino IDE Serial Monitor for error messages
4. Verify all file paths are correct
