# ESP32 Deployment Guide

This guide explains how to deploy the SensorFusion-HAR model on ESP32 hardware with MPU6050 IMU sensor.

## Hardware Requirements

- **ESP32** microcontroller (e.g., ESP32-DevKitC, ESP32-WROVER)
- **MPU6050** IMU sensor (3-axis accelerometer + 3-axis gyroscope)
- **USB cable** for programming and power
- **Breadboard** and jumper wires for connections

## Wiring Diagram

```
ESP32          MPU6050
-----          -------
3.3V   ------> VCC
GND    ------> GND
GPIO21 ------> SDA
GPIO22 ------> SCL
```

Note: Default I2C pins on ESP32 are GPIO21 (SDA) and GPIO22 (SCL). You can modify these in the code if needed.

## Software Requirements

- **Arduino IDE** (latest version)
- **ESP32 Board Package** (install via Arduino Board Manager)
- **TensorFlowLite_ESP32** library (install via Arduino Library Manager)
- **Python 3.10 or 3.11** (for model export - TensorFlow not available for Python 3.14)

## Installation Steps

### 1. Install Arduino IDE and ESP32 Support

1. Download and install Arduino IDE from https://www.arduino.cc/en/software
2. Open Arduino IDE
3. Go to File → Preferences
4. Add this URL to "Additional Board Manager URLs":
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
5. Go to Tools → Board → Boards Manager
6. Search for "esp32" and install "esp32 by Espressif Systems"

### 2. Install TensorFlowLite_ESP32 Library

1. In Arduino IDE, go to Sketch → Include Library → Manage Libraries
2. Search for "TensorFlowLite_ESP32"
3. Install the library by TensorFlow

### 3. Export Model to TFLite

**Important:** TensorFlow is not available for Python 3.14. You must use Python 3.10 or 3.11 for this step.

On a system with Python 3.10/3.11:

```bash
pip install tensorflow torch onnx onnx-tf

python export_tflite.py \
    --checkpoint checkpoints_v2/best_sensorfusion_esp32_v2_useful11_final.pt \
    --output esp32_deploy/sensorfusion_esp32_v2.tflite \
    --header esp32_deploy/model_data.h \
    --validate
```

This will:
- Convert PyTorch model to TFLite format
- Generate C header file with model bytes
- Validate that TFLite output matches PyTorch output

**Alternative (if TensorFlow unavailable):**
```bash
python export_weights_esp32.py
```
This exports weights as C arrays, but requires manual implementation of the full model architecture in C++.

### 4. Update ESP32 Code

The `esp32_har.ino` file has been updated with:
- Normalization constants from training
- 11 activity labels
- Serial JSON output for dashboard integration

Verify the normalization constants match your training stats in `esp32_deploy/normalization.json`.

### 5. Flash ESP32

1. Open `esp32_deploy/esp32_har.ino` in Arduino IDE
2. Select your ESP32 board: Tools → Board → ESP32 Arduino → [Your Board]
3. Select correct port: Tools → Port → [Your ESP32 Port]
4. Click Upload button

## Testing

### Option 1: Test with Dataset Samples

```bash
python esp32_dataset_test.py --port COM3 --dataset uci --samples-per-class 10
```

This sends dataset samples to ESP32 via Serial and compares predictions to ground truth.

### Option 2: Test with Real Sensor Data

After flashing, open Serial Monitor (115200 baud) to see live predictions:
```
Activity: Walking  (conf: 85.3%, 12.5 ms)
{"activity":"Walking","confidence":0.8532,"class":0,"inference_time_ms":12.5}
```

### Option 3: Test with Dashboard

Run the Serial-to-WebSocket bridge:

```bash
python esp32_bridge.py --port COM3 --ws-url ws://localhost:8443/ws/dashboard
```

Then open the dashboard at `https://<your-ip>:8443/` to see ESP32 predictions.

## Configuration

### Sampling Rate

Default: 50 Hz (20ms per sample)

Modify in code:
```cpp
#define SAMPLE_RATE_HZ     50
```

### Window Size

Default: 50 samples (1 second of data)

Modify in code:
```cpp
#define TIME_STEPS         50
```

### Confidence Threshold

Default: 0.30 (30% confidence required)

Modify in code:
```cpp
#define CONFIDENCE_THRESHOLD 0.3f
```

### Normalization Constants

These are loaded from training. Update if you train with different data:

```cpp
static const float NORM_MEAN[NUM_CHANNELS] = {
    1.9352865f, 3.705785f, 0.23090644f, -0.02979615f, -0.04969823f, 0.04553811f
};
static const float NORM_STD[NUM_CHANNELS] = {
    5.3360248f, 7.3998132f, 4.6662188f, 0.47728997f, 0.55818295f, 0.37370721f
};
```

## Troubleshooting

### Model Not Loading

**Error:** "ERROR: Model schema version mismatch"

**Solution:** Ensure TFLite conversion was done with compatible TensorFlow version. Re-export model.

**Error:** "ERROR: AllocateTensors() failed"

**Solution:** Increase TENSOR_ARENA_SIZE:
```cpp
#define TENSOR_ARENA_SIZE  (60 * 1024)  // 60 KB
```

### Serial Communication Issues

**Error:** No data in Serial Monitor

**Solution:**
- Check baud rate (must be 115200)
- Verify USB cable supports data transfer (not just charging)
- Check Serial port number in Arduino IDE

**Error:** Garbled data in Serial Monitor

**Solution:** Ensure baud rate matches (115200)

### MPU6050 Issues

**Error:** No sensor readings

**Solution:**
- Check wiring (SDA, SCL, VCC, GND)
- Verify I2C address (default 0x68)
- Check MPU6050 is powered (3.3V)

**Error:** Constant zero readings

**Solution:**
- MPU6050 may need wake-up call (included in code)
- Check power supply

### Inference Too Slow

**Issue:** Inference takes > 100ms

**Solution:**
- Use INT8 quantization instead of FP32
- Reduce reservoir size in model architecture
- Optimize model size

### Poor Accuracy

**Issue:** Predictions are incorrect

**Solution:**
- Verify normalization constants match training
- Check sensor placement (chest vs pocket)
- Recalibrate sensors
- Test with dataset samples to validate model

## Performance Metrics

Expected performance on ESP32:

| Metric | Value |
|--------|-------|
| Inference Time | 10-30 ms |
| Memory Usage | ~40 KB arena |
| Model Size | ~75 KB (FP32) / ~25 KB (INT8) |
| Power Consumption | ~100-200 mA |
| Battery Life (2000mAh) | ~10-20 hours |

## Integration with Dashboard

The ESP32 can send predictions to the dashboard via Serial:

1. Run the bridge script:
   ```bash
   python esp32_bridge.py --port COM3 --ws-url ws://localhost:8443/ws/dashboard
   ```

2. Open dashboard at `https://<your-ip>:8443/`

3. Dashboard will show ESP32 predictions alongside phone predictions

## Advanced Features

### INT8 Quantization

To reduce model size and improve speed:

```python
# In export_tflite.py, add:
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
```

### Custom Activity Labels

Edit `esp32_har.ino`:
```cpp
static const char* ACTIVITY_LABELS[NUM_CLASSES] = {
    "Walking",
    "Sitting",
    // ... your activities
};
```

### Multiple Sensors

To add additional sensors (e.g., magnetometer):
1. Update `NUM_CHANNELS`
2. Modify normalization constants
3. Update model input shape
4. Re-train and export model

## References

- [ESP32 Arduino Core](https://github.com/espressif/arduino-esp32)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [MPU6050 Datasheet](https://www.invensense.com/wp-content/uploads/2015/02/MPU-6000-Datasheet1.pdf)

## Support

For issues specific to this deployment:
1. Check this README
2. Review `ESP32_DEPLOYMENT_SUMMARY.md`
3. Check Arduino IDE Serial Monitor for errors
4. Verify wiring and connections

For general model issues:
- Review the main project README
- Check training logs
- Validate with dataset samples
