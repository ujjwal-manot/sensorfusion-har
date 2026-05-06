# ESP32 Tuning Guide for SensorFusion-HAR

## Completed Tuning Changes

### 1. Normalization Stats Updated
**File**: `esp32_har.ino` (lines 56-63)

Updated from old corrupted stats to corrected pocket_v3 stats (all m/s² units):
```c
static const float NORM_MEAN[NUM_CHANNELS] = {
    -3.1886296272f, 1.2004078627f, 2.4552721977f,
    -0.0335379131f, -0.0226027351f, 0.0529510267f
};
static const float NORM_STD[NUM_CHANNELS] = {
    6.3020558357f, 6.4955196381f, 3.8114185333f,
    1.0866706371f, 0.8430932164f, 1.2929021120f
};
```

### 2. Tensor Arena Size Increased
**File**: `esp32_har.ino` (line 32)

- **Before**: 40 KB
- **After**: 60 KB
- **Reason**: 11-class model requires more memory than 7-class model

### 3. Confidence Threshold Increased
**File**: `esp32_har.ino` (line 33)

- **Before**: 0.30
- **After**: 0.40
- **Reason**: Matches server.py threshold for consistency

## TFLite Model Conversion (Required)

The `model_data.h` is currently a placeholder. You need to convert the PyTorch checkpoint to TFLite format.

### Option 1: Colab (Recommended)
Use the provided Colab notebook which has TensorFlow pre-installed:

1. Open `final_esp32_v2_useful11_package/sensorfusion_har_ESP32_v2_useful11_final.ipynb` in Google Colab
2. Install dependencies in the first cell:
   ```python
   !pip install -q numpy scipy scikit-learn matplotlib onnx onnx2tf tensorflow ai-edge-litert
   ```
3. Run the TFLite conversion cells
4. Download the generated `model_data.h` and replace the placeholder in `esp32_deploy/`

### Option 2: Local (Python 3.10-3.11 only)
If you have Python 3.10 or 3.11 locally:

```bash
pip install tensorflow onnx onnx2tf
python export_tflite.py --checkpoint checkpoints_v2/best_sensorfusion_esp32_v2_pocket_v3.pt --header esp32_deploy/model_data.h
```

**Note**: Python 3.14 (your current version) cannot install TensorFlow cleanly.

## MPU6050 Sensor Configuration

Current settings in `esp32_har.ino` (lines 83-116):

### Accelerometer
- **Range**: +/- 8g
- **LSB Sensitivity**: 4096 LSB/g
- **Bandwidth**: 44 Hz (DLPF_CFG = 3)

### Gyroscope
- **Range**: +/- 500 deg/s
- **LSB Sensitivity**: 65.5 LSB/(deg/s)
- **Bandwidth**: 42 Hz (DLPF_CFG = 3)

### Sampling
- **Rate**: 50 Hz (SMPLRT_DIV = 19)
- **I2C Clock**: 400 kHz

### Tuning Options

**If sensor noise is high:**
- Increase DLPF to reduce bandwidth (DLPF_CFG = 4 for 21 Hz, DLPF_CFG = 5 for 10 Hz)
- Trade-off: Higher latency, lower noise

**If dynamic range is insufficient:**
- Increase accel range to +/- 16g (ACCEL_CONFIG = 0x18)
- Increase gyro range to +/- 1000 deg/s (GYRO_CONFIG = 0x10)
- Trade-off: Lower resolution

**If sampling jitter is high:**
- Increase I2C clock to 800 kHz (if ESP32 supports it)
- Use DMA for I2C transfers
- Trade-off: Higher power consumption

## Additional Tuning Parameters

### Buffer Overlap
**Current**: 50% overlap (lines 344-351)
- Keeps last 25 samples for next inference
- Provides smoother predictions
- Can adjust: Higher overlap = smoother but slower response

### Inference Timing
Monitor inference time from Serial output:
- Target: < 50 ms for real-time
- If > 50 ms: Consider INT8 quantization or model pruning

### Memory Usage
Monitor arena usage from Serial output:
- Current allocation: 60 KB
- If arena_used > 50 KB: Increase TENSOR_ARENA_SIZE
- If arena_used < 30 KB: Can reduce to save RAM

## Testing Steps

### 1. Compile Firmware
```bash
# In Arduino IDE or PlatformIO
# Open esp32_har.ino
# Select ESP32 board (e.g., ESP32 Dev Module)
# Upload to ESP32
```

### 2. Monitor Serial Output
```bash
# Open Serial Monitor at 115200 baud
# Expected output:
# ========================================
# SensorFusion-HAR Lite — ESP32 + MPU6050
# 11-Class Human Activity Recognition
# ========================================
# Initializing MPU6050... OK
# Loading TFLite model... OK
# Input tensor shape: [1, 50, 6]
# Output tensor shape: [1, 11]
# Arena used: XXXXX bytes
# Starting activity recognition...
```

### 3. Test Activities
Perform each activity and verify predictions:
- Walking
- Sitting
- Standing
- Lying Down
- Stairs Up
- Stairs Down
- Jogging
- Jumping
- Cycling
- Running
- Waist Bending

### 4. Validate Confidence
- Most predictions should have confidence > 0.40
- If consistently low: Check sensor mounting and normalization

## Troubleshooting

### Model Loading Failed
- **Cause**: `model_data.h` is still placeholder
- **Fix**: Complete TFLite conversion (see above)

### Arena Allocation Failed
- **Cause**: Insufficient memory
- **Fix**: Increase TENSOR_ARENA_SIZE (try 80 KB)

### Low Confidence Predictions
- **Cause**: Normalization mismatch or sensor drift
- **Fix**: Verify NORM_MEAN/NORM_STD match training stats
- **Fix**: Recalibrate MPU6050 offsets

### Wrong Activity Predictions
- **Cause**: Sensor placement different from training
- **Fix**: Mount sensor in pocket (same as training data)
- **Fix**: Re-train with sensor in new location

### Slow Inference (> 100 ms)
- **Cause**: FP32 model too large
- **Fix**: Use INT8 quantization (requires calibration data)
- **Fix**: Reduce model complexity

## Performance Targets

- **Inference time**: < 50 ms
- **Confidence threshold**: > 0.40
- **Prediction accuracy**: ~89% (matches training)
- **Memory usage**: < 80 KB arena
- **Power consumption**: < 150 mA (with MPU6050)

## Next Steps

1. ✅ Normalization stats updated
2. ✅ Tensor arena size increased
3. ✅ Confidence threshold adjusted
4. ⏳ TFLite model conversion (Colab)
5. ⏳ Firmware compilation and testing
6. ⏳ Real-world validation
