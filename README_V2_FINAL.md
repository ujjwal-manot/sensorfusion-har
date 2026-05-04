# SensorFusion-HAR ESP32 V2 Useful-11 Final Package

This package contains the final V2 ESP32-oriented Human Activity Recognition model and all supporting artifacts produced after the V1 baseline. V1 is preserved separately and was not overwritten.

## Final result

- **Model tag:** `sensorfusion_esp32_v2_useful11_final`
- **Checkpoint:** `checkpoints_v2/best_sensorfusion_esp32_v2_useful11_final.pt`
- **ONNX:** `exports/esp32_v2/sensorfusion_esp32_v2_useful11_final.onnx`
- **Input shape:** `(1, 50, 6)`
- **Classes:** `11`
- **Parameters:** `75,290`
- **FP32 parameter estimate:** `294.1 KB`
- **Estimated INT8 weights:** about `74 KB`
- **Best stage:** head-only stage, epoch `17/18`
- **Full fine-tune:** disabled for the final run because earlier unfreezing destabilized the model.

## Metrics

| Metric | Value |
|---|---:|
| Accuracy | `0.8935640138` |
| Macro F1 | `0.9027877942` |
| Minimum class F1 | `0.8342412451` |

## Per-class F1

| Class | F1 |
|---|---:|
| Walking | `0.9264897782` |
| Sitting | `0.8709036743` |
| Standing | `0.8399629972` |
| Lying Down | `0.9448022079` |
| Stairs Up | `0.8445993031` |
| Stairs Down | `0.8342412451` |
| Jogging | `0.9565217391` |
| Jumping | `0.8545034642` |
| Cycling | `0.9677926159` |
| Running | `0.8997772829` |
| Waist Bending | `0.9910714286` |

Every retained class is above `0.80` F1. The final macro F1 is above `0.90`.

## Improvement over V1

| Item | V1 baseline | V2 final |
|---|---:|---:|
| Accuracy | about `0.7898` | `0.8936` |
| Macro F1 | about `0.8088` | `0.9028` |
| Minimum F1 | about `0.6318` | `0.8342` |
| Activities | 11 mixed utility classes | 11 more useful classes |

V2 replaces weaker/less useful activities `Ironing` and `Vacuum Cleaning` with `Running` and `Waist Bending`, while keeping `Jumping` and lifting it above the target boundary.

## Final activity labels

Index order is fixed and must match the model output logits:

```text
0  Walking
1  Sitting
2  Standing
3  Lying Down
4  Stairs Up
5  Stairs Down
6  Jogging
7  Jumping
8  Cycling
9  Running
10 Waist Bending
```

## Datasets used

- **UCI-HAR:** total acceleration plus body gyroscope.
- **PAMAP2:** hand accelerometer plus gyroscope.
- **MHEALTH:** URL-downloaded UCI MHEALTH dataset, right-arm accelerometer plus gyroscope.

A RealWorld HAR local waist acc+gyro parser was implemented and tested, but the small RealWorld sample did not improve final metrics, so the final selected model excludes RealWorld data.

## Signal format

The model expects fixed ESP32-style windows:

```text
shape: (50, 6)
channels: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
```

Normalize each channel before inference:

```text
mean = [-3.1886296272, 1.2004078627, 2.4552721977, -0.0335379131, -0.0226027351, 0.0529510267]
std  = [ 6.3020558357, 6.4955196381, 3.8114185333,  1.0866706371,  0.8430932164, 1.2929021120]
```

Use `esp32_v2_useful11_config.h` for the labels and normalization constants.

## Novel architecture features

The V2 model preserves the original V1 architectural ideas and adds ESP32-safe improvements.

### Preserved features

- Learnable spectral-radius echo-state reservoir.
- Differential reservoir states.
- Learned differential reservoir gate.
- Depthwise-separable temporal encoder.
- Spectral gated fusion.
- Patch micro-attention.
- Scaled binary classifier head.
- Masked Sensor Modeling pretraining.
- Reservoir-manifold mixup support.

### Added V2 features

- Total/raw acceleration path for UCI-HAR to preserve gravity cues.
- URL-based MHEALTH integration.
- Orientation/statistical micro-path for posture separation.
- Feature fusion between attention features and orientation features.
- Multi-scale depthwise temporal kernels with `3/5/9` temporal receptive fields.
- Optional prototype margin loss.
- Minority-aware augmentation and weighted sampling.
- Best-epoch checkpoint selection optimized toward minimum per-class F1.
- Dataset manifest and confidence-aware final reporting.

## Training configuration of selected model

```text
seed: 42
msm_epochs: 10
head_epochs: 18
ft_epochs: 0
batch_size: 128
reservoir_size: 64
focal_gamma: 1.2
label_smoothing: 0.01
prototype_weight: 0.0
include_mhealth: true
include_realworld: false
artifact_suffix: useful11_final
```

## Included package files

```text
README_V2_FINAL.md
requirements_v2_final.txt
train_esp32_v2_expanded_local.py
sensorfusion_har_ESP32_v2_useful11_final.ipynb
esp32_v2_useful11_config.h
tests/unit/test_esp32_v2_pipeline.py
checkpoints_v2/best_sensorfusion_esp32_v2_useful11_final.pt
exports/esp32_v2/sensorfusion_esp32_v2_useful11_final.onnx
exports/esp32_v2/sensorfusion_esp32_v2_useful11_final.onnx.data
exports/esp32_v2/calib_data_v2.npy
exports/esp32_v2/normalization_stats.json
outputs/esp32_v2/summary_v2_useful11_final.json
outputs/esp32_v2/dataset_manifest_v2.json
outputs/esp32_v2/v2_useful11_final_confusion_f1.png
```

## Re-run final training locally

```powershell
C:\Python314\python.exe train_esp32_v2_expanded_local.py --msm-epochs 10 --head-epochs 18 --ft-epochs 0 --batch-size 128 --threads 8 --reservoir-size 64 --focal-gamma 1.2 --label-smoothing 0.01 --prototype-weight 0.0 --no-realworld --artifact-suffix useful11_final
```

## Verify tests

```powershell
C:\Python314\python.exe -m pytest tests\unit\test_esp32_v2_pipeline.py -q
```

Expected result:

```text
6 passed
```

## ESP32 deployment notes

The local machine uses Python 3.14, where TensorFlow/onnx2tf were not installed. Therefore this final local package contains the PyTorch checkpoint, ONNX model, normalization statistics, calibration data, and a Colab notebook path for TFLite FP32/INT8 conversion.

For ESP32-S3 deployment:

1. Use `sensorfusion_har_ESP32_v2_useful11_final.ipynb` in Colab or Python 3.10/3.11.
2. Convert the ONNX model to TFLite FP32 and INT8 using the calibration data.
3. Convert the INT8 `.tflite` file to a C array header.
4. Use `esp32_v2_useful11_config.h` for labels and normalization.
5. Feed normalized `(50, 6)` windows to the model.

## Important honesty note

A RealWorld HAR adapter was added and tested. With the small local RealWorld jumping sample, final performance dropped from `0.9028` macro F1 to `0.8639`, so RealWorld was not selected for the final best model. The final result uses UCI-HAR + PAMAP2 + MHEALTH and satisfies the requested per-class F1 boundary.
