# Final ESP32 11-Class SensorFusion-HAR Package

## Best 11-Class Result

- Accuracy: 78.98%
- Macro F1: 0.8088
- Min per-class F1: 0.6318
- Parameters: 51,482
- FP32 parameter size estimate: 201.1 KB
- Estimated INT8 parameter size: ~50 KB

## Per-Class F1

| Activity | F1 |
|---|---:|
| Walking | 0.9328 |
| Sitting | 0.6318 |
| Standing | 0.6328 |
| Lying Down | 0.6603 |
| Stairs Up | 0.8895 |
| Stairs Down | 0.8403 |
| Jogging | 0.8917 |
| Jumping | 0.8057 |
| Cycling | 0.9310 |
| Ironing | 0.8385 |
| Vacuum Cleaning | 0.8422 |

## Included Files

- `checkpoints/best_esp32_11class.pt`  
  Best PyTorch checkpoint with labels and normalization metadata.

- `exports/esp32/sensorfusion_esp32_11class.onnx` + `.onnx.data`  
  Exported ONNX model. This is the portable model artifact for conversion.

- `exports/esp32/calib_data.npy`  
  Representative calibration windows for INT8 quantization.

- `exports/esp32/normalization_stats.json`  
  Mean/std and label metadata needed by ESP32 firmware.

- `outputs/esp32/summary_11class.json`  
  Full metrics and epoch history.

- `outputs/esp32/11class_confusion_f1.png`  
  Confusion matrix and per-class F1 plot.

- `train_esp32_local.py`  
  Working local training/export script used to produce this package.

- `train_esp32_10class_local.py`  
  Optional 10-class variant script that removes Jumping for a potentially stronger deployment model.

- `sensorfusion_har_ESP32.ipynb`  
  Notebook version; use the training script as the most reliable source of truth.

## ESP32-S3 Status

The current exported model is ESP32-S3-sized. The trained model has ~51k parameters, so an INT8 TFLite Micro model should be around ~50 KB for weights plus tensor arena memory.

## Note on TFLite/Header

This Windows environment uses Python 3.14, and TensorFlow/onnx2tf are not available for this Python version. The ONNX model and calibration data are included so the final TFLite INT8 and C header can be generated in Colab or a Python 3.10/3.11 TensorFlow environment.
