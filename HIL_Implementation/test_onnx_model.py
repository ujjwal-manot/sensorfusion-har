"""
Test ONNX model with onnxruntime (works on Python 3.14)

This script validates the ONNX model works correctly without requiring TensorFlow.
For ESP32 deployment, you still need to generate TFLite via Colab (Python 3.10-3.11).
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path

# Load ONNX model
onnx_path = Path(__file__).parent.parent / "exports" / "esp32_v2" / "sensorfusion_esp32_v2_pocket_final.onnx"

if not onnx_path.exists():
    print(f"Error: ONNX model not found at {onnx_path}")
    print("Available ONNX files:")
    exports_dir = Path(__file__).parent.parent / "exports" / "esp32_v2"
    for f in exports_dir.glob("*.onnx"):
        print(f"  - {f.name}")
    exit(1)

print(f"Loading ONNX model: {onnx_path}")
session = ort.InferenceSession(str(onnx_path))

# Get input/output info
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_shape = session.get_inputs()[0].shape
output_shape = session.get_outputs()[0].shape

print(f"\nModel Info:")
print(f"  Input: {input_name}, shape: {input_shape}")
print(f"  Output: {output_name}, shape: {output_shape}")

# Test inference
test_input = np.random.randn(1, 50, 6).astype(np.float32)
outputs = session.run([output_name], {input_name: test_input})
prediction = outputs[0]

print(f"\nInference Test:")
print(f"  Input shape: {test_input.shape}")
print(f"  Output shape: {prediction.shape}")
print(f"  Predicted class: {np.argmax(prediction[0])}")
print(f"  Max probability: {np.max(prediction[0]):.4f}")

print("\n✓ ONNX model works correctly!")
print("\nNOTE: For ESP32 deployment, you must generate TFLite via Colab.")
print("See HIL_Implementation/generate_tflite_model.ipynb for instructions.")
