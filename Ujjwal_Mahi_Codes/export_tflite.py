"""
Export PyTorch SensorFusionESP32 model to TFLite format for ESP32 deployment.

Usage:
    python export_tflite.py --checkpoint checkpoints_v2/best_sensorfusion_esp32_v2_pocket_v3.pt
                            --output esp32_deploy/sensorfusion_esp32_v2.tflite
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def load_model(checkpoint_path, device='cpu'):
    """Load the PyTorch model from checkpoint."""
    # Add current directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent / "final_esp32_v2_useful11_package"))

    from train_esp32_v2_expanded_local import SensorFusionESP32

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get number of classes and labels
    ckpt_labels = state.get("labels", state.get("activity_labels", None))
    num_classes = len(ckpt_labels) if ckpt_labels else state.get("num_classes", 7)

    print(f"Loading model with {num_classes} classes")
    if ckpt_labels:
        print(f"Labels: {ckpt_labels}")

    # Instantiate model
    model = SensorFusionESP32(input_channels=6, reservoir_size=64, num_classes=num_classes)

    # Load weights
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.eval()
    model.to(device)

    print(f"Model loaded from {checkpoint_path}")
    print(f"Parameters: {model.count_parameters():,}")

    return model, num_classes, ckpt_labels


def export_to_onnx(model, output_path, input_shape=(1, 50, 6)):
    """Export PyTorch model to ONNX format."""
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    print(f"Exported to ONNX: {output_path}")


def onnx_to_tflite(onnx_path, tflite_path):
    """Convert ONNX model to TFLite using TensorFlow."""
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow not installed. Install with: pip install tensorflow")
        print("Falling back to direct PyTorch export (may have limited TFLite support)")
        return None

    # Import onnx-tf
    try:
        import onnx
        from onnx_tf.backend import prepare
    except ImportError:
        print("ERROR: onnx-tf not installed. Install with: pip install onnx-tf")
        return None

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    # Export to TensorFlow SavedModel
    tf_model_dir = tflite_path.parent / "tf_model"
    tf_rep.export_graph(str(tf_model_dir))

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_dir))
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # Save TFLite model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Converted to TFLite: {tflite_path}")
    print(f"TFLite model size: {len(tflite_model) / 1024:.2f} KB")

    # Cleanup
    import shutil
    shutil.rmtree(tf_model_dir, ignore_errors=True)

    return tflite_path


def direct_tflite_export(model, tflite_path):
    """
    Direct export from PyTorch to TFLite using torch.export (if available).
    This is the preferred method for newer PyTorch versions.
    """
    try:
        import torch.export as export
        import torch._export
    except (ImportError, AttributeError):
        print("torch.export not available, using ONNX route instead")
        return None

    # Use torch.export to get a captured graph
    dummy_input = torch.randn(1, 50, 6)
    exported_program = export.export(model, args=(dummy_input,))

    # Try to convert to TFLite
    try:
        from torch._export import tflite as tflite_exporter
        tflite_model = tflite_exporter.convert(exported_program)

        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        print(f"Direct TFLite export: {tflite_path}")
        print(f"TFLite model size: {len(tflite_model) / 1024:.2f} KB")
        return tflite_path
    except Exception as e:
        print(f"Direct TFLite export failed: {e}")
        return None


def validate_outputs(pytorch_model, tflite_path, num_classes, device='cpu'):
    """Validate that TFLite output matches PyTorch output."""
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test with random input
        dummy_input = torch.randn(1, 50, 6).numpy().astype(np.float32)

        # PyTorch output
        with torch.no_grad():
            pytorch_output = pytorch_model(torch.from_numpy(dummy_input).to(device))
            pytorch_probs = torch.softmax(pytorch_output, dim=1).cpu().numpy()[0]

        # TFLite output
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        tflite_probs = tf.nn.softmax(tflite_output[0]).numpy()

        # Compare
        max_diff = np.max(np.abs(pytorch_probs - tflite_probs))
        argmax_match = np.argmax(pytorch_probs) == np.argmax(tflite_probs)

        print(f"\nValidation Results:")
        print(f"  Max probability difference: {max_diff:.6f}")
        print(f"  Prediction match: {argmax_match}")
        print(f"  PyTorch prediction: {np.argmax(pytorch_probs)}")
        print(f"  TFLite prediction: {np.argmax(tflite_probs)}")

        if max_diff < 0.1 and argmax_match:
            print("  ✓ Validation passed")
            return True
        else:
            print("  ⚠ Validation warning - outputs differ")
            return False

    except Exception as e:
        print(f"Validation failed: {e}")
        return False


def generate_c_header(tflite_path, header_path, model_name="model_tflite"):
    """Generate C header file from TFLite model bytes."""
    with open(tflite_path, 'rb') as f:
        model_bytes = f.read()

    # Generate C array
    hex_array = ', '.join([f'0x{b:02x}' for b in model_bytes])
    array_lines = [hex_array[i:i+80] for i in range(0, len(hex_array), 80)]

    header_content = f"""/*
 * Auto-generated TFLite model data for SensorFusion-HAR ESP32 deployment.
 * Generated from: {tflite_path.name}
 * Model size: {len(model_bytes)} bytes ({len(model_bytes) / 1024:.2f} KB)
 */

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

alignas(8) const unsigned char {model_name}[] = {{
{chr(10).join(['    ' + line for line in array_lines])}
}};

const unsigned int {model_name}_len = sizeof({model_name});

#endif  // MODEL_DATA_H
"""

    with open(header_path, 'w') as f:
        f.write(header_content)

    print(f"Generated C header: {header_path}")
    print(f"Model bytes: {len(model_bytes)}")


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to TFLite')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints_v2/best_sensorfusion_esp32_v2_pocket_v3.pt',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str,
                        default='esp32_deploy/sensorfusion_esp32_v2.tflite',
                        help='Output TFLite model path')
    parser.add_argument('--header', type=str,
                        default='esp32_deploy/model_data.h',
                        help='Output C header path')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for PyTorch model (cpu/cuda)')
    parser.add_argument('--validate', action='store_true',
                        help='Validate TFLite output matches PyTorch')

    args = parser.parse_args()

    # Convert paths
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    header_path = Path(args.header)

    # Create output directories
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SensorFusion-HAR TFLite Export")
    print("=" * 60)

    # Load model
    model, num_classes, labels = load_model(checkpoint_path, args.device)

    # Try direct TFLite export first
    tflite_path = direct_tflite_export(model, output_path)

    if tflite_path is None:
        # Fall back to ONNX route
        onnx_path = output_path.with_suffix('.onnx')
        print("\nTrying ONNX intermediate format...")

        export_to_onnx(model, onnx_path)
        tflite_path = onnx_to_tflite(onnx_path, output_path)

        if tflite_path is None:
            print("\nERROR: Could not export to TFLite")
            print("Please ensure TensorFlow and onnx-tf are installed:")
            print("  pip install tensorflow onnx-tf")
            return 1

        # Cleanup ONNX
        onnx_path.unlink(missing_ok=True)

    # Validate if requested
    if args.validate:
        validate_outputs(model, tflite_path, num_classes, args.device)

    # Generate C header
    generate_c_header(tflite_path, header_path)

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"TFLite model: {tflite_path}")
    print(f"C header: {header_path}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
