"""
Export PyTorch SensorFusionESP32 model weights to C arrays for ESP32 deployment.
This avoids TFLite conversion by exporting weights as raw arrays.

Usage:
    python export_weights_esp32.py --checkpoint checkpoints_v2/best_sensorfusion_esp32_v2_pocket_v3.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def load_model(checkpoint_path, device='cpu'):
    """Load the PyTorch model from checkpoint."""
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent / "final_esp32_v2_useful11_package"))

    from train_esp32_v2_expanded_local import SensorFusionESP32

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_labels = state.get("labels", state.get("activity_labels", None))
    num_classes = len(ckpt_labels) if ckpt_labels else state.get("num_classes", 7)

    print(f"Loading model with {num_classes} classes")
    if ckpt_labels:
        print(f"Labels: {ckpt_labels}")

    model = SensorFusionESP32(input_channels=6, reservoir_size=64, num_classes=num_classes)

    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.eval()
    model.to(device)

    print(f"Model loaded from {checkpoint_path}")
    print(f"Parameters: {model.count_parameters():,}")

    return model, num_classes, ckpt_labels, state


def tensor_to_c_array(tensor, name, dtype='float'):
    """Convert a PyTorch tensor to C array string."""
    tensor_np = tensor.detach().cpu().numpy().flatten()
    
    if dtype == 'float':
        values = ', '.join([f'{v:.8g}' for v in tensor_np])
    elif dtype == 'int':
        values = ', '.join([f'{int(v)}' for v in tensor_np])
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    array_def = f"const {dtype} {name}[] = {{{values}}};"
    size_def = f"const int {name}_size = {len(tensor_np)};"
    
    return array_def, size_def, tensor_np.shape


def generate_weights_header(model, labels, normalization_stats, output_path):
    """Generate C header with all model weights."""
    lines = []
    
    lines.append("/*")
    lines.append(" * Auto-generated model weights for SensorFusion-HAR ESP32 deployment")
    lines.append(" * This file contains the trained weights for manual inference")
    lines.append(" */")
    lines.append("#ifndef MODEL_WEIGHTS_H")
    lines.append("#define MODEL_WEIGHTS_H")
    lines.append("")
    
    # Activity labels
    lines.append("// Activity labels")
    lines.append(f"const int NUM_CLASSES = {len(labels)};")
    labels_str = ', '.join([f'"{l}"' for l in labels])
    lines.append(f"const char* ACTIVITY_LABELS[NUM_CLASSES] = {{{labels_str}}};")
    lines.append("")
    
    # Normalization stats
    lines.append("// Normalization statistics")
    lines.append(f"const float NORM_MEAN[6] = {{")
    lines.append(f"    {', '.join([f'{v:.8g}' for v in normalization_stats['mean']])}")
    lines.append("};")
    lines.append(f"const float NORM_STD[6] = {{")
    lines.append(f"    {', '.join([f'{v:.8g}' for v in normalization_stats['std']])}")
    lines.append("};")
    lines.append("")
    
    # Export key weights (simplified for ESP32)
    # Note: This is a simplified export - full model would need all layers
    
    state_dict = model.state_dict()
    
    lines.append("// Model weights (simplified subset for demonstration)")
    lines.append("// Full implementation would export all layer weights")
    lines.append("")
    
    # Export classifier weights (most important)
    if 'classifier.weight' in state_dict:
        array_def, size_def, shape = tensor_to_c_array(state_dict['classifier.weight'], 'classifier_weight')
        lines.append(f"// Classifier weight: shape {shape}")
        lines.append(size_def)
        lines.append(array_def)
        lines.append("")
    
    if 'classifier.bias' in state_dict:
        array_def, size_def, shape = tensor_to_c_array(state_dict['classifier.bias'], 'classifier_bias')
        lines.append(f"// Classifier bias: shape {shape}")
        lines.append(size_def)
        lines.append(array_def)
        lines.append("")
    
    lines.append("#endif  // MODEL_WEIGHTS_H")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Generated weights header: {output_path}")


def generate_normalization_json(normalization_stats, labels, output_path):
    """Generate JSON file with normalization stats for reference."""
    data = {
        'normalization': normalization_stats,
        'labels': labels,
        'num_classes': len(labels),
        'input_shape': [1, 50, 6],
        'output_shape': [1, len(labels)]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated normalization JSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Export model weights for ESP32')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints_v2/best_sensorfusion_esp32_v2_pocket_v3.pt',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--weights-header', type=str,
                        default='esp32_deploy/model_weights.h',
                        help='Output weights header path')
    parser.add_argument('--normalization-json', type=str,
                        default='esp32_deploy/normalization.json',
                        help='Output normalization JSON path')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for PyTorch model (cpu/cuda)')

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    weights_header_path = Path(args.weights_header)
    normalization_json_path = Path(args.normalization_json)

    weights_header_path.parent.mkdir(parents=True, exist_ok=True)
    normalization_json_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SensorFusion-HAR Weight Export for ESP32")
    print("=" * 60)

    model, num_classes, labels, state = load_model(checkpoint_path, args.device)

    # Get normalization stats
    norm_stats = state.get("normalization", state.get("normalization_stats", None))
    if norm_stats is None:
        # Try to load from file
        norm_file = Path("exports/esp32_v2/normalization_stats.json")
        if norm_file.exists():
            with open(norm_file) as f:
                norm_stats = json.load(f)
        else:
            print("WARNING: No normalization stats found, using defaults")
            norm_stats = {
                'mean': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'std': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }

    # Generate outputs
    generate_weights_header(model, labels, norm_stats, weights_header_path)
    generate_normalization_json(norm_stats, labels, normalization_json_path)

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"Weights header: {weights_header_path}")
    print(f"Normalization JSON: {normalization_json_path}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
