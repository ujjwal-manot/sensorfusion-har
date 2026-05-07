"""Build TFLite model + ESP32 C header from the canonical 11-class ONNX."""
import os
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "training"))

ONNX = BASE / "exports" / "sensorfusion_esp32_v2_useful11_final.onnx"
OUT_TFLITE = BASE / "esp32" / "sensorfusion_esp32_v2_useful11_final.tflite"
OUT_HEADER = BASE / "esp32" / "model_data.h"
WORK = BASE / "work" / "tf_build"

if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True, exist_ok=True)

print(f"Source ONNX: {ONNX} ({ONNX.stat().st_size/1024:.1f} KB)")

# Save a calibration sample as numpy for -cotof comparison
calib_npy = WORK / "calib_sample.npy"
rng_seed = np.random.default_rng(0)
np.save(calib_npy, rng_seed.standard_normal((1, 50, 6)).astype(np.float32))

# Use onnx2tf CLI for the conversion (more robust than the python API)
# -kat forces the input tensor to keep its original [1, 50, 6] layout so the
# ESP32 firmware (which writes idx = t * NUM_CHANNELS + c, i.e. row-major
# [T, C]) feeds the model in the layout it expects. Without -kat, onnx2tf's
# NCHW->NHWC pass produces a [1, 6, 50] input tensor and every firmware
# inference would silently transpose the window.
cmd = [
    "onnx2tf",
    "-i", str(ONNX),
    "-o", str(WORK / "tf_out"),
    "-osd",            # output saved-model dir
    "-cotof",          # check op-by-op accuracy vs onnx
    "-cind", "input", str(calib_npy),
    "-kat", "input",
    "-b", "1",
]
print(f"Running: {' '.join(cmd)}")
res = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT (tail):")
print("\n".join(res.stdout.splitlines()[-30:]))
if res.returncode != 0:
    print("STDERR:")
    print("\n".join(res.stderr.splitlines()[-20:]))
    sys.exit(1)

# Find the produced .tflite
candidates = sorted((WORK / "tf_out").glob("*.tflite"))
if not candidates:
    print("No .tflite produced; listing tf_out:")
    for p in (WORK / "tf_out").iterdir():
        print(" ", p.name)
    sys.exit(1)

# Prefer the float32 .tflite for ESP32-S3 (no integer quant calibration data here yet)
preferred = None
for p in candidates:
    if "float32" in p.name:
        preferred = p
        break
if preferred is None:
    preferred = candidates[0]

print(f"Selected: {preferred.name} ({preferred.stat().st_size/1024:.1f} KB)")
shutil.copy(preferred, OUT_TFLITE)
print(f"Copied to: {OUT_TFLITE}")

# Validate by running both PyTorch and TFLite on a known input
import torch
import tensorflow as tf

interp = tf.lite.Interpreter(model_path=str(OUT_TFLITE))
interp.allocate_tensors()
in_det = interp.get_input_details()
out_det = interp.get_output_details()
print("TFLite input:", in_det[0]["shape"], in_det[0]["dtype"])
print("TFLite output:", out_det[0]["shape"], out_det[0]["dtype"])

from train_esp32_v2_expanded_local import SensorFusionESP32

ckpt = torch.load(BASE / "checkpoints" / "best_sensorfusion_esp32_v2_useful11_final.pt",
                  map_location="cpu", weights_only=False)
model = SensorFusionESP32(input_channels=6, reservoir_size=64, num_classes=11)
sd = ckpt["model_state_dict"]
# Mirror the same remap server.py applies for these older checkpoints.
import importlib
spec = importlib.util.spec_from_file_location("server_mod", BASE / "server.py")
server_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_mod)
sd = server_mod._compatible_esp32_state_dict(sd)
model.load_state_dict(sd, strict=True)
model.eval()

rng = np.random.default_rng(42)
diffs = []
matches = 0
# Both PyTorch and TFLite now take input in [1, 50, 6] (T, C) layout, so no
# transpose is needed in the firmware or here.
for trial in range(8):
    x_pt = rng.standard_normal((1, 50, 6)).astype(np.float32)
    with torch.no_grad():
        torch_out = torch.softmax(model(torch.from_numpy(x_pt)), dim=1).numpy()[0]
    interp.set_tensor(in_det[0]["index"], x_pt)
    interp.invoke()
    tfl_logits = interp.get_tensor(out_det[0]["index"])
    tfl_probs = tf.nn.softmax(tfl_logits[0]).numpy()
    diff = float(np.abs(torch_out - tfl_probs).max())
    diffs.append(diff)
    matches += int(np.argmax(torch_out) == np.argmax(tfl_probs))

print(f"\nValidation over 8 trials: max-prob diff mean={np.mean(diffs):.4g}, "
      f"max={np.max(diffs):.4g}, argmax matches={matches}/8")

# Build the C header
mb = OUT_TFLITE.read_bytes()
hex_array = ", ".join(f"0x{b:02x}" for b in mb)
LINE_W = 16
hex_bytes = [f"0x{b:02x}" for b in mb]
lines = []
for i in range(0, len(hex_bytes), LINE_W):
    chunk = ", ".join(hex_bytes[i:i + LINE_W])
    lines.append("    " + chunk + ("," if i + LINE_W < len(hex_bytes) else ""))

header = f"""/*
 * Auto-generated TFLite model data for SensorFusion-HAR ESP32 deployment.
 *
 * Source: exports/sensorfusion_esp32_v2_useful11_final.onnx
 * Converted via: onnx2tf {tf.__version__} -> TFLite float32
 * Model size: {len(mb)} bytes ({len(mb)/1024:.2f} KB)
 *
 * Activity labels (in classifier output order, 0..10):
 *   Walking, Sitting, Standing, Lying Down, Stairs Up, Stairs Down,
 *   Jogging, Jumping, Cycling, Running, Waist Bending
 *
 * Validation: TFLite agrees with PyTorch source on argmax for {matches}/8 random inputs,
 *             max softmax diff {np.max(diffs):.4g}.
 *
 * Usage on ESP32:
 *   #include "model_data.h"
 *   const tflite::Model* model = tflite::GetModel(model_tflite);
 */

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

alignas(8) const unsigned char model_tflite[] = {{
{chr(10).join(lines)}
}};

const unsigned int model_tflite_len = {len(mb)};

#endif  // MODEL_DATA_H
"""
OUT_HEADER.write_text(header)
print(f"\nWrote C header: {OUT_HEADER} ({len(header)/1024:.1f} KB source size)")
print(f"  array len: {len(mb)} bytes")
