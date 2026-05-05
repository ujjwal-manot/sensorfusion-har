import sys, os, torch
sys.path.insert(0, ".")
from train_esp32_v2_expanded_local import SensorFusionESP32, TARGET_TIME_STEPS, INPUT_CHANNELS

ckpt_path = "checkpoints_v2/best_sensorfusion_esp32_v2_pocket_final.pt"
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
labels = state.get("labels", [])
num_classes = len(labels) if labels else 7
model = SensorFusionESP32(input_channels=INPUT_CHANNELS, reservoir_size=64, num_classes=num_classes)
model.load_state_dict(state["model_state_dict"])
model.eval()

dummy = torch.randn(1, TARGET_TIME_STEPS, INPUT_CHANNELS)
os.makedirs("exports/esp32_v2", exist_ok=True)
onnx_path = "exports/esp32_v2/sensorfusion_esp32_v2_pocket_final.onnx"

torch.onnx.export(
    model, dummy, onnx_path, opset_version=18,
    input_names=["input"], output_names=["logits"],
)
import onnx
m = onnx.load(onnx_path)
onnx.checker.check_model(m)
kb = os.path.getsize(onnx_path) // 1024
print(f"ONNX OK: {onnx_path} ({kb} KB)")
print(f"Labels: {labels}")
print(f"Metrics: acc={state['metrics']['acc']:.4f} macro_f1={state['metrics']['macro_f1']:.4f} min_f1={state['metrics']['min_f1']:.4f}")
print(f"Stage: {state.get('stage')}  Epoch: {state.get('epoch')}")
