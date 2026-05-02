import os, shutil, zipfile, torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model.sensorfusion import SensorFusionHAR
from model.dataset import UCIHARDataset
from model.visualize import plot_tsne, plot_attention_maps, plot_noise_robustness, plot_confidence_calibration
from model.transitions import evaluate_transition_accuracy, plot_transition_analysis
from model.drift import evaluate_drift_robustness, plot_drift_robustness
from model.adversarial import evaluate_adversarial_robustness, plot_adversarial_robustness
from model.energy import count_macs, estimate_energy, compare_models_energy, plot_energy_comparison

OUT = "outputs"
PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data/UCI HAR Dataset"
if not os.path.exists(DATA_DIR):
    DATA_DIR = UCIHARDataset.download("data")
train_ds = UCIHARDataset(DATA_DIR, split="train")
test_ds = UCIHARDataset(DATA_DIR, split="test")
mean = train_ds.X.mean(dim=(0, 1))
std = train_ds.X.std(dim=(0, 1))
train_ds.X = (train_ds.X - mean) / std
test_ds.X = (test_ds.X - mean) / std

checkpoint = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=False)
num_classes = checkpoint["model_state_dict"]["classifier.head.linear.weight"].shape[0]
model = SensorFusionHAR(input_channels=6, reservoir_size=32, num_classes=num_classes).to(device)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()

ACTIVITY_LABELS = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
if num_classes > 6:
    ACTIVITY_LABELS = [f"Class {i}" for i in range(num_classes)]

plots = {
    "01_tsne": lambda: plot_tsne(model, test_ds, device, stage="all", n_samples=1000, activity_labels=ACTIVITY_LABELS, save_path=os.path.join(PLOTS, "01_tsne.png")),
    "02_attention_maps": lambda: plot_attention_maps(model, test_ds, device, n_samples=4, activity_labels=ACTIVITY_LABELS, save_path=os.path.join(PLOTS, "02_attention_maps.png")),
    "03_noise_robustness": lambda: plot_noise_robustness(model, test_ds, device, save_path=os.path.join(PLOTS, "03_noise_robustness.png")),
    "04_confidence_calibration": lambda: plot_confidence_calibration(model, test_ds, device, n_bins=10, save_path=os.path.join(PLOTS, "04_confidence_calibration.png")),
}

for name, fn in plots.items():
    try:
        fn()
        plt.close("all")
        print(f"Saved {name}.png")
    except Exception as e:
        print(f"Skipped {name}: {e}")

try:
    transition_results = evaluate_transition_accuracy(model, test_ds, device)
    fig = plot_transition_analysis(transition_results, activity_labels=ACTIVITY_LABELS, save_path=os.path.join(PLOTS, "05_transitions.png"))
    plt.close("all")
    print("Saved 05_transitions.png")
except Exception as e:
    print(f"Skipped transitions: {e}")

try:
    drift_results = evaluate_drift_robustness(model, test_ds, device)
    fig = plot_drift_robustness(drift_results, save_path=os.path.join(PLOTS, "06_drift.png"))
    plt.close("all")
    print("Saved 06_drift.png")
except Exception as e:
    print(f"Skipped drift: {e}")

try:
    adv_results = evaluate_adversarial_robustness(model, test_ds, device, epsilons=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    fig = plot_adversarial_robustness(adv_results, save_path=os.path.join(PLOTS, "07_adversarial.png"))
    plt.close("all")
    print("Saved 07_adversarial.png")
except Exception as e:
    print(f"Skipped adversarial: {e}")

try:
    from model.sensorfusion import NoReservoirModel, NoAttentionModel, NoGateModel
    energy_models = {
        "SensorFusionHAR": SensorFusionHAR(input_channels=6, reservoir_size=32, num_classes=num_classes),
        "No Reservoir": NoReservoirModel(num_classes=num_classes),
        "No Attention": NoAttentionModel(num_classes=num_classes),
        "No Gate": NoGateModel(num_classes=num_classes),
    }
    energy_comparison = compare_models_energy(energy_models, input_shape=(1, 128, 6))
    fig = plot_energy_comparison(energy_comparison, save_path=os.path.join(PLOTS, "08_energy.png"))
    plt.close("all")
    print("Saved 08_energy.png")
except Exception as e:
    print(f"Skipped energy: {e}")

shutil.copy2("checkpoints/best_model.pt", os.path.join(OUT, "best_model.pt"))
print("Copied best_model.pt")

onnx_src = "sensorfusion-har/sensorfusion_har.onnx"
if not os.path.exists(onnx_src):
    onnx_src = "checkpoints/sensorfusion_har.onnx"
if os.path.exists(onnx_src):
    shutil.copy2(onnx_src, os.path.join(OUT, "sensorfusion_har.onnx"))
    print("Copied sensorfusion_har.onnx")
else:
    export_model = SensorFusionHAR(input_channels=6, reservoir_size=32, num_classes=num_classes)
    export_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    export_model.eval()
    dummy_input = torch.randn(1, 128, 6)
    onnx_out = os.path.join(OUT, "sensorfusion_har.onnx")
    torch.onnx.export(
        export_model, dummy_input, onnx_out,
        input_names=["sensor_input"], output_names=["activity_logits"],
        dynamic_axes={"sensor_input": {0: "batch"}, "activity_logits": {0: "batch"}},
        opset_version=13,
    )
    print("Exported sensorfusion_har.onnx")

onnx_data = "sensorfusion_har.onnx.data"
if os.path.exists(onnx_data):
    shutil.copy2(onnx_data, os.path.join(OUT, "sensorfusion_har.onnx.data"))
    print("Copied sensorfusion_har.onnx.data")

try:
    binary_export = model.classifier.head.export_binary()
    torch.save(binary_export, os.path.join(OUT, "binary_weights.pt"))
    print("Saved binary_weights.pt")
except Exception as e:
    print(f"Skipped binary export: {e}")

zip_path = "sensorfusion_har_outputs.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(OUT):
        for file in files:
            fpath = os.path.join(root, file)
            zf.write(fpath, os.path.relpath(fpath, OUT))

print(f"\nAll outputs zipped to: {os.path.abspath(zip_path)}")
with zipfile.ZipFile(zip_path, "r") as zf:
    for info in zf.infolist():
        print(f"  {info.filename} ({info.file_size:,} bytes)")
