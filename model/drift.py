import numpy as np
import torch
import matplotlib.pyplot as plt


def simulate_bias_drift(data, drift_rate=0.01, channels=None):
    drifted = data.clone()
    seq_len = data.shape[1] if data.dim() == 3 else data.shape[0]

    if channels is None:
        channels = list(range(data.shape[-1]))

    ramp = torch.arange(seq_len, dtype=data.dtype, device=data.device).float() * drift_rate

    if data.dim() == 3:
        for ch in channels:
            drifted[:, :, ch] = drifted[:, :, ch] + ramp.unsqueeze(0)
    else:
        for ch in channels:
            drifted[:, ch] = drifted[:, ch] + ramp

    return drifted


def simulate_scale_drift(data, drift_rate=0.001, channels=None):
    drifted = data.clone()
    seq_len = data.shape[1] if data.dim() == 3 else data.shape[0]

    if channels is None:
        channels = list(range(data.shape[-1]))

    scale = 1.0 + torch.arange(seq_len, dtype=data.dtype, device=data.device).float() * drift_rate

    if data.dim() == 3:
        for ch in channels:
            drifted[:, :, ch] = drifted[:, :, ch] * scale.unsqueeze(0)
    else:
        for ch in channels:
            drifted[:, ch] = drifted[:, ch] * scale

    return drifted


def simulate_noise_drift(data, initial_snr=40, final_snr=10, channels=None):
    drifted = data.clone()
    seq_len = data.shape[1] if data.dim() == 3 else data.shape[0]

    if channels is None:
        channels = list(range(data.shape[-1]))

    snr_schedule = torch.linspace(initial_snr, final_snr, seq_len)

    if data.dim() == 3:
        signal_power = (data ** 2).mean(dim=0)
        for t in range(seq_len):
            snr_linear = 10 ** (snr_schedule[t] / 10)
            for ch in channels:
                noise_power = signal_power[t, ch] / snr_linear
                noise_std = torch.sqrt(noise_power.clamp(min=1e-12))
                drifted[:, t, ch] = drifted[:, t, ch] + torch.randn(data.shape[0]) * noise_std
    else:
        signal_power = data ** 2
        for t in range(seq_len):
            snr_linear = 10 ** (snr_schedule[t] / 10)
            for ch in channels:
                noise_power = signal_power[t, ch] / snr_linear
                noise_std = torch.sqrt(noise_power.clamp(min=1e-12))
                drifted[t, ch] = drifted[t, ch] + torch.randn(1).item() * noise_std

    return drifted


def evaluate_drift_robustness(model, dataset, device, drift_types=None, drift_levels=None):
    if drift_types is None:
        drift_types = ["bias", "scale", "noise"]

    if drift_levels is None:
        drift_levels = {
            "bias": [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
            "scale": [0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02],
            "noise": [(40, 40), (40, 30), (40, 20), (40, 15), (40, 10), (40, 5), (40, 0)],
        }

    all_x, all_y = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        all_x.append(x)
        all_y.append(y.item() if hasattr(y, "item") else y)

    X = torch.stack(all_x)
    Y = np.array(all_y)

    model.eval()
    batch_size = 256

    def predict(data):
        preds = []
        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                batch = data[start:start + batch_size].to(device)
                out = model(batch)
                preds.append(out.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    clean_preds = predict(X)
    clean_accuracy = (clean_preds == Y).mean()

    results = {
        "clean_accuracy": clean_accuracy,
        "drift_types": drift_types,
        "drift_levels": drift_levels,
    }

    for dtype in drift_types:
        levels = drift_levels[dtype]
        accuracies = []

        for level in levels:
            if dtype == "bias":
                if level == 0:
                    accuracies.append(clean_accuracy)
                    continue
                X_drifted = simulate_bias_drift(X, drift_rate=level)
            elif dtype == "scale":
                if level == 0:
                    accuracies.append(clean_accuracy)
                    continue
                X_drifted = simulate_scale_drift(X, drift_rate=level)
            elif dtype == "noise":
                initial_snr, final_snr = level
                if initial_snr == final_snr == 40:
                    accuracies.append(clean_accuracy)
                    continue
                X_drifted = simulate_noise_drift(X, initial_snr=initial_snr, final_snr=final_snr)
            else:
                continue

            preds = predict(X_drifted)
            accuracies.append((preds == Y).mean())

        results[f"{dtype}_accuracy"] = accuracies

    return results


def plot_drift_robustness(results, save_path=None):
    drift_types = results["drift_types"]
    drift_levels = results["drift_levels"]

    n_types = len(drift_types)
    fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 5))
    if n_types == 1:
        axes = [axes]

    type_config = {
        "bias": {"color": "#4CAF50", "xlabel": "Bias Drift Rate", "title": "Bias Drift"},
        "scale": {"color": "#2196F3", "xlabel": "Scale Drift Rate", "title": "Scale Drift"},
        "noise": {"color": "#FF9800", "xlabel": "Final SNR (dB)", "title": "Noise Drift"},
    }

    for ax, dtype in zip(axes, drift_types):
        config = type_config.get(dtype, {"color": "#9C27B0", "xlabel": dtype, "title": dtype})
        levels = drift_levels[dtype]
        accuracies = results[f"{dtype}_accuracy"]

        if dtype == "noise":
            x_vals = [lvl[1] for lvl in levels]
        else:
            x_vals = levels

        ax.plot(x_vals, accuracies, "o-", color=config["color"], linewidth=2, markersize=6)
        ax.axhline(y=results["clean_accuracy"], color="#F44336", linestyle="--",
                    linewidth=1.5, alpha=0.7, label="Clean Accuracy")

        ax.set_xlabel(config["xlabel"], fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(config["title"], fontsize=12)
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        if dtype == "noise":
            ax.invert_xaxis()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
