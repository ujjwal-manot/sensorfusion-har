import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score

DEFAULT_ACTIVITY_LABELS = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
DEFAULT_COLORS = ["#4CAF50", "#FF9800", "#2196F3", "#9C27B0", "#F44336", "#00BCD4"]


def _get_labels_and_colors(activity_labels, colors, num_classes):
    if activity_labels is None:
        activity_labels = DEFAULT_ACTIVITY_LABELS[:num_classes]
        if num_classes > len(DEFAULT_ACTIVITY_LABELS):
            activity_labels = [f"Class {i}" for i in range(num_classes)]
    if colors is None:
        colors = DEFAULT_COLORS[:num_classes]
        if num_classes > len(DEFAULT_COLORS):
            cmap = plt.cm.get_cmap("tab20", num_classes)
            colors = [cmap(i) for i in range(num_classes)]
    return activity_labels, colors


def _extract_embeddings(model, dataset, device, n_samples):
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    all_x, all_y = [], []
    for idx in indices:
        x, y = dataset[idx]
        all_x.append(x)
        all_y.append(y)

    X = torch.stack(all_x).to(device)
    Y = torch.tensor(all_y).numpy() if not isinstance(all_y[0], int) else np.array([y.item() if hasattr(y, 'item') else y for y in all_y])

    model.eval()
    with torch.no_grad():
        h_reservoir = model.reservoir(X)

        h_dsconv_in = h_reservoir.transpose(1, 2)
        h_dsconv = model.dsconv(h_dsconv_in)

        h_attention = model.attention(h_dsconv)

        h_final = model.classifier(h_attention)

    stages = {
        "reservoir": h_reservoir.cpu().numpy().reshape(len(indices), -1),
        "dsconv": h_dsconv.cpu().numpy().reshape(len(indices), -1),
        "attention": h_attention.cpu().numpy().reshape(len(indices), -1),
        "final": h_final.cpu().numpy().reshape(len(indices), -1),
    }

    return stages, Y


def plot_tsne(model, dataset, device, stage="all", n_samples=1000, activity_labels=None, colors=None, save_path=None):
    stages, labels = _extract_embeddings(model, dataset, device, n_samples)
    num_classes = len(np.unique(labels))
    activity_labels, colors = _get_labels_and_colors(activity_labels, colors, num_classes)

    if stage == "all":
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        stage_names = ["reservoir", "dsconv", "attention", "final"]
        for ax, sname in zip(axes.flatten(), stage_names):
            embeddings = stages[sname]
            perplexity = min(30, len(embeddings) - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            reduced = tsne.fit_transform(embeddings)
            for cls_idx in range(num_classes):
                mask = labels == cls_idx
                label = activity_labels[cls_idx] if cls_idx < len(activity_labels) else f"Class {cls_idx}"
                color = colors[cls_idx] if cls_idx < len(colors) else None
                ax.scatter(reduced[mask, 0], reduced[mask, 1], c=color, label=label, s=10, alpha=0.7)
            ax.set_title(sname.upper())
            ax.legend(fontsize=7, markerscale=2)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        embeddings = stages[stage]
        perplexity = min(30, len(embeddings) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced = tsne.fit_transform(embeddings)
        for cls_idx in range(num_classes):
            mask = labels == cls_idx
            label = activity_labels[cls_idx] if cls_idx < len(activity_labels) else f"Class {cls_idx}"
            color = colors[cls_idx] if cls_idx < len(colors) else None
            ax.scatter(reduced[mask, 0], reduced[mask, 1], c=color, label=label, s=10, alpha=0.7)
        ax.set_title(stage.upper())
        ax.legend(markerscale=2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_attention_maps(model, dataset, device, n_samples=4, activity_labels=None, colors=None, save_path=None):
    num_classes = len(np.unique([dataset[i][1].item() if hasattr(dataset[i][1], 'item') else dataset[i][1] for i in range(min(100, len(dataset)))]))
    activity_labels, colors = _get_labels_and_colors(activity_labels, colors, num_classes)

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    captured_weights = []

    def hook_fn(module, input, output):
        _, attn_weights = module(input[0], input[1], input[2], need_weights=True, average_attn_weights=True)
        captured_weights.append(attn_weights.detach().cpu())

    hook_handle = None
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            hook_handle = module.register_forward_hook(hook_fn)
            break

    fig, axes = plt.subplots(1, n_samples, figsize=(12, 4))
    if n_samples == 1:
        axes = [axes]

    model.eval()
    for i, idx in enumerate(indices):
        x, y = dataset[idx]
        x = x.unsqueeze(0).to(device)
        label_idx = y.item() if hasattr(y, 'item') else y

        captured_weights.clear()
        with torch.no_grad():
            model(x)

        if captured_weights:
            attn_map = captured_weights[0].squeeze(0).numpy()
        else:
            attn_map = np.zeros((8, 8))

        ax = axes[i]
        im = ax.imshow(attn_map, cmap="inferno", aspect="auto", vmin=0)
        title = activity_labels[label_idx] if label_idx < len(activity_labels) else f"Class {label_idx}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if hook_handle:
        hook_handle.remove()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_noise_robustness(model, dataset, device, snr_levels=None, activity_labels=None, save_path=None):
    if snr_levels is None:
        snr_levels = [40, 30, 20, 15, 10, 5, 0]

    all_x, all_y = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        all_x.append(x)
        all_y.append(y.item() if hasattr(y, 'item') else y)

    X = torch.stack(all_x)
    Y = np.array(all_y)

    num_classes = len(np.unique(Y))
    if activity_labels is None:
        activity_labels = DEFAULT_ACTIVITY_LABELS[:num_classes]
        if num_classes > len(DEFAULT_ACTIVITY_LABELS):
            activity_labels = [f"Class {i}" for i in range(num_classes)]

    model.eval()
    results = {"snr_levels": snr_levels, "accuracy": [], "f1": [], "accuracy_std": [], "f1_std": []}

    n_trials = 5
    for snr in snr_levels:
        trial_accs = []
        trial_f1s = []
        for _ in range(n_trials):
            signal_power = (X ** 2).mean()
            noise_power = signal_power / (10 ** (snr / 10))
            noise = torch.randn_like(X) * torch.sqrt(noise_power)
            X_noisy = X + noise

            batch_size = 256
            preds = []
            with torch.no_grad():
                for start in range(0, len(X_noisy), batch_size):
                    batch = X_noisy[start:start + batch_size].to(device)
                    out = model(batch)
                    preds.append(out.argmax(dim=1).cpu().numpy())
            preds = np.concatenate(preds)

            acc = (preds == Y).mean()
            f1 = f1_score(Y, preds, average="weighted")
            trial_accs.append(acc)
            trial_f1s.append(f1)

        results["accuracy"].append(np.mean(trial_accs))
        results["f1"].append(np.mean(trial_f1s))
        results["accuracy_std"].append(np.std(trial_accs))
        results["f1_std"].append(np.std(trial_f1s))

    acc_arr = np.array(results["accuracy"])
    acc_std = np.array(results["accuracy_std"])
    f1_arr = np.array(results["f1"])
    f1_std = np.array(results["f1_std"])

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(snr_levels, acc_arr, "o-", color="#4CAF50", label="Accuracy", linewidth=2)
    ax.fill_between(snr_levels, acc_arr - acc_std, acc_arr + acc_std, color="#4CAF50", alpha=0.2)
    ax.plot(snr_levels, f1_arr, "s-", color="#2196F3", label="F1 Score", linewidth=2)
    ax.fill_between(snr_levels, f1_arr - f1_std, f1_arr + f1_std, color="#2196F3", alpha=0.2)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Score")
    ax.set_title("Noise Robustness")
    ax.legend()
    ax.set_xlim(max(snr_levels), min(snr_levels))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return results, fig


def plot_confidence_calibration(model, dataset, device, n_bins=10, activity_labels=None, save_path=None):
    all_x, all_y = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        all_x.append(x)
        all_y.append(y.item() if hasattr(y, 'item') else y)

    X = torch.stack(all_x)
    Y = np.array(all_y)

    model.eval()
    batch_size = 256
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size].to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    confidences = np.max(all_probs, axis=1)
    predictions = np.argmax(all_probs, axis=1)
    accuracies = (predictions == Y).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    ece = 0.0
    total = len(Y)

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)

        count = mask.sum()
        bin_counts.append(count)

        if count > 0:
            avg_acc = accuracies[mask].mean()
            avg_conf = confidences[mask].mean()
            bin_accs.append(avg_acc)
            bin_confs.append(avg_conf)
            ece += (count / total) * abs(avg_acc - avg_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((low + high) / 2)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    ax.bar(bin_centers, bin_accs, width=1.0 / n_bins, alpha=0.6, color="#2196F3", edgecolor="white", label="Observed")
    ax.plot([0, 1], [0, 1], "--", color="#F44336", linewidth=2, label="Perfect Calibration")
    ax.plot(bin_confs, bin_accs, "o-", color="#4CAF50", linewidth=2, label="Model")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Confidence Calibration (ECE = {ece:.4f})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ece, fig
