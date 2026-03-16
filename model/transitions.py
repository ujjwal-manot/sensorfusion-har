import numpy as np
import torch
import matplotlib.pyplot as plt


DEFAULT_ACTIVITY_LABELS = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]


def detect_transitions(labels, window_size=128, step_size=64):
    transitions = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            transitions.append((i, labels[i - 1], labels[i]))
    return transitions


def _classify_windows(dataset, threshold=0.95):
    stable_indices = []
    transition_indices = []
    transition_info = []

    n = len(dataset)

    if hasattr(dataset, "_raw_window_labels"):
        # Pre-populate stable_indices, then reclassify transitions
        stable_indices = list(range(n))
        for idx in range(len(dataset)):
            raw_labels = dataset._raw_window_labels[idx]
            unique, counts = np.unique(raw_labels, return_counts=True)
            dominant_ratio = counts.max() / len(raw_labels)

            if dominant_ratio < threshold:
                if idx not in transition_indices:
                    transition_indices.append(idx)
                    from_act = unique[np.argsort(counts)[-2]] if len(unique) > 1 else unique[0]
                    to_act = unique[np.argmax(counts)]
                    transition_info.append((from_act, to_act))
                if idx in stable_indices:
                    stable_indices.remove(idx)
        return stable_indices, transition_indices, transition_info

    n = len(dataset)
    labels_seq = []
    for i in range(n):
        _, y = dataset[i]
        labels_seq.append(y.item() if hasattr(y, "item") else y)
    labels_seq = np.array(labels_seq)

    stable_indices = []
    transition_indices = []
    transition_info = []

    for i in range(n):
        is_transition = False
        from_act = labels_seq[i]
        to_act = labels_seq[i]

        if i > 0 and labels_seq[i] != labels_seq[i - 1]:
            is_transition = True
            from_act = labels_seq[i - 1]
            to_act = labels_seq[i]

        if i < n - 1 and labels_seq[i] != labels_seq[i + 1]:
            is_transition = True
            from_act = labels_seq[i]
            to_act = labels_seq[i + 1]

        if is_transition:
            transition_indices.append(i)
            transition_info.append((from_act, to_act))
        else:
            stable_indices.append(i)

    return stable_indices, transition_indices, transition_info


def evaluate_transition_accuracy(model, dataset, device, transition_window=5):
    stable_indices, transition_indices, transition_info = _classify_windows(dataset)

    all_x, all_y = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        all_x.append(x)
        all_y.append(y.item() if hasattr(y, "item") else y)

    X = torch.stack(all_x)
    Y = np.array(all_y)

    model.eval()
    batch_size = 256
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size].to(device)
            out = model(batch)
            all_preds.append(out.argmax(dim=1).cpu().numpy())
    all_preds = np.concatenate(all_preds)

    stable_correct = 0
    stable_total = 0
    if stable_indices:
        stable_idx = np.array(stable_indices)
        stable_correct = (all_preds[stable_idx] == Y[stable_idx]).sum()
        stable_total = len(stable_idx)

    transition_correct = 0
    transition_total = 0
    if transition_indices:
        trans_idx = np.array(transition_indices)
        transition_correct = (all_preds[trans_idx] == Y[trans_idx]).sum()
        transition_total = len(trans_idx)

    pair_accuracy = {}
    for i, tidx in enumerate(transition_indices):
        if i < len(transition_info):
            pair = transition_info[i]
            key = (pair[0], pair[1])
            if key not in pair_accuracy:
                pair_accuracy[key] = {"correct": 0, "total": 0}
            pair_accuracy[key]["total"] += 1
            if all_preds[tidx] == Y[tidx]:
                pair_accuracy[key]["correct"] += 1

    pair_results = {}
    for pair, counts in pair_accuracy.items():
        if counts["total"] > 0:
            pair_results[pair] = counts["correct"] / counts["total"]

    num_classes = len(np.unique(Y))

    results = {
        "stable_accuracy": stable_correct / stable_total if stable_total > 0 else 0.0,
        "transition_accuracy": transition_correct / transition_total if transition_total > 0 else 0.0,
        "stable_count": stable_total,
        "transition_count": transition_total,
        "pair_accuracy": pair_results,
        "overall_accuracy": (all_preds == Y).mean(),
        "num_classes": num_classes,
    }

    return results


def plot_transition_analysis(results, activity_labels=None, save_path=None):
    num_classes = results.get("num_classes", 6)
    if activity_labels is None:
        if num_classes <= len(DEFAULT_ACTIVITY_LABELS):
            activity_labels = DEFAULT_ACTIVITY_LABELS[:num_classes]
        else:
            activity_labels = [f"Class {i}" for i in range(num_classes)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    categories = ["Stable", "Transition", "Overall"]
    accuracies = [
        results["stable_accuracy"],
        results["transition_accuracy"],
        results["overall_accuracy"],
    ]
    counts = [
        results["stable_count"],
        results["transition_count"],
        results["stable_count"] + results["transition_count"],
    ]
    bar_colors = ["#4CAF50", "#F44336", "#2196F3"]

    bars = ax1.bar(categories, accuracies, color=bar_colors, alpha=0.8, edgecolor="white")
    for bar, acc, count in zip(bars, accuracies, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{acc:.3f}\n(n={count})", ha="center", va="bottom", fontsize=9)

    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Stable vs Transition Accuracy", fontsize=12)
    ax1.set_ylim(0, 1.15)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    pair_accuracy = results.get("pair_accuracy", {})
    if pair_accuracy:
        sorted_pairs = sorted(pair_accuracy.items(), key=lambda p: p[1], reverse=True)
        pair_labels = []
        pair_accs = []
        for (from_act, to_act), acc in sorted_pairs:
            from_name = activity_labels[from_act] if from_act < len(activity_labels) else f"C{from_act}"
            to_name = activity_labels[to_act] if to_act < len(activity_labels) else f"C{to_act}"
            pair_labels.append(f"{from_name} -> {to_name}")
            pair_accs.append(acc)

        y_pos = np.arange(len(pair_labels))
        ax2.barh(y_pos, pair_accs, color="#FF9800", alpha=0.8, edgecolor="white")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(pair_labels, fontsize=8)
        ax2.set_xlabel("Accuracy", fontsize=11)
        ax2.set_title("Accuracy by Transition Pair", fontsize=12)
        ax2.set_xlim(0, 1.05)
        ax2.grid(True, alpha=0.3, axis="x")
    else:
        ax2.text(0.5, 0.5, "No transition pairs detected",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Accuracy by Transition Pair", fontsize=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
