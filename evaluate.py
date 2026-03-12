import os
import json
import time
import argparse
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

from model import SensorFusionHAR
from model.dataset import UCIHARDataset
from model.dsconv import DSConvEncoder


ACTIVITY_LABELS = ["WALKING", "WALKING_UP", "WALKING_DOWN", "SITTING", "STANDING", "LAYING"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/UCI HAR Dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--ablation_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark_runs", type=int, default=1000)
    return parser.parse_args()


def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            all_preds.append(out.argmax(dim=1).cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = (all_preds == all_labels).mean()
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    return acc, f1_macro, f1_per_class, all_preds, all_labels


def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    header = "          " + "".join("{:>10s}".format(ACTIVITY_LABELS[i][:8]) for i in range(len(ACTIVITY_LABELS)))
    print(header)
    for i in range(cm.shape[0]):
        row = "{:<10s}".format(ACTIVITY_LABELS[i][:8])
        row += "".join("{:>10d}".format(cm[i, j]) for j in range(cm.shape[1]))
        print(row)


def print_results(acc, f1_macro, f1_per_class, preds, labels):
    print("Accuracy: {:.2f}%".format(acc * 100))
    print("F1 Macro: {:.4f}".format(f1_macro))
    print("\nPer-class F1:")
    for i, label in enumerate(ACTIVITY_LABELS):
        print("  {:<16s} {:.4f}".format(label, f1_per_class[i]))
    print("\nConfusion Matrix:")
    print_confusion_matrix(labels, preds)


def load_data(args, device):
    stats_path = os.path.join(os.path.dirname(args.checkpoint), "normalization_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            norm_stats = json.load(f)
    else:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        norm_stats = checkpoint.get("normalization_stats", None)
        if norm_stats is None:
            print("WARNING: No normalization stats found. Computing from training set...")
            means, stds = UCIHARDataset.get_normalization_stats(args.data_dir)
            norm_stats = {"means": means, "stds": stds}

    test_ds = UCIHARDataset(args.data_dir, split="test")
    mean_t = torch.tensor(norm_stats["means"], dtype=torch.float32)
    std_t = torch.tensor(norm_stats["stds"], dtype=torch.float32)
    test_ds.X = (test_ds.X - mean_t) / std_t
    return test_ds, norm_stats


def load_train_data(args, norm_stats):
    train_ds = UCIHARDataset(args.data_dir, split="train")
    mean_t = torch.tensor(norm_stats["means"], dtype=torch.float32)
    std_t = torch.tensor(norm_stats["stds"], dtype=torch.float32)
    train_ds.X = (train_ds.X - mean_t) / std_t
    return train_ds


def train_variant(model, train_loader, test_loader, epochs, lr, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        acc, f1_macro, _, _, _ = evaluate(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0:
            print("    Epoch {}/{}: acc={:.2f}% f1={:.4f}".format(epoch, epochs, acc * 100, f1_macro))

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


class NoReservoirModel(nn.Module):

    def __init__(self):
        super().__init__()
        from model.dsconv import DSConvEncoder
        from model.attention import PatchMicroAttention
        from model.binary_head import BinaryClassifier
        self.input_proj = nn.Linear(6, 64)
        self.dsconv = DSConvEncoder(in_channels=64)
        self.attention = PatchMicroAttention(in_channels=128, seq_len=32)
        self.classifier = BinaryClassifier(in_features=64, num_classes=6)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.dsconv(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NoAttentionModel(nn.Module):

    def __init__(self):
        super().__init__()
        from model.reservoir import EchoStateNetwork
        from model.dsconv import DSConvEncoder
        from model.binary_head import BinaryClassifier
        self.reservoir = EchoStateNetwork(6, 64)
        self.dsconv = DSConvEncoder(in_channels=64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = BinaryClassifier(in_features=128, num_classes=6)

    def forward(self, x):
        x = self.reservoir(x)
        x = x.transpose(1, 2)
        x = self.dsconv(x)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NoBinaryHeadModel(nn.Module):

    def __init__(self):
        super().__init__()
        from model.reservoir import EchoStateNetwork
        from model.dsconv import DSConvEncoder
        from model.attention import PatchMicroAttention
        self.reservoir = EchoStateNetwork(6, 64)
        self.dsconv = DSConvEncoder(in_channels=64)
        self.attention = PatchMicroAttention(in_channels=128, seq_len=32)
        self.bn = nn.BatchNorm1d(64)
        self.head = nn.Linear(64, 6)

    def forward(self, x):
        x = self.reservoir(x)
        x = x.transpose(1, 2)
        x = self.dsconv(x)
        x = self.attention(x)
        x = self.head(self.bn(x))
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NoDSConvModel(nn.Module):

    def __init__(self):
        super().__init__()
        from model.reservoir import EchoStateNetwork
        from model.attention import PatchMicroAttention
        from model.binary_head import BinaryClassifier
        self.reservoir = EchoStateNetwork(6, 64)
        self.conv = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.attention = PatchMicroAttention(in_channels=128, seq_len=32)
        self.classifier = BinaryClassifier(in_features=64, num_classes=6)

    def forward(self, x):
        x = self.reservoir(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def run_ablation(args, device, test_ds, norm_stats):
    train_ds = load_train_data(args, norm_stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    variants = [
        ("Full Model", SensorFusionHAR(input_channels=6, reservoir_size=64, num_classes=6)),
        ("No Reservoir", NoReservoirModel()),
        ("No Attention", NoAttentionModel()),
        ("No Binary Head", NoBinaryHeadModel()),
        ("No DS-Conv", NoDSConvModel()),
    ]

    results = []

    for name, model in variants:
        print("\n--- {} ---".format(name))
        model = model.to(device)
        params = model.count_parameters()
        print("  Parameters: {:,}".format(params))

        model = train_variant(model, train_loader, test_loader, args.ablation_epochs, args.lr, device)
        acc, f1_macro, f1_per_class, _, _ = evaluate(model, test_loader, device)
        results.append((name, params, acc, f1_macro))
        print("  Result: acc={:.2f}% f1={:.4f}".format(acc * 100, f1_macro))

    print("\n" + "=" * 70)
    print("Ablation Study Results")
    print("=" * 70)
    print("{:<20s}{:>12s}{:>12s}{:>12s}".format("Variant", "Params", "Accuracy", "F1 Macro"))
    print("-" * 56)
    for name, params, acc, f1_macro in results:
        print("{:<20s}{:>12,}{:>11.2f}%{:>12.4f}".format(name, params, acc * 100, f1_macro))
    print("-" * 56)


def run_benchmark(model, device, test_ds, args):
    model.eval()
    sample = test_ds.X[0:1].to(device)

    for _ in range(50):
        with torch.no_grad():
            model(sample)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    for _ in range(args.benchmark_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)

    param_count = model.count_parameters()
    model_kb = model.model_size_kb()

    reservoir_flops = 128 * (6 * 64 + 64 * 64)
    dsconv_block1 = 128 * (64 * 5 + 64 * 128)
    dsconv_block2 = 64 * (128 * 5 + 128 * 128)
    dsconv_block3 = 32 * (128 * 3 + 128 * 128)
    dsconv_flops = dsconv_block1 + dsconv_block2 + dsconv_block3
    patch_dim = 128 * 4
    num_patches = 8
    proj_flops = num_patches * patch_dim * 64
    attn_flops = num_patches * num_patches * 64 * 3
    ffn_flops = num_patches * (64 * 128 + 128 * 64)
    attention_flops = proj_flops + attn_flops + ffn_flops
    classifier_flops = 64 * 6
    total_flops = reservoir_flops + dsconv_flops + attention_flops + classifier_flops

    print("\n" + "=" * 50)
    print("Benchmark Results ({} runs)".format(args.benchmark_runs))
    print("=" * 50)
    print("Device: {}".format(device))
    print("Model size: {:.2f} KB ({:,} params)".format(model_kb, param_count))
    print("FLOPs estimate: {:,}".format(total_flops))
    print("")
    print("Latency (single sample):")
    print("  Mean:  {:.3f} ms".format(latencies.mean()))
    print("  Std:   {:.3f} ms".format(latencies.std()))
    print("  P95:   {:.3f} ms".format(np.percentile(latencies, 95)))
    print("  P99:   {:.3f} ms".format(np.percentile(latencies, 99)))
    print("  Min:   {:.3f} ms".format(latencies.min()))
    print("  Max:   {:.3f} ms".format(latencies.max()))
    print("=" * 50)


def main():
    args = parse_args()
    device = get_device(args.device)

    if not os.path.isdir(args.data_dir):
        print("ERROR: Dataset not found at {}".format(args.data_dir))
        print("Run train.py first to download the dataset.")
        return

    if not os.path.isfile(args.checkpoint) and not args.ablation:
        print("ERROR: Checkpoint not found at {}".format(args.checkpoint))
        print("Run train.py first to train the model.")
        return

    test_ds, norm_stats = load_data(args, device)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if args.ablation:
        run_ablation(args, device, test_ds, norm_stats)
        return

    print("Loading model from {}".format(args.checkpoint))
    model = SensorFusionHAR(input_channels=6, reservoir_size=64, num_classes=6).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Model: SensorFusionHAR")
    print("Parameters: {:,}".format(model.count_parameters()))
    print("Model size: {:.2f} KB".format(model.model_size_kb()))
    print("Checkpoint epoch: {}".format(checkpoint.get("epoch", "N/A")))
    print("")

    acc, f1_macro, f1_per_class, preds, labels = evaluate(model, test_loader, device)
    print_results(acc, f1_macro, f1_per_class, preds, labels)

    if args.benchmark:
        run_benchmark(model, device, test_ds, args)


if __name__ == "__main__":
    main()
