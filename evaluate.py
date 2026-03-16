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
from model.dataset_pamap2 import PAMAP2Dataset, ACTIVITY_NAMES as PAMAP2_LABELS
from model.dsconv import DSConvEncoder


UCIHAR_LABELS = ["WALKING", "WALKING_UP", "WALKING_DOWN", "SITTING", "STANDING", "LAYING"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ucihar", choices=["ucihar", "pamap2"])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--ablation_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark_runs", type=int, default=1000)
    return parser.parse_args()


def get_dataset_config(args):
    if args.dataset == "pamap2":
        data_dir = args.data_dir or "data/PAMAP2_Dataset"
        return data_dir, PAMAP2Dataset, PAMAP2_LABELS, 12
    data_dir = args.data_dir or "data/UCI HAR Dataset"
    return data_dir, UCIHARDataset, UCIHAR_LABELS, 6


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


def print_confusion_matrix(y_true, y_pred, label_names):
    cm = confusion_matrix(y_true, y_pred)
    trunc = [l[:10] for l in label_names]
    header = "          " + "".join("{:>12s}".format(t) for t in trunc)
    print(header)
    for i in range(cm.shape[0]):
        row = "{:<10s}".format(trunc[i])
        row += "".join("{:>12d}".format(cm[i, j]) for j in range(cm.shape[1]))
        print(row)


def print_results(acc, f1_macro, f1_per_class, preds, labels, label_names):
    print("Accuracy: {:.2f}%".format(acc * 100))
    print("F1 Macro: {:.4f}".format(f1_macro))
    print("\nPer-class F1:")
    for i, label in enumerate(label_names):
        print("  {:<20s} {:.4f}".format(label, f1_per_class[i]))
    print("\nConfusion Matrix:")
    print_confusion_matrix(labels, preds, label_names)


def load_data(args, device, DatasetCls, data_dir):
    stats_path = os.path.join(os.path.dirname(args.checkpoint), "normalization_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            norm_stats = json.load(f)
    else:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        norm_stats = checkpoint.get("normalization_stats", None)
        if norm_stats is None:
            print("WARNING: No normalization stats found. Computing from training set...")
            means, stds = DatasetCls.get_normalization_stats(data_dir)
            norm_stats = {"means": means, "stds": stds}

    test_ds = DatasetCls(data_dir, split="test")
    mean_t = torch.tensor(norm_stats["means"], dtype=torch.float32)
    std_t = torch.tensor(norm_stats["stds"], dtype=torch.float32)
    test_ds.X = (test_ds.X - mean_t) / std_t
    return test_ds, norm_stats


def load_train_data(DatasetCls, data_dir, norm_stats):
    train_ds = DatasetCls(data_dir, split="train")
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


class _AblationBase(nn.Module):

    def _make_reservoir(self):
        from model.reservoir import EchoStateNetwork
        self.reservoir = EchoStateNetwork(
            6, 32, learnable_sr=True, use_diff_states=True, reservoir_dropout=0.1
        )
        self.diff_gate = nn.Parameter(torch.zeros(32))

    def _merge_reservoir_states(self, h):
        rs = self.reservoir.reservoir_size
        alpha = torch.sigmoid(self.diff_gate)
        return h[:, :, :rs] + alpha * h[:, :, rs:]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NoReservoirModel(_AblationBase):

    def __init__(self, num_classes=6):
        super().__init__()
        from model.dsconv import DSConvEncoder
        from model.attention import PatchMicroAttention
        from model.binary_head import BinaryClassifier
        from model.sensorfusion import SpectralGatedFusion
        self.input_proj = nn.Linear(6, 32)
        self.dsconv = DSConvEncoder(in_channels=32)
        self.gate = SpectralGatedFusion(reservoir_dim=32, dsconv_channels=48, seq_len=32)
        self.attention = PatchMicroAttention(in_channels=48, seq_len=32, d_model=32, ff_dim=48)
        self.classifier = BinaryClassifier(in_features=32, num_classes=num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        dsconv_out = self.dsconv(x)
        x = self.gate(x, dsconv_out)
        x = self.attention(x)
        x = self.classifier(x)
        return x


class NoAttentionModel(_AblationBase):

    def __init__(self, num_classes=6):
        super().__init__()
        from model.dsconv import DSConvEncoder
        from model.binary_head import BinaryClassifier
        from model.sensorfusion import SpectralGatedFusion
        self._make_reservoir()
        self.dsconv = DSConvEncoder(in_channels=32)
        self.gate = SpectralGatedFusion(reservoir_dim=32, dsconv_channels=48, seq_len=32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = BinaryClassifier(in_features=48, num_classes=num_classes)

    def forward(self, x):
        h = self.reservoir(x)
        h = self._merge_reservoir_states(h)
        h = h.transpose(1, 2)
        dsconv_out = self.dsconv(h)
        x = self.gate(h, dsconv_out)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return x


class NoBinaryHeadModel(_AblationBase):

    def __init__(self, num_classes=6):
        super().__init__()
        from model.dsconv import DSConvEncoder
        from model.attention import PatchMicroAttention
        from model.sensorfusion import SpectralGatedFusion
        self._make_reservoir()
        self.dsconv = DSConvEncoder(in_channels=32)
        self.gate = SpectralGatedFusion(reservoir_dim=32, dsconv_channels=48, seq_len=32)
        self.attention = PatchMicroAttention(in_channels=48, seq_len=32, d_model=32, ff_dim=48)
        self.bn = nn.BatchNorm1d(32)
        self.head = nn.Linear(32, num_classes)

    def forward(self, x):
        h = self.reservoir(x)
        h = self._merge_reservoir_states(h)
        h = h.transpose(1, 2)
        dsconv_out = self.dsconv(h)
        x = self.gate(h, dsconv_out)
        x = self.attention(x)
        x = self.head(self.bn(x))
        return x


class NoDSConvModel(_AblationBase):

    def __init__(self, num_classes=6):
        super().__init__()
        from model.attention import PatchMicroAttention
        from model.binary_head import BinaryClassifier
        from model.sensorfusion import SpectralGatedFusion
        self._make_reservoir()
        self.conv = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Conv1d(48, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Conv1d(48, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
        )
        self.gate = SpectralGatedFusion(reservoir_dim=32, dsconv_channels=48, seq_len=32)
        self.attention = PatchMicroAttention(in_channels=48, seq_len=32, d_model=32, ff_dim=48)
        self.classifier = BinaryClassifier(in_features=32, num_classes=num_classes)

    def forward(self, x):
        h = self.reservoir(x)
        h = self._merge_reservoir_states(h)
        h = h.transpose(1, 2)
        conv_out = self.conv(h)
        x = self.gate(h, conv_out)
        x = self.attention(x)
        x = self.classifier(x)
        return x


class NoGateModel(_AblationBase):

    def __init__(self, num_classes=6):
        super().__init__()
        from model.dsconv import DSConvEncoder
        from model.attention import PatchMicroAttention
        from model.binary_head import BinaryClassifier
        self._make_reservoir()
        self.dsconv = DSConvEncoder(in_channels=32)
        self.attention = PatchMicroAttention(in_channels=48, seq_len=32, d_model=32, ff_dim=48)
        self.classifier = BinaryClassifier(in_features=32, num_classes=num_classes)

    def forward(self, x):
        h = self.reservoir(x)
        h = self._merge_reservoir_states(h)
        h = h.transpose(1, 2)
        x = self.dsconv(h)
        x = self.attention(x)
        x = self.classifier(x)
        return x


def run_ablation(args, device, test_ds, norm_stats, DatasetCls, data_dir, num_classes):
    train_ds = load_train_data(DatasetCls, data_dir, norm_stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    variants = [
        ("Full Model", SensorFusionHAR(input_channels=6, reservoir_size=32, num_classes=num_classes)),
        ("No Reservoir", NoReservoirModel(num_classes)),
        ("No Attention", NoAttentionModel(num_classes)),
        ("No Binary Head", NoBinaryHeadModel(num_classes)),
        ("No DS-Conv", NoDSConvModel(num_classes)),
        ("No Gate", NoGateModel(num_classes)),
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

    reservoir_flops = 128 * (6 * 32 + 32 * 32)
    dsconv_block1 = 128 * (32 * 5 + 32 * 48)
    dsconv_block2 = 64 * (48 * 5 + 48 * 48)
    dsconv_block3 = 32 * (48 * 3 + 48 * 48)
    dsconv_flops = dsconv_block1 + dsconv_block2 + dsconv_block3
    patch_dim = 48 * 4
    num_patches = 8
    proj_flops = num_patches * patch_dim * 32
    attn_flops = num_patches * num_patches * 32 * 3
    ffn_flops = num_patches * (32 * 48 + 48 * 32)
    attention_flops = proj_flops + attn_flops + ffn_flops
    num_classes = model(sample).shape[1]
    classifier_flops = 32 * num_classes
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

    data_dir, DatasetCls, activity_labels, num_classes = get_dataset_config(args)

    if not os.path.isdir(data_dir):
        print("ERROR: Dataset not found at {}".format(data_dir))
        print("Run train.py first to download the dataset.")
        return

    if not os.path.isfile(args.checkpoint) and not args.ablation:
        print("ERROR: Checkpoint not found at {}".format(args.checkpoint))
        print("Run train.py first to train the model.")
        return

    test_ds, norm_stats = load_data(args, device, DatasetCls, data_dir)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if args.ablation:
        run_ablation(args, device, test_ds, norm_stats, DatasetCls, data_dir, num_classes)
        return

    print("Loading model from {}".format(args.checkpoint))
    model = SensorFusionHAR(input_channels=6, reservoir_size=32, num_classes=num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    print("Model: SensorFusionHAR ({} classes)".format(num_classes))
    print("Parameters: {:,}".format(model.count_parameters()))
    print("Model size: {:.2f} KB".format(model.model_size_kb()))
    print("Checkpoint epoch: {}".format(checkpoint.get("epoch", "N/A")))
    print("")

    acc, f1_macro, f1_per_class, preds, labels = evaluate(model, test_loader, device)
    print_results(acc, f1_macro, f1_per_class, preds, labels, activity_labels)

    if args.benchmark:
        run_benchmark(model, device, test_ds, args)


if __name__ == "__main__":
    main()
