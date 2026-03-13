import os
import json
import time
import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

from model import SensorFusionHAR
from model.dataset import UCIHARDataset
from model.dataset_pamap2 import PAMAP2Dataset, ACTIVITY_NAMES as PAMAP2_LABELS


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


UCIHAR_LABELS = ["WALKING", "WALKING_UP", "WALKING_DOWN", "SITTING", "STANDING", "LAYING"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ucihar", choices=["ucihar", "pamap2"])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    return parser.parse_args()


def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def get_dataset_config(args):
    if args.dataset == "pamap2":
        data_dir = args.data_dir or "data/PAMAP2_Dataset"
        return data_dir, PAMAP2Dataset, PAMAP2_LABELS, 12
    data_dir = args.data_dir or "data/UCI HAR Dataset"
    return data_dir, UCIHARDataset, UCIHAR_LABELS, 6


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item() * X.size(0)
            all_preds.append(out.argmax(dim=1).cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = (all_preds == all_labels).mean()
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    return total_loss / len(all_labels), acc, f1_macro, f1_per_class, all_preds, all_labels


def print_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    trunc = [l[:10] for l in labels]
    header = "          " + "".join("{:>12s}".format(t) for t in trunc)
    print(header)
    for i in range(cm.shape[0]):
        row = "{:<10s}".format(trunc[i])
        row += "".join("{:>12d}".format(cm[i, j]) for j in range(cm.shape[1]))
        print(row)


def main():
    set_seed(42)
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    data_dir, DatasetCls, activity_labels, num_classes = get_dataset_config(args)

    if not os.path.isdir(data_dir):
        print("Dataset not found at {}. Downloading...".format(data_dir))
        parent = os.path.dirname(data_dir) or "."
        extracted = DatasetCls.download(parent)
        if os.path.isdir(extracted):
            data_dir = extracted
        if not os.path.isdir(data_dir):
            print("ERROR: Could not find dataset at {}".format(data_dir))
            return

    print("Loading {} dataset...".format(args.dataset.upper()))
    train_ds = DatasetCls(data_dir, split="train")
    test_ds = DatasetCls(data_dir, split="test")
    print("Train samples: {}  Test samples: {}".format(len(train_ds), len(test_ds)))

    print("Computing normalization stats...")
    means, stds = DatasetCls.get_normalization_stats(data_dir)
    norm_stats = {"means": means, "stds": stds}
    stats_path = os.path.join(args.checkpoint_dir, "normalization_stats.json")
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print("Normalization stats saved to {}".format(stats_path))

    mean_t = torch.tensor(means, dtype=torch.float32)
    std_t = torch.tensor(stds, dtype=torch.float32)
    train_ds.X = (train_ds.X - mean_t) / std_t
    test_ds.X = (test_ds.X - mean_t) / std_t

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = SensorFusionHAR(input_channels=6, reservoir_size=32, num_classes=num_classes).to(device)
    param_count = model.count_parameters()
    model_kb = model.model_size_kb()
    print("\nModel: SensorFusionHAR ({} classes)".format(num_classes))
    print("Parameters: {:,}".format(param_count))
    print("Model size: {:.2f} KB".format(model_kb))
    print("Device: {}\n".format(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_path = os.path.join(args.checkpoint_dir, "best_model.pt")

    print("{:<8s}{:>12s}{:>12s}{:>12s}{:>12s}{:>10s}".format(
        "Epoch", "Train Loss", "Test Loss", "Test Acc", "F1 Macro", "LR"
    ))
    print("-" * 66)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, f1_macro, f1_per_class, preds, labels = eval_epoch(
            model, test_loader, criterion, device
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print("{:<8d}{:>12.4f}{:>12.4f}{:>11.2f}%{:>12.4f}{:>10.6f}".format(
            epoch, train_loss, test_loss, test_acc * 100, f1_macro, lr
        ))

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "accuracy": test_acc,
                "f1_macro": f1_macro,
                "normalization_stats": norm_stats,
                "dataset": args.dataset,
                "num_classes": num_classes,
                "activity_labels": activity_labels,
            }, best_path)

    print("\n" + "=" * 66)
    print("Training complete!")
    print("Best test accuracy: {:.2f}%".format(best_acc * 100))
    print("Best model saved to: {}".format(best_path))

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, final_acc, final_f1, final_f1_per_class, final_preds, final_labels = eval_epoch(
        model, test_loader, criterion, device
    )

    print("\nFinal Evaluation (Best Model):")
    print("Accuracy: {:.2f}%".format(final_acc * 100))
    print("F1 Macro: {:.4f}".format(final_f1))
    print("\nPer-class F1:")
    for i, label in enumerate(activity_labels):
        print("  {:<20s} {:.4f}".format(label, final_f1_per_class[i]))

    print("\nConfusion Matrix:")
    print_confusion_matrix(final_labels, final_preds, activity_labels)
    print("\nModel size: {:.2f} KB ({:,} parameters)".format(model_kb, param_count))


if __name__ == "__main__":
    main()
