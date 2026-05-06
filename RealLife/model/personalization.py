import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score


def few_shot_personalize(model, support_X, support_y, device, k_shots=10, lr=0.01, steps=50):
    personalized = copy.deepcopy(model)
    personalized = personalized.to(device)

    for param in personalized.parameters():
        param.requires_grad = False

    for param in personalized.classifier.parameters():
        param.requires_grad = True

    unique_classes = support_y.unique()
    selected_indices = []
    for cls in unique_classes:
        cls_mask = (support_y == cls).nonzero(as_tuple=True)[0]
        n_select = min(k_shots, len(cls_mask))
        perm = torch.randperm(len(cls_mask))[:n_select]
        selected_indices.append(cls_mask[perm])

    selected_indices = torch.cat(selected_indices)
    sx = support_X[selected_indices].to(device)
    sy = support_y[selected_indices].to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in personalized.classifier.parameters() if p.requires_grad],
        lr=lr,
    )

    personalized.train()
    for _ in range(steps):
        logits = personalized(sx)
        loss = criterion(logits, sy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    personalized.eval()
    return personalized


def evaluate_personalization(model, dataset_cls, data_dir, device, k_shots_list=None, num_trials=5):
    if k_shots_list is None:
        k_shots_list = [5, 10, 20, 50]

    subjects = dataset_cls.get_subjects(data_dir)

    results = {
        "per_subject": {},
        "per_k": defaultdict(lambda: {"accuracy": [], "f1": []}),
        "overall": {},
    }

    for subject in subjects:
        train_ds, test_ds = dataset_cls.loso_split(data_dir, subject)

        if len(test_ds) == 0:
            continue

        results["per_subject"][subject] = {}

        for k in k_shots_list:
            trial_accs = []
            trial_f1s = []

            for trial in range(num_trials):
                personalized = few_shot_personalize(
                    model,
                    train_ds.X,
                    train_ds.y,
                    device,
                    k_shots=k,
                )

                test_loader = DataLoader(
                    TensorDataset(test_ds.X, test_ds.y),
                    batch_size=64,
                    shuffle=False,
                )

                all_preds = []
                all_labels = []

                with torch.no_grad():
                    for bx, by in test_loader:
                        bx = bx.to(device)
                        preds = personalized(bx).argmax(dim=1).cpu()
                        all_preds.append(preds)
                        all_labels.append(by)

                all_preds = torch.cat(all_preds).numpy()
                all_labels = torch.cat(all_labels).numpy()

                acc = (all_preds == all_labels).mean()
                f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

                trial_accs.append(acc)
                trial_f1s.append(f1)

            mean_acc = float(np.mean(trial_accs))
            std_acc = float(np.std(trial_accs))
            mean_f1 = float(np.mean(trial_f1s))
            std_f1 = float(np.std(trial_f1s))

            results["per_subject"][subject][k] = {
                "accuracy_mean": mean_acc,
                "accuracy_std": std_acc,
                "f1_mean": mean_f1,
                "f1_std": std_f1,
            }

            results["per_k"][k]["accuracy"].append(mean_acc)
            results["per_k"][k]["f1"].append(mean_f1)

            print(f"Subject {subject} | k={k} | Acc: {mean_acc:.4f} +/- {std_acc:.4f} | F1: {mean_f1:.4f} +/- {std_f1:.4f}")

    for k in k_shots_list:
        if results["per_k"][k]["accuracy"]:
            results["overall"][k] = {
                "accuracy_mean": float(np.mean(results["per_k"][k]["accuracy"])),
                "accuracy_std": float(np.std(results["per_k"][k]["accuracy"])),
                "f1_mean": float(np.mean(results["per_k"][k]["f1"])),
                "f1_std": float(np.std(results["per_k"][k]["f1"])),
            }

    results["per_k"] = dict(results["per_k"])
    return results
