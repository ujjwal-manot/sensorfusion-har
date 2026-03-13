import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def fgsm_attack(model, x, y, epsilon, device):
    x_adv = x.clone().detach().to(device).requires_grad_(True)
    y = y.to(device)

    model.eval()
    output = model(x_adv)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()

    grad_sign = x_adv.grad.data.sign()
    x_adv = x_adv.detach() + epsilon * grad_sign
    return x_adv.detach()


def pgd_attack(model, x, y, epsilon, alpha, num_steps, device):
    x = x.to(device)
    y = y.to(device)

    delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = (x + delta).detach()

    model.eval()
    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()

        grad_sign = x_adv.grad.data.sign()
        x_adv = x_adv.detach() + alpha * grad_sign

        perturbation = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = (x + perturbation).detach()

    return x_adv


def evaluate_adversarial_robustness(model, dataset, device, epsilons=None, attack="both"):
    if epsilons is None:
        epsilons = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    all_x, all_y = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        all_x.append(x)
        all_y.append(y.item() if hasattr(y, "item") else y)

    X = torch.stack(all_x)
    Y = torch.tensor(all_y, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        batch_size = 256
        clean_preds = []
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size].to(device)
            out = model(batch)
            clean_preds.append(out.argmax(dim=1).cpu())
        clean_preds = torch.cat(clean_preds)
    clean_accuracy = (clean_preds == Y).float().mean().item()

    results = {
        "epsilons": epsilons,
        "clean_accuracy": clean_accuracy,
        "fgsm_accuracy": [],
        "pgd_accuracy": [],
    }

    batch_size = 128
    for eps in epsilons:
        if eps == 0:
            if attack in ("fgsm", "both"):
                results["fgsm_accuracy"].append(clean_accuracy)
            if attack in ("pgd", "both"):
                results["pgd_accuracy"].append(clean_accuracy)
            continue

        if attack in ("fgsm", "both"):
            fgsm_correct = 0
            fgsm_total = 0
            for start in range(0, len(X), batch_size):
                x_batch = X[start:start + batch_size]
                y_batch = Y[start:start + batch_size]
                x_adv = fgsm_attack(model, x_batch, y_batch, eps, device)
                with torch.no_grad():
                    preds = model(x_adv).argmax(dim=1).cpu()
                fgsm_correct += (preds == y_batch).sum().item()
                fgsm_total += len(y_batch)
            results["fgsm_accuracy"].append(fgsm_correct / fgsm_total)

        if attack in ("pgd", "both"):
            pgd_alpha = eps / 4.0
            pgd_steps = 10
            pgd_correct = 0
            pgd_total = 0
            for start in range(0, len(X), batch_size):
                x_batch = X[start:start + batch_size]
                y_batch = Y[start:start + batch_size]
                x_adv = pgd_attack(model, x_batch, y_batch, eps, pgd_alpha, pgd_steps, device)
                with torch.no_grad():
                    preds = model(x_adv).argmax(dim=1).cpu()
                pgd_correct += (preds == y_batch).sum().item()
                pgd_total += len(y_batch)
            results["pgd_accuracy"].append(pgd_correct / pgd_total)

    return results


def plot_adversarial_robustness(results, save_path=None):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    epsilons = results["epsilons"]

    if results.get("fgsm_accuracy"):
        ax.plot(epsilons, results["fgsm_accuracy"], "o-", color="#4CAF50",
                label="FGSM", linewidth=2, markersize=6)

    if results.get("pgd_accuracy"):
        ax.plot(epsilons, results["pgd_accuracy"], "s-", color="#F44336",
                label="PGD", linewidth=2, markersize=6)

    ax.axhline(y=results["clean_accuracy"], color="#2196F3", linestyle="--",
               linewidth=1.5, label="Clean Accuracy", alpha=0.7)

    ax.set_xlabel("Perturbation (epsilon)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Adversarial Robustness", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
