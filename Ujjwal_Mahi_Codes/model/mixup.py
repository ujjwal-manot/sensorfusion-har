import numpy as np
import torch


def reservoir_manifold_mixup(model, x1, x2, y1, y2, criterion, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    h1 = model.reservoir_states(x1)
    h2 = model.reservoir_states(x2)
    h_mixed = lam * h1 + (1.0 - lam) * h2
    output = model.forward_from_reservoir(h_mixed)
    return lam * criterion(output, y1) + (1.0 - lam) * criterion(output, y2)
