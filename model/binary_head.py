import torch
import torch.nn as nn

__all__ = ["BinaryClassifier"]


class BinaryLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        w = self.linear.weight
        binary_w = w + (torch.sign(w) - w).detach()
        return nn.functional.linear(x, binary_w, self.linear.bias)


class BinaryClassifier(nn.Module):

    def __init__(self, in_features=64, num_classes=6):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.head = BinaryLinear(in_features, num_classes)

    def forward(self, x):
        return self.head(self.bn(x))
