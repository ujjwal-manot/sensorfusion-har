import torch
import torch.nn as nn

__all__ = ["BinaryLinear", "BinaryClassifier"]


class BinaryLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.scale = nn.Parameter(torch.ones(out_features))
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        w = self.linear.weight
        binary_w = w + (torch.sign(w) - w).detach()
        scaled_w = binary_w * self.scale.unsqueeze(1)
        return nn.functional.linear(x, scaled_w, self.linear.bias)

    def export_binary(self):
        with torch.no_grad():
            signs = torch.sign(self.linear.weight)
            packed = (signs > 0).to(torch.uint8)
            return {
                "packed_weights": packed,
                "scale": self.scale.data.clone(),
                "bias": self.linear.bias.data.clone(),
                "in_features": self.linear.in_features,
                "out_features": self.linear.out_features,
            }


class BinaryClassifier(nn.Module):

    def __init__(self, in_features=32, num_classes=6):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.head = BinaryLinear(in_features, num_classes)

    def forward(self, x):
        return self.head(self.bn(x))
