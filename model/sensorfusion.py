import torch
import torch.nn as nn
import torch.ao.quantization

from .reservoir import EchoStateNetwork
from .dsconv import DSConvEncoder
from .attention import PatchMicroAttention
from .binary_head import BinaryClassifier

__all__ = ["SensorFusionHAR"]


class SensorFusionHAR(nn.Module):

    def __init__(self, input_channels=6, reservoir_size=32, num_classes=6):
        super().__init__()
        self.reservoir = EchoStateNetwork(input_channels, reservoir_size)
        self.dsconv = DSConvEncoder(in_channels=reservoir_size)
        self.attention = PatchMicroAttention(in_channels=48, seq_len=32, d_model=32, ff_dim=48)
        self.classifier = BinaryClassifier(in_features=32, num_classes=num_classes)

    def forward(self, x):
        x = self.reservoir(x)
        x = x.transpose(1, 2)
        x = self.dsconv(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_kb(self):
        return self.count_parameters() * 4 / 1024

    def quantized_size_kb(self):
        return self.count_parameters() * 1 / 1024

    def quantize(self):
        saved_classifier = self.classifier
        self.classifier = nn.Identity()
        quantized = torch.ao.quantization.quantize_dynamic(
            self, {nn.Linear}, dtype=torch.qint8
        )
        quantized.classifier = saved_classifier
        self.classifier = saved_classifier
        return quantized
