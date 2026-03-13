import torch
import torch.nn as nn
import torch.ao.quantization

from .reservoir import EchoStateNetwork
from .dsconv import DSConvEncoder
from .attention import PatchMicroAttention
from .binary_head import BinaryClassifier

__all__ = ["SensorFusionHAR"]


class GatedResidualFusion(nn.Module):

    def __init__(self, reservoir_dim, dsconv_channels, seq_len):
        super().__init__()
        self.channel_proj = nn.Conv1d(reservoir_dim, dsconv_channels, kernel_size=1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(seq_len)
        self.gate_proj = nn.Linear(dsconv_channels, 1)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, reservoir_out, dsconv_out):
        res_projected = self.channel_proj(reservoir_out)
        res_aligned = self.temporal_pool(res_projected)
        gate_input = dsconv_out.mean(dim=2)
        gate = torch.sigmoid(self.gate_proj(gate_input)).unsqueeze(2)
        return dsconv_out + gate * res_aligned


class SensorFusionHAR(nn.Module):

    def __init__(self, input_channels=6, reservoir_size=32, num_classes=6):
        super().__init__()
        self.reservoir = EchoStateNetwork(input_channels, reservoir_size)
        self.dsconv = DSConvEncoder(in_channels=reservoir_size)
        self.gate = GatedResidualFusion(reservoir_dim=reservoir_size, dsconv_channels=48, seq_len=32)
        self.attention = PatchMicroAttention(in_channels=48, seq_len=32, d_model=32, ff_dim=48)
        self.classifier = BinaryClassifier(in_features=32, num_classes=num_classes)

    def forward(self, x):
        x = self.reservoir(x)
        x = x.transpose(1, 2)
        dsconv_out = self.dsconv(x)
        x = self.gate(x, dsconv_out)
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
        import copy
        model_copy = copy.deepcopy(self)
        saved_classifier = model_copy.classifier
        model_copy.classifier = nn.Identity()
        quantized = torch.ao.quantization.quantize_dynamic(
            model_copy, {nn.Linear}, dtype=torch.qint8
        )
        quantized.classifier = saved_classifier
        return quantized
