import torch
import torch.nn as nn
import torch.ao.quantization

from .reservoir import EchoStateNetwork
from .dsconv import DSConvEncoder
from .attention import PatchMicroAttention
from .binary_head import BinaryClassifier

__all__ = ["SensorFusionHAR", "GatedResidualFusion", "SpectralGatedFusion"]


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


class SpectralGatedFusion(nn.Module):

    def __init__(self, reservoir_dim, dsconv_channels, seq_len):
        super().__init__()
        self.channel_proj = nn.Conv1d(reservoir_dim, dsconv_channels, kernel_size=1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(seq_len)
        freq_bins = seq_len // 2 + 1
        self.freq_gate = nn.Linear(freq_bins, freq_bins)
        nn.init.zeros_(self.freq_gate.bias)

    def forward(self, reservoir_out, dsconv_out):
        res_projected = self.channel_proj(reservoir_out)
        res_aligned = self.temporal_pool(res_projected)
        dsconv_freq = torch.fft.rfft(dsconv_out, dim=2)
        freq_energy = dsconv_freq.abs().mean(dim=1)
        gate = torch.sigmoid(self.freq_gate(freq_energy))
        res_freq = torch.fft.rfft(res_aligned, dim=2)
        gated_freq = res_freq * gate.unsqueeze(1)
        gated_time = torch.fft.irfft(gated_freq, n=dsconv_out.shape[2], dim=2)
        return dsconv_out + gated_time


class SensorFusionHAR(nn.Module):

    def __init__(self, input_channels=6, reservoir_size=32, num_classes=6):
        super().__init__()
        self.reservoir = EchoStateNetwork(
            input_channels, reservoir_size,
            learnable_sr=True, use_diff_states=True, reservoir_dropout=0.1
        )

        if self.reservoir.use_diff_states:
            self.diff_gate = nn.Parameter(torch.zeros(reservoir_size))
        else:
            self.diff_gate = None

        self.dsconv = DSConvEncoder(in_channels=reservoir_size)
        self.gate = SpectralGatedFusion(
            reservoir_dim=reservoir_size, dsconv_channels=48, seq_len=32
        )
        self.attention = PatchMicroAttention(
            in_channels=48, seq_len=32, d_model=32, ff_dim=48
        )
        self.classifier = BinaryClassifier(in_features=32, num_classes=num_classes)

    def _merge_reservoir_states(self, h):
        if self.diff_gate is not None:
            rs = self.reservoir.reservoir_size
            alpha = torch.sigmoid(self.diff_gate)
            return h[:, :, :rs] + alpha * h[:, :, rs:]
        return h

    def forward(self, x, return_aux=False):
        h = self.reservoir(x)
        h = self._merge_reservoir_states(h)
        h = h.transpose(1, 2)
        dsconv_out = self.dsconv(h)
        x = self.gate(h, dsconv_out)

        if return_aux:
            x, attn_weights, attn_entropy = self.attention(x, return_attention=True)
        else:
            x = self.attention(x)

        x = self.classifier(x)

        if return_aux:
            aux = {
                "attention_entropy": attn_entropy,
                "attention_weights": attn_weights,
                "spectral_radius": self.reservoir.effective_spectral_radius,
            }
            return x, aux
        return x

    def forward_from_reservoir(self, h):
        h = self._merge_reservoir_states(h)
        h = h.transpose(1, 2)
        dsconv_out = self.dsconv(h)
        x = self.gate(h, dsconv_out)
        x = self.attention(x)
        return self.classifier(x)

    def reservoir_states(self, x):
        return self.reservoir(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_kb(self):
        return self.count_parameters() * 4 / 1024

    def quantized_size_kb(self):
        return self.count_parameters() * 1 / 1024

    def architecture_summary(self):
        return {
            "learnable_spectral_radius": self.reservoir.sr_logit is not None,
            "effective_sr": float(self.reservoir.effective_spectral_radius.item()) if self.reservoir.sr_logit is not None else float(self.reservoir._base_sr.item()),
            "differential_states": self.reservoir.use_diff_states,
            "reservoir_dropout": self.reservoir.reservoir_dropout,
            "fusion_type": type(self.gate).__name__,
            "scaled_binary_weights": hasattr(self.classifier.head, "scale"),
            "trainable_parameters": self.count_parameters(),
            "model_size_fp32_kb": round(self.model_size_kb(), 2),
            "model_size_int8_kb": round(self.quantized_size_kb(), 2),
        }

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
