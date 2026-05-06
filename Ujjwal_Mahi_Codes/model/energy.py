import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import OrderedDict


ENERGY_COSTS = {
    "mul_fp32": 3.7e-12,
    "add_fp32": 0.9e-12,
    "mul_int8": 0.2e-12,
    "add_int8": 0.03e-12,
}


def count_macs(model, input_shape=(1, 128, 6)):
    mac_counts = OrderedDict()
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                batch_dims = inp[0].shape[:-1]
                batch_elements = 1
                for d in batch_dims:
                    batch_elements *= d
                macs = in_features * out_features * batch_elements
                mac_counts[name] = macs

            elif isinstance(module, nn.Conv1d):
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = module.kernel_size[0]
                groups = module.groups
                out_length = out.shape[2]
                batch_size = out.shape[0]

                if groups == in_channels and groups == out_channels:
                    macs = out_channels * out_length * kernel_size * batch_size
                elif groups == 1:
                    macs = out_channels * out_length * in_channels * kernel_size * batch_size
                else:
                    channels_per_group = in_channels // groups
                    out_per_group = out_channels // groups
                    macs = groups * out_per_group * out_length * channels_per_group * kernel_size * batch_size
                mac_counts[name] = macs

            elif isinstance(module, nn.MultiheadAttention):
                embed_dim = module.embed_dim
                seq_len = inp[0].shape[1] if module.batch_first else inp[0].shape[0]
                batch_size = inp[0].shape[0] if module.batch_first else inp[0].shape[1]

                qkv_macs = 4 * seq_len * embed_dim * embed_dim * batch_size
                attn_macs = 2 * seq_len * seq_len * embed_dim * batch_size
                macs = qkv_macs + attn_macs
                mac_counts[name] = macs

        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.MultiheadAttention)):
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device("cpu")
    dummy = torch.randn(*input_shape).to(device)

    model.eval()
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    total_macs = sum(mac_counts.values())
    mac_counts["total"] = total_macs

    return mac_counts


def estimate_energy(mac_counts, precision="fp32"):
    mul_key = f"mul_{precision}"
    add_key = f"add_{precision}"

    if mul_key not in ENERGY_COSTS or add_key not in ENERGY_COSTS:
        raise ValueError(f"Unknown precision: {precision}. Supported: fp32, int8")

    mul_cost = ENERGY_COSTS[mul_key]
    add_cost = ENERGY_COSTS[add_key]

    energy_per_layer = OrderedDict()
    total_joules = 0.0

    for name, macs in mac_counts.items():
        if name == "total":
            continue
        layer_energy = macs * (mul_cost + add_cost)
        energy_per_layer[name] = layer_energy
        total_joules += layer_energy

    energy_per_layer["total_joules"] = total_joules
    energy_per_layer["total_millijoules"] = total_joules * 1000

    return energy_per_layer


def compare_models_energy(models_dict, input_shape=(1, 128, 6)):
    comparison = OrderedDict()

    for name, model in models_dict.items():
        macs = count_macs(model, input_shape)
        energy_fp32 = estimate_energy(macs, precision="fp32")
        energy_int8 = estimate_energy(macs, precision="int8")

        param_count = sum(p.numel() for p in model.parameters())

        comparison[name] = {
            "total_macs": macs["total"],
            "total_params": param_count,
            "energy_fp32_mj": energy_fp32["total_millijoules"],
            "energy_int8_mj": energy_int8["total_millijoules"],
            "macs_per_layer": {k: v for k, v in macs.items() if k != "total"},
            "energy_per_layer_fp32": {k: v for k, v in energy_fp32.items()
                                      if k not in ("total_joules", "total_millijoules")},
        }

    return comparison


def plot_energy_comparison(comparison, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    model_names = list(comparison.keys())
    macs = [comparison[n]["total_macs"] for n in model_names]
    energy_fp32 = [comparison[n]["energy_fp32_mj"] for n in model_names]
    energy_int8 = [comparison[n]["energy_int8_mj"] for n in model_names]

    ax1 = axes[0]
    y_pos = np.arange(len(model_names))
    bar_height = 0.35

    bars_fp32 = ax1.barh(y_pos - bar_height / 2, energy_fp32, bar_height,
                          color="#4CAF50", alpha=0.8, label="FP32")
    bars_int8 = ax1.barh(y_pos + bar_height / 2, energy_int8, bar_height,
                          color="#2196F3", alpha=0.8, label="INT8")

    for bar, val in zip(bars_fp32, energy_fp32):
        ax1.text(bar.get_width() + max(energy_fp32) * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=8)
    for bar, val in zip(bars_int8, energy_int8):
        ax1.text(bar.get_width() + max(energy_fp32) * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=8)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(model_names, fontsize=10)
    ax1.set_xlabel("Energy per Inference (mJ)", fontsize=11)
    ax1.set_title("Energy Comparison", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="x")

    ax2 = axes[1]
    bars_macs = ax2.barh(y_pos, [m / 1e6 for m in macs], 0.5,
                          color="#FF9800", alpha=0.8)

    for bar, val in zip(bars_macs, macs):
        ax2.text(bar.get_width() + max(macs) / 1e6 * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val / 1e6:.2f}M", va="center", fontsize=8)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(model_names, fontsize=10)
    ax2.set_xlabel("MACs (Millions)", fontsize=11)
    ax2.set_title("Computation Cost", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
