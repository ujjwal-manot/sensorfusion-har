import torch
import pytest
from collections import OrderedDict
from model.energy import count_macs, estimate_energy, compare_models_energy, ENERGY_COSTS


class TestCountMacs:

    def test_returns_ordered_dict(self, model):
        macs = count_macs(model)
        assert isinstance(macs, OrderedDict)
        assert "total" in macs

    def test_total_positive(self, model):
        macs = count_macs(model)
        assert macs["total"] > 0

    def test_has_layer_keys(self, model):
        macs = count_macs(model)
        keys = [k for k in macs.keys() if k != "total"]
        assert len(keys) > 0

    def test_custom_input_shape(self, model):
        macs = count_macs(model, input_shape=(1, 128, 6))
        assert macs["total"] > 0

    def test_hooks_removed_after_counting(self, model):
        """After count_macs, model should have no leftover hooks."""
        count_macs(model)
        # Model should still work normally
        out = model(torch.randn(1, 128, 6))
        assert out.shape == (1, 6)

    def test_12_class_model(self, model_12cls):
        macs = count_macs(model_12cls)
        assert macs["total"] > 0


class TestEstimateEnergy:

    def test_fp32(self, model):
        macs = count_macs(model)
        energy = estimate_energy(macs, precision="fp32")
        assert "total_joules" in energy
        assert "total_millijoules" in energy
        assert energy["total_joules"] > 0

    def test_int8_less_than_fp32(self, model):
        macs = count_macs(model)
        fp32 = estimate_energy(macs, "fp32")
        int8 = estimate_energy(macs, "int8")
        assert int8["total_joules"] < fp32["total_joules"]

    def test_invalid_precision(self, model):
        macs = count_macs(model)
        with pytest.raises(ValueError):
            estimate_energy(macs, "fp16")

    def test_millijoules_conversion(self, model):
        macs = count_macs(model)
        energy = estimate_energy(macs, "fp32")
        assert abs(energy["total_millijoules"] - energy["total_joules"] * 1000) < 1e-15

    def test_energy_per_layer_sums_to_total(self, model):
        macs = count_macs(model)
        energy = estimate_energy(macs, "fp32")
        layer_sum = sum(v for k, v in energy.items() if k not in ("total_joules", "total_millijoules"))
        assert abs(layer_sum - energy["total_joules"]) < 1e-15


class TestCompareModelsEnergy:

    def test_compare(self, model):
        from model.sensorfusion import SensorFusionHAR
        m2 = SensorFusionHAR(6, 32, 6)
        result = compare_models_energy({"full": model, "clone": m2})
        assert "full" in result
        assert "clone" in result
        assert result["full"]["total_macs"] > 0

    def test_compare_returns_all_fields(self, model):
        result = compare_models_energy({"test": model})
        entry = result["test"]
        assert "total_macs" in entry
        assert "total_params" in entry
        assert "energy_fp32_mj" in entry
        assert "energy_int8_mj" in entry
        assert "macs_per_layer" in entry
        assert "energy_per_layer_fp32" in entry

    def test_int8_energy_less(self, model):
        result = compare_models_energy({"test": model})
        assert result["test"]["energy_int8_mj"] < result["test"]["energy_fp32_mj"]

    def test_custom_input_shape(self, model):
        result = compare_models_energy({"test": model}, input_shape=(1, 128, 6))
        assert result["test"]["total_macs"] > 0
