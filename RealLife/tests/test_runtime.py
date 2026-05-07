import importlib.util
from pathlib import Path

import numpy as np

SERVER_PATH = Path(__file__).resolve().parents[1] / "server.py"
SERVER_SPEC = importlib.util.spec_from_file_location("reallife_server", SERVER_PATH)
server = importlib.util.module_from_spec(SERVER_SPEC)
SERVER_SPEC.loader.exec_module(server)


def _stationary_window(samples=50, hz=50):
    step_ms = 1000.0 / hz
    return [
        {
            "ax": 0.0,
            "ay": 0.0,
            "az": 9.81,
            "gx": 0.0,
            "gy": 0.0,
            "gz": 0.0,
            "t": i * step_ms,
            "source": "pytest",
            "accelMode": "accelerationIncludingGravity",
            "gyroMode": "synthetic",
        }
        for i in range(samples)
    ]


def test_checkpoint_selection_prefers_11_class_pocket_model():
    assert server.CHECKPOINT_PATH.name == "best_sensorfusion_esp32_v2_pocket_v3.pt"
    assert server._checkpoint_labels(server.CHECKPOINT_PATH) == server.EXPECTED_LABELS


def test_model_loads_with_checkpoint_normalization():
    server.load_model()
    assert server.MODEL_INFO["model_loaded"] is True
    assert server.MODEL_INFO["architecture"] == "SensorFusionESP32"
    assert server.MODEL_INFO["normalization_source"] == "checkpoint"
    assert server.WINDOW_SIZE == 50
    assert server.STRIDE == 25
    assert len(server.CHANNEL_STATS) == 6


def test_validate_phone_sample_rejects_invalid_values():
    sample, error = server.validate_phone_sample({"ax": 1.0})
    assert sample is None
    assert "missing keys" in error

    invalid = {"ax": np.nan, "ay": 0, "az": 9.81, "gx": 0, "gy": 0, "gz": 0, "t": 0}
    sample, error = server.validate_phone_sample(invalid)
    assert sample is None
    assert error == "non-finite sensor value"


def test_validate_phone_sample_preserves_sensor_metadata():
    valid = {
        "ax": "0.0",
        "ay": "0.0",
        "az": "9.81",
        "gx": "0.0",
        "gy": "0.0",
        "gz": "0.0",
        "t": "123.0",
        "source": "DeviceMotion",
        "accelMode": "accelerationIncludingGravity",
        "gyroMode": "rotationRate",
    }
    sample, error = server.validate_phone_sample(valid)
    assert error is None
    assert sample["az"] == 9.81
    assert sample["source"] == "DeviceMotion"
    assert sample["accelMode"] == "accelerationIncludingGravity"


def test_window_diagnostics_identifies_gravity_and_rate():
    diag = server.window_diagnostics(_stationary_window())
    assert diag["gravity_like"] is True
    assert diag["sensor_health"] == "ok"
    assert abs(diag["sample_rate_hz"] - 50.0) < 0.1
    assert abs(diag["acc_mag_mean"] - 9.81) < 0.01


def test_run_inference_returns_stable_runtime_payload():
    server.load_model()
    result = server.run_inference(_stationary_window())
    labels = set(server.model_info_payload()["labels"])
    assert result["prediction"] in labels | {"Uncertain"}
    assert set(result["probabilities"]) == labels
    assert result["model_info"]["model_loaded"] is True
    assert result["mode"] == "torch"
    assert result["diagnostics"]["gravity_like"] is True


def test_stationary_generic_sensor_lying_is_corrected_to_sitting():
    labels = server.EXPECTED_LABELS
    probs = {label: 0.01 for label in labels}
    probs["Lying Down"] = 0.90
    diagnostics = {
        "acc_mag_std": 0.56,
        "gyro_std": 0.394,
    }
    data = [{
        "source": "Generic Sensor API",
        "accelMode": "Accelerometer",
    }]
    label, conf, corrected, correction = server.apply_realtime_corrections(
        "Lying Down", 0.78, probs, diagnostics, data
    )
    assert label == "Sitting"
    assert conf >= 0.62
    assert corrected["Sitting"] == conf
    assert correction == "stationary_pocket_generic_sensor_lying_to_sitting"
