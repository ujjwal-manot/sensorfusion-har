import numpy as np
import torch
import zipfile

import train_esp32_v2_expanded_local as v2


def test_v2_uci_signal_files_prefer_total_acceleration():
    assert v2.UCI_TOTAL_SIGNAL_FILES[:3] == [
        "total_acc_x_{}.txt",
        "total_acc_y_{}.txt",
        "total_acc_z_{}.txt",
    ]


def test_v2_mhealth_windows_from_subject_array_maps_to_six_axis_windows():
    raw = np.zeros((160, 24), dtype=np.float32)
    raw[:, 0:3] = np.random.randn(160, 3)
    raw[:, 14:17] = np.random.randn(160, 3)
    raw[:, 17:20] = np.random.randn(160, 3)
    raw[:, -1] = 4

    X, y = v2.mhealth_windows_from_subject_array(
        raw,
        mapping={4: 0},
        target_len=50,
        window_size=100,
        step_size=50,
    )

    assert X.shape == (2, 50, 6)
    assert y.tolist() == [0, 0]
    assert np.isfinite(X).all()


def test_orientation_micro_path_shape():
    path = v2.OrientationMicroPath(input_channels=6, d_model=64)
    out = path(torch.randn(4, 50, 6))
    assert out.shape == (4, 64)


def test_multiscale_temporal_edge_shape():
    layer = v2.MultiScaleTemporalEdge(channels=64)
    out = layer(torch.randn(2, 64, 25))
    assert out.shape == (2, 64, 25)


def test_v2_model_aux_contains_features_for_prototype_loss():
    model = v2.SensorFusionESP32(input_channels=6, reservoir_size=64, num_classes=8)
    logits, aux = model(torch.randn(3, 50, 6), return_aux=True)
    assert logits.shape == (3, 8)
    assert "features" in aux
    assert aux["features"].shape == (3, 64)


def test_realworld_zip_pair_to_windows_uses_waist_csv(tmp_path):
    acc_zip = tmp_path / "acc_jumping_csv.zip"
    gyr_zip = tmp_path / "gyr_jumping_csv.zip"
    header = "id,attr_time,attr_x,attr_y,attr_z\n"
    acc_rows = [f"{i},{1000 + i * 20},{9.0 + i * 0.01},{0.1},{1.0}\n" for i in range(120)]
    gyr_rows = [f"{i},{1000 + i * 20},{0.01},{0.02 + i * 0.001},{0.03}\n" for i in range(120)]
    with zipfile.ZipFile(acc_zip, "w") as zf:
        zf.writestr("acc_jumping_waist.csv", header + "".join(acc_rows))
    with zipfile.ZipFile(gyr_zip, "w") as zf:
        zf.writestr("Gyroscope_jumping_waist.csv", header + "".join(gyr_rows))

    X, y = v2.realworld_zip_pair_to_windows(
        str(acc_zip),
        str(gyr_zip),
        label=7,
        target_len=50,
        window_size=100,
        step_size=50,
    )

    assert X.shape == (1, 50, 6)
    assert y.tolist() == [7]
    assert np.isfinite(X).all()
