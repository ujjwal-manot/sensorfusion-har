import pytest
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def seed_rng():
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def synthetic_batch(device):
    return torch.randn(8, 128, 6, device=device)


@pytest.fixture
def synthetic_labels():
    return torch.randint(0, 6, (8,))


@pytest.fixture
def synthetic_dataset(synthetic_batch, synthetic_labels):
    return torch.utils.data.TensorDataset(synthetic_batch, synthetic_labels)


@pytest.fixture
def model():
    from model.sensorfusion import SensorFusionHAR
    return SensorFusionHAR(input_channels=6, reservoir_size=32, num_classes=6)


@pytest.fixture
def model_12cls():
    from model.sensorfusion import SensorFusionHAR
    return SensorFusionHAR(input_channels=6, reservoir_size=32, num_classes=12)


@pytest.fixture
def reservoir():
    from model.reservoir import EchoStateNetwork
    return EchoStateNetwork(
        input_channels=6, reservoir_size=32,
        learnable_sr=True, use_diff_states=True, reservoir_dropout=0.1,
    )


@pytest.fixture
def reservoir_nodiff():
    from model.reservoir import EchoStateNetwork
    return EchoStateNetwork(
        input_channels=6, reservoir_size=32,
        learnable_sr=False, use_diff_states=False,
    )


@pytest.fixture
def dsconv_encoder():
    from model.dsconv import DSConvEncoder
    return DSConvEncoder(in_channels=32)


@pytest.fixture
def attention():
    from model.attention import PatchMicroAttention
    return PatchMicroAttention(in_channels=48, seq_len=32, d_model=32, ff_dim=48)


@pytest.fixture
def binary_classifier():
    from model.binary_head import BinaryClassifier
    return BinaryClassifier(in_features=32, num_classes=6)
