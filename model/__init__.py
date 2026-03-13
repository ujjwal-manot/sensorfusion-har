from .sensorfusion import SensorFusionHAR, GatedResidualFusion
from .reservoir import EchoStateNetwork
from .dsconv import DepthwiseSeparableBlock, DSConvEncoder
from .attention import PatchMicroAttention
from .binary_head import BinaryLinear, BinaryClassifier
from .dataset_pamap2 import PAMAP2Dataset
from .contrastive import SensorSimCLR, nt_xent_loss, pretrain_contrastive, transfer_weights
from .masked_pretrain import MaskedSensorModel, create_mask, masked_pretrain, transfer_masked_weights
from .multitask import GradientReversalLayer, MultiTaskHAR, SubjectLabeledDataset, train_multitask
from .curriculum import CurriculumScheduler, CurriculumTrainer
from .personalization import few_shot_personalize, evaluate_personalization
from .adversarial import fgsm_attack, pgd_attack, evaluate_adversarial_robustness, plot_adversarial_robustness
from .transitions import detect_transitions, evaluate_transition_accuracy, plot_transition_analysis
from .drift import simulate_bias_drift, simulate_scale_drift, simulate_noise_drift, evaluate_drift_robustness, plot_drift_robustness
from .energy import count_macs, estimate_energy, compare_models_energy, plot_energy_comparison
