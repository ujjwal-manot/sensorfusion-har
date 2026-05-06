# SensorFusion-HAR: Complete Technical Documentation for Thesis

## Abstract
SensorFusion-HAR is a lightweight real-time HAR system for ESP32 deployment. Uses 5-stage pipeline: ESN with learnable spectral radius, DS-Conv, spectral gated fusion, patch micro-attention, scaled binary quantization. Achieves 83.41% accuracy on 11-class merged dataset with 75,290 parameters (294 KB FP32, ~75 KB INT8), sub-30ms ESP32 inference.

## System Architecture
**Pipeline:** Input(50×6) → ESN(128) → DS-Conv(64) → Multi-Scale(64) → Spectral Gate(64) → Patch Attention(64) → Fusion(64) → Binary Head(11)

**Data Flow:**
1. Input: 50 timesteps × 6 channels (ax,ay,az,gx,gy,gz) at 50Hz
2. ESN: Expands to 128 dims (64 reservoir + 64 differential states)
3. DS-Conv: Depthwise separable 1D-CNN, stride-2 downsampling to 25 timesteps
4. Multi-Scale: Multi-kernel (3,5,9) depthwise convolutions
5. Spectral Gate: Frequency-domain gating between ESN and DS-Conv
6. Attention: 5-patch micro-attention, 2-head self-attention
7. Fusion: Concat with orientation stats (128→64)
8. Classification: Scaled binary linear layer

## Model Architecture Details

### SensorFusionESP32
- **Input:** 6 channels, reservoir_size=64, num_classes=11
- **Total Parameters:** 75,290
- **FP32 Size:** 294.10 KB
- **INT8 Size:** ~75 KB (estimated)
- **Inference:** <10ms CPU, 10-30ms ESP32

### Component 1: EchoStateNetworkEdge
**Purpose:** Temporal feature extraction via reservoir computing
**Parameters:** input_channels=6, reservoir_size=64, spectral_radius=0.9 (learnable), sparsity=0.80, dropout=0.10
**Trainable:** sr_logit (1 parameter)
**Fixed:** W_in (6×64), W_res (64×64, 80% sparse)
**Forward:** h_t = tanh(x_t @ W_in + h_{t-1} @ W_res_scaled), output = concat(states, diffs) shape (batch,50,128)
**Learnable SR:** effective_sr = sigmoid(sr_logit), W_res_scaled = W_res × (effective_sr / base_sr)

### Component 2: DSConvEncoderEdge
**Purpose:** Spatial feature extraction with parameter efficiency
**Parameters:** in_channels=64, out_channels=64, dropout=0.15
**Architecture:** DW-Conv(k=5) → PW-Conv(k=1) → BN → ReLU → Dropout → DW-Conv(k=5,s=2) → PW-Conv(k=1) → BN → ReLU → Dropout
**Input/Output:** (batch,64,50) → (batch,64,25)
**Parameters:** 9,088 (vs 40,960 standard conv, 4.5× reduction)

### Component 3: MultiScaleTemporalEdge
**Purpose:** Multi-scale temporal feature extraction
**Parameters:** channels=64
**Architecture:** DW-Conv(k=3,5,9) → Mix-Conv(k=1) → BN → ReLU, residual connection
**Input/Output:** (batch,64,25) → (batch,64,25)

### Component 4: EdgeSpectralGatedFusion
**Purpose:** Frequency-domain selective gating
**Parameters:** reservoir_dim=64, channels=64, seq_len=25
**Architecture:** Conv1d(64→64) → AdaptiveAvgPool1d(25) → Linear(2→1)
**Forward:** low = avg_pool(dsconv), high = dsconv - low, gate = sigmoid(band_gate(energy)), output = dsconv + gate × res

### Component 5: PatchMicroAttentionEdge
**Purpose:** Temporal pattern recognition via attention
**Parameters:** in_channels=64, seq_len=25, patch_len=5, d_model=64, attn_drop=0.10
**Patches:** 5 patches (25/5), each 5×64=320 features
**Architecture:** Linear(320→64) → Q/K/V Linear(64→64) → Attention → Linear(64→64) → LayerNorm
**Parameters:** 36,992
**Complexity:** O(25×320)=8,000 vs standard O(625×64)=40,000 (5× reduction)
**Entropy:** -(attn × log(attn)).sum().mean() for regularization

### Component 6: OrientationMicroPath
**Purpose:** Orientation and statistical features
**Parameters:** input_channels=6, d_model=64
**Statistics:** mean(6), std(6), RMS(6), acc_mean(3), acc_std(3), acc_deriv_RMS(3) = 27 features
**Architecture:** Linear(27→64) → ReLU → LayerNorm
**Input/Output:** (batch,50,6) → (batch,64)

### Component 7: Feature Fusion
**Purpose:** Combine attention + orientation features
**Architecture:** Linear(128→64) → ReLU → Dropout(0.25) → LayerNorm
**Input/Output:** (batch,128) → (batch,64)

### Component 8: ScaledBinaryLinear
**Purpose:** Memory-efficient classification
**Parameters:** in_features=64, out_features=11
**Architecture:** weight(64×11), bias(11), scale(11×1)
**Binary Quantization:** bw = sign(weight) × abs(scale).clamp_min(1e-4)
**Parameters:** 726
**Memory:** FP32=2,816 bytes, Binary=88 bytes (32× reduction)

## Dataset Specifications

### Merged Dataset Composition
**Datasets:** UCI-HAR (total_acc+gyro), PAMAP2 (hand acc+gyro), MHEALTH (right-arm acc+gyro), RealWorld HAR (waist acc+gyro)
**Total Windows:** ~25,000 (80/20 train/test split)
**Train:** ~20,000, **Test:** ~5,000

### UCI-HAR
**Subjects:** 30, **Activities:** 6, **Sampling:** 50Hz, **Window:** 128→50
**Files:** total_acc_x/y/z, body_gyro_x/y/z
**Unit Conversion:** acc: g→m/s² (×9.81), gyro: rad/s
**Mapping:** Walking→0, Stairs Up→4, Stairs Down→5, Sitting→1, Standing→2, Laying→3
**Samples:** Train=7,352, Test=2,947

### PAMAP2
**Subjects:** 9, **Activities:** 12 (subset), **Sampling:** 100Hz→50Hz, **Window:** 128→50
**Sensors:** Chest acc (cols 21-23), Chest gyro (cols 27-29)
**Mapping:** Lying→3, Sitting→1, Standing→2, Walking→0, Running→9, Cycling→8, Nordic Walking→0, Stairs Up→4, Stairs Down→5
**Samples:** Train~8,000, Test~2,000

### MHEALTH
**Subjects:** 10, **Activities:** 21 (subset), **Sampling:** 50Hz, **Window:** 100→50
**Sensors:** Right-arm acc+gyro (cols 14-19)
**Mapping:** Act3→1, Act4→0, Act5→4, Act9→8, Act10→6, Act11→9
**Samples:** Train~5,000, Test~1,500

### RealWorld HAR
**Subjects:** 15, **Activities:** 8, **Sampling:** 50Hz, **Window:** 50
**Sensor:** Chest-mounted acc+gyro
**Mapping:** walking→0, sitting→1, standing→2, lying→3, climbing_up→4, climbing_down→5, running→9, jumping→7
**Samples:** ~2,000

### Class Distribution (Train/Test)
| Class | Label | Train | Test |
|-------|-------|-------|------|
| Walking | 0 | ~3,500 | ~875 |
| Sitting | 1 | ~1,800 | ~450 |
| Standing | 2 | ~1,900 | ~475 |
| Lying Down | 3 | ~2,000 | ~500 |
| Stairs Up | 4 | ~1,200 | ~300 |
| Stairs Down | 5 | ~1,200 | ~300 |
| Jogging | 6 | ~2,500 | ~625 |
| Jumping | 7 | ~800 | ~200 |
| Cycling | 8 | ~2,200 | ~550 |
| Running | 9 | ~2,400 | ~600 |
| Waist Bending | 10 | ~700 | ~175 |

### Normalization Statistics
**Mean:** [1.9352865, 3.705785, 0.23090644, -0.02979615, -0.04969823, 0.04553811]
**Std:** [5.3360248, 7.3998132, 4.6662188, 0.47728997, 0.55818295, 0.37370721]
**Channels:** [ax, ay, az, gx, gy, gz]
**Units:** m/s² (acc), rad/s (gyro)

## Training Configuration

### Stage 1: MSM Pre-training
**Purpose:** Self-supervised via Masked Sensor Modeling
**Epochs:** 10, **Batch Size:** 128, **LR:** 3e-4, **Weight Decay:** 1e-4
**Scheduler:** CosineAnnealingLR, **Mask Ratio:** 0.20
**Loss:** MSE on masked positions
**Output:** checkpoints_v2/esp32_v2_msm_pretrain.pt
**Transfer:** Reservoir + DS-Conv weights to supervised model

### Stage 2: Head Training
**Purpose:** Train classifier/attention with frozen backbone
**Epochs:** 20 (default), **Batch Size:** 128, **LR:** 1e-3, **Weight Decay:** 1e-4
**Scheduler:** CosineAnnealingWarmRestarts(T_0=15, T_mult=1, eta_min=1e-5)
**SWA Start:** Epoch 30, **SWA LR:** 2e-4, **SWA Anneal:** 5 epochs
**Frozen:** reservoir.*, dsconv.*
**Early Stopping:** Patience=10, Metric=4×min_f1+macro_f1+acc
**SWA:** Averages weights from epoch 30+, updates BatchNorm

### Stage 3: Fine-tuning
**Purpose:** End-to-end training with all layers unfrozen
**Epochs:** 25, **Batch Size:** 128, **LR:** 1e-4, **Weight Decay:** 1e-4
**Scheduler:** CosineAnnealingLR
**Note:** Skipped in pocket_v3 due to degradation

## Loss Functions

### Focal Loss
**Purpose:** Address class imbalance
**Parameters:** alpha (inverse class freq), gamma=1.5, label_smoothing=0.02
**Formula:** FL = -α_t × (1-p_t)^γ × log(p_t)
**Alpha Calculation:** alpha = sum(counts)/(num_classes×counts), normalized by mean

### Attention Entropy Regularization
**Purpose:** Prevent attention head collapse
**Weight:** 0.01
**Formula:** loss = criterion(logits, targets) - 0.01 × entropy
**Entropy:** -(attn × log(attn)).sum(dim=-1).mean()

### Prototype Margin Loss
**Purpose:** Encourage feature separation
**Parameters:** num_classes=11, feature_dim=64, margin=1.0, weight=0.01, warmup=5 epochs
**Formula:** loss = own.mean() + relu(margin + own - nearest_other).mean()
**Warmup:** Linearly increase weight 0→0.01 over 5 epochs

## Data Augmentation

### Augmentation Pipeline
**Minority Classes:** p=0.65, **Majority Classes:** p=0.35
**Minority Threshold:** 2,000 samples

**Augmentations:**
1. Gaussian Noise: σ=0.03
2. Scaling: 1.0 + N(0, 0.08) per sample
3. Channel Dropout: 20% prob, zero random channel
4. 3D Rotation: 30% prob, random rotation matrix R=Rz@Ry@Rx

### Reservoir Manifold Mixup
**Purpose:** Augmentation in frozen reservoir state space
**Alpha:** 0.2, **Probability:** 0.30
**Formula:** h_mix = λ×h1 + (1-λ)×h2, λ~Beta(α,α)
**Loss:** λ×criterion(logits,y1) + (1-λ)×criterion(logits,y2)

### Sampling Strategy
**Weighted Random Sampler:** sample_weights = inv[bincount(labels)]
**Effect:** Oversample minority, undersample majority classes

## Optimizer Configuration
**Optimizer:** AdamW
**LR:** MSM=3e-4, Head=1e-3, Fine-tune=1e-4
**Weight Decay:** 1e-4
**Betas:** (0.9, 0.999)
**Gradient Clipping:** Max norm=1.0

## Performance Metrics (pocket_v3)

### Overall Performance
**Accuracy:** 83.41%
**Macro F1:** 82.72%
**Min F1:** 61.07%
**Parameters:** 75,290
**FP32 Size:** 294.10 KB

### Per-Class F1 Scores
| Class | F1 Score |
|-------|----------|
| Walking | 92.23% |
| Sitting | 68.69% |
| Standing | 73.96% |
| Lying Down | 91.96% |
| Stairs Up | 81.23% |
| Stairs Down | 89.26% |
| Jogging | 97.19% |
| Jumping | 61.07% |
| Cycling | 95.95% |
| Running | 96.50% |
| Waist Bending | 61.84% |

### Training History (Head Stage)
**Epoch 1:** acc=61.11%, macro_f1=59.11%, min_f1=27.24%, loss=0.324
**Epoch 5:** acc=71.70%, macro_f1=72.39%, min_f1=45.88%, loss=0.125
**Epoch 10:** acc=80.41%, macro_f1=79.44%, min_f1=55.50%, loss=0.097
**Epoch 15:** acc=83.41%, macro_f1=82.72%, min_f1=61.07%, loss=0.084 (best)
**Epoch 20:** acc=82.09%, macro_f1=81.37%, min_f1=55.69%, loss=0.086

### Training Configuration (pocket_v3)
**MSM Epochs:** 10, **Head Epochs:** 20, **FT Epochs:** 25
**Batch Size:** 128
**Include MHEALTH:** True, **Include RealWorld:** True
**Prototype Weight:** 0.01, **Prototype Warmup:** 5

## ESP32 Implementation

### Hardware
**Microcontroller:** ESP32
**IMU Sensor:** MPU6050
**Connection:** I2C (GPIO21=SDA, GPIO22=SCL)
**Power:** 3.3V

### Firmware (esp32_har.ino)
**Sampling Rate:** 50 Hz (20ms per sample)
**Window Size:** 50 samples (1 second)
**Tensor Arena:** 40 KB
**Confidence Threshold:** 0.30

### Normalization Constants
**NORM_MEAN:** [1.9352865, 3.705785, 0.23090644, -0.02979615, -0.04969823, 0.04553811]
**NORM_STD:** [5.3360248, 7.3998132, 4.6662188, 0.47728997, 0.55818295, 0.37370721]

### Activity Labels (11)
Walking, Sitting, Standing, Lying Down, Stairs Up, Stairs Down, Jogging, Jumping, Cycling, Running, Waist Bending

### Serial Output Format
**Human:** `Activity: Walking (conf: 85.3%, 12.5 ms)`
**JSON:** `{"activity":"Walking","confidence":0.8532,"class":0,"inference_time_ms":12.5}`

### ESP32 Performance
**Inference Time:** 10-30 ms
**Memory Usage:** ~40 KB arena
**Model Size:** ~75 KB (FP32) / ~25 KB (INT8)
**Power:** ~100-200 mA
**Battery (2000mAh):** ~10-20 hours

## Dashboard System

### Server (server.py)
**Framework:** FastAPI
**WebSocket Endpoints:** /ws/phone, /ws/dashboard
**HTTPS Port:** 8443
**HTTP Redirect Port:** 8765
**Model Loading:** Prioritizes useful11_final.pt (11 classes, matches architecture)

### Dashboard V3 (dashboard_v3.html)
**Section 1 - Input Details:**
- Live sensor waveforms (accelerometer, gyroscope)
- Statistics: sampling rate, buffer fill, source
- Per-channel statistics (accel mean/std, gyro std)
- Quality indicators: signal quality, noise level, normalization status

**Section 2 - Model Working:**
- Attention weights heatmap visualization
- Feature activations bar chart (64 features)
- Processing pipeline animation (Input→Reservoir→DS-Conv→Attention→Output)
- Spectral radius display

**Section 3 - Output:**
- Activity prediction with confidence
- Probability distribution across all classes
- Stability metrics: inference time, confidence threshold, majority vote, prediction count

### Model Internals
**Server Modification:** return_aux=True in model forward
**Extracted Data:**
- attention_weights: (batch, num_patches, num_patches)
- features: (batch, 64)
- spectral_radius: scalar

## Deployment Scripts

### export_tflite.py
**Purpose:** Convert PyTorch model to TFLite
**Workflow:** PyTorch → ONNX → TFLite → C header
**Validation:** Compare PyTorch vs TFLite outputs
**Note:** Requires TensorFlow (unavailable on Python 3.14, use Python 3.10/3.11)

### export_weights_esp32.py
**Purpose:** Export weights as C arrays (alternative to TFLite)
**Output:** model_weights.h with normalization stats, labels, classifier weights
**Note:** Requires manual C++ implementation of full model architecture

### esp32_dataset_test.py
**Purpose:** Test ESP32 with dataset samples
**Features:**
- Supports UCI-HAR, MHEALTH, RealWorld datasets
- Normalizes samples using training stats
- Simulation mode (--simulate) for testing without hardware
- Generates classification report and confusion matrix

### esp32_bridge.py
**Purpose:** Serial-to-WebSocket bridge
**Features:**
- Reads JSON predictions from ESP32 Serial
- Forwards to WebSocket server for dashboard
- Simulation mode for testing without hardware
- Auto-reconnection on disconnect

### collect_pocket_data.py
**Purpose:** Template for pocket-specific data collection
**Note:** Requires WebSocket connection implementation for actual use

## Novel Research Contributions

| # | Contribution | Type | Params Added |
|---|-------------|------|--------------|
| 1 | Learnable Spectral Radius | Architecture | +1 |
| 2 | Differential Reservoir State Encoding | Architecture | +32 |
| 3 | Spectral-Domain Gated Fusion | Architecture | +257 |
| 4 | Scaled Binary Quantization | Architecture | +6 |
| 5 | Reservoir Manifold Mixup | Training | 0 |
| 6 | Attention Entropy Regularization | Training | 0 |
| 7 | Stochastic Reservoir Masking | Training | 0 |

**Total Additional Parameters:** 296
**Zero-Cost Contributions:** 3 (Mixup, Entropy Reg, Masking)

## Complete Hyperparameters

### Model Hyperparameters
- TARGET_TIME_STEPS: 50
- INPUT_CHANNELS: 6
- NUM_CLASSES: 11
- RESERVOIR_SIZE: 64
- SPECTRAL_RADIUS_INIT: 0.9
- RESERVOIR_SPARSITY: 0.80
- RESERVOIR_DROPOUT: 0.10
- DS_CONV_DROPOUT: 0.15
- ATTENTION_DROPOUT: 0.10
- CLASSIFIER_DROPOUT: 0.25
- PATCH_LEN: 5
- D_MODEL: 64
- ENTROPY_WEIGHT: 0.01
- MIXUP_ALPHA: 0.2
- MIXUP_PROB: 0.30

### Training Hyperparameters
- SEED: 42
- MSM_EPOCHS: 10
- HEAD_EPOCHS: 20
- FT_EPOCHS: 25
- BATCH_SIZE: 128
- MSM_LR: 3e-4
- HEAD_LR: 1e-3
- FT_LR: 1e-4
- WEIGHT_DECAY: 1e-4
- FOCAL_GAMMA: 1.5
- LABEL_SMOOTHING: 0.02
- PROTOTYPE_WEIGHT: 0.01
- PROTOTYPE_WARMUP: 5
- PROTOTYPE_MARGIN: 1.0
- PATIENCE: 10
- SWA_START: 30
- SWA_LR: 2e-4
- SWA_ANNEAL_EPOCHS: 5

### Augmentation Hyperparameters
- NOISE_STD: 0.03
- SCALE_STD: 0.08
- CHANNEL_DROPROUT_PROB: 0.20
- ROTATION_PROB: 0.30
- MINORITY_THRESHOLD: 2000
- P_MAJOR: 0.35
- P_MINOR: 0.65
- MSM_MASK_RATIO: 0.20

### Dataset Hyperparameters
- UCI_G_TO_MS2: 9.81
- TARGET_HZ: 50
- WINDOW_SIZE: 50
- STRIDE: 25
- TEST_SIZE: 0.20

### ESP32 Hyperparameters
- SAMPLE_RATE_HZ: 50
- SAMPLE_PERIOD_MS: 20
- TIME_STEPS: 50
- NUM_CHANNELS: 6
- NUM_CLASSES: 11
- TENSOR_ARENA_SIZE: 40,096 bytes
- CONFIDENCE_THRESHOLD: 0.30
- MPU6050_ADDR: 0x68


## Citation
```bibtex
@article{sensorfusionhar2026,
  title={SensorFusion-HAR: A Lightweight Multi-Paradigm Pipeline with Novel Reservoir Dynamics and Spectral Fusion for Real-Time Human Activity Recognition},
  year={2026}
}
```

## License
MIT
