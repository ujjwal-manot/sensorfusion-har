# SensorFusion-HAR 

A lightweight real-time Human Activity Recognition system using a novel 5-stage pipeline -- reservoir computing with learnable dynamics, depthwise separable convolutions, spectral-domain gated fusion, patch micro-attention with entropy regularization, and scaled binary quantization -- achieving sub-10ms latency in ~23KB (INT8) with 7 novel architectural and training contributions.

## Architecture

```
Input (128x6) -> [ESN] -> [DS-Conv1D x3] -> [Spectral Gate] -> [Patch Attention] -> [Binary Head] -> N classes
     MEMS data   32-dim     48-ch features   freq-domain gate   32-dim pooled       scaled STE weights
```

**5-Stage Pipeline:**

1. **Echo State Network (ESN):** Fixed random reservoir expanding 6-channel MEMS input to 32 dimensions. Supports learnable spectral radius (gradient-based optimization of reservoir dynamics), differential state encoding (captures instantaneous rate-of-change), stochastic reservoir masking (variational dropout on frozen states), and data-driven spectral initialization via autocorrelation eigenvectors. 1 trainable parameter (spectral radius logit).

2. **Depthwise Separable 1D-CNN:** Three stacked DS-Conv1D blocks (32->48->48->48 channels, strides 1/2/2) reduce the parameter count by ~8x compared to standard convolutions.

3. **Spectral Gated Fusion:** Frequency-domain selective gating between reservoir dynamics and convolutional features. Computes FFT of both streams, learns per-frequency gate values via a linear layer, and applies gating in the spectral domain before IFFT. This allows selective passing of reservoir information at activity-relevant frequencies while suppressing noise bands. Replaces scalar gating with 17-bin frequency resolution.

4. **Patch Micro-Attention:** 8 fixed-size patches with 2-head self-attention (d_model=32) and learnable positional embeddings. Supports attention entropy regularization during training to prevent head collapse. Approximately 128x cheaper than full-sequence attention.

5. **Scaled Binary Quantized Head:** Classification weights quantized to +1/-1 via Straight-Through Estimator (STE) with per-channel learned magnitude scaling. Multiply-accumulate operations reduce to scaled additions and subtractions. Includes binary weight export for bit-packed deployment.

## Novel Research Contributions

| # | Contribution | Type | Params Added |
|---|---|---|---|
| 1 | Learnable Spectral Radius -- first gradient-based spectral radius optimization in reservoir computing | Architecture | +1 |
| 2 | Differential Reservoir State Encoding with per-neuron gated fusion | Architecture | +32 |
| 3 | Spectral-Domain Gated Fusion between ESN and DS-Conv using learned frequency gates | Architecture | +257 |
| 4 | Scaled Binary Quantization with per-channel magnitude factors | Architecture | +6 |
| 5 | Reservoir Manifold Mixup -- data augmentation via interpolation in frozen reservoir state space | Training | Zero |
| 6 | Attention Entropy Regularization preventing patch attention collapse | Training | Zero |
| 7 | Stochastic Reservoir Masking -- variational dropout on non-trainable reservoir states | Training | Zero |

All 7 contributions add only 296 parameters total. 3 of 7 add zero inference cost. The full model stays under 24K parameters.

## Additional Features

- Spectral Reservoir Initialization via autocorrelation eigenvectors
- Masked Sensor Modeling (MSM) -- self-supervised pre-training for IMU data
- SimCLR contrastive pre-training with NT-Xent loss
- Multi-task adversarial training with gradient reversal for subject invariance
- Curriculum learning by activity complexity (static -> dynamic)
- Few-shot personalization -- fine-tune binary head on K samples
- FGSM/PGD adversarial robustness evaluation
- Activity transition detection -- stable vs boundary accuracy
- Sensor drift simulation -- bias, scale, and noise degradation
- MAC-based energy estimation using Horowitz 2014 energy costs
- 7 sensor augmentation methods (jitter, scaling, rotation, permutation, time warp, magnitude warp, channel dropout)
- LOSO (Leave-One-Subject-Out) cross-validation
- Cross-dataset transfer evaluation (UCI-HAR to PAMAP2)
- Phone-to-laptop WebSocket streaming with live dashboard
- t-SNE, attention maps, noise robustness, confidence calibration visualizations
- ONNX export for deployment
- 7-variant ablation study (Full, No Reservoir, No Attention, No Binary Head, No DS-Conv, No Gate, No Spectral Gate)

## Project Structure

```
sensorfusion-har/
├── model/
│   ├── reservoir.py           Echo State Network + learnable spectral radius + differential states
│   ├── dsconv.py              Depthwise Separable Conv1D
│   ├── attention.py           Patch Micro-Attention + entropy regularization
│   ├── binary_head.py         Scaled Binary Quantized Classifier (STE)
│   ├── sensorfusion.py        Full pipeline + Spectral Gated Fusion
│   ├── mixup.py               Reservoir Manifold Mixup
│   ├── dataset.py             UCI-HAR dataset loader
│   ├── dataset_pamap2.py      PAMAP2 dataset loader
│   ├── augmentation.py        7-method sensor augmentation
│   ├── contrastive.py         SimCLR contrastive pre-training
│   ├── masked_pretrain.py     Masked Sensor Modeling (MSM)
│   ├── multitask.py           Multi-task adversarial training
│   ├── curriculum.py          Curriculum learning scheduler
│   ├── personalization.py     Few-shot personalization
│   ├── adversarial.py         FGSM/PGD robustness testing
│   ├── transitions.py         Activity transition detection
│   ├── drift.py               Sensor drift simulation
│   ├── energy.py              MAC-based energy estimation
│   ├── visualize.py           t-SNE, attention maps, calibration
│   └── __init__.py
├── static/
│   ├── phone.html             Phone sensor streaming UI
│   └── dashboard.html         Real-time dashboard
├── server.py                  FastAPI WebSocket server
├── train.py                   Training script (UCI-HAR + PAMAP2)
├── evaluate.py                Evaluation + ablation + benchmark
├── sensorfusion_har.ipynb     66-cell notebook (full analysis)
├── GUIDE.md                   Comprehensive learning guide
├── requirements.txt
└── README.md
```

## Quick Start

### Installation

```bash
git clone https://github.com/ujjwal-manot/sensorfusion-har.git
cd sensorfusion-har
pip install -r requirements.txt
```

### Training

```bash
python train.py --dataset ucihar --epochs 100 --batch_size 64 --lr 0.001
python train.py --dataset pamap2 --epochs 100 --batch_size 64 --lr 0.001
```

Training automatically uses all novel features: learnable spectral radius, differential states, spectral gated fusion, attention entropy regularization, and reservoir manifold mixup.

Datasets are downloaded automatically on first run.

### Evaluation

```bash
python evaluate.py --dataset ucihar
python evaluate.py --dataset pamap2
python evaluate.py --ablation --ablation_epochs 50
python evaluate.py --benchmark --benchmark_runs 1000
```

### Live Demo

```bash
python server.py
```

- Dashboard: `http://localhost:8765`
- Phone: `http://<your-ip>:8765/phone`
- Both devices must be on the same network.

### Notebook

The 66-cell notebook covers: dataset exploration, augmentation, model architecture, training, evaluation, ablation study, t-SNE, attention maps, noise robustness, calibration, SimCLR, Masked Sensor Modeling, multi-task adversarial training, curriculum learning, LOSO, few-shot personalization, adversarial robustness (FGSM/PGD), activity transition detection, sensor drift simulation, energy estimation, cross-dataset transfer, spectral reservoir init, baseline comparison, ONNX export, and inference benchmark.

## Datasets

**UCI-HAR:** 6 activities (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying) from 30 subjects. 128 timesteps x 6 channels (3-axis accelerometer + 3-axis gyroscope) at 50Hz.

**PAMAP2:** 12 activities (Lying, Sitting, Standing, Walking, Running, Cycling, Nordic Walking, Ascending Stairs, Descending Stairs, Vacuum Cleaning, Ironing, Rope Jumping) from 9 subjects. 128 timesteps x 6 channels at 100Hz.

## Model Specifications

| Metric | Value |
|---|---|
| Trainable Parameters | ~23.1K |
| Model Size (FP32) | ~90 KB |
| Model Size (INT8) | ~23 KB |
| MACs per Inference | ~766K |
| Energy per Inference (FP32) | ~0.0035 mJ |
| Energy per Inference (INT8) | ~0.0002 mJ |
| Inference Latency | <10 ms (CPU) |
| Input Shape | (batch, 128, 6) |

## Ablation Variants

| Configuration | Description |
|---|---|
| Full Pipeline | ESN (learnable SR + diff states) + DS-Conv + Spectral Gate + Attention + Scaled Binary Head |
| No Reservoir | Linear projection replaces ESN |
| No Attention | Global average pooling replaces patch attention |
| No Binary Head | Standard linear replaces binary classifier |
| No DS-Conv | Standard Conv1D replaces depthwise separable |
| No Gate | Removes spectral gated fusion |

## Citation

```bibtex
@article{sensorfusionhar2026,
  title={SensorFusion-HAR: A Lightweight Multi-Paradigm Pipeline with Novel Reservoir Dynamics and Spectral Fusion for Real-Time Human Activity Recognition},
  year={2026}
}
```

## License

MIT
