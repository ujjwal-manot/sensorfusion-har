# SensorFusion-HAR

A lightweight real-time Human Activity Recognition system using a novel 5-stage pipeline -- reservoir computing, depthwise separable convolutions, gated residual fusion, patch micro-attention, and binary quantization -- achieving sub-10ms latency in ~22KB (INT8) with 10 novel research contributions.

## Architecture

```
Input (128x6) -> [ESN] -> [DS-Conv1D x3] -> [Gated Fusion] -> [Patch Attention] -> [Binary Head] -> N classes
     MEMS data   32-dim     48-ch features   residual gate     32-dim pooled       STE weights
```

**5-Stage Pipeline:**

1. **Echo State Network (ESN):** Fixed random reservoir expanding 6-channel MEMS input to 32 dimensions. Supports data-driven spectral initialization using the autocorrelation eigenvectors of training data. Zero trainable parameters.

2. **Depthwise Separable 1D-CNN:** Three stacked DS-Conv1D blocks (32->48->48->48 channels, strides 1/2/2) reduce the parameter count by ~8x compared to standard convolutions.

3. **Gated Residual Fusion:** A learnable sigmoid gate that controls how much raw reservoir dynamics bypass the convolutional encoder via a residual connection. Adds ~1.6K parameters.

4. **Patch Micro-Attention:** 8 fixed-size patches with 2-head self-attention (d_model=32). Approximately 128x cheaper than full-sequence attention.

5. **Binary Quantized Head:** Classification weights quantized to +1/-1 via Straight-Through Estimator (STE). All multiply-accumulate operations reduce to additions and subtractions.

## Novel Research Contributions

| # | Contribution | Type | Inference Cost |
|---|---|---|---|
| 1 | Gated Residual Fusion between ESN and DS-Conv | Architecture | +1.6K params |
| 2 | Spectral Reservoir Initialization via autocorrelation eigenvectors | Initialization | Zero |
| 3 | Masked Sensor Modeling (MSM) -- BERT-style pre-training for IMU data | Self-supervised | Zero |
| 4 | Multi-Task Adversarial Training with gradient reversal for subject invariance | Training | Zero |
| 5 | Curriculum Learning by activity complexity (static -> dynamic) | Training | Zero |
| 6 | Few-Shot Personalization -- fine-tune binary head on K samples | Deployment | Zero |
| 7 | FGSM/PGD adversarial robustness evaluation on sensor data | Evaluation | Zero |
| 8 | Activity transition detection -- stable vs boundary accuracy | Evaluation | Zero |
| 9 | Sensor drift simulation -- bias, scale, and noise degradation | Evaluation | Zero |
| 10 | MAC-based energy estimation using Horowitz 2014 energy costs | Evaluation | Zero |

9 of 10 contributions add zero inference cost. The full model stays under 23K parameters.

## Features

- Sub-10ms CPU inference latency
- ~22KB INT8 quantized model
- Dual dataset: UCI-HAR (6 classes) and PAMAP2 (12 complex activities)
- 7 sensor augmentation methods (jitter, scaling, rotation, permutation, time warp, magnitude warp, channel dropout)
- 3 pre-training methods: supervised, SimCLR contrastive, Masked Sensor Modeling
- LOSO (Leave-One-Subject-Out) cross-validation
- Cross-dataset transfer evaluation (UCI-HAR to PAMAP2)
- Phone-to-laptop WebSocket streaming with live dashboard
- Browser-based sensor access via DeviceMotion API
- t-SNE, attention maps, noise robustness, confidence calibration visualizations
- ONNX export for deployment
- 6-variant ablation study (Full, No Reservoir, No Attention, No Binary Head, No DS-Conv, No Gate)

## Project Structure

```
sensorfusion-har/
├── model/
│   ├── reservoir.py           Echo State Network + spectral init
│   ├── dsconv.py              Depthwise Separable Conv1D
│   ├── attention.py           Patch Micro-Attention
│   ├── binary_head.py         Binary Quantized Classifier (STE)
│   ├── sensorfusion.py        Full pipeline + Gated Residual Fusion
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
| Trainable Parameters | ~22.8K |
| Model Size (FP32) | ~89 KB |
| Model Size (INT8) | ~22 KB |
| MACs per Inference | ~765K |
| Energy per Inference (FP32) | ~0.0035 mJ |
| Energy per Inference (INT8) | ~0.0002 mJ |
| Inference Latency | <10 ms (CPU) |
| Input Shape | (batch, 128, 6) |

## Ablation Variants

| Configuration | Description |
|---|---|
| Full Pipeline | ESN + DS-Conv + Gate + Attention + Binary Head |
| No Reservoir | Linear projection replaces ESN |
| No Attention | Global average pooling replaces patch attention |
| No Binary Head | Standard linear replaces binary classifier |
| No DS-Conv | Standard Conv1D replaces depthwise separable |
| No Gate | Removes gated residual fusion |

## Citation

```bibtex
@article{sensorfusionhar2026,
  title={SensorFusion-HAR: A Lightweight Multi-Paradigm Pipeline with Novel Training Strategies for Real-Time Human Activity Recognition},
  year={2026}
}
```

## License

MIT
