# SensorFusion-HAR

A lightweight real-time Human Activity Recognition system using a novel 4-stage pipeline -- reservoir computing, depthwise separable convolutions, patch micro-attention, and binary quantization -- targeting sub-10ms latency in ~21KB (INT8).

## Architecture

SensorFusion-HAR combines four computational paradigms into a single inference pipeline:

1. **Echo State Network (ESN):** A fixed random reservoir that expands 6-channel MEMS input into a 32-dimensional feature space. The reservoir weights are never trained, making this stage parameter-free during backpropagation.

2. **Depthwise Separable 1D-CNN:** Three stacked DS-Conv1D blocks (32->48->48->48 channels) extract spatial features from the reservoir output. Depthwise separable factorization reduces the parameter count by roughly 8x compared to standard convolutions at equivalent channel depth.

3. **Patch Micro-Attention:** The temporal sequence is split into 8 fixed-size patches, and multi-head self-attention (2 heads, d_model=32) is computed across patches. This yields the representational benefits of attention at approximately 128x lower cost than full-sequence attention.

4. **Binary Quantized Head:** The classification head uses +1/-1 weights trained via the Straight-Through Estimator (STE). At inference, all multiply-accumulate operations in this stage reduce to additions and subtractions.

```
Input (128x6) -> [ESN] -> [DS-Conv1D x3] -> [Patch Attention] -> [Binary Head] -> N classes
     MEMS sensors   32-dim reservoir   48-ch features    32-dim pooled      N outputs
```

## Features

- Real-time inference with sub-10ms latency on CPU
- ~21KB INT8 quantized model size
- Dual dataset support: UCI-HAR (6 classes) and PAMAP2 (12 complex activities)
- 7 sensor data augmentation methods (jitter, scaling, rotation, permutation, time warp, magnitude warp, channel dropout)
- SimCLR contrastive pre-training for improved representations
- LOSO (Leave-One-Subject-Out) cross-validation
- Cross-dataset transfer evaluation
- Phone-to-laptop sensor streaming via WebSocket
- Browser-based sensor access using DeviceMotion API
- Live dashboard with waveform visualization and activity timeline
- t-SNE feature visualization at each pipeline stage
- Attention map visualization
- Noise robustness analysis across SNR levels
- Confidence calibration with ECE measurement
- ONNX export for deployment
- Full ablation study framework

## Project Structure

```
sensorfusion-har/
├── model/
│   ├── reservoir.py
│   ├── dsconv.py
│   ├── attention.py
│   ├── binary_head.py
│   ├── sensorfusion.py
│   ├── dataset.py
│   ├── dataset_pamap2.py
│   ├── augmentation.py
│   ├── contrastive.py
│   ├── visualize.py
│   └── __init__.py
├── static/
│   ├── phone.html
│   └── dashboard.html
├── server.py
├── train.py
├── evaluate.py
├── sensorfusion_har.ipynb
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

- Dashboard on laptop: `http://localhost:8765`
- Phone sensor page: `http://<your-ip>:8765/phone`
- Both devices must be on the same network.

### Jupyter Notebook

The notebook `sensorfusion_har.ipynb` contains the complete pipeline: dataset exploration, augmentation demo, model summary, training with augmentation, evaluation with confusion matrix, ablation study, t-SNE visualization, attention maps, noise robustness, confidence calibration, SimCLR contrastive pre-training, LOSO evaluation, cross-dataset transfer, baseline comparison with Pareto plot, ONNX export, and inference benchmark.

## Datasets

**UCI-HAR:** 6 activities (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying) from 30 subjects, 128 timesteps x 6 channels (3-axis accelerometer + 3-axis gyroscope).

**PAMAP2:** 12 activities (Lying, Sitting, Standing, Walking, Running, Cycling, Nordic Walking, Ascending Stairs, Descending Stairs, Vacuum Cleaning, Ironing, Rope Jumping) from 9 subjects, 128 timesteps x 6 channels at 100Hz.

## Model Specifications

| Metric | Value |
|---|---|
| Trainable Parameters | ~21K |
| Model Size (FP32) | ~83 KB |
| Model Size (INT8) | ~21 KB |
| Inference Latency | <10 ms (CPU) |
| Input Shape | (batch, 128, 6) |
| Output | N-class logits |

## Ablation Variants

| Configuration | Description |
|---|---|
| Full Pipeline | ESN + DS-Conv + Attention + Binary Head |
| No Reservoir | Linear projection replaces ESN |
| No Attention | Global average pooling replaces patch attention |
| No Binary Head | Standard linear layer replaces binary classifier |
| No DS-Conv | Standard Conv1D replaces depthwise separable blocks |

## Citation

```bibtex
@article{sensorfusionhar2026,
  title={SensorFusion-HAR: A Lightweight Multi-Paradigm Pipeline for Real-Time Human Activity Recognition},
  year={2026}
}
```

## License

MIT
