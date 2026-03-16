# SensorFusion-HAR

## Overview
Ultra-lightweight Human Activity Recognition using a novel 5-stage pipeline:
ESN → DS-Conv → Spectral Gated Fusion → Patch Micro-Attention → Binary Quantized Head

## Key Stats
- 23,103 parameters (90 KB FP32, 22.5 KB INT8)
- 81.81% accuracy (supervised), 92.64% (curriculum learning)
- 7 novel research contributions
- <10ms inference on CPU

## Commands
```bash
# Train
python train.py --dataset ucihar --epochs 100 --batch_size 64 --lr 0.001

# Evaluate + ablation
python evaluate.py --dataset ucihar
python evaluate.py --ablation --ablation_epochs 50

# Benchmark
python evaluate.py --benchmark --benchmark_runs 1000

# Live demo
python server.py
# Dashboard: http://localhost:8765/
# Phone: http://<ip>:8765/phone
```

## Architecture
- model/sensorfusion.py - Main architecture + Spectral Gated Fusion
- model/reservoir.py - Echo State Network with learnable spectral radius
- model/dsconv.py - Depthwise Separable Conv1D blocks
- model/attention.py - Patch Micro-Attention with entropy regularization
- model/binary_head.py - Scaled Binary Quantized Classifier

## Datasets
- UCI-HAR: 30 subjects, 6 activities, auto-downloads
- PAMAP2: 9 subjects, 12 activities, auto-downloads

## Advanced Training
- model/contrastive.py - SimCLR pre-training (+8.18% accuracy)
- model/masked_pretrain.py - Masked Sensor Modeling (+6.82%)
- model/multitask.py - Multi-task adversarial (subject-invariant features)
- model/curriculum.py - Curriculum learning (+10.83%)
- model/personalization.py - Few-shot adaptation

## Testing
```bash
python -m pytest tests/ -v  # if tests exist
python evaluate.py --ablation  # ablation study serves as integration test
```
