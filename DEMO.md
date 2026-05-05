# SensorFusion-HAR — Laboratory Demonstration Guide

## System Overview
Real-time Human Activity Recognition using a smartphone held in a **pants pocket**.  
The model runs on a laptop server (ESP32-deployable INT8 size) and receives raw IMU data from an Android phone over WebSocket.

---

## Novel Technical Contributions

| # | Contribution | Where |
|---|---|---|
| 1 | **Echo State Network (ESN) with learnable spectral radius** — reservoir computing with adaptive dynamics; avoids gradient vanishing over time steps | `EchoStateNetworkEdge` |
| 2 | **Depthwise Separable Conv (DS-Conv) Encoder** — 3–5× fewer parameters than standard conv with comparable expressivity | `DSConvEncoderEdge` |
| 3 | **Spectral Gated Fusion** — band-selective (low/high frequency) gating between reservoir features and convolution features | `EdgeSpectralGatedFusion` |
| 4 | **Patch Micro-Attention with Entropy Regularisation** — efficient self-attention on 5-sample signal patches; entropy term prevents attention collapse | `PatchMicroAttentionEdge` |
| 5 | **Binary Quantized Head (ScaledBinaryLinear)** — 1-bit weight classifier with learned scale; enables ESP32 deployment | `ScaledBinaryLinear` |
| 6 | **OrientationMicroPath** — parallel branch extracting orientation-invariant features (magnitude stats + jerk) for pocket invariance | `OrientationMicroPath` |
| 7 | **Masked Sensor Modelling (MSM) Pre-training** — self-supervised pre-training by masking 20% of input channels; initialises encoder with rich temporal representations | `MSMEncoderEdge` |
| 8 | **3-D Random Rotation Augmentation** — synthetic orientation diversity applied to both accelerometer and gyroscope axes; critical for pocket use where phone orientation is unknown | `augment_batch()` |
| 9 | **Pocket-Specific Normalization Fix** — corrected UCI-HAR total-acceleration from g-force (g) → m/s²; eliminated cross-dataset unit mismatch that caused a hidden ×9.81 bias | `UCITotalHARDataset` |
| 10 | **Stochastic Weight Averaging (SWA)** — averages model weights over later training epochs to find a flatter loss minimum; improves real-world generalisation | Training Stage 1 |
| 11 | **Early Stopping with Patience** — halts training when composite score (4×MinF1 + MacroF1 + Acc) stops improving; prevents overfitting | Training Stage 1 |
| 12 | **Multi-dataset Fusion** — merges UCI-HAR, PAMAP2 (chest sensor), mHealth, and RealWorld-HAR under a unified 7-class schema with per-source remapping | `build_merged_dataset()` |

---

## Model Architecture

```
Input: [Batch, 50 timesteps, 6 channels (ax ay az gx gy gz)]
         ↓
[MSM Pre-trained] EchoStateNetwork (reservoir_size=64, learnable sr)
         ↓
DSConv Encoder (depth-sep conv × 2 + BatchNorm + ReLU + Dropout 0.15)
         ↓
MultiScale Temporal (kernels 3,5,9 → residual fusion)
         ↓
SpectralGatedFusion (band-gate reservoir ↔ DSConv output)
         ↓
PatchMicroAttention (patches=5, d_model=64, attn_drop=0.10)
         ↓
OrientationMicroPath ─────────────────────────────┐
                                                   ↓
                              Feature Fusion (128→64, ReLU, Dropout 0.25, LayerNorm)
                                                   ↓
                              Dropout (0.25) → BatchNorm → BinaryLinear
                                                   ↓
                              Output: 7-class logits
```

**Parameters:** ~75,000  
**FP32 size:** 293 KB  
**INT8 (quantized):** ~73 KB → fits ESP32-S3 flash  

---

## Classes Recognised

Only activities with F1 ≥ 0.80 on the combined test set are reported:

| Class | Activity | Expected F1 |
|-------|----------|-------------|
| 0 | Walking | ≥ 0.90 |
| 1 | Lying Down | ≥ 0.90 |
| 2 | Stairs Up | ≥ 0.80 |
| 3 | Stairs Down | ≥ 0.85 |
| 4 | Jogging | ≥ 0.95 |
| 5 | Cycling | ≥ 0.95 |
| 6 | Running | ≥ 0.95 |

Dropped (< 0.80 F1 in all experiments): Sitting, Standing, Jumping, Waist Bending — these are merged into the "Uncertain" output when confidence < 0.40.

---

## Data Sources

| Dataset | Sensor Position | Activities Used | Windows |
|---------|----------------|----------------|---------|
| UCI-HAR | Waist | Walking, Stairs Up/Down, Lying | ~10,300 |
| PAMAP2 | **Chest** (pocket-representative) | Walking, Running, Cycling, Stairs, Lying | ~30,200 |
| mHealth | Chest | Walking, Jogging, Running, Cycling, Lying, Stairs | ~3,600 |
| RealWorld HAR | Chest/Thigh | Walking, Running, Stairs | varies |

---

## Running the Demo

### 1. Start the server
```bash
python server.py
```
Server starts on `http://0.0.0.0:8765`

### 2. Open the dashboard
On the demo laptop browser: `http://localhost:8765`

### 3. Connect the phone
On Android Chrome, navigate to: `http://<laptop-IP>:8765/phone`  
> Requires HTTP or a trusted certificate. Chrome flags: `chrome://flags/#unsafely-treat-insecure-origin-as-secure`

### 4. Tap "Start Streaming"
The phone streams 50 Hz IMU data. The dashboard updates every 500 ms with:
- Predicted activity
- Confidence score
- Per-class probability bars
- Inference time (ms)

### 5. Perform activities
Put the phone in your **pants pocket** and walk, jog, run, go up/down stairs, cycle, or lie down.

---

## ESP32 Deployment Path

```
PyTorch .pt  →  ONNX (opset 18)  →  TFLite FP32  →  TFLite INT8 (calibrated)  →  ESP32-S3
```

Generated files after training:
- `exports/esp32_v2/sensorfusion_esp32_v2_pocket_final.onnx`
- `exports/esp32_v2/calib_data_v2.npy` (calibration data for INT8 quantization)
- `exports/esp32_v2/normalization_stats.json` (mean/std for on-device normalization)

---

## Key Inference Pipeline

```
Phone IMU @ 50 Hz
    → WebSocket to Server
    → Ring buffer (50 samples = 1 second window)
    → Per-channel normalization (z-score with training stats)
    → SensorFusionESP32.forward() [eval mode, dropout disabled]
    → Softmax → argmax + confidence
    → Temporal majority vote (window=5) for smoothing
    → If confidence < 0.40 → "Uncertain"
    → WebSocket broadcast to dashboard
```
