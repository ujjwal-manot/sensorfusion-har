# Run on Google Colab T4 GPU

## Quick Start (60-90 seconds to start training)

### Step 1: Upload Notebook to Colab
1. Open [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Select `sensorfusion_har_COLAB.ipynb` from this folder

### Step 2: Set T4 GPU Runtime
1. Click **Runtime → Change runtime type**
2. Under "Hardware accelerator" select **T4 GPU**
3. Click **Save**

### Step 3: Run All Cells
1. Click **Runtime → Run all** (or press `Ctrl+F9`)
2. First cell will show: `GPU: Tesla T4` and install dependencies (~2 min)
3. Training starts automatically

## What's Running

| Stage | Epochs | Est. Time on T4 |
|-------|--------|-----------------|
| UCI-HAR Training | 100 | ~15 min |
| MSM Pre-training | 12 | ~3 min |
| SimCLR Contrastive | 8 | ~2 min |
| Curriculum Learning | 60 | ~12 min |
| ESP32 Fine-tuning | 40 | ~8 min |
| ONNX → TFLite Export | - | ~2 min |
| **Total** | | **~42 min** |

## Outputs (Auto-downloaded)

After completion, two zip files download:
- `sensorfusion_har_outputs.zip` - Plots, ONNX, TFLite models
- `sensorfusion_har_checkpoints.zip` - All `.pt` checkpoints

## Key Files Generated

```
outputs/
├── plots/                    # 8 visualization PNGs
├── best_model.pt            # Main checkpoint
└── summary.json             # Metrics

checkpoints/
├── best_model.pt            # UCI-HAR model
├── best_simclr.pt         # Contrastive pretrained
└── best_esp32_model.pt    # Lite model

exports/esp32/
├── sensorfusion_lite.onnx       # ONNX FP32
├── sensorfusion_lite_fp32.tflite
├── sensorfusion_lite_int8.tflite  # ESP32 deploy
└── sensorfusion_lite_model.h      # C header
```

## Troubleshooting

**"GPU not available"**
- Runtime → Change runtime type → Select T4 GPU → Save
- Runtime → Restart runtime

**Out of memory**
- Runtime → Restart runtime
- Reduce batch_size in training cells

**Want to stop?**
- Click the stop button (■) or Runtime → Interrupt execution

## Alternative: Run Python Scripts Directly

If notebook cells fail, run these in a code cell:

```python
# Full training pipeline
!python train.py --dataset ucihar --epochs 100 --batch_size 64
!python evaluate.py --dataset ucihar --ablation --benchmark
!python download_outputs.py
```
