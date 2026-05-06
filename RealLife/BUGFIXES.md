# Fixes applied — Real-Life Implementation

Running log of every change to the original code drop, with rationale.

## Round 1 — surface fixes

### 1. `server.py` — wrong import path for the trained model
The original imported the model class from
`final_esp32_v2_useful11_package.train_esp32_v2_expanded_local`, but that
folder doesn't exist anywhere in the repo. The actual training script lives
under `training/`. The `try/except ImportError` silently swallowed the error
and the server fell back to the heuristic classifier — meaning the trained
checkpoint was never actually used at runtime.

### 2. `server.py` — `normalize()` was given the wrong shape in the HIL path
`normalize()` is written for a 2-D `(T, C)` window. The HIL simulation called
it as `normalize(sample[np.newaxis, :, :])` (3-D), so the channel indexing
inside the function silently overwrote the wrong axis. Fixed by calling
`normalize(sample)` and adding the batch axis afterwards.

### 3. `hil_server.py` — full rewrite, four bugs
- `args.baud_rate` was read but only `--baud` is defined, so every run hit
  `AttributeError`.
- The nested key lookup `data['normalization']['mean']` did not match the
  on-disk JSON layout (`{"mean": [...], "std": [...]}` at the top level).
  Always silently fell back to defaults.
- Path lookups used `Path("exports/...")` (relative to CWD), so the script
  only worked when launched from one specific folder.
- The class started a `websockets.serve(...)` server but its handler called
  `websocket.accept()`, `.receive_text()`, `.send_text()` — those are
  FastAPI/Starlette methods. The dashboard could never receive frames.

### 4. `training/train_esp32_v2_expanded_local.py` — duplicate dict keys
`PAMAP2_TO_MERGED` had keys 7 and 8 each appearing twice. Python silently
keeps the last one, dropping two intended class mappings.

### 5. `esp32/esp32_har.ino` — header comment lied about the label set
The header said "10-Class HAR: ... Soft Fall, Hard Collapse" but the actual
`NUM_CLASSES` is 11 with different labels. Comment now matches reality.

### 6. `generate_cert.py` — `datetime.utcnow()` deprecated on Python 3.12+
Replaced with timezone-aware `datetime.now(datetime.timezone.utc)`.


## Round 2 — deeper QA pass

### 7. `train_esp32_v2_expanded_local.py` — class count mismatched the deployed system
Training script had `NUM_CLASSES = 7` and a 7-entry `ESP32_ACTIVITY_LABELS`,
but every other component in the deployment surface (`server.py`,
`esp32_har.ino`, `esp32_hil.ino`, `esp32_v2_useful11_config.h`, both
dashboards, the report) is 11-class. Training the old layout produces a
checkpoint whose final-layer dimension cannot be loaded into the deployed
model. Fixed `NUM_CLASSES = 11` with the canonical label order:

    Walking, Sitting, Standing, Lying Down, Stairs Up, Stairs Down,
    Jogging, Jumping, Cycling, Running, Waist Bending

### 8. `train_esp32_v2_expanded_local.py` — every dataset-to-merged mapping was broken
- `UCITotalHARDataset.__init__` does `self.y = labels - 1`, so labels are
  0-indexed. `UCIHAR_TO_MERGED = {1: 0, 2: 4, 3: 5, 5: 1, 6: 2}` had keys
  consistent with neither the 0-indexed output (key 6 doesn't exist) nor
  the original 1-indexed labels (key 5 = "Standing" was being mapped to
  merged class 1 = "Sitting"). Either way, the wrong samples were being
  assigned to the wrong merged classes during every training run.
- `MHEALTH_TO_MERGED` and `PAMAP2_TO_MERGED` targeted the deleted 7-class
  layout, producing labels out of `[0, NUM_CLASSES)`.

All three mappings rewritten with correct source-label indexing for each
dataset, targeting the 11-class space. Verified all 11 merged classes are
reachable from at least one source and that every mapping value is in
`[0, 11)`.

### 9. `esp32_har.ino` — normalisation constants disagreed with the canonical config
Production firmware had hardcoded `NORM_MEAN`/`NORM_STD` arrays whose
values disagreed with `esp32_v2_useful11_config.h`. The HIL firmware
(`esp32_hil.ino`) uses the canonical constants. Real-life and HIL
inference would silently disagree. Fixed by `#include`-ing the config
header in the production sketch and aliasing the names. Header copied
into `esp32/` so the Arduino include resolves.

### 10. `esp32_har.ino` — boot messages still said "Lite" / "10-Class"
Updated to match the actual deployed model.

### 11. `esp32_dataset_test.py` — multiple bugs
- Same `data['normalization']['mean']` JSON layout bug.
- Same ghost `final_esp32_v2_useful11_package` folder in `sys.path`.
- Default `--norm-json` pointed at a non-existent path; updated.
- Wrote the binary payload to the ESP32 without first sending the
  `HIL_SYNC\n` line that the firmware requires, then tried to parse the
  firmware's `HIL_RES:<class>:<conf>:<ms>` reply as JSON. Both halves
  fixed: sync line sent, both reply formats accepted.
- Same training-script `sys.path` fix.

### 12. `export_tflite.py` — broken direct path
- Same ghost-folder bug.
- The "direct PyTorch -> TFLite" function tried to import
  `torch._export.tflite`, which has never existed. Always silently returned
  None; broken code path now stubbed.
- `generate_c_header()` assumed `tflite_path` was always a `Path`; the
  caller passes a string in some code paths. Now coerced.

### 13. `model/augmentation.py` — silent dtype upcast in SensorAugmentor
`np.random.normal(...)` returns float64, so augmentations upcast float32
inputs to float64 silently. PyTorch dataloaders then fail with "expected
Float but got Double". Now restores the input dtype after augmentations.

### 14. `train_esp32_v2_expanded_local.py` — sys.path missed the repo root
The script inserted only its own directory (`training/`) into `sys.path`,
so `from model.dataset import UCIHARDataset` failed when the script was run
as `python training/train_esp32_v2_expanded_local.py` from the repo root.
Fixed by also inserting the parent directory.


## Verified working after Round 2

- Every Python file parses without syntax errors.
- The `model` package imports cleanly.
- `SensorFusionESP32` instantiates and forwards correctly with 75,290
  parameters (matches the report).
- `SensorFusionHAR` and `SensorFusionLite` both forward correctly.
- `MaskedSensorModelESP32` reconstructs masked windows correctly.
- `FocalLoss`, `make_balanced_loader`, `augment_batch`, and the
  reservoir-manifold-mixup helper all run on synthetic data.
- `server.py`, `hil_server.py`, `esp32_bridge.py`, `esp32_dataset_test.py`,
  `export_tflite.py`, `generate_cert.py`, and the training script all
  start cleanly with `--help`.
- Training script reaches the dataset loader and fails only with a real
  `FileNotFoundError` for the missing UCI HAR data — not a programming
  bug.
