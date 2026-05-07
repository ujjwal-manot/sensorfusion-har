# flash_esp32.ps1
# ================
# Full pipeline: Python 3.12 venv → TFLite export → arduino-cli compile → flash
#
# Usage:
#   .\flash_esp32.ps1                  # auto-detect COM port
#   .\flash_esp32.ps1 -Port COM5       # specify port
#   .\flash_esp32.ps1 -SkipConvert    # skip TFLite conversion (use existing model_data.h)
#
param(
    [string]$Port = "",
    [switch]$SkipConvert
)

$ErrorActionPreference = "Stop"
$ROOT  = Split-Path -Parent $PSScriptRoot   # sensorfusion-har root
$DEPLOY = "$ROOT\esp32_deploy"
$SKETCH = "$DEPLOY\esp32_serial_inference"
$CKPT   = "$ROOT\RealLife\checkpoints\best_sensorfusion_esp32_v2_useful11_pocket_v3.pt"
$HEADER = "$SKETCH\model_data.h"
$TFLITE = "$DEPLOY\pocket_v3.tflite"

Write-Host "`n=== ESP32 Flash Pipeline ===" -ForegroundColor Cyan

# ── Step 1: TFLite conversion ────────────────────────────────────────────────
if (-not $SkipConvert) {
    Write-Host "`n[1/4] TFLite conversion" -ForegroundColor Yellow

    # Find Python 3.12
    $py312 = $null
    foreach ($candidate in @("py", "python3.12", "python")) {
        try {
            $ver = & $candidate -c "import sys; print(sys.version_info[:2])" 2>$null
            if ($ver -match "3, 12") { $py312 = $candidate; break }
        } catch {}
    }
    if (-not $py312) {
        Write-Host "  Python 3.12 not found — trying winget install..." -ForegroundColor Yellow
        winget install Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
        $py312 = "py -3.12"
    }
    Write-Host "  Using: $py312"

    # Create/activate venv
    $VENV = "$ROOT\.venv312"
    if (-not (Test-Path "$VENV\Scripts\pip.exe")) {
        Write-Host "  Creating Python 3.12 venv..."
        & $py312 -m venv $VENV
    }
    $pip = "$VENV\Scripts\pip.exe"
    $python = "$VENV\Scripts\python.exe"

    # Install deps (skip if already present)
    $installed = & $pip list 2>$null
    if ($installed -notmatch "tensorflow") {
        Write-Host "  Installing tensorflow (this takes a few minutes)..."
        & $pip install --quiet tensorflow onnx onnx-tf
    }
    if ($installed -notmatch "onnx2tf") {
        Write-Host "  Installing onnx2tf..."
        & $pip install --quiet onnx2tf
    }

    Write-Host "  Running export_tflite.py..."
    & $python "$ROOT\export_tflite.py" `
        --checkpoint $CKPT `
        --output $TFLITE `
        --header $HEADER `
        --validate
    if ($LASTEXITCODE -ne 0) { throw "TFLite export failed" }
    Write-Host "  TFLite: $TFLITE" -ForegroundColor Green
    Write-Host "  Header: $HEADER" -ForegroundColor Green
} else {
    Write-Host "`n[1/4] Skipping TFLite conversion (using existing $HEADER)"
    if (-not (Test-Path $HEADER)) { throw "model_data.h not found at $HEADER" }
}

# ── Step 2: Detect ESP32 port ─────────────────────────────────────────────────
Write-Host "`n[2/4] Detecting ESP32 port" -ForegroundColor Yellow
if (-not $Port) {
    $candidates = Get-WMIObject Win32_SerialPort | Where-Object {
        $_.Description -match "CP210|CH340|FTDI|USB Serial|Silicon Labs|ESP32"
    }
    if ($candidates.Count -eq 0) {
        $candidates = Get-WMIObject Win32_SerialPort | Where-Object {
            $_.Description -notmatch "Bluetooth"
        }
    }
    if ($candidates.Count -eq 1) {
        $Port = $candidates[0].DeviceID
        Write-Host "  Auto-detected: $Port ($($candidates[0].Description))" -ForegroundColor Green
    } elseif ($candidates.Count -gt 1) {
        Write-Host "  Multiple ports found:"
        $candidates | ForEach-Object { Write-Host "    $($_.DeviceID)  $($_.Description)" }
        $Port = Read-Host "  Enter COM port (e.g. COM5)"
    } else {
        throw "No non-Bluetooth serial ports found. Connect the ESP32 first."
    }
}
Write-Host "  Port: $Port"

# ── Step 3: arduino-cli setup + compile ──────────────────────────────────────
Write-Host "`n[3/4] Compiling with arduino-cli" -ForegroundColor Yellow
$cli = Get-Command arduino-cli -ErrorAction SilentlyContinue
if (-not $cli) {
    # Try common install paths
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Arduino CLI\arduino-cli.exe",
        "$env:ProgramFiles\Arduino CLI\arduino-cli.exe",
        "C:\Program Files\Arduino CLI\arduino-cli.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $cli = $c; break }
    }
    if (-not $cli) { throw "arduino-cli not found. Install it or add to PATH." }
}

Write-Host "  arduino-cli: $cli"
Write-Host "  Updating board index..."
& $cli core update-index 2>$null
Write-Host "  Installing esp32 core..."
& $cli core install esp32:esp32 2>$null
Write-Host "  Installing TensorFlowLite_ESP32 library..."
& $cli lib install "TensorFlowLite_ESP32" 2>$null

Write-Host "  Compiling sketch..."
& $cli compile --fqbn esp32:esp32:esp32 `
    --build-property "build.partitions=huge_app" `
    $SKETCH
if ($LASTEXITCODE -ne 0) { throw "Compilation failed" }
Write-Host "  Compiled OK" -ForegroundColor Green

# ── Step 4: Flash ─────────────────────────────────────────────────────────────
Write-Host "`n[4/4] Flashing ESP32 on $Port" -ForegroundColor Yellow
& $cli upload --fqbn esp32:esp32:esp32 -p $Port $SKETCH
if ($LASTEXITCODE -ne 0) { throw "Upload failed" }
Write-Host "  Flashed OK" -ForegroundColor Green

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host "`n=== Done! ===" -ForegroundColor Cyan
Write-Host "To activate ESP32 inference, restart the server with:"
Write-Host "  `$env:ESP32_PORT='$Port'; python RealLife\server.py" -ForegroundColor White
Write-Host "The server will send windows to ESP32 and display results on the dashboard."
