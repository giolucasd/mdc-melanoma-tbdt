# Train Baseline Model - PowerShell Script
# Usage: .\train_baseline.ps1 [experiment_name]

param(
    [string]$ExperimentName = "baseline_v1"
)

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Training Melanoma Classification Model" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Experiment: $ExperimentName" -ForegroundColor Yellow
Write-Host "Config: configs/baseline.yaml" -ForegroundColor Yellow
Write-Host "Output: outputs/$ExperimentName/" -ForegroundColor Yellow
Write-Host ""

# Check if data exists
if (-not (Test-Path "data\train.csv")) {
    Write-Host "ERROR: Training data not found!" -ForegroundColor Red
    Write-Host "Please run: .\scripts\download_data.sh" -ForegroundColor Yellow
    exit 1
}

# Check if config exists
if (-not (Test-Path "configs\baseline.yaml")) {
    Write-Host "ERROR: Config file not found!" -ForegroundColor Red
    exit 1
}

# Run training
Write-Host "Starting training..." -ForegroundColor Green
python -m scripts.train --config configs\baseline.yaml --output-suffix $ExperimentName

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "Training completed successfully!" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Results saved to: outputs\$ExperimentName\" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To analyze the model:" -ForegroundColor Yellow
    Write-Host "  python main.py" -ForegroundColor White
    Write-Host "  Choose option 1, then click 'Load saved model'" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "Training failed with error code: $LASTEXITCODE" -ForegroundColor Red
}
