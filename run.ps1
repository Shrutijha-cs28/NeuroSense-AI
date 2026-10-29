<#
.SYNOPSIS
    NeuroSenseAI automation script (setup, train-all, test, app).
.DESCRIPTION
    RESEARCH USE ONLY. Not a medical device. No clinical claims.
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("setup", "train-all", "test", "app")]
    [string]$Command = "test"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python.exe"
}

switch ($Command) {
    "setup" {
        Write-Host "==> Initializing Python virtual environment and installing dependencies..." -ForegroundColor Cyan
        if (-not (Test-Path ".venv")) {
            py -3.13 -m venv .venv
        }
        & $PythonExe -m pip install --upgrade pip
        & $PythonExe -m pip install -r requirements.txt
        Write-Host "==> Setup completed successfully." -ForegroundColor Green
    }
    "train-all" {
        Write-Host "==> Running Alzheimer's handwriting training (nested CV + SHAP)..." -ForegroundColor Cyan
        & $PythonExe -m src.alzheimer.train
        
        Write-Host "==> Running Parkinson's voice training (GroupKFold + SHAP)..." -ForegroundColor Cyan
        & $PythonExe -m src.parkinsons.train
        
        Write-Host "==> Running Parkinson's gait training (StratifiedGroupKFold + SHAP)..." -ForegroundColor Cyan
        & $PythonExe -m src.parkinsons.train_gait

        Write-Host "==> Generating synthetic matched fusion datasets..." -ForegroundColor Cyan
        & $PythonExe scripts/generate_synthetic_fusion_data.py

        Write-Host "==> Running synthetic multi-modal fusion plumbing benchmarks..." -ForegroundColor Cyan
        & $PythonExe -m src.common.train_fusion --disease alzheimer --data data/synthetic/alzheimer_matched_synthetic.csv
        & $PythonExe -m src.common.train_fusion --disease parkinsons --data data/synthetic/parkinsons_matched_synthetic.csv
        
        Write-Host "==> All training and report generation completed successfully." -ForegroundColor Green
    }
    "test" {
        Write-Host "==> Running complete pytest suite (schema, leakage, models, fusion)..." -ForegroundColor Cyan
        & $PythonExe -m pytest tests -v
    }
    "app" {
        Write-Host "==> Launching Streamlit Research Dashboard..." -ForegroundColor Cyan
        & $PythonExe -m streamlit run app.py
    }
}
