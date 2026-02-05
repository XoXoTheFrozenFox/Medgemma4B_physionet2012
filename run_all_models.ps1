$ErrorActionPreference = "Stop"

# ============================================================
# CONFIG
# ============================================================
$PROJECT_ROOT = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019"
$CONDA_ENV    = "MedGemma"
$ENV_FILE     = Join-Path $PROJECT_ROOT ".env"

# ============================================================
# LOAD .env (HF_TOKEN)
# ============================================================
function Import-DotEnv {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        throw ".env file not found: $Path`nCreate it with: HF_TOKEN=hf_..."
    }

    Get-Content -LiteralPath $Path | ForEach-Object {
        $line = $_.Trim()

        # ignore blanks + comments
        if ($line -eq "" -or $line.StartsWith("#")) { return }

        # parse KEY=VALUE
        if ($line -match '^\s*([^=]+?)\s*=\s*(.*)\s*$') {
            $key = $matches[1].Trim()
            $val = $matches[2].Trim()

            # strip quotes
            if (($val.StartsWith('"') -and $val.EndsWith('"')) -or ($val.StartsWith("'") -and $val.EndsWith("'"))) {
                $val = $val.Substring(1, $val.Length - 2)
            }

            [System.Environment]::SetEnvironmentVariable($key, $val, "Process")
        }
    }
}

Import-DotEnv -Path $ENV_FILE

if (-not $env:HF_TOKEN -or $env:HF_TOKEN.Trim() -eq "") {
    throw "HF_TOKEN missing. Add HF_TOKEN=... to $ENV_FILE"
}

# ============================================================
# ACTIVATE CONDA (AUTO-DETECT, NO HARD PATHS)
# ============================================================
function Get-CondaHook {
    # If conda is available, use its base
    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCmd) {
        try {
            $base = (conda info --base).Trim()

            $hook = Join-Path $base "shell\condabin\conda-hook.ps1"
            if (Test-Path -LiteralPath $hook) { return $hook }

            $hook2 = Join-Path $base "condabin\conda-hook.ps1"
            if (Test-Path -LiteralPath $hook2) { return $hook2 }
        } catch { }
    }

    # Common fallback locations
    $candidates = @(
        "$env:USERPROFILE\miniconda3",
        "$env:USERPROFILE\Miniconda3",
        "$env:USERPROFILE\anaconda3",
        "$env:USERPROFILE\Anaconda3",
        "C:\ProgramData\miniconda3",
        "C:\ProgramData\anaconda3"
    )

    foreach ($base in $candidates) {
        $hook = Join-Path $base "shell\condabin\conda-hook.ps1"
        if (Test-Path -LiteralPath $hook) { return $hook }

        $hook2 = Join-Path $base "condabin\conda-hook.ps1"
        if (Test-Path -LiteralPath $hook2) { return $hook2 }
    }

    return $null
}

$CONDA_HOOK = Get-CondaHook
if (-not $CONDA_HOOK) {
    throw @"
Could not find conda-hook.ps1.

Fix:
1) Open Anaconda/Miniconda Prompt and run: where conda
2) In PowerShell run: where conda
3) Ensure conda is installed and on PATH.
"@
}

Write-Host "Using conda hook: $CONDA_HOOK"
. $CONDA_HOOK

Write-Host "Activating conda env: $CONDA_ENV"
conda activate $CONDA_ENV

Write-Host "Python in env:"
python --version

# ============================================================
# REAL SCRIPT PATHS
# ============================================================
$QLORA_SCRIPT_2012 = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\model_scripts\train_Qlora.py"
$LORA_SCRIPT_2012  = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\model_scripts\train_lora.py"

$QLORA_SCRIPT_2019 = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\model_scripts\train_Qlora.py"
$LORA_SCRIPT_2019  = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\model_scripts\train_lora.py"

# ============================================================
# DATASETS + OUTPUTS
# ============================================================
$TRAIN_JSONL_Q1 = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\dataset\train.jsonl"
$VAL_JSONL_Q1   = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\dataset\val.jsonl"
$OUT_DIR_Q1     = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\results\Qlora"

$TRAIN_JSONL_Q2 = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\dataset\train.jsonl"
$VAL_JSONL_Q2   = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\dataset\val.jsonl"
$OUT_DIR_Q2     = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\results\Qlora"

$TRAIN_JSONL_L1 = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\dataset\train.jsonl"
$VAL_JSONL_L1   = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\dataset\val.jsonl"
$OUT_DIR_L1     = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2012\results\lora"

$TRAIN_JSONL_L2 = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\dataset\train.jsonl"
$VAL_JSONL_L2   = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\dataset\val.jsonl"
$OUT_DIR_L2     = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019\physionet2019\results\lora"

# ============================================================
# TRAINING PARAMS
# ============================================================
$MAX_LEN    = 1024
$BATCH      = 1
$GRAD_ACCUM = 8
$EPOCHS     = 1

# ============================================================
# HELPERS
# ============================================================
function Assert-Path([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing: $Path" }
}

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Run-Train {
    param(
        [string]$Script,
        [string]$Train,
        [string]$Val,
        [string]$Out,
        [string]$Label
    )

    Assert-Path $Script
    Assert-Path $Train
    Assert-Path $Val
    Ensure-Dir  $Out

    Write-Host ""
    Write-Host "=============================="
    Write-Host "RUN: $Label"
    Write-Host "Script: $Script"
    Write-Host "Out: $Out"
    Write-Host "=============================="

    & python $Script `
        --train_jsonl $Train `
        --val_jsonl   $Val `
        --out_dir     $Out `
        --max_len     $MAX_LEN `
        --batch       $BATCH `
        --grad_accum  $GRAD_ACCUM `
        --epochs      $EPOCHS

    if ($LASTEXITCODE -ne 0) {
        throw "FAILED: $Label (exit code $LASTEXITCODE)"
    }
}

# ============================================================
# RUN IN ORDER
# ============================================================
Run-Train $QLORA_SCRIPT_2012 $TRAIN_JSONL_Q1 $VAL_JSONL_Q1 $OUT_DIR_Q1 "QLoRA 2012"
Run-Train $QLORA_SCRIPT_2019 $TRAIN_JSONL_Q2 $VAL_JSONL_Q2 $OUT_DIR_Q2 "QLoRA 2019"
Run-Train $LORA_SCRIPT_2012  $TRAIN_JSONL_L1 $VAL_JSONL_L1 $OUT_DIR_L1 "LoRA 2012"
Run-Train $LORA_SCRIPT_2019  $TRAIN_JSONL_L2 $VAL_JSONL_L2 $OUT_DIR_L2 "LoRA 2019"

Write-Host ""
Write-Host "✅ ALL TRAINING COMPLETE"
