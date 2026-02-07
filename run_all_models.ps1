# run_all_models.ps1
# ------------------------------------------------------------
# Robust runner for your 4 jobs (QLoRA 2012/2019 + LoRA 2012/2019)
# - Loads HF_TOKEN from .env
# - Activates conda env (robust)
# - Passes args only if each training script supports them (--help scan)
# - Adds eval OOM reducers where supported: --eval_steps 0, --eval_max_len 512, --no_eval
# ------------------------------------------------------------

$ErrorActionPreference = "Stop"

# ============================================================
# CONFIG
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#   .\run_all_models.ps1
# ============================================================

$BASE_DIR   = "C:\CODE ON GITHUB\Medgemma4B_physionet2012-2019"   # <-- change to your Windows path
$CONDA_ENV  = "MedGemma"
$ENV_FILE   = Join-Path $BASE_DIR ".env"

# Training params
$MAX_LEN     = $env:MAX_LEN     ? [int]$env:MAX_LEN     : 1024
$BATCH       = $env:BATCH       ? [int]$env:BATCH       : 1
$GRAD_ACCUM  = $env:GRAD_ACCUM  ? [int]$env:GRAD_ACCUM  : 8
$EPOCHS      = $env:EPOCHS      ? [int]$env:EPOCHS      : 1
$LR          = $env:LR          ? [double]$env:LR       : 2e-4

$EVAL_STEPS   = $env:EVAL_STEPS   ? [int]$env:EVAL_STEPS   : 0
$EVAL_MAX_LEN = $env:EVAL_MAX_LEN ? [int]$env:EVAL_MAX_LEN : 512
$LOG_STEPS    = $env:LOG_STEPS    ? [int]$env:LOG_STEPS    : 10
$NO_EVAL      = $env:NO_EVAL      ? [int]$env:NO_EVAL      : 0

# Helpful VRAM fragmentation settings (mirrors your python env usage)
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128" }
if (-not $env:TOKENIZERS_PARALLELISM)  { $env:TOKENIZERS_PARALLELISM  = "false" }

# ============================================================
# HELPERS
# ============================================================

function Assert-File($Path) {
  if (-not (Test-Path -LiteralPath $Path)) {
    throw "ERROR: missing file: $Path"
  }
}

function Ensure-Dir($Path) {
  New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Import-DotEnv($Path) {
  Assert-File $Path
  Get-Content -LiteralPath $Path | ForEach-Object {
    $line = $_.Trim()
    if (-not $line) { return }
    if ($line.StartsWith("#")) { return }
    if ($line -match '^[A-Za-z_][A-Za-z0-9_]*=') {
      $key = $line.Split("=",2)[0]
      $val = $line.Split("=",2)[1]

      # strip quotes
      if (($val.StartsWith('"') -and $val.EndsWith('"')) -or ($val.StartsWith("'") -and $val.EndsWith("'"))) {
        $val = $val.Substring(1, $val.Length-2)
      }
      Set-Item -Path "Env:$key" -Value $val
    }
  }
}

function Get-CondaBase {
  try {
    $base = (conda info --base 2>$null)
    if ($base) { return $base.Trim() }
  } catch {}
  return $null
}

function Enable-Conda {
  $base = Get-CondaBase
  if (-not $base) {
    throw "ERROR: conda not found. Install Miniconda/Anaconda and ensure conda is on PATH."
  }

  $hook = Join-Path $base "shell\condabin\conda-hook.ps1"
  if (-not (Test-Path -LiteralPath $hook)) {
    throw "ERROR: conda-hook.ps1 not found at: $hook"
  }

  # Load conda function into this shell
  & $hook | Out-Null
}

function Script-SupportsFlag($Script, $Flag) {
  try {
    $help = & python $Script --help 2>&1 | Out-String
    return ($help -match [regex]::Escape($Flag))
  } catch {
    return $false
  }
}

function Run-Train($Script, $Train, $Val, $Out, $Label) {
  Assert-File $Script
  Assert-File $Train
  Assert-File $Val
  Ensure-Dir $Out

  Write-Host ""
  Write-Host "=============================="
  Write-Host "RUN:    $Label"
  Write-Host "Script: $Script"
  Write-Host "Train:  $Train"
  Write-Host "Val:    $Val"
  Write-Host "Out:    $Out"
  Write-Host "=============================="

  # Build args; only add if supported
  $argsList = New-Object System.Collections.Generic.List[string]
  $argsList.Add("--train_jsonl"); $argsList.Add($Train)
  $argsList.Add("--val_jsonl");   $argsList.Add($Val)
  $argsList.Add("--out_dir");     $argsList.Add($Out)
  $argsList.Add("--max_len");     $argsList.Add("$MAX_LEN")
  $argsList.Add("--batch");       $argsList.Add("$BATCH")
  $argsList.Add("--grad_accum");  $argsList.Add("$GRAD_ACCUM")
  $argsList.Add("--epochs");      $argsList.Add("$EPOCHS")

  if (Script-SupportsFlag $Script "--lr") {
    $argsList.Add("--lr"); $argsList.Add("$LR")
  }

  if (Script-SupportsFlag $Script "--log_steps") {
    $argsList.Add("--log_steps"); $argsList.Add("$LOG_STEPS")
  }

  if (Script-SupportsFlag $Script "--hf_token") {
    $argsList.Add("--hf_token"); $argsList.Add($env:HF_TOKEN)
  }

  if (Script-SupportsFlag $Script "--eval_steps") {
    $argsList.Add("--eval_steps"); $argsList.Add("$EVAL_STEPS")
  }

  if (Script-SupportsFlag $Script "--eval_max_len") {
    $argsList.Add("--eval_max_len"); $argsList.Add("$EVAL_MAX_LEN")
  }

  if ($NO_EVAL -eq 1 -and (Script-SupportsFlag $Script "--no_eval")) {
    $argsList.Add("--no_eval")
  }

  $logFile = Join-Path $Out ("run_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
  Write-Host "[info] logging -> $logFile"

  # Run and tee output to log
  $pinfo = New-Object System.Diagnostics.ProcessStartInfo
  $pinfo.FileName = "python"
  $pinfo.Arguments = ('"{0}" {1}' -f $Script, ($argsList | ForEach-Object { if ($_ -match '\s') { '"{0}"' -f $_ } else { $_ } }) -join " ")
  $pinfo.RedirectStandardOutput = $true
  $pinfo.RedirectStandardError  = $true
  $pinfo.UseShellExecute        = $false

  $p = New-Object System.Diagnostics.Process
  $p.StartInfo = $pinfo
  [void]$p.Start()

  $outStream = New-Object System.IO.StreamWriter($logFile, $false)
  try {
    while (-not $p.HasExited) {
      Start-Sleep -Milliseconds 100
      while (-not $p.StandardOutput.EndOfStream) {
        $line = $p.StandardOutput.ReadLine()
        if ($null -ne $line) { Write-Host $line; $outStream.WriteLine($line) }
      }
      while (-not $p.StandardError.EndOfStream) {
        $line = $p.StandardError.ReadLine()
        if ($null -ne $line) { Write-Host $line; $outStream.WriteLine($line) }
      }
    }

    # flush remaining
    while (-not $p.StandardOutput.EndOfStream) {
      $line = $p.StandardOutput.ReadLine()
      if ($null -ne $line) { Write-Host $line; $outStream.WriteLine($line) }
    }
    while (-not $p.StandardError.EndOfStream) {
      $line = $p.StandardError.ReadLine()
      if ($null -ne $line) { Write-Host $line; $outStream.WriteLine($line) }
    }

    if ($p.ExitCode -ne 0) {
      throw "Training failed with exit code $($p.ExitCode). See log: $logFile"
    }
  }
  finally {
    $outStream.Flush()
    $outStream.Close()
    $p.Dispose()
  }
}

# ============================================================
# MAIN
# ============================================================

Import-DotEnv $ENV_FILE
if (-not $env:HF_TOKEN) { throw "ERROR: HF_TOKEN missing in $ENV_FILE (add HF_TOKEN=hf_...)" }

Enable-Conda
conda activate $CONDA_ENV | Out-Null

Write-Host "Conda env active: $(& python --version 2>&1)"
Write-Host ("HF_TOKEN loaded: {0}********" -f $env:HF_TOKEN.Substring(0, [Math]::Min(8, $env:HF_TOKEN.Length)))

# Script paths
$QLORA_SCRIPT_2012 = Join-Path $BASE_DIR "physionet2012\model_scripts\train_Qlora.py"
$LORA_SCRIPT_2012  = Join-Path $BASE_DIR "physionet2012\model_scripts\train_lora.py"
$QLORA_SCRIPT_2019 = Join-Path $BASE_DIR "physionet2019\model_scripts\train_Qlora.py"
$LORA_SCRIPT_2019  = Join-Path $BASE_DIR "physionet2019\model_scripts\train_lora.py"

# Data + outputs
$TRAIN_JSONL_Q1 = Join-Path $BASE_DIR "physionet2012\dataset\train.jsonl"
$VAL_JSONL_Q1   = Join-Path $BASE_DIR "physionet2012\dataset\val.jsonl"
$OUT_DIR_Q1     = Join-Path $BASE_DIR "physionet2012\results\Qlora"

$TRAIN_JSONL_Q2 = Join-Path $BASE_DIR "physionet2019\dataset\train.jsonl"
$VAL_JSONL_Q2   = Join-Path $BASE_DIR "physionet2019\dataset\val.jsonl"
$OUT_DIR_Q2     = Join-Path $BASE_DIR "physionet2019\results\Qlora"

$TRAIN_JSONL_L1 = Join-Path $BASE_DIR "physionet2012\dataset\train.jsonl"
$VAL_JSONL_L1   = Join-Path $BASE_DIR "physionet2012\dataset\val.jsonl"
$OUT_DIR_L1     = Join-Path $BASE_DIR "physionet2012\results\lora"

$TRAIN_JSONL_L2 = Join-Path $BASE_DIR "physionet2019\dataset\train.jsonl"
$VAL_JSONL_L2   = Join-Path $BASE_DIR "physionet2019\dataset\val.jsonl"
$OUT_DIR_L2     = Join-Path $BASE_DIR "physionet2019\results\lora"

# Run order
Run-Train $QLORA_SCRIPT_2012 $TRAIN_JSONL_Q1 $VAL_JSONL_Q1 $OUT_DIR_Q1 "QLoRA 2012"
Run-Train $QLORA_SCRIPT_2019 $TRAIN_JSONL_Q2 $VAL_JSONL_Q2 $OUT_DIR_Q2 "QLoRA 2019"
Run-Train $LORA_SCRIPT_2012  $TRAIN_JSONL_L1 $VAL_JSONL_L1 $OUT_DIR_L1 "LoRA 2012"
Run-Train $LORA_SCRIPT_2019  $TRAIN_JSONL_L2 $VAL_JSONL_L2 $OUT_DIR_L2 "LoRA 2019"

Write-Host ""
Write-Host "✅ ALL TRAINING COMPLETE"
