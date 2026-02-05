#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CONFIG
# ============================================================

BASE_DIR="/home/heystekgrobler/Documents/Medgemma4B_physionet2012-2019"
CONDA_ENV="MedGemma"
ENV_FILE="$BASE_DIR/.env"

# Optional fallback if conda not on PATH:
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"

# Training params
MAX_LEN=1024
BATCH=1
GRAD_ACCUM=8
EPOCHS=1

# ============================================================
# LOAD .env (HF_TOKEN) - safe parser (no process substitution)
# ============================================================

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env not found at: $ENV_FILE"
  echo "Create it with: HF_TOKEN=hf_..."
  exit 1
fi

# Load only simple KEY=VALUE lines, ignore comments/blanks
while IFS= read -r line || [[ -n "$line" ]]; do
  # trim
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"

  [[ -z "$line" ]] && continue
  [[ "${line:0:1}" == "#" ]] && continue

  if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
    key="${line%%=*}"
    val="${line#*=}"

    # strip surrounding quotes if present
    if [[ ( "${val:0:1}" == "\"" && "${val: -1}" == "\"" ) || ( "${val:0:1}" == "'" && "${val: -1}" == "'" ) ]]; then
      val="${val:1:${#val}-2}"
    fi

    export "$key=$val"
  fi
done < "$ENV_FILE"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN missing. Add HF_TOKEN=... to $ENV_FILE"
  exit 1
fi

# ============================================================
# ACTIVATE CONDA (robust)
# ============================================================

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE_DETECTED="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$CONDA_BASE_DETECTED" && -f "$CONDA_BASE_DETECTED/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE_DETECTED/etc/profile.d/conda.sh"
  else
    # fallback to default
    if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
      echo "ERROR: Could not find conda.sh via conda info --base or CONDA_BASE."
      echo "Set CONDA_BASE to your conda install folder."
      exit 1
    fi
  fi
else
  # conda not on PATH, try CONDA_BASE directly
  if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
  else
    echo "ERROR: conda not found on PATH and conda.sh not found at: $CONDA_BASE/etc/profile.d/conda.sh"
    echo 'Fix: add conda to PATH or export CONDA_BASE="$HOME/anaconda3" (or your install path).'
    exit 1
  fi
fi

conda activate "$CONDA_ENV"

echo "Conda env active: $(python --version 2>&1)"
echo "HF_TOKEN loaded: ${HF_TOKEN:0:8}********"

# ============================================================
# REAL SCRIPT PATHS
# ============================================================

QLORA_SCRIPT_2012="$BASE_DIR/physionet2012/model_scripts/train_Qlora.py"
LORA_SCRIPT_2012="$BASE_DIR/physionet2012/model_scripts/train_lora.py"

QLORA_SCRIPT_2019="$BASE_DIR/physionet2019/model_scripts/train_Qlora.py"
LORA_SCRIPT_2019="$BASE_DIR/physionet2019/model_scripts/train_lora.py"

# ============================================================
# DATASETS + OUTPUTS
# ============================================================

TRAIN_JSONL_Q1="$BASE_DIR/physionet2012/dataset/train.jsonl"
VAL_JSONL_Q1="$BASE_DIR/physionet2012/dataset/val.jsonl"
OUT_DIR_Q1="$BASE_DIR/physionet2012/results/Qlora"

TRAIN_JSONL_Q2="$BASE_DIR/physionet2019/dataset/train.jsonl"
VAL_JSONL_Q2="$BASE_DIR/physionet2019/dataset/val.jsonl"
OUT_DIR_Q2="$BASE_DIR/physionet2019/results/Qlora"

TRAIN_JSONL_L1="$BASE_DIR/physionet2012/dataset/train.jsonl"
VAL_JSONL_L1="$BASE_DIR/physionet2012/dataset/val.jsonl"
OUT_DIR_L1="$BASE_DIR/physionet2012/results/lora"

TRAIN_JSONL_L2="$BASE_DIR/physionet2019/dataset/train.jsonl"
VAL_JSONL_L2="$BASE_DIR/physionet2019/dataset/val.jsonl"
OUT_DIR_L2="$BASE_DIR/physionet2019/results/lora"

# ============================================================
# HELPERS
# ============================================================

assert_file() {
  [[ -f "$1" ]] || { echo "ERROR: missing file: $1"; exit 1; }
}

ensure_dir() {
  mkdir -p "$1"
}

run_train() {
  local script="$1" train="$2" val="$3" out="$4" label="$5"

  assert_file "$script"
  assert_file "$train"
  assert_file "$val"
  ensure_dir "$out"

  echo
  echo "=============================="
  echo "RUN:   $label"
  echo "Script $script"
  echo "Out    $out"
  echo "=============================="

  python "$script" \
    --train_jsonl "$train" \
    --val_jsonl   "$val" \
    --out_dir     "$out" \
    --max_len     "$MAX_LEN" \
    --batch       "$BATCH" \
    --grad_accum  "$GRAD_ACCUM" \
    --epochs      "$EPOCHS"
}

# ============================================================
# RUN IN ORDER (QLoRA x2, then LoRA x2)
# ============================================================

run_train "$QLORA_SCRIPT_2012" "$TRAIN_JSONL_Q1" "$VAL_JSONL_Q1" "$OUT_DIR_Q1" "QLoRA 2012"
run_train "$QLORA_SCRIPT_2019" "$TRAIN_JSONL_Q2" "$VAL_JSONL_Q2" "$OUT_DIR_Q2" "QLoRA 2019"
run_train "$LORA_SCRIPT_2012"  "$TRAIN_JSONL_L1" "$VAL_JSONL_L1" "$OUT_DIR_L1" "LoRA 2012"
run_train "$LORA_SCRIPT_2019"  "$TRAIN_JSONL_L2" "$VAL_JSONL_L2" "$OUT_DIR_L2" "LoRA 2019"

echo
echo "✅ ALL TRAINING COMPLETE"
