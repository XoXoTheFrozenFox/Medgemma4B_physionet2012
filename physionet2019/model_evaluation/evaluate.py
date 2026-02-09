#!/usr/bin/env python
# evaluate.py
# ------------------------------------------------------------
# Evaluate MedGemma LoRA / QLoRA adapters on val.jsonl (prompt/target)
#
# Updated for PhysioNet 2019-style tasks (sepsis early warning / binary label),
# while KEEPING backward compatibility with your older "status/drivers/..." schema.
#
# Key features:
# - Loads processor/tokenizer from adapter_dir when available (avoids HF download)
# - Supports LoRA (fp16/bf16) and QLoRA (4-bit NF4) base loading
# - Forces “JSON-only” instruction onto prompts (without editing dataset)
# - Optionally enforces required output keys (helps keep schema stable)
# - Blocks '<unused94>thought' token/sequence where tokenizer supports it
# - Decodes ONLY generated continuation (not prompt echo)
# - Robustly extracts the FIRST valid JSON object (handles repeated JSON blocks)
# - Writes adapter_loaded into outputs + metrics
# - 2019 metrics:
#     * sepsis label classification (accuracy/precision/recall/F1/balanced acc/specificity)
#     * AUROC/AUPRC/Brier when probability-like field is present
#     * optional factors list overlap (F1/Jaccard)
# - Still supports legacy schema metrics (status/drivers/checks/narrative)
#
# FIX (no transformers upgrade needed):
# - Some older transformers versions return an AutoProcessor WITHOUT .tokenizer.
#   -> We now load AutoTokenizer separately (adapter_dir first, else base_model),
#      and we build chat-template inputs using:
#         tokenizer.apply_chat_template (if available) else processor.apply_chat_template
#         else a safe Gemma-style fallback wrapper.
# ------------------------------------------------------------

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from hf_auth import get_hf_token, try_with_token


# ---------------------------
# Dtype / helpers
# ---------------------------
def pick_compute_dtype():
    # For eval: bf16 if supported else fp16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_jsonl(path: str) -> List[dict]:
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def load_val(path: str, max_samples: int, seed: int) -> List[dict]:
    recs = read_jsonl(path)
    rng = np.random.default_rng(seed)
    rng.shuffle(recs)
    return recs[:max_samples]


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            try:
                end_idx = max(i for i, ln in enumerate(lines) if ln.strip() == "```")
                inner = "\n".join(lines[1:end_idx]).strip()
                return inner
            except Exception:
                return t
    return t


def extract_first_json(text: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Robust JSON extractor that returns (obj, blob) for the FIRST complete JSON object.
    Handles cases where model outputs repeated JSON blocks back-to-back.

    - strips code fences
    - finds first '{'
    - scans forward with brace-depth while respecting strings/escapes
    - returns the first balanced {...} that parses as JSON
    """
    text = _strip_code_fences(text)

    start = text.find("{")
    if start == -1:
        return None, None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : i + 1]
                try:
                    return json.loads(blob), blob
                except Exception:
                    return None, blob

    return None, None


def parse_target_field(target: Any) -> Any:
    # target can be: dict, JSON-string, or plain string
    if isinstance(target, dict):
        return target
    if isinstance(target, str):
        t = target.strip()
        if t.startswith("{") and t.endswith("}"):
            try:
                return json.loads(t)
            except Exception:
                return target
        return target
    return target


def safe_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()]
    if isinstance(x, str):
        # comma-separated strings
        return [s.strip() for s in x.split(",") if s.strip()]
    return [str(x)]


def set_f1(pred: List[str], true: List[str]) -> Tuple[float, float, float]:
    p = set(pred or [])
    t = set(true or [])
    if not p and not t:
        return 1.0, 1.0, 1.0
    tp = len(p & t)
    fp = len(p - t)
    fn = len(t - p)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def lcs_len(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[m]


def rouge_l(pred: str, ref: str) -> float:
    p = str(pred or "").split()
    r = str(ref or "").split()
    if not p or not r:
        return 0.0
    lcs = lcs_len(p, r)
    prec = lcs / len(p)
    rec = lcs / len(r)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def jaccard(pred: List[str], true: List[str]) -> float:
    p = set(pred or [])
    t = set(true or [])
    if not p and not t:
        return 1.0
    denom = len(p | t)
    return (len(p & t) / denom) if denom else 0.0


def required_keys_rate(pred: Optional[dict], required: List[str]) -> float:
    if not required:
        return 1.0
    if not isinstance(pred, dict):
        return 0.0
    present = sum(1 for k in required if k in pred)
    return present / max(1, len(required))


# ---------------------------
# PhysioNet 2019 schema helpers
# ---------------------------
DEFAULT_LABEL_KEYS = [
    "sepsis_label",
    "SepsisLabel",
    "sepsis",
    "label",
    "y",
    "target_label",
    "sepsis_within_6h",
    "sepsis_within_3h",
    "sepsis_within_12h",
]

DEFAULT_PROB_KEYS = [
    "prob",
    "probability",
    "risk",
    "score",
    "confidence",
    "sepsis_risk",
    "sepsis_probability",
]

DEFAULT_FACTORS_KEYS = [
    "top_factors",
    "factors",
    "drivers",
    "reasons",
    "evidence",
]


def _to_int_label(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, np.integer)):
        return 1 if int(v) != 0 else 0
    if isinstance(v, (float, np.floating)):
        # treat >=0.5 as positive if it's a float
        return 1 if float(v) >= 0.5 else 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "pos", "positive", "sepsis"):
            return 1
        if s in ("0", "false", "no", "n", "neg", "negative", "no_sepsis", "nonsepsis", "non-sepsis"):
            return 0
        # fallback: try int parse
        try:
            iv = int(float(s))
            return 1 if iv != 0 else 0
        except Exception:
            return None
    return None


def _to_prob(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, np.integer, float, np.floating)):
        x = float(v)
        # normalize common formats
        if x > 1.0 and x <= 100.0:
            x = x / 100.0
        # clamp
        x = max(0.0, min(1.0, x))
        return x
    if isinstance(v, str):
        s = v.strip().lower().replace("%", "")
        try:
            x = float(s)
            if x > 1.0 and x <= 100.0:
                x = x / 100.0
            x = max(0.0, min(1.0, x))
            return x
        except Exception:
            return None
    return None


def extract_label(obj: Any, keys: List[str]) -> Optional[int]:
    if not isinstance(obj, dict):
        return None
    for k in keys:
        if k in obj:
            lab = _to_int_label(obj.get(k))
            if lab is not None:
                return lab
    return None


def extract_prob(obj: Any, keys: List[str]) -> Optional[float]:
    if not isinstance(obj, dict):
        return None
    for k in keys:
        if k in obj:
            p = _to_prob(obj.get(k))
            if p is not None:
                return p
    return None


def extract_factors(obj: Any, keys: List[str]) -> List[str]:
    if not isinstance(obj, dict):
        return []
    for k in keys:
        if k in obj:
            return safe_list(obj.get(k))
    return []


def detect_schema(val_recs: List[dict]) -> str:
    """
    Returns:
      - "legacy_status" if target dict looks like your old schema
      - "p2019_sepsis" otherwise (default)
    """
    for rec in val_recs[:50]:
        tgt = parse_target_field(rec.get("target", ""))
        if isinstance(tgt, dict):
            if "status" in tgt or ("drivers" in tgt and "what_to_check_next" in tgt):
                return "legacy_status"
            # likely 2019 if it has a sepsis-ish key
            if any(k in tgt for k in DEFAULT_LABEL_KEYS):
                return "p2019_sepsis"
    return "p2019_sepsis"


# ---------------------------
# Plotting
# ---------------------------
def plot_training_curves(trainer_state_path: str, out_png: str):
    if not os.path.exists(trainer_state_path):
        return
    try:
        with open(trainer_state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        return

    hist = state.get("log_history", [])

    steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    for item in hist:
        if "loss" in item and "step" in item:
            steps.append(item["step"])
            train_loss.append(item["loss"])
        if "eval_loss" in item and "step" in item:
            eval_steps.append(item["step"])
            eval_loss.append(item["eval_loss"])

    if not steps:
        return

    plt.figure()
    plt.plot(steps, train_loss, label="train_loss")
    if eval_steps:
        plt.plot(eval_steps, eval_loss, label="eval_loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training / Eval Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_confusion(cm: np.ndarray, labels: List[str], out_png: str, title: str):
    cm = np.asarray(cm)

    n = max(1, len(labels))
    fig_w = min(14.0, max(6.0, 0.8 * n))
    fig_h = min(12.0, max(5.5, 0.7 * n))

    plt.figure(figsize=(fig_w, fig_h))

    vmax = float(cm.max()) if cm.size else 1.0
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=vmax)

    plt.title(title)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.xlabel("pred")
    plt.ylabel("true")

    thresh = vmax * 0.55 if vmax > 0 else 0.0
    font_size = 10 if n <= 8 else 8 if n <= 14 else 6

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt_color = "white" if val >= thresh else "black"
            plt.text(j, i, f"{int(val)}", ha="center", va="center", color=txt_color, fontsize=font_size)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_bar(metrics: Dict[str, float], keys: List[str], out_png: str, title: str):
    plt.figure()
    vals = [metrics.get(k, 0.0) for k in keys]
    plt.bar(range(len(keys)), vals)
    plt.xticks(range(len(keys)), keys, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


# ---------------------------
# Chat template input builder (MedGemma text-only)
# ---------------------------
def _messages_text_only(prompt: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def _maybe_get_chat_applier(tokenizer, processor):
    """
    Returns an object that has apply_chat_template:
      prefer tokenizer (newer best practice), else processor, else None.
    """
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer
    if processor is not None and hasattr(processor, "apply_chat_template"):
        return processor
    return None


def _gemma_fallback_wrap(prompt: str) -> str:
    """
    Fallback if apply_chat_template is missing in old transformers.
    Gemma IT commonly uses <start_of_turn>/<end_of_turn> wrappers.
    If your local tokenizer doesn't know these tokens, it'll just tokenize as text;
    still usable as a last-resort to keep evaluation running.
    """
    p = prompt.rstrip()
    return f"<start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n"


def _apply_chat(processor, tokenizer, prompt: str, max_len: int):
    """
    Returns dict with torch tensors on CPU: input_ids, attention_mask (+ optional token_type_ids).
    Compatible with older transformers:
      - tries apply_chat_template with various signatures
      - falls back to manual wrapper + tokenizer() if needed
    """
    if tokenizer is None:
        raise RuntimeError("Tokenizer is required but was not loaded.")

    msgs = _messages_text_only(prompt)
    applier = _maybe_get_chat_applier(tokenizer, processor)

    # Helper: normalize to 2D torch tensors on CPU
    def _to_2d_tensors(enc: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "input_ids" not in enc:
            raise RuntimeError("Encoding has no input_ids.")
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask", None)
        ttype = enc.get("token_type_ids", None)

        # already tensors?
        if torch.is_tensor(input_ids):
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attn is None:
                attn = torch.ones_like(input_ids, dtype=torch.long)
            elif torch.is_tensor(attn) and attn.dim() == 1:
                attn = attn.unsqueeze(0)
            outb = {"input_ids": input_ids.to(torch.long), "attention_mask": attn.to(torch.long)}
            if ttype is not None:
                if torch.is_tensor(ttype) and ttype.dim() == 1:
                    ttype = ttype.unsqueeze(0)
                if torch.is_tensor(ttype):
                    outb["token_type_ids"] = ttype.to(torch.long)
            return outb

        # lists (maybe nested)
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        if attn is not None and isinstance(attn, list) and attn and isinstance(attn[0], list):
            attn = attn[0]
        if ttype is not None and isinstance(ttype, list) and ttype and isinstance(ttype[0], list):
            ttype = ttype[0]

        if attn is None:
            attn = [1] * len(input_ids)

        outb = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attn], dtype=torch.long),
        }
        if ttype is not None:
            outb["token_type_ids"] = torch.tensor([ttype], dtype=torch.long)
        return outb

    # 1) Try apply_chat_template if present
    if applier is not None:
        # Attempt A: tokenize=True, return_dict=True (newer)
        try:
            out = applier.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                truncation=True,
                max_length=max_len,
            )
            return _to_2d_tensors(out)
        except TypeError:
            pass
        except Exception:
            pass

        # Attempt B: tokenize=True, return_tensors="pt" (some versions)
        try:
            out = applier.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
            )
            # out might be a BatchEncoding; dict() is safe
            if hasattr(out, "data"):
                out = out.data
            return _to_2d_tensors(out)
        except TypeError:
            pass
        except Exception:
            pass

        # Attempt C: get formatted text then tokenize normally
        try:
            formatted = applier.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=False,
            )
            enc = tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
            )
            if hasattr(enc, "data"):
                enc = enc.data
            return _to_2d_tensors(enc)
        except Exception:
            pass

    # 2) Final fallback: manual wrapper + tokenizer()
    formatted = _gemma_fallback_wrap(prompt)
    enc = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    )
    if hasattr(enc, "data"):
        enc = enc.data
    return _to_2d_tensors(enc)


# ---------------------------
# Run spec parsing
# ---------------------------
@dataclass
class RunSpec:
    name: str
    adapter_dir: str
    mode: str  # "lora" or "qlora"


def parse_run(s: str) -> RunSpec:
    # format: NAME|ADAPTER_DIR|MODE
    parts = s.split("|")
    if len(parts) != 3:
        raise ValueError("Run spec must be: NAME|ADAPTER_DIR|MODE   where MODE is lora or qlora")
    name, adapter_dir, mode = parts[0].strip(), parts[1].strip(), parts[2].strip().lower()
    if mode not in ("lora", "qlora"):
        raise ValueError("MODE must be 'lora' or 'qlora'")
    return RunSpec(name=name, adapter_dir=adapter_dir, mode=mode)


# ---------------------------
# Model loader for LoRA vs QLoRA
# ---------------------------
def _load_processor(base_model: str, adapter_dir: str, token: str):
    # Prefer adapter_dir if it contains processor files (keeps special tokens consistent)
    try:
        return AutoProcessor.from_pretrained(adapter_dir)
    except Exception:
        return try_with_token(AutoProcessor.from_pretrained, base_model, token=token)


def _load_tokenizer(base_model: str, adapter_dir: str, token: str):
    # Prefer adapter_dir if tokenizer files exist there (even if processor lacks tokenizer)
    try:
        return AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    except Exception:
        return try_with_token(AutoTokenizer.from_pretrained, base_model, token=token, use_fast=True)


def load_model_with_adapter(
    base_model: str,
    adapter_dir: str,
    mode: str,
    token: str,
    compute_dtype,
):
    processor = _load_processor(base_model=base_model, adapter_dir=adapter_dir, token=token)
    tok = _load_tokenizer(base_model=base_model, adapter_dir=adapter_dir, token=token)

    # Patch processor.tokenizer if missing (helps downstream code & consistency)
    try:
        if getattr(processor, "tokenizer", None) is None:
            processor.tokenizer = tok
    except Exception:
        pass

    # Ensure pad token
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        try:
            tok.pad_token = tok.eos_token
        except Exception:
            pass

    quant_cfg = None
    torch_dtype = compute_dtype

    if mode == "qlora":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype if compute_dtype in (torch.float16, torch.bfloat16) else torch.float16,
        )

    base = try_with_token(
        AutoModelForImageTextToText.from_pretrained,
        base_model,
        token=token,
        device_map={"": 0},
        torch_dtype=torch_dtype if mode == "lora" else None,
        quantization_config=quant_cfg if mode == "qlora" else None,
        low_cpu_mem_usage=True,
    )

    adapter_loaded = False

    # Attach adapter if present
    try:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, adapter_dir)
        adapter_loaded = True
    except Exception:
        model = base

    model.eval()
    return model, processor, tok, adapter_loaded


def build_bad_words_ids(tok) -> Optional[List[List[int]]]:
    """
    Blocks the exact sequence we saw in outputs.
    Uses tokenizer() so it works even if it's multiple tokens.
    """
    bad_sequences = ["<unused94>thought"]
    bad_ids: List[List[int]] = []

    for s in bad_sequences:
        try:
            ids = tok(s, add_special_tokens=False).input_ids
            if isinstance(ids, list) and len(ids) > 0:
                bad_ids.append([int(i) for i in ids])
        except Exception:
            continue

    return bad_ids if bad_ids else None


def strengthen_prompt_for_json(prompt: str, required_keys: List[str]) -> str:
    """
    Adds a hard constraint without changing your dataset files.
    Also tells the model to output EXACTLY ONE json object (prevents repeats).
    If required_keys is provided, we explicitly instruct keys to be present.
    """
    req = ""
    if required_keys:
        req = (
            "\n"
            + "Your JSON must include these keys: "
            + ", ".join([f'"{k}"' for k in required_keys])
            + "."
        )

    return (
        (prompt or "").rstrip()
        + "\n\n"
        + "IMPORTANT: Output ONLY valid JSON. Start with '{' and end with '}'. "
        + "Output exactly ONE JSON object and stop. Do not repeat the JSON. "
        + "Do not include any other words, headings, code fences, or analysis."
        + req
    )


# ---------------------------
# Evaluation core (legacy + 2019)
# ---------------------------
def _safe_specificity(y_true: List[int], y_pred: List[int]) -> float:
    # specificity = TN / (TN + FP)
    if not y_true:
        return 0.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return 0.0
    tn, fp = float(cm[0, 0]), float(cm[0, 1])
    denom = tn + fp
    return (tn / denom) if denom > 0 else 0.0


def evaluate_one_run(
    run: RunSpec,
    base_model: str,
    val_recs: List[dict],
    out_dir: str,
    token: str,
    compute_dtype,
    max_new: int,
    required_keys: List[str],
    seed: int,
    max_len: int,
    task: str,
    label_keys: List[str],
    prob_keys: List[str],
    factors_keys: List[str],
):
    run_dir = os.path.join(out_dir, run.name)
    ensure_dir(run_dir)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")

    model, processor, tok, adapter_loaded = load_model_with_adapter(
        base_model=base_model,
        adapter_dir=run.adapter_dir,
        mode=run.mode,
        token=token,
        compute_dtype=compute_dtype,
    )
    bad_words_ids = build_bad_words_ids(tok)

    # Determine schema
    auto_schema = detect_schema(val_recs)
    schema = auto_schema if task == "auto" else task
    if schema not in ("legacy_status", "p2019_sepsis"):
        schema = "p2019_sepsis"

    # Outputs
    outputs_path = os.path.join(run_dir, "sample_outputs.jsonl")

    json_ok = 0
    req_ok_rates: List[float] = []

    # perf
    latencies: List[float] = []
    gen_lengths: List[int] = []

    # deterministic-ish (reserved)
    _ = np.random.default_rng(seed)

    # Legacy accumulators
    status_true, status_pred = [], []
    drivers_f1, checks_f1 = [], []
    drivers_j, checks_j = [], []
    rouge_scores = []

    # 2019 accumulators
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    factors_f1: List[float] = []
    factors_j: List[float] = []
    rouge_expl: List[float] = []

    with open(outputs_path, "w", encoding="utf-8") as outf:
        for rec in val_recs:
            base_prompt = rec.get("prompt", "")
            prompt = strengthen_prompt_for_json(base_prompt, required_keys)
            target_raw = rec.get("target", "")
            tgt = parse_target_field(target_raw)

            inputs = _apply_chat(processor, tok, prompt, max_len=max_len)
            prompt_len = int(inputs["input_ids"].shape[-1])  # BEFORE cuda

            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    temperature=0.0,
                    eos_token_id=getattr(tok, "eos_token_id", None),
                    pad_token_id=getattr(tok, "pad_token_id", None),
                    bad_words_ids=bad_words_ids,
                )

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            latency = t1 - t0
            latencies.append(latency)

            # Decode ONLY generated continuation
            gen = out[0][prompt_len:]
            gen_len = int(gen.shape[-1]) if hasattr(gen, "shape") else len(gen.tolist())
            gen_lengths.append(gen_len)

            gen_text_full = tok.decode(gen, skip_special_tokens=True).strip()

            # Robust: parse first JSON object only
            pred, blob = extract_first_json(gen_text_full)
            raw_out_to_store = blob if blob is not None else gen_text_full

            ok = isinstance(pred, dict)
            json_ok += int(ok)

            req_ok_rates.append(required_keys_rate(pred, required_keys))

            # -----------------------
            # Legacy schema metrics
            # -----------------------
            if schema == "legacy_status" and ok and isinstance(tgt, dict):
                t_status = str(tgt.get("status", "unknown"))
                p_status = str(pred.get("status", "unknown"))
                status_true.append(t_status)
                status_pred.append(p_status)

                t_dr = safe_list(tgt.get("drivers", []))
                p_dr = safe_list(pred.get("drivers", []))
                _, _, f1d = set_f1(p_dr, t_dr)
                drivers_f1.append(f1d)
                drivers_j.append(jaccard(p_dr, t_dr))

                t_ck = safe_list(tgt.get("what_to_check_next", []))
                p_ck = safe_list(pred.get("what_to_check_next", []))
                _, _, f1c = set_f1(p_ck, t_ck)
                checks_f1.append(f1c)
                checks_j.append(jaccard(p_ck, t_ck))

                rouge_scores.append(
                    rouge_l(str(pred.get("narrative", "")), str(tgt.get("narrative", "")))
                )

            # -----------------------
            # PhysioNet 2019 sepsis metrics
            # -----------------------
            if schema == "p2019_sepsis" and isinstance(tgt, dict):
                t_lab = extract_label(tgt, label_keys)
                # allow a fallback if target is a plain number/string
                if t_lab is None and not isinstance(tgt, dict):
                    t_lab = _to_int_label(tgt)

                p_lab = extract_label(pred, label_keys) if ok else None
                p_prb = extract_prob(pred, prob_keys) if ok else None

                # If only prob is present, derive label by threshold 0.5
                if p_lab is None and p_prb is not None:
                    p_lab = 1 if p_prb >= 0.5 else 0

                # If only label is present, set prob to label (still useful for Brier-ish)
                if p_prb is None and p_lab is not None:
                    p_prb = float(p_lab)

                # Accumulate only if we have ground truth + prediction label
                if t_lab is not None and p_lab is not None:
                    y_true.append(int(t_lab))
                    y_pred.append(int(p_lab))
                    if p_prb is not None:
                        y_prob.append(float(p_prb))

                    # optional factors overlap (if present on both)
                    t_fac = extract_factors(tgt, factors_keys)
                    p_fac = extract_factors(pred, factors_keys) if ok else []
                    if t_fac or p_fac:
                        _, _, ff1 = set_f1(p_fac, t_fac)
                        factors_f1.append(ff1)
                        factors_j.append(jaccard(p_fac, t_fac))

                    # optional explanation rouge (if present on both)
                    t_exp = tgt.get("explanation", tgt.get("rationale", tgt.get("narrative", "")))
                    p_exp = pred.get("explanation", pred.get("rationale", pred.get("narrative", ""))) if ok else ""
                    if str(t_exp).strip() or str(p_exp).strip():
                        rouge_expl.append(rouge_l(str(p_exp), str(t_exp)))

            outf.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "target": tgt,
                        "raw_output": raw_out_to_store,
                        "pred_json": pred,
                        "latency_s": latency,
                        "gen_len_tokens": gen_len,
                        "prompt_len_tokens": prompt_len,
                        "adapter_loaded": adapter_loaded,
                        "schema": schema,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ---------------------------
    # Aggregate metrics
    # ---------------------------
    metrics: Dict[str, Any] = {
        "run_name": run.name,
        "mode": run.mode,
        "adapter_dir": run.adapter_dir,
        "adapter_loaded": bool(adapter_loaded),
        "samples": len(val_recs),
        "schema": schema,
        "json_parse_rate": float(json_ok / max(1, len(val_recs))),
        "required_keys_rate_mean": float(np.mean(req_ok_rates)) if req_ok_rates else 0.0,
        "latency_s_mean": float(np.mean(latencies)) if latencies else 0.0,
        "gen_len_tokens_mean": float(np.mean(gen_lengths)) if gen_lengths else 0.0,
    }

    # Legacy metrics
    if schema == "legacy_status":
        metrics.update(
            {
                "status_accuracy": float(np.mean([1 if a == b else 0 for a, b in zip(status_true, status_pred)]))
                if status_true else 0.0,
                "drivers_f1_mean": float(np.mean(drivers_f1)) if drivers_f1 else 0.0,
                "checks_f1_mean": float(np.mean(checks_f1)) if checks_f1 else 0.0,
                "drivers_jaccard_mean": float(np.mean(drivers_j)) if drivers_j else 0.0,
                "checks_jaccard_mean": float(np.mean(checks_j)) if checks_j else 0.0,
                "rougeL_narrative_mean": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
            }
        )

    # PhysioNet 2019 metrics
    if schema == "p2019_sepsis":
        if y_true:
            acc = float(np.mean([1 if a == b else 0 for a, b in zip(y_true, y_pred)]))
            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            f1v = float(f1_score(y_true, y_pred, zero_division=0))
            bal = float(balanced_accuracy_score(y_true, y_pred))
            spec = float(_safe_specificity(y_true, y_pred))

            metrics.update(
                {
                    "n_scored": int(len(y_true)),
                    "sepsis_accuracy": acc,
                    "sepsis_precision": prec,
                    "sepsis_recall": rec,
                    "sepsis_f1": f1v,
                    "sepsis_balanced_accuracy": bal,
                    "sepsis_specificity": spec,
                    "factors_f1_mean": float(np.mean(factors_f1)) if factors_f1 else 0.0,
                    "factors_jaccard_mean": float(np.mean(factors_j)) if factors_j else 0.0,
                    "rougeL_explanation_mean": float(np.mean(rouge_expl)) if rouge_expl else 0.0,
                }
            )

            # Probabilistic metrics only if we have probs for all scored samples
            if len(y_prob) == len(y_true):
                try:
                    metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
                except Exception:
                    metrics["auroc"] = 0.0
                try:
                    metrics["auprc"] = float(average_precision_score(y_true, y_prob))
                except Exception:
                    metrics["auprc"] = 0.0
                try:
                    metrics["brier"] = float(brier_score_loss(y_true, y_prob))
                except Exception:
                    metrics["brier"] = 0.0
        else:
            metrics.update(
                {
                    "n_scored": 0,
                    "sepsis_accuracy": 0.0,
                    "sepsis_precision": 0.0,
                    "sepsis_recall": 0.0,
                    "sepsis_f1": 0.0,
                    "sepsis_balanced_accuracy": 0.0,
                    "sepsis_specificity": 0.0,
                    "auroc": 0.0,
                    "auprc": 0.0,
                    "brier": 0.0,
                    "factors_f1_mean": 0.0,
                    "factors_jaccard_mean": 0.0,
                    "rougeL_explanation_mean": 0.0,
                }
            )

    # write metrics
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # confusion matrix/report
    if schema == "legacy_status":
        labels = sorted(list(set(status_true) | set(status_pred))) if status_true else []
        if labels:
            cm = confusion_matrix(status_true, status_pred, labels=labels)
            plot_confusion(cm, labels, os.path.join(run_dir, "status_confusion.png"), "Status confusion matrix")

            rep = classification_report(status_true, status_pred, labels=labels, zero_division=0)
            with open(os.path.join(run_dir, "status_report.txt"), "w", encoding="utf-8") as f:
                f.write(rep)

    if schema == "p2019_sepsis" and y_true:
        labels = ["0", "1"]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plot_confusion(cm, labels, os.path.join(run_dir, "sepsis_confusion.png"), "Sepsis confusion matrix (0/1)")

        rep = classification_report(y_true, y_pred, labels=[0, 1], zero_division=0)
        with open(os.path.join(run_dir, "sepsis_report.txt"), "w", encoding="utf-8") as f:
            f.write(rep)

    # training curves (if exists)
    plot_training_curves(
        os.path.join(run.adapter_dir, "trainer_state.json"),
        os.path.join(run_dir, "training_loss.png"),
    )

    # metric bar
    if schema == "legacy_status":
        bar_keys = [
            "json_parse_rate",
            "required_keys_rate_mean",
            "status_accuracy",
            "drivers_f1_mean",
            "checks_f1_mean",
            "rougeL_narrative_mean",
        ]
        plot_bar(metrics, bar_keys, os.path.join(run_dir, "metrics_bar.png"), f"Key metrics: {run.name}")

    if schema == "p2019_sepsis":
        bar_keys = [
            "json_parse_rate",
            "required_keys_rate_mean",
            "sepsis_accuracy",
            "sepsis_f1",
            "sepsis_balanced_accuracy",
        ]
        if "auroc" in metrics:
            bar_keys.append("auroc")
        if "auprc" in metrics:
            bar_keys.append("auprc")
        plot_bar(metrics, bar_keys, os.path.join(run_dir, "metrics_bar.png"), f"Key metrics: {run.name}")

    # summary markdown
    md = [
        f"# Evaluation summary — {run.name}",
        "",
        f"- Mode: **{run.mode}**",
        f"- Adapter: **{run.adapter_dir}**",
        f"- Adapter loaded: **{metrics['adapter_loaded']}**",
        f"- Samples (loaded): **{metrics['samples']}**",
        f"- Schema: **{metrics['schema']}**",
        f"- JSON parse rate: **{metrics['json_parse_rate']:.3f}**",
        f"- Required keys rate (mean): **{metrics['required_keys_rate_mean']:.3f}**",
        f"- Mean latency (s): **{metrics['latency_s_mean']:.3f}**",
        f"- Mean generated length (tokens): **{metrics['gen_len_tokens_mean']:.1f}**",
        "",
    ]

    if schema == "legacy_status":
        md += [
            "## Legacy (status/drivers/checks) metrics",
            f"- Status accuracy: **{metrics.get('status_accuracy', 0.0):.3f}**",
            f"- Drivers F1 (mean): **{metrics.get('drivers_f1_mean', 0.0):.3f}**",
            f"- Checks F1 (mean): **{metrics.get('checks_f1_mean', 0.0):.3f}**",
            f"- ROUGE-L narrative (mean): **{metrics.get('rougeL_narrative_mean', 0.0):.3f}**",
            "",
            "Artifacts:",
            "- metrics.json",
            "- metrics_bar.png",
            "- status_confusion.png (if status labels exist)",
            "- status_report.txt (if status labels exist)",
            "- training_loss.png (if trainer_state.json exists)",
            "- sample_outputs.jsonl (raw output + parsed json + latency)",
        ]

    if schema == "p2019_sepsis":
        md += [
            "## PhysioNet 2019 (sepsis) metrics",
            f"- Scored samples (with label+pred): **{metrics.get('n_scored', 0)}**",
            f"- Accuracy: **{metrics.get('sepsis_accuracy', 0.0):.3f}**",
            f"- Precision: **{metrics.get('sepsis_precision', 0.0):.3f}**",
            f"- Recall: **{metrics.get('sepsis_recall', 0.0):.3f}**",
            f"- F1: **{metrics.get('sepsis_f1', 0.0):.3f}**",
            f"- Balanced accuracy: **{metrics.get('sepsis_balanced_accuracy', 0.0):.3f}**",
            f"- Specificity: **{metrics.get('sepsis_specificity', 0.0):.3f}**",
        ]
        if "auroc" in metrics:
            md.append(f"- AUROC: **{metrics.get('auroc', 0.0):.3f}**")
        if "auprc" in metrics:
            md.append(f"- AUPRC: **{metrics.get('auprc', 0.0):.3f}**")
        if "brier" in metrics:
            md.append(f"- Brier: **{metrics.get('brier', 0.0):.3f}**")

        md += [
            f"- Factors F1 (mean): **{metrics.get('factors_f1_mean', 0.0):.3f}**",
            f"- Factors Jaccard (mean): **{metrics.get('factors_jaccard_mean', 0.0):.3f}**",
            f"- ROUGE-L explanation (mean): **{metrics.get('rougeL_explanation_mean', 0.0):.3f}**",
            "",
            "Artifacts:",
            "- metrics.json",
            "- metrics_bar.png",
            "- sepsis_confusion.png (if labels exist)",
            "- sepsis_report.txt (if labels exist)",
            "- training_loss.png (if trainer_state.json exists)",
            "- sample_outputs.jsonl (raw output + parsed json + latency)",
        ]

    with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    return metrics


def write_combined_report(all_metrics: List[dict], out_dir: str):
    ensure_dir(out_dir)

    with open(os.path.join(out_dir, "all_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Build a comparison table that works for both schemas
    lines = [
        "# Adapter comparison",
        "",
        "| run | mode | schema | adapter_loaded | json_parse | req_keys | acc | f1 | auroc | auprc | latency_s | gen_len |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in all_metrics:
        schema = m.get("schema", "")
        acc = m.get("sepsis_accuracy", m.get("status_accuracy", 0.0))
        f1v = m.get("sepsis_f1", 0.0)
        auroc = m.get("auroc", 0.0)
        auprc = m.get("auprc", 0.0)
        lines.append(
            f"| {m['run_name']} | {m['mode']} | {schema} | {int(bool(m.get('adapter_loaded')))} | "
            f"{m.get('json_parse_rate', 0.0):.3f} | {m.get('required_keys_rate_mean', 0.0):.3f} | "
            f"{acc:.3f} | {f1v:.3f} | {auroc:.3f} | {auprc:.3f} | "
            f"{m.get('latency_s_mean', 0.0):.3f} | {m.get('gen_len_tokens_mean', 0.0):.1f} |"
        )

    with open(os.path.join(out_dir, "comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Heuristic "best":
    def _score(x):
        return (
            x.get("sepsis_f1", 0.0),
            x.get("sepsis_accuracy", x.get("status_accuracy", 0.0)),
            x.get("json_parse_rate", 0.0),
        )

    best = sorted(all_metrics, key=_score, reverse=True)[0] if all_metrics else None
    if best:
        with open(os.path.join(out_dir, "best.txt"), "w", encoding="utf-8") as f:
            f.write(f"Best (heuristic): {best['run_name']}  mode={best['mode']}  schema={best.get('schema','')}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="google/medgemma-1.5-4b-it")
    ap.add_argument("--val_jsonl", type=str, default="data/val.jsonl")
    ap.add_argument("--out_dir", type=str, default="reports_eval")
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--max_len", type=int, default=1024, help="Max input tokens for truncation")
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hf_token", type=str, default="")

    # Task schema:
    ap.add_argument("--task", type=str, default="auto", choices=["auto", "p2019_sepsis", "legacy_status"])

    # Required keys (used for required_keys_rate + prompt strengthening)
    ap.add_argument(
        "--required_keys",
        type=str,
        default="sepsis_label",
        help="Comma-separated keys to require in model output JSON (also used in prompt strengthening).",
    )

    # Label/prob/factors keys for 2019 parsing (auto tries multiple keys)
    ap.add_argument("--label_keys", type=str, default=",".join(DEFAULT_LABEL_KEYS))
    ap.add_argument("--prob_keys", type=str, default=",".join(DEFAULT_PROB_KEYS))
    ap.add_argument("--factors_keys", type=str, default=",".join(DEFAULT_FACTORS_KEYS))

    # Repeatable runs:
    ap.add_argument("--run", action="append", default=[], help='Repeat: "NAME|ADAPTER_DIR|MODE" where MODE is lora/qlora')

    # Single-run fallback (optional)
    ap.add_argument("--adapter_dir", type=str, default="")
    ap.add_argument("--mode", type=str, default="", choices=["", "lora", "qlora"])

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available (check your torch install).")

    ensure_dir(args.out_dir)

    token = get_hf_token(args.hf_token)
    compute_dtype = pick_compute_dtype()

    required_keys = [k.strip() for k in args.required_keys.split(",") if k.strip()]
    label_keys = [k.strip() for k in args.label_keys.split(",") if k.strip()]
    prob_keys = [k.strip() for k in args.prob_keys.split(",") if k.strip()]
    factors_keys = [k.strip() for k in args.factors_keys.split(",") if k.strip()]

    runs: List[RunSpec] = []
    for s in args.run:
        runs.append(parse_run(s))

    if not runs and args.adapter_dir and args.mode:
        runs.append(
            RunSpec(
                name=os.path.basename(args.adapter_dir.rstrip("/\\")),
                adapter_dir=args.adapter_dir,
                mode=args.mode,
            )
        )

    if not runs:
        raise SystemExit(
            "No runs specified. Use --run \"NAME|ADAPTER_DIR|MODE\" (repeatable) "
            "or --adapter_dir + --mode."
        )

    val_recs = load_val(args.val_jsonl, args.max_samples, args.seed)

    all_metrics = []
    for run in runs:
        print(f"[run] {run.name}  mode={run.mode}  adapter={run.adapter_dir}")
        m = evaluate_one_run(
            run=run,
            base_model=args.base_model,
            val_recs=val_recs,
            out_dir=args.out_dir,
            token=token,
            compute_dtype=compute_dtype,
            max_new=args.max_new,
            required_keys=required_keys,
            seed=args.seed,
            max_len=args.max_len,
            task=args.task,
            label_keys=label_keys,
            prob_keys=prob_keys,
            factors_keys=factors_keys,
        )
        all_metrics.append(m)
        print(f"[ok] wrote metrics -> {os.path.join(args.out_dir, run.name, 'metrics.json')}")

    write_combined_report(all_metrics, args.out_dir)
    print("[ok] wrote comparison ->", os.path.join(args.out_dir, "comparison.md"))
    print("[ok] wrote all_metrics.json ->", os.path.join(args.out_dir, "all_metrics.json"))


if __name__ == "__main__":
    main()
