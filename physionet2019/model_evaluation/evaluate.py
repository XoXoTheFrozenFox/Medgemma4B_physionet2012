#!/usr/bin/env python
# evaluate.py
# ------------------------------------------------------------
# Evaluate MedGemma LoRA / QLoRA adapters on val.jsonl (prompt/target)
#
# CRITICAL FIXES (your current failure mode):
# 1) prompt_len_tokens=2 / raw_output="" / pred_json=null:
#    - FORCE tokenizer + processor from BASE MODEL (adapter checkpoints often lack tokenizer/chat_template)
#    - Inject a default Gemma chat_template if missing
#    - FAIL FAST if prompt_len is tiny (so you don’t wait 300s)
#
# 2) Outputs take too long:
#    - Default max_new=256
#    - Default max_retries=0 (no doubling to 1024)
#    - Optional JSON brace early-stopping during generation
#
# Keeps:
# - LoRA / QLoRA base loading
# - JSON extraction and metrics (p2019_sepsis + legacy_status)
# ------------------------------------------------------------

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

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
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

from hf_auth import get_hf_token, try_with_token


# ---------------------------
# Speed knobs (safe)
# ---------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ---------------------------
# Helpers
# ---------------------------
def pick_compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def ensure_dir(p: str):
    if not p:
        return
    if os.path.exists(p) and not os.path.isdir(p):
        raise RuntimeError(f"Path exists but is not a directory: {p}")
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


def stop_at_json_object(text: str) -> str:
    t = _strip_code_fences(text or "")
    start = t.find("{")
    if start == -1:
        return t

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(t)):
        ch = t[i]
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
                return t[start : i + 1]
    return t


def parse_target_field(target: Any) -> Any:
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
        return 1 if float(v) >= 0.5 else 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "pos", "positive", "sepsis"):
            return 1
        if s in ("0", "false", "no", "n", "neg", "negative", "no_sepsis", "nonsepsis", "non-sepsis"):
            return 0
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
        if x > 1.0 and x <= 100.0:
            x = x / 100.0
        return max(0.0, min(1.0, x))
    if isinstance(v, str):
        s = v.strip().lower().replace("%", "")
        try:
            x = float(s)
            if x > 1.0 and x <= 100.0:
                x = x / 100.0
            return max(0.0, min(1.0, x))
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
    for rec in val_recs[:50]:
        tgt = parse_target_field(rec.get("target", ""))
        if isinstance(tgt, dict):
            if any(k in tgt for k in DEFAULT_LABEL_KEYS):
                return "p2019_sepsis"
            if "status" in tgt or ("drivers" in tgt and "what_to_check_next" in tgt):
                return "legacy_status"
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
# Chat template + encoding
# ---------------------------
GEMMA_DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<start_of_turn>model\n"
    "{% endif %}"
)


def ensure_chat_template(tokenizer, processor):
    # If tokenizer/processor lacks a chat_template, inject a sane Gemma template.
    try:
        ct = getattr(tokenizer, "chat_template", None)
        if not ct:
            tokenizer.chat_template = GEMMA_DEFAULT_CHAT_TEMPLATE
    except Exception:
        pass
    try:
        if processor is not None:
            pct = getattr(processor, "chat_template", None)
            # some processors proxy to tokenizer; safe to set anyway
            if not pct and hasattr(processor, "tokenizer") and processor.tokenizer is not None:
                if not getattr(processor.tokenizer, "chat_template", None):
                    processor.tokenizer.chat_template = GEMMA_DEFAULT_CHAT_TEMPLATE
    except Exception:
        pass


def _messages_variants(prompt: str) -> List[List[Dict[str, Any]]]:
    p = prompt or ""
    return [
        [{"role": "user", "content": p}],
        [{"role": "user", "content": [{"type": "text", "text": p}]}],
    ]


def _to_2d_tensors(enc: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "input_ids" not in enc:
        raise RuntimeError("Encoding has no input_ids.")
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask", None)
    ttype = enc.get("token_type_ids", None)

    if torch.is_tensor(input_ids):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attn is None:
            attn = torch.ones_like(input_ids, dtype=torch.long)
        elif torch.is_tensor(attn) and attn.dim() == 1:
            attn = attn.unsqueeze(0)
        outb = {"input_ids": input_ids.to(torch.long), "attention_mask": attn.to(torch.long)}
        if ttype is not None and torch.is_tensor(ttype):
            if ttype.dim() == 1:
                ttype = ttype.unsqueeze(0)
            outb["token_type_ids"] = ttype.to(torch.long)
        return outb

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


def encode_best(processor, tokenizer, prompt: str, max_len: int) -> Tuple[Dict[str, torch.Tensor], int, str, Dict[str, int], Dict[str, str]]:
    """
    Returns: inputs_cpu, prompt_len, best_strategy, candidate_lens, candidate_errors
    """
    if tokenizer is None:
        raise RuntimeError("Tokenizer is required but was not loaded.")

    ensure_chat_template(tokenizer, processor)

    candidates: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    lens: Dict[str, int] = {}
    errs: Dict[str, str] = {}

    msgs_list = _messages_variants(prompt)

    # A) tokenizer.apply_chat_template
    if hasattr(tokenizer, "apply_chat_template"):
        for tag, msgs in [("tok_chat_plain", msgs_list[0]), ("tok_chat_mm", msgs_list[1])]:
            try:
                out = tokenizer.apply_chat_template(
                    msgs,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    truncation=True,
                    max_length=max_len,
                )
                if hasattr(out, "data"):
                    out = out.data
                enc = _to_2d_tensors(out)
                candidates.append((tag, enc))
                lens[tag] = int(enc["input_ids"].shape[-1])
            except Exception as e:
                errs[tag] = repr(e)

    # B) processor.apply_chat_template (optional)
    if processor is not None and hasattr(processor, "apply_chat_template"):
        for tag, msgs in [("proc_chat_plain", msgs_list[0]), ("proc_chat_mm", msgs_list[1])]:
            try:
                out = processor.apply_chat_template(
                    msgs,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    truncation=True,
                    max_length=max_len,
                )
                if hasattr(out, "data"):
                    out = out.data
                enc = _to_2d_tensors(out)
                candidates.append((tag, enc))
                lens[tag] = int(enc["input_ids"].shape[-1])
            except Exception as e:
                errs[tag] = repr(e)

    # C) raw tokenizer(prompt) fallback (works even without chat template)
    try:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
        if hasattr(enc, "data"):
            enc = enc.data
        enc2 = _to_2d_tensors(enc)
        candidates.append(("raw_tok", enc2))
        lens["raw_tok"] = int(enc2["input_ids"].shape[-1])
    except Exception as e:
        errs["raw_tok"] = repr(e)

    if not candidates:
        raise RuntimeError(f"All encoding strategies failed. errors={errs}")

    tag_best, enc_best = max(candidates, key=lambda x: int(x[1]["input_ids"].shape[-1]))
    best_len = int(enc_best["input_ids"].shape[-1])
    return enc_best, best_len, tag_best, lens, errs


# ---------------------------
# Generation: optional early stop when JSON closes
# ---------------------------
class JsonBraceStopper(StoppingCriteria):
    def __init__(self, start_len: int, lbrace_id: int, rbrace_id: int, min_new: int = 16):
        self.start_len = int(start_len)
        self.lbrace_id = int(lbrace_id)
        self.rbrace_id = int(rbrace_id)
        self.min_new = int(min_new)
        self._seen_open = False
        self._depth = 0
        self._last_proc = self.start_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        ids = input_ids[0].tolist()
        cur_len = len(ids)
        if cur_len <= self._last_proc:
            return False

        new_ids = ids[self._last_proc : cur_len]
        self._last_proc = cur_len

        for tid in new_ids:
            if tid == self.lbrace_id:
                self._seen_open = True
                self._depth += 1
            elif tid == self.rbrace_id and self._seen_open:
                self._depth -= 1
                if self._depth <= 0:
                    # ensure we produced at least some tokens
                    if (cur_len - self.start_len) >= self.min_new:
                        return True
        return False


def build_bad_words_ids(tok) -> Optional[List[List[int]]]:
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
    req = ""
    if required_keys:
        req = "\n" + "Your JSON must include these keys: " + ", ".join([f'"{k}"' for k in required_keys]) + "."
    extra = ""
    if any(k.strip().lower() == "sepsis_label" for k in required_keys):
        extra = "\nIf you output sepsis_label, it MUST be an integer 0 or 1 (not a string)."
    return (
        (prompt or "").rstrip()
        + "\n\n"
        + "IMPORTANT: Output ONLY valid JSON. Start with '{' and end with '}'. "
        + "Output exactly ONE JSON object and stop. Do not repeat the JSON. "
        + "Do not include any other words, headings, code fences, or analysis."
        + req
        + extra
    )


def _trim_generated_ids(gen_ids: torch.Tensor, eos_id: Optional[int], strip_ids: Optional[Set[int]] = None) -> List[int]:
    ids = gen_ids.tolist() if torch.is_tensor(gen_ids) else list(gen_ids)
    if eos_id is not None:
        try:
            eos_int = int(eos_id)
            if eos_int in ids:
                ids = ids[: ids.index(eos_int)]
        except Exception:
            pass
    if strip_ids:
        while ids and int(ids[-1]) in strip_ids:
            ids.pop()
        while ids and int(ids[0]) in strip_ids:
            ids.pop(0)
    return ids


def generate_once(
    model,
    tok,
    inputs_cuda: Dict[str, torch.Tensor],
    prompt_len: int,
    max_new_tokens: int,
    eos_id,
    pad_id,
    bad_words_ids,
    strip_ids: Set[int],
    json_stop_after: bool,
    use_json_early_stop: bool,
) -> Tuple[str, int]:
    stopping = None
    if use_json_early_stop:
        # Only enable if "{" and "}" are single-token encodings
        try:
            lb = tok.encode("{", add_special_tokens=False)
            rb = tok.encode("}", add_special_tokens=False)
            if len(lb) == 1 and len(rb) == 1:
                stopping = StoppingCriteriaList([JsonBraceStopper(prompt_len, lb[0], rb[0], min_new=16)])
        except Exception:
            stopping = None

    out = model.generate(
        **inputs_cuda,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        bad_words_ids=bad_words_ids,
        stopping_criteria=stopping,
    )

    gen_full = out[0][prompt_len:]
    trimmed_ids = _trim_generated_ids(gen_full, eos_id=eos_id, strip_ids=strip_ids)
    gen_len_eff = len(trimmed_ids)

    gen_text = tok.decode(trimmed_ids, skip_special_tokens=True).strip() if trimmed_ids else ""
    if json_stop_after and gen_text:
        gen_text = stop_at_json_object(gen_text)

    return gen_text, gen_len_eff


def _safe_specificity(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true:
        return 0.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return 0.0
    tn, fp = float(cm[0, 0]), float(cm[0, 1])
    denom = tn + fp
    return (tn / denom) if denom > 0 else 0.0


# ---------------------------
# Run spec parsing
# ---------------------------
@dataclass
class RunSpec:
    name: str
    adapter_dir: str
    mode: str  # "lora" or "qlora"


def parse_run(s: str) -> RunSpec:
    parts = s.split("|")
    if len(parts) != 3:
        raise ValueError("Run spec must be: NAME|ADAPTER_DIR|MODE   where MODE is lora or qlora")
    name, adapter_dir, mode = parts[0].strip(), parts[1].strip(), parts[2].strip().lower()
    if mode not in ("lora", "qlora"):
        raise ValueError("MODE must be 'lora' or 'qlora'")
    return RunSpec(name=name, adapter_dir=adapter_dir, mode=mode)


# ---------------------------
# Model loader: FORCE base tokenizer/processor
# ---------------------------
def load_tokenizer_and_processor(base_model: str, token: str):
    processor = try_with_token(AutoProcessor.from_pretrained, base_model, token=token)
    tok = try_with_token(AutoTokenizer.from_pretrained, base_model, token=token, use_fast=True)
    try:
        if getattr(processor, "tokenizer", None) is None:
            processor.tokenizer = tok
    except Exception:
        pass
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        try:
            tok.pad_token = tok.eos_token
        except Exception:
            pass
    ensure_chat_template(tok, processor)
    return tok, processor


def load_model_with_adapter(base_model: str, adapter_dir: str, mode: str, token: str, compute_dtype):
    tok, processor = load_tokenizer_and_processor(base_model=base_model, token=token)

    quant_cfg = None
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
        torch_dtype=compute_dtype if mode == "lora" else None,
        quantization_config=quant_cfg if mode == "qlora" else None,
        low_cpu_mem_usage=True,
    )

    adapter_loaded = False
    try:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, adapter_dir)
        adapter_loaded = True
    except Exception:
        model = base

    model.eval()
    return model, processor, tok, adapter_loaded


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


# ---------------------------
# Evaluation
# ---------------------------
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
    print_every: int,
    fail_on_short_prompt: bool,
    min_ok_prompt_len: int,
    json_stop_after: bool,
    use_json_early_stop: bool,
):
    ensure_dir(out_dir)
    run_dir = os.path.join(out_dir, run.name)
    ensure_dir(run_dir)

    outputs_path = os.path.join(run_dir, "sample_outputs.jsonl")
    ensure_dir(os.path.dirname(outputs_path))

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

    auto_schema = detect_schema(val_recs)
    schema = auto_schema if task == "auto" else task
    if schema not in ("legacy_status", "p2019_sepsis"):
        schema = "p2019_sepsis"

    json_ok = 0
    req_ok_rates: List[float] = []
    latencies: List[float] = []
    gen_lengths: List[int] = []

    status_true, status_pred = [], []
    drivers_f1, checks_f1 = [], []
    drivers_j, checks_j = [], []
    rouge_scores = []

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []

    _ = np.random.default_rng(seed)

    eos_id = getattr(tok, "eos_token_id", None)
    pad_id = getattr(tok, "pad_token_id", None)

    strip_ids: Set[int] = set(getattr(tok, "all_special_ids", []) or [])
    if pad_id is not None:
        strip_ids.add(int(pad_id))
    if eos_id is not None:
        strip_ids.add(int(eos_id))

    with open(outputs_path, "w", encoding="utf-8") as outf:
        for idx, rec in enumerate(val_recs):
            base_prompt = rec.get("prompt", "")
            prompt = strengthen_prompt_for_json(base_prompt, required_keys)
            tgt = parse_target_field(rec.get("target", ""))

            inputs_cpu, prompt_len, encode_strategy, enc_lens, enc_errs = encode_best(
                processor=processor,
                tokenizer=tok,
                prompt=prompt,
                max_len=max_len,
            )

            if fail_on_short_prompt and prompt_len < min_ok_prompt_len:
                # Hard stop: your previous run wasted 300s per sample because prompt_len was 2.
                raise RuntimeError(
                    f"FATAL: prompt_len_tokens={prompt_len} (<{min_ok_prompt_len}). "
                    f"Encoding is broken. strategy={encode_strategy} lens={enc_lens} errors={enc_errs}"
                )

            inputs = {k: v.to("cuda") for k, v in inputs_cpu.items()}

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                gen_text, gen_len_eff = generate_once(
                    model=model,
                    tok=tok,
                    inputs_cuda=inputs,
                    prompt_len=prompt_len,
                    max_new_tokens=max_new,
                    eos_id=eos_id,
                    pad_id=pad_id,
                    bad_words_ids=bad_words_ids,
                    strip_ids=strip_ids,
                    json_stop_after=json_stop_after,
                    use_json_early_stop=use_json_early_stop,
                )
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            latency = t1 - t0
            latencies.append(latency)
            gen_lengths.append(gen_len_eff)

            pred, blob = extract_first_json(gen_text)
            ok = isinstance(pred, dict)
            json_ok += int(ok)
            req_ok_rates.append(required_keys_rate(pred, required_keys))

            if schema == "legacy_status" and ok and isinstance(tgt, dict):
                status_true.append(str(tgt.get("status", "unknown")))
                status_pred.append(str(pred.get("status", "unknown")))
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

                rouge_scores.append(rouge_l(str(pred.get("narrative", "")), str(tgt.get("narrative", ""))))

            if schema == "p2019_sepsis":
                t_lab = extract_label(tgt, label_keys) if isinstance(tgt, dict) else _to_int_label(tgt)
                p_lab = extract_label(pred, label_keys) if ok else None
                p_prb = extract_prob(pred, prob_keys) if ok else None

                if p_lab is None and p_prb is not None:
                    p_lab = 1 if p_prb >= 0.5 else 0
                if p_prb is None and p_lab is not None:
                    p_prb = float(p_lab)

                if t_lab is not None and p_lab is not None:
                    y_true.append(int(t_lab))
                    y_pred.append(int(p_lab))
                    if p_prb is not None:
                        y_prob.append(float(p_prb))

            row = {
                "prompt": prompt,
                "target": tgt,
                "raw_output": (blob if blob is not None else gen_text),
                "pred_json": pred,
                "latency_s": latency,
                "gen_len_tokens": gen_len_eff,
                "prompt_len_tokens": prompt_len,
                "adapter_loaded": bool(adapter_loaded),
                "schema": schema,
                "encode_strategy": encode_strategy,
                "encode_candidates_lens": enc_lens,
                "encode_candidates_errors": enc_errs,
            }
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")

            if print_every and ((idx + 1) % print_every == 0):
                print(
                    f"[{run.name}] {idx+1}/{len(val_recs)}  "
                    f"prompt_len={prompt_len}  gen_len={gen_len_eff}  "
                    f"json_ok={json_ok}/{idx+1}  latency={latency:.2f}s"
                )

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

    if schema == "legacy_status":
        metrics.update(
            {
                "status_accuracy": float(np.mean([1 if a == b else 0 for a, b in zip(status_true, status_pred)]))
                if status_true
                else 0.0,
                "drivers_f1_mean": float(np.mean(drivers_f1)) if drivers_f1 else 0.0,
                "checks_f1_mean": float(np.mean(checks_f1)) if checks_f1 else 0.0,
                "drivers_jaccard_mean": float(np.mean(drivers_j)) if drivers_j else 0.0,
                "checks_jaccard_mean": float(np.mean(checks_j)) if checks_j else 0.0,
                "rougeL_narrative_mean": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
            }
        )

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
                }
            )

            if len(y_prob) == len(y_true) and len(y_true) > 1:
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
                metrics["auroc"] = 0.0
                metrics["auprc"] = 0.0
                metrics["brier"] = 0.0
        else:
            metrics.update({"n_scored": 0, "sepsis_accuracy": 0.0, "sepsis_precision": 0.0, "sepsis_recall": 0.0,
                            "sepsis_f1": 0.0, "sepsis_balanced_accuracy": 0.0, "sepsis_specificity": 0.0,
                            "auroc": 0.0, "auprc": 0.0, "brier": 0.0})

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if schema == "p2019_sepsis" and y_true:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plot_confusion(cm, ["0", "1"], os.path.join(run_dir, "sepsis_confusion.png"), "Sepsis confusion matrix (0/1)")
        rep = classification_report(y_true, y_pred, labels=[0, 1], zero_division=0)
        with open(os.path.join(run_dir, "sepsis_report.txt"), "w", encoding="utf-8") as f:
            f.write(rep)

    plot_training_curves(
        os.path.join(run.adapter_dir, "trainer_state.json"),
        os.path.join(run_dir, "training_loss.png"),
    )

    return metrics


def write_combined_report(all_metrics: List[dict], out_dir: str):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "all_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="google/medgemma-1.5-4b-it")
    ap.add_argument("--val_jsonl", type=str, default="data/val.jsonl")
    ap.add_argument("--out_dir", type=str, default="reports_eval")

    # FAST DEFAULTS
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--max_samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hf_token", type=str, default="")
    ap.add_argument("--print_every", type=int, default=1)

    ap.add_argument("--task", type=str, default="auto", choices=["auto", "p2019_sepsis", "legacy_status"])
    ap.add_argument("--required_keys", type=str, default="sepsis_label")

    ap.add_argument("--label_keys", type=str, default=",".join(DEFAULT_LABEL_KEYS))
    ap.add_argument("--prob_keys", type=str, default=",".join(DEFAULT_PROB_KEYS))
    ap.add_argument("--factors_keys", type=str, default=",".join(DEFAULT_FACTORS_KEYS))

    ap.add_argument("--run", action="append", default=[], help='Repeat: "NAME|ADAPTER_DIR|MODE" where MODE is lora/qlora')
    ap.add_argument("--adapter_dir", type=str, default="")
    ap.add_argument("--mode", type=str, default="", choices=["", "lora", "qlora"])

    # IMPORTANT: stop wasting time when prompt_len is broken
    ap.add_argument("--fail_on_short_prompt", action="store_true", help="Abort if prompt_len_tokens is tiny (recommended).")
    ap.add_argument("--min_ok_prompt_len", type=int, default=64)

    # JSON handling
    ap.add_argument("--no_json_stop_after", action="store_true", help="Do not cut decoded output at first complete JSON.")
    ap.add_argument("--json_early_stop", action="store_true", help="Stop generation early when JSON braces close.")

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
        raise SystemExit("No runs specified. Use --adapter_dir + --mode, or --run \"NAME|ADAPTER_DIR|MODE\".")

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
            print_every=args.print_every,
            fail_on_short_prompt=bool(args.fail_on_short_prompt),
            min_ok_prompt_len=int(args.min_ok_prompt_len),
            json_stop_after=(not args.no_json_stop_after),
            use_json_early_stop=bool(args.json_early_stop),
        )
        all_metrics.append(m)
        print(f"[ok] metrics -> {os.path.join(args.out_dir, run.name, 'metrics.json')}")

    write_combined_report(all_metrics, args.out_dir)
    print("[ok] all_metrics.json ->", os.path.join(args.out_dir, "all_metrics.json"))


if __name__ == "__main__":
    main()
