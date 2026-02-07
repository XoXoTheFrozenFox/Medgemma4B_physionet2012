#!/usr/bin/env python
# evaluate.py
# ------------------------------------------------------------
# Evaluate MedGemma LoRA / QLoRA adapters on val.jsonl (prompt/target)
# Fixes included:
# - Loads processor/tokenizer from adapter_dir when available (avoids HF download)
# - Supports LoRA (fp16/bf16) and QLoRA (4-bit NF4) base loading
# - Forces “JSON-only” instruction onto prompts (without editing dataset)
# - Blocks '<unused94>thought' token/sequence where tokenizer supports it
# - Decodes ONLY generated continuation (not prompt echo)
# - Robustly extracts the FIRST valid JSON object (handles repeated JSON blocks)
# - Writes adapter_loaded into outputs + metrics
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
from sklearn.metrics import confusion_matrix, classification_report

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
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
    t = text.strip()
    # remove ```json ... ``` or ``` ... ```
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            # find closing fence
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

        # not in string
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
    # target can be:
    # - dict
    # - JSON string
    # - plain string
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
        return [str(v) for v in x]
    # allow comma-separated strings
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
    p = pred.split()
    r = ref.split()
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
# Plotting
# ---------------------------
def plot_training_curves(trainer_state_path: str, out_png: str):
    if not os.path.exists(trainer_state_path):
        return
    with open(trainer_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
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
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
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


def _apply_chat(processor, prompt: str):
    # returns dict with torch tensors on CPU
    msgs = _messages_text_only(prompt)
    out = processor.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )

    # normalize: list[list[int]] -> list[int]
    input_ids = out["input_ids"]
    attn = out.get("attention_mask", None)

    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if attn is None:
        attn = [1] * len(input_ids)
    elif isinstance(attn, list) and attn and isinstance(attn[0], list):
        attn = attn[0]

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attn], dtype=torch.long),
    }


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
def load_model_with_adapter(
    base_model: str,
    adapter_dir: str,
    mode: str,
    token: str,
    compute_dtype,
):
    # Always load processor from adapter_dir if present (keeps special tokens consistent)
    try:
        processor = AutoProcessor.from_pretrained(adapter_dir)
    except Exception:
        processor = try_with_token(AutoProcessor.from_pretrained, base_model, token=token)

    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        raise RuntimeError("Processor has no tokenizer; upgrade transformers.")

    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

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
    return model, processor, adapter_loaded


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


def strengthen_prompt_for_json(prompt: str) -> str:
    """
    Adds a hard constraint without changing your dataset files.
    Also tells the model to output EXACTLY ONE json object (prevents repeats).
    """
    return (
        prompt.rstrip()
        + "\n\n"
        + "IMPORTANT: Output ONLY valid JSON. Start with '{' and end with '}'. "
          "Output exactly ONE JSON object and stop. Do not repeat the JSON. "
          "Do not include any other words, headings, code fences, or analysis."
    )


# ---------------------------
# Evaluation core
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
):
    run_dir = os.path.join(out_dir, run.name)
    ensure_dir(run_dir)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")

    model, processor, adapter_loaded = load_model_with_adapter(
        base_model=base_model,
        adapter_dir=run.adapter_dir,
        mode=run.mode,
        token=token,
        compute_dtype=compute_dtype,
    )
    tok = processor.tokenizer
    bad_words_ids = build_bad_words_ids(tok)

    # Outputs
    outputs_path = os.path.join(run_dir, "sample_outputs.jsonl")

    json_ok = 0
    req_ok_rates = []
    status_true, status_pred = [], []
    drivers_f1, checks_f1 = [], []
    drivers_j, checks_j = [], []
    rouge_scores = []

    # perf
    latencies = []
    gen_lengths = []

    # deterministic-ish (reserved; keep for future sampling modes)
    _ = np.random.default_rng(seed)

    with open(outputs_path, "w", encoding="utf-8") as outf:
        for rec in val_recs:
            base_prompt = rec.get("prompt", "")
            prompt = strengthen_prompt_for_json(base_prompt)
            target_raw = rec.get("target", "")
            tgt = parse_target_field(target_raw)

            inputs = _apply_chat(processor, prompt)
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
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id,
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

            # Robust: parse first JSON object only (handles repeated JSON blocks)
            pred, blob = extract_first_json(gen_text_full)
            raw_out_to_store = blob if blob is not None else gen_text_full

            ok = isinstance(pred, dict)
            json_ok += int(ok)

            req_ok_rates.append(required_keys_rate(pred, required_keys))

            if ok and isinstance(tgt, dict):
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

            outf.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "target": tgt,
                        "raw_output": raw_out_to_store,  # first JSON blob if found, else full text
                        "pred_json": pred,
                        "latency_s": latency,
                        "gen_len_tokens": gen_len,
                        "prompt_len_tokens": prompt_len,
                        "adapter_loaded": adapter_loaded,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics = {
        "run_name": run.name,
        "mode": run.mode,
        "adapter_dir": run.adapter_dir,
        "adapter_loaded": bool(adapter_loaded),
        "samples": len(val_recs),
        "json_parse_rate": float(json_ok / max(1, len(val_recs))),
        "required_keys_rate_mean": float(np.mean(req_ok_rates)) if req_ok_rates else 0.0,
        "status_accuracy": float(np.mean([1 if a == b else 0 for a, b in zip(status_true, status_pred)]))
        if status_true else 0.0,
        "drivers_f1_mean": float(np.mean(drivers_f1)) if drivers_f1 else 0.0,
        "checks_f1_mean": float(np.mean(checks_f1)) if checks_f1 else 0.0,
        "drivers_jaccard_mean": float(np.mean(drivers_j)) if drivers_j else 0.0,
        "checks_jaccard_mean": float(np.mean(checks_j)) if checks_j else 0.0,
        "rougeL_narrative_mean": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
        "latency_s_mean": float(np.mean(latencies)) if latencies else 0.0,
        "gen_len_tokens_mean": float(np.mean(gen_lengths)) if gen_lengths else 0.0,
    }

    # write metrics
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # confusion matrix/report (status)
    labels = sorted(list(set(status_true) | set(status_pred))) if status_true else []
    if labels:
        cm = confusion_matrix(status_true, status_pred, labels=labels)
        plot_confusion(cm, labels, os.path.join(run_dir, "status_confusion.png"), "Status confusion matrix")

        rep = classification_report(status_true, status_pred, labels=labels, zero_division=0)
        with open(os.path.join(run_dir, "status_report.txt"), "w", encoding="utf-8") as f:
            f.write(rep)

    # training curves (if exists)
    plot_training_curves(
        os.path.join(run.adapter_dir, "trainer_state.json"),
        os.path.join(run_dir, "training_loss.png"),
    )

    # metric bar
    bar_keys = [
        "json_parse_rate",
        "required_keys_rate_mean",
        "status_accuracy",
        "drivers_f1_mean",
        "checks_f1_mean",
        "rougeL_narrative_mean",
    ]
    plot_bar(metrics, bar_keys, os.path.join(run_dir, "metrics_bar.png"), f"Key metrics: {run.name}")

    # summary markdown
    md = [
        f"# Evaluation summary — {run.name}",
        "",
        f"- Mode: **{run.mode}**",
        f"- Adapter: **{run.adapter_dir}**",
        f"- Adapter loaded: **{metrics['adapter_loaded']}**",
        f"- Samples: **{metrics['samples']}**",
        f"- JSON parse rate: **{metrics['json_parse_rate']:.3f}**",
        f"- Required keys rate (mean): **{metrics['required_keys_rate_mean']:.3f}**",
        f"- Status accuracy: **{metrics['status_accuracy']:.3f}**",
        f"- Drivers F1 (mean): **{metrics['drivers_f1_mean']:.3f}**",
        f"- Checks F1 (mean): **{metrics['checks_f1_mean']:.3f}**",
        f"- ROUGE-L narrative (mean): **{metrics['rougeL_narrative_mean']:.3f}**",
        f"- Mean latency (s): **{metrics['latency_s_mean']:.3f}**",
        f"- Mean generated length (tokens): **{metrics['gen_len_tokens_mean']:.1f}**",
        "",
        "Artifacts:",
        "- metrics.json",
        "- metrics_bar.png",
        "- status_confusion.png (if status labels exist)",
        "- status_report.txt (if status labels exist)",
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

    lines = [
        "# Adapter comparison",
        "",
        "| run | mode | adapter_loaded | json_parse | req_keys | status_acc | drivers_f1 | checks_f1 | rougeL | latency_s | gen_len |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in all_metrics:
        lines.append(
            f"| {m['run_name']} | {m['mode']} | {int(bool(m.get('adapter_loaded')))} | "
            f"{m['json_parse_rate']:.3f} | {m['required_keys_rate_mean']:.3f} | "
            f"{m['status_accuracy']:.3f} | {m['drivers_f1_mean']:.3f} | {m['checks_f1_mean']:.3f} | "
            f"{m['rougeL_narrative_mean']:.3f} | {m['latency_s_mean']:.3f} | {m['gen_len_tokens_mean']:.1f} |"
        )

    with open(os.path.join(out_dir, "comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    best = (
        sorted(
            all_metrics,
            key=lambda x: (x.get("rougeL_narrative_mean", 0.0), x.get("drivers_f1_mean", 0.0)),
            reverse=True,
        )[0]
        if all_metrics
        else None
    )

    if best:
        with open(os.path.join(out_dir, "best.txt"), "w", encoding="utf-8") as f:
            f.write(f"Best (heuristic): {best['run_name']}  mode={best['mode']}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="google/medgemma-1.5-4b-it")
    ap.add_argument("--val_jsonl", type=str, default="data/val.jsonl")
    ap.add_argument("--out_dir", type=str, default="reports_eval")
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hf_token", type=str, default="")
    ap.add_argument("--required_keys", type=str, default="status,drivers,what_to_check_next,narrative")

    # Repeatable runs:
    # --run "NAME|ADAPTER_DIR|MODE"
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
        )
        all_metrics.append(m)
        print(f"[ok] wrote metrics -> {os.path.join(args.out_dir, run.name, 'metrics.json')}")

    write_combined_report(all_metrics, args.out_dir)
    print("[ok] wrote comparison ->", os.path.join(args.out_dir, "comparison.md"))
    print("[ok] wrote all_metrics.json ->", os.path.join(args.out_dir, "all_metrics.json"))


if __name__ == "__main__":
    main()
