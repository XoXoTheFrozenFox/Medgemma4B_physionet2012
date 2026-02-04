# train_qlora_4b.py
# ------------------------------------------------------------
# MedGemma 4B QLoRA fine-tuning (4-bit NF4)
# - Quantized base model (4-bit) + LoRA adapters (QLoRA)
# - Correct loss masking: trains ONLY on assistant tokens
# - Pad-safe collator: label padding uses -100
# - Adds token_type_ids (REQUIRED for Gemma3 training)
# - Avoids LoRA injection into vision tower by targeting non-vision module paths
# - Fixes warnings:
#     * use_cache=True incompatible with gradient checkpointing -> force-disable use_cache everywhere + silence logger
#     * PyTorch checkpoint use_reentrant warning -> enable GC with use_reentrant=False when supported + safe patch
# ------------------------------------------------------------

import argparse
import math
import os
import random
import warnings
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

from hf_auth import get_hf_token, try_with_token

DEFAULT_MODEL = "google/medgemma-1.5-4b-it"


# ------------------------------------------------------------
# PyTorch checkpoint warning fix (PyTorch 2.9 will require explicit use_reentrant)
# ------------------------------------------------------------
def patch_torch_checkpoint_default_use_reentrant_false():
    try:
        import inspect
        import torch.utils.checkpoint as ckpt

        sig = inspect.signature(ckpt.checkpoint)
        if "use_reentrant" not in sig.parameters:
            return

        if getattr(ckpt.checkpoint, "_patched_use_reentrant_default", False):
            return

        _orig = ckpt.checkpoint

        def _wrapped(function, *args, **kwargs):
            kwargs.setdefault("use_reentrant", False)
            return _orig(function, *args, **kwargs)

        _wrapped._patched_use_reentrant_default = True
        ckpt.checkpoint = _wrapped
    except Exception:
        return


def enable_gc_no_reentrant(model):
    if not hasattr(model, "gradient_checkpointing_enable"):
        return
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()


# ------------------------------------------------------------
# HARD disable use_cache everywhere (top-level + nested configs)
# ------------------------------------------------------------
def force_disable_use_cache_everywhere(model):
    """
    Some multimodal wrappers keep nested configs (e.g., language_model.config.use_cache=True)
    even if model.config.use_cache=False. This recursively forces use_cache=False anywhere found.
    """
    visited = set()

    def _set_use_cache(obj):
        if obj is None:
            return
        oid = id(obj)
        if oid in visited:
            return
        visited.add(oid)

        # direct flag
        if hasattr(obj, "use_cache"):
            try:
                obj.use_cache = False
            except Exception:
                pass

        # common nested config holders
        for attr in (
            "config",
            "generation_config",
            "text_config",
            "language_config",
            "vision_config",
            "model_config",
        ):
            if hasattr(obj, attr):
                try:
                    _set_use_cache(getattr(obj, attr))
                except Exception:
                    pass

    # model + common sub-objects
    _set_use_cache(model)
    for attr in ("model", "base_model", "language_model", "vision_tower", "vision_model"):
        if hasattr(model, attr):
            try:
                _set_use_cache(getattr(model, attr))
            except Exception:
                pass

    # configs attached to submodules
    try:
        for m in model.modules():
            if hasattr(m, "config"):
                _set_use_cache(m.config)
            if hasattr(m, "generation_config"):
                _set_use_cache(m.generation_config)
    except Exception:
        pass


@contextmanager
def suppress_use_cache_gc_warning():
    """
    That 'use_cache=True incompatible with gradient checkpointing' line is a logger.warning,
    not a Python warning. We silence the relevant Transformers loggers *only* during GC enable.
    """
    names = [
        "transformers.modeling_utils",
        "transformers",
    ]
    loggers = [logging.getLogger(n) for n in names]
    old_levels = [lg.level for lg in loggers]
    try:
        for lg in loggers:
            lg.setLevel(logging.ERROR)
        yield
    finally:
        for lg, lvl in zip(loggers, old_levels):
            lg.setLevel(lvl)


# ------------------------------------------------------------
# Repro / dtype
# ------------------------------------------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def print_gpu_mem(prefix=""):
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
        a = torch.cuda.memory_allocated() / (1024**3)
        r = torch.cuda.memory_reserved() / (1024**3)
        m = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"{prefix}CUDA mem: allocated={a:.2f}GB reserved={r:.2f}GB max_alloc={m:.2f}GB")
    except Exception:
        pass


# ------------------------------------------------------------
# Chat helpers (text-only)
# ------------------------------------------------------------
def _messages_text_only(prompt: str, target: str | None = None) -> List[Dict[str, Any]]:
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if target is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return msgs


def _ensure_1d_list(x):
    if isinstance(x, dict) and "input_ids" in x:
        x = x["input_ids"]
    if torch.is_tensor(x):
        x = x.tolist()
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
        return x[0]
    return x


def _apply_chat(processor, messages, max_len: int, add_generation_prompt: bool):
    if not hasattr(processor, "apply_chat_template"):
        raise RuntimeError("Processor does not support apply_chat_template; please upgrade transformers.")

    out = processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        truncation=True,
        max_length=max_len,
    )

    input_ids = _ensure_1d_list(out["input_ids"])
    attn = out.get("attention_mask", None)
    attn = _ensure_1d_list(attn) if attn is not None else None
    if attn is None:
        attn = [1] * len(input_ids)

    return {"input_ids": input_ids, "attention_mask": attn}


def tokenize_fn(processor, example, max_len: int):
    prompt = example["prompt"]
    target = example["target"]

    full_msgs = _messages_text_only(prompt, target)
    prompt_msgs = _messages_text_only(prompt, None)

    full = _apply_chat(processor, full_msgs, max_len=max_len, add_generation_prompt=False)
    pref = _apply_chat(processor, prompt_msgs, max_len=max_len, add_generation_prompt=True)

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    # Gemma3 requires token_type_ids during training; text-only => zeros.
    token_type_ids = [0] * len(input_ids)

    labels = list(input_ids)
    prompt_len = min(len(pref["input_ids"]), len(labels))
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
    }


# ------------------------------------------------------------
# Collator (label-safe + token_type_ids-safe)
# ------------------------------------------------------------
@dataclass
class CausalLMPadCollator:
    pad_token_id: int
    label_pad_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attn, ttype, labels = [], [], [], []
        for f in features:
            ids = f["input_ids"]
            am = f.get("attention_mask", [1] * len(ids))
            lb = f["labels"]
            tt = f.get("token_type_ids", [0] * len(ids))

            pad_n = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_n)
            attn.append(am + [0] * pad_n)
            ttype.append(tt + [0] * pad_n)
            labels.append(lb + [self.label_pad_id] * pad_n)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "token_type_ids": torch.tensor(ttype, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ------------------------------------------------------------
# Target modules (avoid vision tower)
# ------------------------------------------------------------
def pick_target_module_paths(model):
    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    banned_tokens = ("vision", "visual", "image", "encoder")

    targets = []
    for name, _module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in suffixes:
            lname = name.lower()
            if any(bt in lname for bt in banned_tokens):
                continue
            targets.append(name)

    if not targets:
        for name, _ in model.named_modules():
            leaf = name.split(".")[-1]
            if leaf in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lname = name.lower()
                if any(bt in lname for bt in banned_tokens):
                    continue
                targets.append(name)

    if not targets:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    return targets


def _supports_trainingargs_kw(cls, kw: str) -> bool:
    try:
        import inspect
        return kw in inspect.signature(cls.__init__).parameters
    except Exception:
        return False


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, default="data/train.jsonl")
    ap.add_argument("--val_jsonl", type=str, default="data/val.jsonl")
    ap.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--out_dir", type=str, default="medgemma4b_icu_qlora_4bit")

    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--eval_steps", type=int, default=0)
    ap.add_argument("--hf_token", type=str, default="")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "linear", "constant"])
    ap.add_argument("--log_steps", type=int, default=10)
    ap.add_argument("--save_merged", action="store_true")

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")

    patch_torch_checkpoint_default_use_reentrant_false()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    torch.backends.cuda.matmul.allow_tf32 = True
    set_all_seeds(args.seed)

    token = get_hf_token(args.hf_token)
    compute_dtype = pick_compute_dtype()
    print("compute_dtype:", compute_dtype)

    # Processor
    processor = try_with_token(AutoProcessor.from_pretrained, args.model_name, token=token)
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        raise SystemExit("AutoProcessor did not expose a tokenizer. Please upgrade transformers.")

    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise SystemExit("Tokenizer has no pad_token_id and no eos_token_id; cannot pad.")
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype if compute_dtype in (torch.float16, torch.bfloat16) else torch.float16,
    )

    print_gpu_mem("[before load] ")
    model = try_with_token(
        AutoModelForImageTextToText.from_pretrained,
        args.model_name,
        quantization_config=bnb_cfg,
        device_map={"": 0},
        token=token,
    )
    print_gpu_mem("[after load]  ")

    # Force-disable cache flags everywhere (top + nested)
    force_disable_use_cache_everywhere(model)

    # QLoRA prep (k-bit) — do NOT enable GC inside this helper
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    # After PEFT kbit prep, nested configs may reappear -> force again
    force_disable_use_cache_everywhere(model)

    # Enable gradient checkpointing (silence the specific logger line even if it tries to flip)
    with suppress_use_cache_gc_warning():
        enable_gc_no_reentrant(model)

    # And force again (belt + suspenders)
    force_disable_use_cache_everywhere(model)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    target_modules = pick_target_module_paths(model)
    print("LoRA target modules count:", len(target_modules))
    print("LoRA target sample:", target_modules[:8])

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    # After wrapping with PEFT, force-disable cache again
    force_disable_use_cache_everywhere(model)

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    train_ds = load_dataset("json", data_files={"train": args.train_jsonl})["train"].shuffle(seed=args.seed)
    val_ds = load_dataset("json", data_files={"val": args.val_jsonl})["val"]

    train_tok = train_ds.map(
        lambda ex: tokenize_fn(processor, ex, args.max_len),
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    )
    val_tok = val_ds.map(
        lambda ex: tokenize_fn(processor, ex, args.max_len),
        remove_columns=val_ds.column_names,
        desc="Tokenizing val",
    )

    collator = CausalLMPadCollator(pad_token_id=pad_id)

    eval_strategy = "epoch" if args.eval_steps == 0 else "steps"
    save_strategy = "epoch" if args.eval_steps == 0 else "steps"

    steps_per_epoch = math.ceil(len(train_tok) / max(1, args.batch * args.grad_accum))
    total_steps = max(1, steps_per_epoch * max(1, int(args.epochs)))
    warmup_steps = int(total_steps * float(args.warmup_ratio))

    ta_kwargs = dict(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=args.log_steps,
        save_total_limit=2,
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        optim="adamw_torch",
        lr_scheduler_type=args.scheduler,
        report_to="none",
        remove_unused_columns=False,
        group_by_length=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
        warmup_steps=warmup_steps,
    )

    if _supports_trainingargs_kw(TrainingArguments, "eval_strategy"):
        ta_kwargs["eval_strategy"] = eval_strategy
    else:
        ta_kwargs["evaluation_strategy"] = eval_strategy

    if _supports_trainingargs_kw(TrainingArguments, "save_strategy"):
        ta_kwargs["save_strategy"] = save_strategy

    if args.eval_steps != 0:
        ta_kwargs["eval_steps"] = args.eval_steps
        ta_kwargs["save_steps"] = args.eval_steps

    train_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
    )

    # Final force-disable just before training (in case anything toggled)
    force_disable_use_cache_everywhere(trainer.model)

    print_gpu_mem("[before train] ")
    trainer.train()
    print_gpu_mem("[after train]  ")

    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    print(f"[ok] saved QLoRA adapter -> {args.out_dir}")
    print("[ok] training logs at:", os.path.join(args.out_dir, "trainer_state.json"))

    if args.save_merged:
        merged_dir = os.path.join(args.out_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        try:
            merged = model.merge_and_unload()
            merged.save_pretrained(merged_dir, safe_serialization=True)
            processor.save_pretrained(merged_dir)
            print(f"[ok] saved merged full model -> {merged_dir}")
        except Exception as e:
            print("[warn] merge failed (adapter still saved). Error:", repr(e))


if __name__ == "__main__":
    warnings.filterwarnings("default")
    main()
