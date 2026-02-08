#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
medgemma_api.py
------------------------------------------------------------
FastAPI backend for MedGemma 4B + local QLoRA adapter (PEFT)

STABILITY-FIRST VERSION (no offload, CPU fallback):
- If --use-4bit is requested on CUDA:
    * load 4-bit NF4 and run a NaN sanity check on next-token logits
    * if NaNs are detected (your case: nan_count == vocab), automatically FALL BACK to CPU FP32
- No accelerate device_map="auto" offload (removes cuda/cpu mismatch crashes)
- CPU FP32 path is slow but stable, and works with PEFT adapters.

Endpoints:
- GET  /health
- POST /v1/generate
- POST /v1/generate_stream (SSE; returns full text at end)

"""

import os
import re
import json
import time
import gc
import argparse
import asyncio
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Literal
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessor,
    LogitsProcessorList,
)
from peft import PeftModel

logger = logging.getLogger("medgemma_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
PING_INTERVAL_S = float(os.environ.get("SSE_PING_INTERVAL_S", "2.0"))

# -----------------------------
# Helpers
# -----------------------------
def has_adapter_files(dir_path: str) -> bool:
    p = Path(dir_path)
    return (p / "adapter_model.safetensors").exists() and (p / "adapter_config.json").exists()


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"^\s*(assistant|ASSISTANT)\s*:\s*", "", s)
    s = s.strip()
    s = re.sub(r"\n{4,}", "\n\n\n", s)
    return s


def build_messages(user_text: str, system_text: Optional[str] = None) -> List[dict]:
    msgs: List[dict] = []
    if system_text and system_text.strip():
        msgs.append({"role": "system", "content": system_text.strip()})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def sse_line(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\r\n\r\n"


def _normalize_messages_for_processor(messages: List[dict]) -> List[dict]:
    """
    Many processor chat templates expect:
      content: [{"type":"text","text":"..."}]
    """
    out: List[dict] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            new_list = []
            for c in content:
                if isinstance(c, str):
                    new_list.append({"type": "text", "text": c})
                elif isinstance(c, dict):
                    if "type" not in c and "text" in c:
                        new_list.append({"type": "text", "text": str(c.get("text", ""))})
                    else:
                        new_list.append(c)
                else:
                    new_list.append({"type": "text", "text": str(c)})
            content_norm = new_list
        elif isinstance(content, str):
            content_norm = [{"type": "text", "text": content}]
        else:
            content_norm = [{"type": "text", "text": str(content)}]
        out.append({"role": role, "content": content_norm})
    return out


def _trim_to_attention(inputs: Dict[str, torch.Tensor], pad_id: Optional[int]) -> Tuple[Dict[str, torch.Tensor], int]:
    if "input_ids" not in inputs:
        raise RuntimeError("Missing input_ids")
    input_ids = inputs["input_ids"]
    if input_ids.dim() != 2:
        raise RuntimeError(f"input_ids must be 2D [B,T], got {tuple(input_ids.shape)}")

    if "attention_mask" in inputs and inputs["attention_mask"] is not None and inputs["attention_mask"].dim() == 2:
        prompt_len = int(inputs["attention_mask"][0].sum().item())
        prompt_len = max(1, min(prompt_len, int(input_ids.shape[-1])))
    else:
        seq = input_ids[0].tolist()
        prompt_len = len(seq)
        if pad_id is not None:
            while prompt_len > 1 and seq[prompt_len - 1] == int(pad_id):
                prompt_len -= 1

    trimmed: Dict[str, torch.Tensor] = {}
    for k, v in inputs.items():
        if torch.is_tensor(v) and v.dim() == 2 and v.shape[:2] == input_ids.shape[:2]:
            trimmed[k] = v[:, :prompt_len]
        else:
            trimmed[k] = v
    return trimmed, prompt_len


def _topk_tokens(tokenizer, scores_1d: torch.Tensor, k: int = 10) -> List[Dict[str, Any]]:
    vals, idx = torch.topk(scores_1d, k=min(k, scores_1d.numel()))
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        try:
            piece = tokenizer.decode([int(i)], skip_special_tokens=False)
        except Exception:
            piece = str(i)
        out.append({"id": int(i), "logit": float(v), "piece": piece})
    return out


# -----------------------------
# Logits processors
# -----------------------------
class SuppressTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, token_ids: List[int]):
        super().__init__()
        self.token_ids = sorted({int(t) for t in token_ids if t is not None})

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.token_ids:
            scores[:, self.token_ids] = float("-inf")
        return scores


class NaNSafeLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores = torch.nan_to_num(scores, nan=float("-inf"), posinf=1e4, neginf=float("-inf"))
        return scores


# -----------------------------
# Stopping criteria
# -----------------------------
class StopOnAnySequence(StoppingCriteria):
    def __init__(self, stop_sequences_ids: List[List[int]]):
        super().__init__()
        self.stop_sequences_ids = [seq for seq in stop_sequences_ids if seq]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.stop_sequences_ids:
            return False
        ids = input_ids[0].tolist()
        for seq in self.stop_sequences_ids:
            if len(ids) >= len(seq) and ids[-len(seq):] == seq:
                return True
        return False


# -----------------------------
# Prompting
# -----------------------------
SYSTEM_BASE = (
    "You are a careful clinical assistant. Do NOT diagnose. "
    "Summarize the provided note, extract tasks, and list red-flag checks. "
    "If information is missing, ask concise follow-up questions."
)

END_MARK = "<END_OF_REPORT>"

ModeName = Literal[
    "All-in-one (SOAP + Tasks + Red flags + Patient summary)",
    "SOAP note only",
    "Task list only",
    "Red flags checklist only",
    "Patient-friendly summary only",
]


def make_user_prompt(mode_name: str, note_text: str, extra: str, *, concise: bool, max_words_hint: int) -> str:
    payload = note_text.strip()
    if extra.strip():
        payload += "\n\nAdditional context:\n" + extra.strip()

    if concise:
        brevity = (
            f"\n\nConstraints:\n"
            f"- Output MUST be plain text (no markdown).\n"
            f"- Be concise (aim for <= ~{max_words_hint} words).\n"
            f"- End the response with this exact marker on its own line:\n{END_MARK}\n"
        )
    else:
        brevity = (
            f"\n\nConstraints:\n"
            f"- Output MUST be plain text (no markdown).\n"
            f"- End the response with this exact marker on its own line:\n{END_MARK}\n"
        )

    if mode_name.startswith("All-in-one"):
        return (
            "Return a SINGLE valid JSON object with these keys:\n"
            "- SOAP: array of strings\n"
            "- Tasks: array of strings (ordered, actionable)\n"
            "- RedFlags: array of strings (confirm/escalate, not diagnoses)\n"
            "- PatientFriendlySummary: string\n"
            "Rules:\n"
            "- Do NOT diagnose.\n"
            "- JSON only (no extra commentary).\n"
            f"{brevity}\n"
            f"NOTE:\n{payload}"
        )

    if mode_name == "SOAP note only":
        return "Write a structured SOAP note (S/O/A/P). Do NOT diagnose." f"{brevity}\nNOTE:\n{payload}"
    if mode_name == "Task list only":
        return "Extract an ordered task list for the clinician. Do NOT diagnose." f"{brevity}\nNOTE:\n{payload}"
    if mode_name == "Red flags checklist only":
        return "Create a red-flag checklist (confirm / escalate). Do NOT diagnose." f"{brevity}\nNOTE:\n{payload}"
    return "Write a patient-friendly summary in plain language. Avoid jargon. Do NOT diagnose." f"{brevity}\nNOTE:\n{payload}"


# -----------------------------
# Settings
# -----------------------------
@dataclass
class ServerSettings:
    base_model: str
    adapter_dir: str
    processor_dir: Optional[str]
    hf_token: Optional[str]
    use_4bit: bool
    device: str
    gpu_index: int
    allow_origins: List[str]
    host: str
    port: int
    gpu_semaphore: int
    warmup: bool
    disable_adapter: bool

    # accepted for CLI compatibility (ignored)
    safe_offload: bool
    offload_dir: str


class ModelManager:
    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self.base_model_obj: Optional[torch.nn.Module] = None
        self.peft_model_obj: Optional[torch.nn.Module] = None
        self.processor = None
        self.tokenizer = None
        self.device: Optional[torch.device] = None
        self.stop_sequences_ids: List[List[int]] = []
        self.sem = asyncio.Semaphore(max(1, int(settings.gpu_semaphore)))

        self.load_mode: str = "unknown"
        self.nan_sanity: Optional[int] = None
        self.adapter_loaded: bool = False
        self.used_cpu_fallback: bool = False

    def _resolve_requested_device(self) -> torch.device:
        if self.settings.device.lower() == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available.")
            return torch.device(f"cuda:{int(self.settings.gpu_index)}")
        if self.settings.device.lower() == "cpu":
            return torch.device("cpu")
        return torch.device(f"cuda:{int(self.settings.gpu_index)}") if torch.cuda.is_available() else torch.device("cpu")

    def _ctx_limit(self) -> int:
        tok_max = getattr(self.tokenizer, "model_max_length", None)
        cfg_max = getattr(getattr(self.base_model_obj, "config", None), "max_position_embeddings", None)

        def _ok(n: Optional[int]) -> bool:
            return isinstance(n, int) and 256 <= n <= 65536

        candidates = [n for n in (tok_max, cfg_max) if _ok(n)]
        if candidates:
            return int(min(candidates))
        return int(os.environ.get("MAX_CTX_DEFAULT", "8192"))

    def _embedding_vocab_size(self) -> Optional[int]:
        m = self.base_model_obj
        if m is None:
            return None
        try:
            emb = m.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight") and emb.weight is not None:
                return int(emb.weight.shape[0])
        except Exception:
            pass
        try:
            return int(getattr(getattr(m, "config", None), "vocab_size", 0) or 0) or None
        except Exception:
            return None

    def _tokenizer_mismatch_guard(self, input_ids: torch.Tensor) -> None:
        v = self._embedding_vocab_size()
        if not v:
            return
        mx = int(input_ids.max().item())
        mn = int(input_ids.min().item())
        if mn < 0 or mx >= v:
            raise RuntimeError(
                f"Tokenizer/model mismatch: input_id range [{mn},{mx}] invalid for embedding_vocab={v}. "
                f"Fix: run server with --processor-dir \"{self.settings.base_model}\"."
            )

    def _build_logits_processor(self) -> LogitsProcessorList:
        assert self.tokenizer is not None
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        procs: List[LogitsProcessor] = [NaNSafeLogitsProcessor()]
        if pad_id is not None and eos_id is not None and int(pad_id) != int(eos_id):
            procs.append(SuppressTokensLogitsProcessor([int(pad_id)]))
        return LogitsProcessorList(procs)

    def _encode_messages(self, messages: List[dict], max_len: int) -> Dict[str, torch.Tensor]:
        if self.processor is None or self.device is None:
            raise RuntimeError("Processor/device not ready")

        msgs_proc = _normalize_messages_for_processor(messages)

        try:
            enc = self.processor.apply_chat_template(
                msgs_proc,
                add_generation_prompt=True,
                tokenize=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError:
            enc = self.processor.apply_chat_template(
                msgs_proc,
                add_generation_prompt=True,
                tokenize=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )

        if torch.is_tensor(enc):
            enc = {"input_ids": enc}
        enc = dict(enc)

        # move tensors to current device
        inputs: Dict[str, torch.Tensor] = {}
        for k, v in enc.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)

        if "attention_mask" not in inputs or inputs["attention_mask"] is None:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer is not None else None
            inputs["attention_mask"] = (inputs["input_ids"] != int(pad_id)).long() if pad_id is not None else torch.ones_like(inputs["input_ids"], dtype=torch.long)

        if "token_type_ids" not in inputs or inputs["token_type_ids"] is None:
            inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"], dtype=torch.long)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer is not None else None
        inputs, prompt_len = _trim_to_attention(inputs, pad_id)

        self._tokenizer_mismatch_guard(inputs["input_ids"])
        inputs["_prompt_len"] = torch.tensor([prompt_len], device=self.device)
        return inputs

    def _get_model(self, use_adapter: bool) -> torch.nn.Module:
        if use_adapter and (self.peft_model_obj is not None) and (not self.settings.disable_adapter):
            return self.peft_model_obj
        if self.base_model_obj is None:
            raise RuntimeError("Base model not loaded")
        return self.base_model_obj

    def _diagnose_next_token_logits(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        if self.tokenizer is None:
            return {"ok": False, "error": "tokenizer missing"}

        fw = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
        }
        if "token_type_ids" in inputs:
            fw["token_type_ids"] = inputs["token_type_ids"]

        with torch.inference_mode():
            out = None
            try:
                out = model(**fw)
            except TypeError:
                fw.pop("token_type_ids", None)
                out = model(**fw)

        logits = getattr(out, "logits", None)
        if logits is None:
            return {"ok": False, "error": "no logits in forward output"}

        last = logits[0, -1, :].detach()
        nan = int(torch.isnan(last).sum().item())
        inf = int(torch.isinf(last).sum().item())

        last_f = last.float()
        safe_last = last_f.nan_to_num(nan=float("-inf"), posinf=1e4, neginf=float("-inf"))

        mx = float(safe_last.max().item()) if safe_last.numel() else None
        mn = float(safe_last.min().item()) if safe_last.numel() else None
        argmax_id = int(torch.argmax(safe_last).item())

        pad_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else None
        eos_id = int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else None

        return {
            "ok": True,
            "nan_count": nan,
            "inf_count": inf,
            "min_logit": mn,
            "max_logit": mx,
            "argmax_id": argmax_id,
            "argmax_piece": self.tokenizer.decode([argmax_id], skip_special_tokens=False),
            "pad_id": pad_id,
            "eos_id": eos_id,
            "pad_is_top": bool(pad_id is not None and argmax_id == pad_id),
            "top10": _topk_tokens(self.tokenizer, safe_last, k=10),
        }

    def _clear_cuda(self) -> None:
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _load_processor(self) -> None:
        token = (self.settings.hf_token or "").strip() or os.environ.get("HF_TOKEN", "").strip() or None
        processor_dir = (self.settings.processor_dir or "").strip() or self.settings.base_model
        self.processor = AutoProcessor.from_pretrained(processor_dir, token=token)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise RuntimeError("AutoProcessor did not expose a tokenizer.")

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise RuntimeError("Tokenizer has no pad_token_id and no eos_token_id.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.stop_sequences_ids = []
        for seq in [END_MARK, "\n" + END_MARK, "\n\n" + END_MARK]:
            ids = self.tokenizer(seq, add_special_tokens=False).input_ids
            if ids:
                self.stop_sequences_ids.append(ids)

    def _try_load_cuda_4bit(self) -> bool:
        """
        Returns True if CUDA 4-bit load is usable (no NaNs in sanity check),
        else returns False (caller should fallback to CPU).
        """
        assert self.device is not None
        token = (self.settings.hf_token or "").strip() or os.environ.get("HF_TOKEN", "").strip() or None

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # Keep compute in fp16 (less weirdness than fp32 in some builds); still slowish.
            bnb_4bit_compute_dtype=torch.float16,
        )

        logger.info("Loading BASE model on CUDA in 4-bit NF4...")
        base = AutoModelForImageTextToText.from_pretrained(
            self.settings.base_model,
            token=token,
            quantization_config=quant_cfg,
            device_map={"": int(self.settings.gpu_index)},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        base.eval()

        self.base_model_obj = base
        self.load_mode = "cuda_4bit_nf4"

        # sanity check
        try:
            msgs = build_messages("Say OK.", system_text="You are helpful.")
            inputs = self._encode_messages(msgs, max_len=256)
            _ = inputs.pop("_prompt_len")
            diag = self._diagnose_next_token_logits(self.base_model_obj, inputs)
            self.nan_sanity = int(diag.get("nan_count", -1))
            if diag.get("ok") and int(diag.get("nan_count", 0)) > 0:
                logger.warning(f"NaNs detected in 4-bit logits (nan_count={diag.get('nan_count')}). CUDA 4-bit is unusable -> falling back to CPU FP32.")
                return False
        except Exception as e:
            logger.warning(f"CUDA 4-bit sanity check failed -> falling back to CPU FP32: {e}")
            return False

        return True

    def _load_cpu_fp32(self) -> None:
        token = (self.settings.hf_token or "").strip() or os.environ.get("HF_TOKEN", "").strip() or None

        logger.info("Loading BASE model on CPU FP32 (slow but stable)...")
        base = AutoModelForImageTextToText.from_pretrained(
            self.settings.base_model,
            token=token,
            device_map=None,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
        base.to(torch.device("cpu"))
        base.eval()

        self.base_model_obj = base
        self.device = torch.device("cpu")
        self.load_mode = "cpu_fp32"
        self.used_cpu_fallback = True

        # sanity check
        try:
            msgs = build_messages("Say OK.", system_text="You are helpful.")
            inputs = self._encode_messages(msgs, max_len=256)
            _ = inputs.pop("_prompt_len")
            diag = self._diagnose_next_token_logits(self.base_model_obj, inputs)
            self.nan_sanity = int(diag.get("nan_count", -1))
            if diag.get("ok") and int(diag.get("nan_count", 0)) > 0:
                logger.warning(f"NaNs detected even on CPU FP32 (nan_count={diag.get('nan_count')}). That indicates a broken install / model load.")
        except Exception as e:
            logger.warning(f"CPU sanity check failed: {e}")

    def _load_adapter(self) -> None:
        self.adapter_loaded = False
        self.peft_model_obj = None

        if self.settings.disable_adapter:
            return

        try:
            logger.info(f"Loading adapter: {self.settings.adapter_dir}")
            peft = PeftModel.from_pretrained(self.base_model_obj, self.settings.adapter_dir)
            # Keep everything on the same device & dtype as base
            if self.device is not None:
                peft.to(self.device)
            # If base is CPU FP32, cast adapter weights too (stability)
            if self.load_mode.startswith("cpu"):
                peft.to(dtype=torch.float32)
            peft.eval()
            self.peft_model_obj = peft
            self.adapter_loaded = True
        except Exception as e:
            logger.warning(f"Adapter load failed (continuing without adapter): {e}")
            self.peft_model_obj = None
            self.adapter_loaded = False

    def load(self) -> None:
        # validate adapter path
        adapter_dir = self.settings.adapter_dir
        if not Path(adapter_dir).exists():
            raise RuntimeError(f"Adapter dir not found: {adapter_dir}")
        if (not self.settings.disable_adapter) and (not has_adapter_files(adapter_dir)):
            raise RuntimeError(f"Adapter files missing in: {adapter_dir} (need adapter_model.safetensors + adapter_config.json)")

        # processor/tokenizer first
        self._load_processor()

        requested = self._resolve_requested_device()
        self.device = requested

        # ignore safe offload (kept only to not break your CLI)
        if self.settings.safe_offload:
            logger.warning("--safe-offload is ignored in this stability build (it caused cuda/cpu mismatch crashes). If CUDA fails, we fall back to CPU.")

        # Try CUDA 4-bit if requested
        loaded_ok = False
        if self.device.type == "cuda" and self.settings.use_4bit:
            try:
                loaded_ok = self._try_load_cuda_4bit()
            except Exception as e:
                logger.warning(f"CUDA 4-bit load failed -> falling back to CPU FP32: {e}")
                loaded_ok = False

        # If CUDA 4-bit not requested or unusable => CPU FP32
        if not loaded_ok:
            # cleanup GPU leftovers
            self.base_model_obj = None
            self.peft_model_obj = None
            gc.collect()
            self._clear_cuda()

            self._load_cpu_fp32()

        # Load adapter (on whichever device we ended up)
        self._load_adapter()

        # Warmup (safe): do a tiny forward only (no generate)
        if self.settings.warmup:
            try:
                msgs = build_messages("Say OK.", system_text="You are helpful.")
                inputs = self._encode_messages(msgs, max_len=128)
                _ = inputs.pop("_prompt_len")
                model = self._get_model(use_adapter=bool(self.adapter_loaded))
                with torch.inference_mode():
                    fw = {"input_ids": inputs["input_ids"], "attention_mask": inputs.get("attention_mask")}
                    if "token_type_ids" in inputs:
                        fw["token_type_ids"] = inputs["token_type_ids"]
                    try:
                        _ = model(**fw)
                    except TypeError:
                        fw.pop("token_type_ids", None)
                        _ = model(**fw)
            except Exception as e:
                logger.warning(f"Warmup forward failed (continuing anyway): {e}")

    def health_payload(self) -> Dict[str, Any]:
        ok = self.base_model_obj is not None
        cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda else None

        param_device = "unknown"
        if ok and self.base_model_obj is not None:
            try:
                p = next(self.base_model_obj.parameters())
                param_device = str(p.device)
            except Exception:
                pass

        mem = {}
        if cuda:
            mem = {
                "cuda_alloc_gb": round(torch.cuda.memory_allocated() / (1024**3), 3),
                "cuda_reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 3),
            }

        tok_mml = getattr(self.tokenizer, "model_max_length", None) if self.tokenizer is not None else None
        cfg_mpe = getattr(getattr(self.base_model_obj, "config", None), "max_position_embeddings", None) if self.base_model_obj is not None else None

        return {
            "ok": ok,
            "cuda_available": cuda,
            "gpu": gpu_name,
            "param_device": param_device,
            **mem,
            "base_model": self.settings.base_model,
            "adapter_dir": self.settings.adapter_dir,
            "adapter_disabled": bool(self.settings.disable_adapter),
            "adapter_loaded": bool(self.adapter_loaded),
            "load_mode": self.load_mode,
            "used_cpu_fallback": bool(self.used_cpu_fallback),
            "nan_sanity": self.nan_sanity,
            "tokenizer_model_max_length": tok_mml,
            "config_max_position_embeddings": cfg_mpe,
            "ctx_max_effective": int(self._ctx_limit()) if ok else None,
            "embedding_vocab": self._embedding_vocab_size(),
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else None,
            "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer else None,
        }


# -----------------------------
# API schemas (keep compatibility with your PS script)
# -----------------------------
class GenerateRequest(BaseModel):
    note: str = Field(..., min_length=1)
    extra_context: str = Field("")
    mode: ModeName = Field("All-in-one (SOAP + Tasks + Red flags + Patient summary)")

    max_input_len: int = Field(1024, ge=256, le=8192)

    # PS expects this field name
    max_total_new_tokens: int = Field(0, ge=0, le=8192)

    # compatibility fields
    chunk_new_tokens: int = Field(256, ge=16, le=2048)
    auto_continue: bool = Field(True)

    temperature: float = Field(0.0, ge=0.0, le=1.5)
    top_p: float = Field(1.0, ge=0.1, le=1.0)
    repetition_penalty: float = Field(1.0, ge=1.0, le=1.5)
    no_repeat_ngram_size: int = Field(0, ge=0, le=10)

    max_time_s: float = Field(0.0, ge=0.0, le=3600.0)

    concise: bool = Field(True)
    use_end_marker_stop: bool = Field(True)


class GenerateResponse(BaseModel):
    text: str
    meta: Dict[str, Any]


# -----------------------------
# FastAPI app
# -----------------------------
def create_app(settings: ServerSettings) -> FastAPI:
    manager = ModelManager(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.to_thread(manager.load)
        yield
        try:
            manager.base_model_obj = None
            manager.peft_model_obj = None
            manager.processor = None
            manager.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    app = FastAPI(title="MedGemma 4B Backend (Stability)", version="3.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _decode_generated(tokenizer, out_ids: torch.Tensor, prompt_len: int) -> Tuple[str, Dict[str, Any]]:
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id

        seq = out_ids[0].tolist()
        gen = seq[prompt_len:]

        pad_count = gen.count(pad_id) if pad_id is not None else 0
        eos_count = gen.count(eos_id) if eos_id is not None else 0
        special = set([t for t in (pad_id, eos_id) if t is not None])
        effective = [t for t in gen if t not in special]

        text = tokenizer.decode(gen, skip_special_tokens=True)
        text = normalize_text(text)

        dbg = {
            "prompt_len": int(prompt_len),
            "raw_new_len": int(len(gen)),
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
            "pad_count_raw_new": int(pad_count),
            "eos_count_raw_new": int(eos_count),
            "effective_non_special_tokens": int(len(effective)),
            "first_32_gen_ids": gen[:32],
        }
        return text, dbg

    def _run_generate_once(req: GenerateRequest, messages: List[dict], *, use_adapter: bool) -> Tuple[str, Dict[str, Any]]:
        if manager.base_model_obj is None or manager.tokenizer is None or manager.device is None:
            raise RuntimeError("Model not loaded")

        inputs = manager._encode_messages(messages, max_len=req.max_input_len)
        prompt_len = int(inputs.pop("_prompt_len")[0].item())

        ctx_max = manager._ctx_limit()
        room = max(1, ctx_max - prompt_len - 8)
        max_new = room if req.max_total_new_tokens <= 0 else min(int(req.max_total_new_tokens), room)
        max_new = max(1, int(max_new))

        do_sample = float(req.temperature) > 0.0
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new,
            do_sample=do_sample,
            temperature=float(req.temperature) if do_sample else None,
            top_p=float(req.top_p) if do_sample else None,
            repetition_penalty=float(req.repetition_penalty) if float(req.repetition_penalty) > 1.0 else None,
            no_repeat_ngram_size=int(req.no_repeat_ngram_size) if int(req.no_repeat_ngram_size) > 0 else None,
            eos_token_id=manager.tokenizer.eos_token_id,
            pad_token_id=manager.tokenizer.pad_token_id,
            use_cache=False,  # safest
            num_beams=1,
            min_new_tokens=1,
            renormalize_logits=True,
            logits_processor=manager._build_logits_processor(),
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        if req.use_end_marker_stop and manager.stop_sequences_ids:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnAnySequence(manager.stop_sequences_ids)])

        if req.max_time_s and float(req.max_time_s) > 0:
            gen_kwargs["max_time"] = float(req.max_time_s)

        model = manager._get_model(use_adapter=use_adapter)

        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        latency = time.time() - t0

        out_ids = out.sequences if hasattr(out, "sequences") else out
        text, decode_dbg = _decode_generated(manager.tokenizer, out_ids, prompt_len)

        finish_reason = "unknown"
        if req.use_end_marker_stop and END_MARK in text:
            before = text.split(END_MARK, 1)[0].rstrip()
            if before.strip():
                text = before
                finish_reason = "end_marker"

        if finish_reason == "unknown":
            eff = int(decode_dbg.get("effective_non_special_tokens", 0))
            finish_reason = "only_special_tokens_generated" if eff == 0 else ("length_cap_reached" if decode_dbg["raw_new_len"] >= max_new else "stopped_early")

        meta: Dict[str, Any] = {
            "device": str(manager.device),
            "load_mode": manager.load_mode,
            "used_cpu_fallback": bool(manager.used_cpu_fallback),
            "ctx_max": int(ctx_max),
            "prompt_tokens": int(prompt_len),
            "raw_generated_tokens": int(decode_dbg["raw_new_len"]),
            "effective_text_tokens": int(decode_dbg["effective_non_special_tokens"]),
            "finish_reason": str(finish_reason),
            "latency_s": round(latency, 4),
            "tok_per_s": round((decode_dbg["raw_new_len"] / latency), 2) if latency > 1e-6 else None,
            "max_total_new_tokens_requested": int(req.max_total_new_tokens),
            "max_total_new_tokens_effective": int(max_new),
            "use_adapter": bool(use_adapter and manager.adapter_loaded and (not manager.settings.disable_adapter)),
            "decode_debug": decode_dbg,
        }

        if finish_reason == "only_special_tokens_generated":
            try:
                meta["logits_diagnostics"] = manager._diagnose_next_token_logits(model, inputs)
            except Exception as e:
                meta["logits_diagnostics"] = {"ok": False, "error": str(e)}

        if req.mode.startswith("All-in-one"):
            try:
                meta["parsed_json"] = json.loads(text)
                meta["parsed_json_ok"] = True
            except Exception:
                meta["parsed_json_ok"] = False

        return text, meta

    @app.get("/health")
    def health():
        return manager.health_payload()

    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        if manager.base_model_obj is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        max_words_hint = 250 if req.max_total_new_tokens <= 0 else max(120, min(450, int(req.max_total_new_tokens * 0.35)))
        user_prompt = make_user_prompt(req.mode, req.note, req.extra_context, concise=req.concise, max_words_hint=max_words_hint)
        messages = build_messages(user_prompt, SYSTEM_BASE)

        async with manager.sem:
            try:
                text, meta = await asyncio.to_thread(_run_generate_once, req, messages, use_adapter=True)

                # fallback to base if adapter collapses
                if meta.get("finish_reason") == "only_special_tokens_generated" and manager.adapter_loaded:
                    text2, meta2 = await asyncio.to_thread(_run_generate_once, req, messages, use_adapter=False)
                    meta["fallback_meta"] = meta2
                    meta["fallback_text_preview"] = (text2[:400] if text2 else "")
                    if meta2.get("finish_reason") != "only_special_tokens_generated" and (text2.strip() != ""):
                        meta["fallback_used"] = True
                        meta["fallback_reason"] = "adapter_collapse_only_specials"
                        text = text2
                    else:
                        meta["fallback_used"] = False

            except RuntimeError as e:
                raise HTTPException(status_code=400, detail=str(e))

        return GenerateResponse(text=text, meta=meta)

    @app.post("/v1/generate_stream")
    async def generate_stream(req: GenerateRequest, request: Request):
        if manager.base_model_obj is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        max_words_hint = 250 if req.max_total_new_tokens <= 0 else max(120, min(450, int(req.max_total_new_tokens * 0.35)))
        user_prompt = make_user_prompt(req.mode, req.note, req.extra_context, concise=req.concise, max_words_hint=max_words_hint)
        messages = build_messages(user_prompt, SYSTEM_BASE)

        async def event_gen():
            yield ": ping\r\n\r\n"

            loop = asyncio.get_running_loop()
            done_evt = asyncio.Event()
            result: Dict[str, Any] = {}

            def worker():
                nonlocal result
                try:
                    text, meta = _run_generate_once(req, messages, use_adapter=True)

                    if meta.get("finish_reason") == "only_special_tokens_generated" and manager.adapter_loaded:
                        text2, meta2 = _run_generate_once(req, messages, use_adapter=False)
                        meta["fallback_meta"] = meta2
                        meta["fallback_text_preview"] = (text2[:400] if text2 else "")
                        if meta2.get("finish_reason") != "only_special_tokens_generated" and (text2.strip() != ""):
                            meta["fallback_used"] = True
                            meta["fallback_reason"] = "adapter_collapse_only_specials"
                            text = text2
                        else:
                            meta["fallback_used"] = False

                    result = {"text": text, "meta": meta}
                except Exception as e:
                    result = {"error": str(e)}
                finally:
                    loop.call_soon_threadsafe(done_evt.set)

            async with manager.sem:
                threading.Thread(target=worker, daemon=True).start()

                while not done_evt.is_set():
                    if await request.is_disconnected():
                        return
                    try:
                        await asyncio.wait_for(done_evt.wait(), timeout=PING_INTERVAL_S)
                    except asyncio.TimeoutError:
                        yield ": ping\r\n\r\n"

                if "error" in result:
                    yield sse_line({"done": True, "error": result["error"]})
                    return

                yield sse_line({"delta": result["text"]})
                yield sse_line({"done": True, "meta": result["meta"]})

        headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)

    return app


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> ServerSettings:
    p = argparse.ArgumentParser(description="Run MedGemma 4B FastAPI backend (Stability)")
    p.add_argument("--base-model", type=str, required=True)
    p.add_argument("--adapter-dir", type=str, required=True)
    p.add_argument("--processor-dir", type=str, default="")
    p.add_argument("--hf-token", type=str, default="")
    p.add_argument("--use-4bit", action="store_true")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--gpu-index", type=int, default=0)

    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu-semaphore", type=int, default=1)
    p.add_argument("--warmup", action="store_true")
    p.add_argument("--allow-origins", type=str, default="*")
    p.add_argument("--disable-adapter", action="store_true")

    # kept only so your existing commands don't break (ignored)
    p.add_argument("--safe-offload", action="store_true")
    p.add_argument("--offload-dir", type=str, default=str(Path(__file__).parent / "hf_offload"))

    args = p.parse_args()

    allow_origins = [o.strip() for o in args.allow_origins.split(",")] if args.allow_origins else ["*"]
    if allow_origins == [""]:
        allow_origins = ["*"]

    return ServerSettings(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        processor_dir=args.processor_dir.strip() or None,
        hf_token=args.hf_token.strip() or None,
        use_4bit=bool(args.use_4bit),
        device=args.device,
        gpu_index=int(args.gpu_index),
        allow_origins=allow_origins,
        host=args.host,
        port=int(args.port),
        gpu_semaphore=int(args.gpu_semaphore),
        warmup=bool(args.warmup),
        disable_adapter=bool(args.disable_adapter),
        safe_offload=bool(args.safe_offload),
        offload_dir=str(args.offload_dir),
    )


def main():
    settings = parse_args()
    app = create_app(settings)
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
