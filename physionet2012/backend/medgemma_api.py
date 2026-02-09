#!/usr/bin/env python
# medgemma_api.py
# ------------------------------------------------------------
# Local FastAPI for MedGemma 1.5 4B IT + QLoRA adapter
#
# FIXES:
# - Fix TemplateError: "Conversation roles must alternate user/assistant/..."
#   -> Do NOT use role="system" with apply_chat_template for Gemma templates.
#   -> Embed system instructions inside a single user message instead.
#
# - Strict JSON output (single object) matching your fine-tuning schema:
#   keys: status, drivers, what_to_check_next, evidence, narrative, disclaimer
#
# - Remove multi-pass continuation stitching (causes "second JSON appended" + truncation issues).
#   -> Use regenerate attempts (attempt/retry) instead.
#
# - Robust JSON extraction: return first valid JSON object with required keys.
#
# - Never ban EOS (banning EOS often leads to truncated JSON).
#   Only ban PAD and "<unused94>thought" sequence.
#
# Keeps:
# - CUDA + optional 4-bit NF4 (bitsandbytes)
# - Presets (quick/normal/detailed)
# - apply_chat_template(tokenize=True, return_dict=True)
# - attention_mask/input_ids ALWAYS 2D
# - PEFT adapter sanity info exposed in /health
# ------------------------------------------------------------

import argparse
import asyncio
import json
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from peft import PeftModel

from hf_auth import get_hf_token, try_with_token

DEFAULT_BASE_MODEL = "google/medgemma-1.5-4b-it"

REQUIRED_JSON_KEYS = [
    "status",
    "drivers",
    "what_to_check_next",
    "evidence",
    "narrative",
    "disclaimer",
]


# -----------------------------
# Presets
# -----------------------------
@dataclass
class Preset:
    name: str
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    do_sample: bool
    max_words_hint: int
    max_attempts: int


PRESETS: Dict[str, Preset] = {
    # For strict JSON, deterministic is usually best; you can override temperature if desired.
    "quick": Preset("quick", 320, 0.0, 1.0, 1.05, False, 150, 2),
    "normal": Preset("normal", 520, 0.0, 1.0, 1.08, False, 350, 2),
    "detailed": Preset("detailed", 900, 0.0, 1.0, 1.10, False, 900, 2),
}


# -----------------------------
# Schemas
# -----------------------------
class AnalyzeRequest(BaseModel):
    preset: Literal["quick", "normal", "detailed"] = "normal"
    note: str = Field(
        ...,
        description="Input text. Ideally includes LAST_12H_WINDOW (+ optional CONTEXT).",
    )

    # Optional overrides:
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None

    debug: bool = False


class AnalyzeResponse(BaseModel):
    preset: str
    reply: str
    meta: Optional[Dict[str, Any]] = None


# -----------------------------
# Helpers
# -----------------------------
def _now_ms() -> int:
    return int(time.time() * 1000)


def pick_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def get_gpu_name() -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def _estimate_ctx_limit(model: Any, fallback: int = 8192) -> int:
    for chain in [
        ("config", "max_position_embeddings"),
        ("language_model", "config", "max_position_embeddings"),
        ("text_config", "max_position_embeddings"),
    ]:
        try:
            obj = model
            for a in chain:
                obj = getattr(obj, a)
            if isinstance(obj, int) and obj > 0:
                return obj
        except Exception:
            continue
    return fallback


def build_instruction_block(p: Preset) -> str:
    # NOTE: we embed this into user message (no role="system")
    # to satisfy Gemma chat-template alternation.
    return (
        "You are an ICU trend summarizer for clinicians.\n"
        "TASK: Given the provided input, output a SINGLE JSON object.\n\n"
        "OUTPUT FORMAT (STRICT):\n"
        "- Output MUST be valid JSON (one object) and NOTHING ELSE.\n"
        "- Keys MUST be exactly:\n"
        " status, drivers, what_to_check_next, evidence, narrative, disclaimer\n"
        "- No extra keys. No markdown. No code fences. No preamble text.\n\n"
        "CONTENT RULES:\n"
        "- Be direct, actionable, safety-conscious.\n"
        f"- Keep it compact (aim ~{p.max_words_hint} words total when possible).\n"
        "- Do NOT invent missing data. If a value is NA / not provided, say it is missing/unknown.\n"
        "- If key info is missing, put the request inside 'what_to_check_next' (snake_case).\n"
        "- 'evidence' must ONLY cite what is present in the input (e.g., 'RR last=32', 'pH last=7.31').\n"
        "- 'drivers' and 'what_to_check_next' should be short snake_case tokens.\n"
        "- 'narrative' should be 1–3 short sentences summarizing status + top concerns.\n"
        "- 'disclaimer' should be a short safety disclaimer.\n\n"
        "SAFETY / SANITIZATION:\n"
        "- Do NOT output hidden reasoning, chain-of-thought, or internal tags.\n"
        "- Never output '<unused94>thought' or any 'thought' tag.\n"
        "- Output only the final JSON object.\n"
    )


def _messages_user_only(user_text: str) -> List[Dict[str, Any]]:
    # Single user message => no alternation issues, Gemma template safe.
    return [{"role": "user", "content": [{"type": "text", "text": user_text}]}]


def _get_tokenizer(processor: Any):
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        raise RuntimeError("AutoProcessor did not expose tokenizer. Upgrade transformers.")
    return tok


def _sanitize_reply(text: str) -> str:
    text = (text or "")
    # Remove markdown code fences if they appear
    text = text.replace("```json", "").replace("```", "")
    # Remove any thought tags
    text = re.sub(r"<unused94>\s*thought", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _supports_generate_kw(model: Any, kw: str) -> bool:
    try:
        import inspect

        return kw in inspect.signature(model.generate).parameters
    except Exception:
        return False


def _extract_json_candidates(s: str) -> List[str]:
    """
    Extract balanced {...} blocks while respecting JSON strings.
    Returns candidates in appearance order.
    """
    s = s or ""
    candidates: List[str] = []
    in_str = False
    esc = False
    depth = 0
    start = None

    for i, ch in enumerate(s):
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
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(s[start : i + 1])
                start = None

    return candidates


def _coerce_and_validate_json(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    if not all(k in obj for k in REQUIRED_JSON_KEYS):
        return None

    cleaned = {k: obj.get(k) for k in REQUIRED_JSON_KEYS}

    if not isinstance(cleaned["drivers"], list):
        cleaned["drivers"] = [] if cleaned["drivers"] is None else [str(cleaned["drivers"])]

    if not isinstance(cleaned["what_to_check_next"], list):
        cleaned["what_to_check_next"] = (
            [] if cleaned["what_to_check_next"] is None else [str(cleaned["what_to_check_next"])]
        )

    if not isinstance(cleaned["evidence"], list):
        cleaned["evidence"] = [] if cleaned["evidence"] is None else [str(cleaned["evidence"])]

    for k in ["status", "narrative", "disclaimer"]:
        if cleaned[k] is None:
            cleaned[k] = ""
        elif not isinstance(cleaned[k], str):
            cleaned[k] = str(cleaned[k])

    return cleaned


def _extract_first_valid_json(text: str) -> Optional[str]:
    for cand in _extract_json_candidates(text):
        try:
            obj = json.loads(cand)
        except Exception:
            continue

        cleaned = _coerce_and_validate_json(obj)
        if cleaned is None:
            continue

        return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))

    return None


# -----------------------------
# Model Server
# -----------------------------
class ModelServer:
    def __init__(
        self,
        base_model: str,
        adapter_dir: str,
        processor_dir: Optional[str],
        use_4bit: bool,
        device: str,
        hf_token: str,
        gpu_semaphore: int,
        ctx_fallback: int = 8192,
    ):
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.processor_dir = processor_dir
        self.use_4bit = use_4bit
        self.device = device
        self.hf_token = hf_token
        self.gpu_sem = asyncio.Semaphore(max(1, gpu_semaphore))
        self.ctx_fallback = ctx_fallback

        self.model = None
        self.processor = None
        self.tok = None
        self.ctx_limit: int = ctx_fallback

        # bad_words_ids: DO NOT ban EOS. Only suppress PAD + "<unused94>thought".
        self.bad_words_ids: List[List[int]] = []
        self.adapter_info: Dict[str, Any] = {}

    def load(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        compute_dtype = pick_compute_dtype()
        token = self.hf_token

        proc_source = self.processor_dir or self.adapter_dir or self.base_model
        self.processor = try_with_token(AutoProcessor.from_pretrained, proc_source, token=token)
        self.tok = _get_tokenizer(self.processor)

        if self.tok.pad_token_id is None:
            if self.tok.eos_token_id is None:
                raise RuntimeError("Tokenizer has no pad_token_id and no eos_token_id.")
            self.tok.pad_token = self.tok.eos_token

        model_kwargs: Dict[str, Any] = {}

        if self.use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=(
                    compute_dtype
                    if compute_dtype in (torch.float16, torch.bfloat16)
                    else torch.float16
                ),
            )

        if self.device.lower() == "cuda" and torch.cuda.is_available():
            model_kwargs["device_map"] = {"": 0}
        else:
            model_kwargs["device_map"] = {"": "cpu"}

        # SDPA if supported; fallback safely
        try:
            model_kwargs["attn_implementation"] = "sdpa"
            self.model = try_with_token(
                AutoModelForImageTextToText.from_pretrained,
                self.base_model,
                token=token,
                **model_kwargs,
            )
        except TypeError:
            model_kwargs.pop("attn_implementation", None)
            self.model = try_with_token(
                AutoModelForImageTextToText.from_pretrained,
                self.base_model,
                token=token,
                **model_kwargs,
            )

        # ---- Load adapter ----
        self.model = PeftModel.from_pretrained(self.model, self.adapter_dir, is_trainable=False)
        self.model.eval()

        # PEFT sanity checks
        is_peft = isinstance(self.model, PeftModel)
        active = getattr(self.model, "active_adapter", None)
        active_list = getattr(self.model, "active_adapters", None)
        if active_list:
            active = active_list

        lora_param_names: List[str] = []
        lora_param_count = 0
        lora_sq_sum = 0.0

        for n, p in self.model.named_parameters():
            if "lora_" in n:
                lora_param_names.append(n)
                lora_param_count += p.numel()
                with torch.no_grad():
                    lora_sq_sum += float((p.detach().float() ** 2).sum().cpu().item())

        lora_norm = (lora_sq_sum**0.5)

        print(
            f"[PEFT] is_peft={is_peft} | active={active} | "
            f"lora_params={lora_param_count:,} | lora_norm={lora_norm:.6f} | "
            f"example_keys={lora_param_names[:3]}"
        )

        self.adapter_info = {
            "is_peft": bool(is_peft),
            "active_adapter": active,
            "lora_param_count": int(lora_param_count),
            "lora_norm": float(lora_norm),
            "example_lora_keys": lora_param_names[:10],
            "peft_config_keys": list(getattr(self.model, "peft_config", {}).keys()),
        }

        if (not is_peft) or (lora_param_count == 0):
            raise RuntimeError(
                "Adapter did not appear to load: no PEFT wrapper or no LoRA parameters found."
            )

        # enable cache for inference speed
        try:
            if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = True
        except Exception:
            pass

        try:
            if hasattr(self.model, "generation_config") and hasattr(
                self.model.generation_config, "use_cache"
            ):
                self.model.generation_config.use_cache = True
        except Exception:
            pass

        self.ctx_limit = _estimate_ctx_limit(self.model, fallback=self.ctx_fallback)

        # bad_words_ids: ban PAD; ban "<unused94>thought" sequence if encodable (never EOS)
        self.bad_words_ids = []
        if self.tok.pad_token_id is not None:
            self.bad_words_ids.append([int(self.tok.pad_token_id)])

        try:
            seq = self.tok.encode("<unused94>thought", add_special_tokens=False)
            if seq and (self.tok.unk_token_id is None or self.tok.unk_token_id not in seq):
                self.bad_words_ids.append([int(x) for x in seq])
        except Exception:
            pass

    def health(self) -> Dict[str, Any]:
        return {
            "ok": self.model is not None,
            "cuda_available": torch.cuda.is_available(),
            "gpu": get_gpu_name(),
            "device_requested": self.device,
            "is_loaded_in_4bit": (
                getattr(self.model, "is_loaded_in_4bit", None) if self.model is not None else None
            ),
            "ctx_limit_est": self.ctx_limit,
            "base_model": self.base_model,
            "adapter_dir": self.adapter_dir,
            "processor_dir": self.processor_dir,
            "adapter": (self.adapter_info or None),
        }

    # -------- always 2D tensors --------
    def _encode_messages(self, messages: List[Dict[str, Any]], max_length: int) -> Dict[str, torch.Tensor]:
        if not hasattr(self.processor, "apply_chat_template"):
            raise RuntimeError("Processor lacks apply_chat_template; update transformers.")

        out = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            truncation=True,
            max_length=max_length,
        )

        def to_2d_long(x) -> torch.Tensor:
            if torch.is_tensor(x):
                if x.dim() == 1:
                    return x.unsqueeze(0).long()
                if x.dim() == 2:
                    return x.long()
                raise ValueError(f"Expected 1D/2D tensor, got dim={x.dim()}")

            if isinstance(x, list):
                if len(x) == 0:
                    return torch.zeros((1, 0), dtype=torch.long)
                if isinstance(x[0], list):
                    return torch.tensor(x, dtype=torch.long)
                return torch.tensor([x], dtype=torch.long)

            raise TypeError(f"Unsupported encoding type: {type(x)}")

        input_ids = to_2d_long(out["input_ids"])

        if "attention_mask" in out and out["attention_mask"] is not None:
            attention_mask = to_2d_long(out["attention_mask"])
        else:
            attention_mask = torch.ones_like(input_ids)

        if "token_type_ids" in out and out["token_type_ids"] is not None:
            token_type_ids = to_2d_long(out["token_type_ids"])
        else:
            token_type_ids = torch.zeros_like(input_ids)

        # explicit 2D guarantees
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D; got {tuple(input_ids.shape)}")
        if attention_mask.dim() != 2:
            raise ValueError(f"attention_mask must be 2D; got {tuple(attention_mask.shape)}")
        if token_type_ids.dim() != 2:
            raise ValueError(f"token_type_ids must be 2D; got {tuple(token_type_ids.shape)}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    @torch.inference_mode()
    def _generate_once(
        self,
        user_text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        do_sample: bool,
        min_new_tokens: int,
        debug: bool,
    ) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        messages = _messages_user_only(user_text)

        margin = 64
        max_in = max(128, self.ctx_limit - int(max_new_tokens) - margin)

        enc = self._encode_messages(messages, max_length=max_in)

        dev = next(self.model.parameters()).device
        for k, v in enc.items():
            enc[k] = v.to(dev)

        input_len = int(enc["input_ids"].shape[1])

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=float(temperature) if do_sample else 1.0,
            top_p=float(top_p) if do_sample else 1.0,
            repetition_penalty=float(repetition_penalty),
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )

        # Never ban EOS. Only suppress PAD / "<unused94>thought".
        if self.bad_words_ids:
            gen_kwargs["bad_words_ids"] = self.bad_words_ids

        if _supports_generate_kw(self.model, "min_new_tokens"):
            gen_kwargs["min_new_tokens"] = int(max(0, min_new_tokens))
        elif _supports_generate_kw(self.model, "min_length"):
            gen_kwargs["min_length"] = int(input_len + max(0, min_new_tokens))

        out = self.model.generate(**enc, **gen_kwargs)

        gen_ids = out[0, input_len:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)

        usage = {
            "prompt_tokens": input_len,
            "completion_tokens": int(gen_ids.numel()),
            "total_tokens": int(input_len + gen_ids.numel()),
        }

        dbg: Dict[str, Any] = {}
        if debug:
            toks = self.tok.convert_ids_to_tokens(gen_ids[:60].tolist())
            dbg["first_tokens"] = toks
            dbg["decoded_raw_unsafe"] = self.tok.decode(gen_ids[:220], skip_special_tokens=False)

        return text, usage, dbg

    def generate_json(
        self,
        preset: Preset,
        note: str,
        overrides: Dict[str, Any],
        debug: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        max_new = int(overrides.get("max_new_tokens") or preset.max_new_tokens)

        temperature = float(
            overrides.get("temperature") if overrides.get("temperature") is not None else preset.temperature
        )
        top_p = float(overrides.get("top_p") if overrides.get("top_p") is not None else preset.top_p)
        repetition_penalty = float(
            overrides.get("repetition_penalty")
            if overrides.get("repetition_penalty") is not None
            else preset.repetition_penalty
        )

        do_sample = preset.do_sample
        if overrides.get("temperature") is not None:
            do_sample = temperature > 0.0

        base_instructions = build_instruction_block(preset)

        attempts = 0
        usage_all = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        previews: List[str] = []
        token_debug: List[Dict[str, Any]] = []
        last_clean = ""

        def record(tag: str, txt: str):
            if debug:
                previews.append(f"{tag}: {repr((txt or '')[:700])}")

        while attempts < preset.max_attempts:
            attempts += 1

            # Regenerate from scratch each attempt (prevents "second JSON object" continuations)
            # On retry: stronger instruction appended inside the SAME user message.
            extra = ""
            if attempts > 1:
                extra = (
                    "\n\nIMPORTANT:\n"
                    "- Output must be EXACTLY one JSON object.\n"
                    "- Do NOT start a second JSON object.\n"
                    "- Ensure all strings are closed and JSON is complete.\n"
                )

            user_text = f"{base_instructions}{extra}\nINPUT:\n{note.strip()}\n"

            # Expand token budget on retry to avoid truncation
            attempt_max_new = max_new if attempts == 1 else int(max_new * 1.7)
            attempt_max_new = max(256, attempt_max_new)

            # Prevent empty short answers
            min_new = 48 if preset.name == "quick" else 72
            if attempts > 1:
                min_new = max(min_new, 120)

            chunk, usage, dbg = self._generate_once(
                user_text=user_text,
                max_new_tokens=attempt_max_new,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                min_new_tokens=min_new,
                debug=debug,
            )

            for k in usage_all:
                usage_all[k] += int(usage.get(k, 0))

            record(f"attempt{attempts}_decoded", chunk)
            if debug:
                token_debug.append({"attempt": attempts, **dbg})

            clean = _sanitize_reply(chunk)
            last_clean = clean

            json_str = _extract_first_valid_json(clean)
            if json_str is not None:
                meta: Dict[str, Any] = {
                    "attempts": attempts,
                    "ctx_limit": self.ctx_limit,
                    "max_new_tokens_requested": int(overrides.get("max_new_tokens") or preset.max_new_tokens),
                    "usage": usage_all,
                }
                if debug:
                    meta["previews"] = previews
                    meta["token_debug"] = token_debug
                return json_str, meta

            # Retry deterministically
            do_sample = False
            temperature = 0.0
            top_p = 1.0

        # Always return valid JSON to the client
        fallback_obj = {
            "status": "unknown",
            "drivers": [],
            "what_to_check_next": [
                "output_was_not_valid_json",
                "increase_max_new_tokens_or_check_input_format",
            ],
            "evidence": [],
            "narrative": "Model output could not be parsed as a valid JSON object with the required keys.",
            "disclaimer": "Demo only. Not medical advice. Clinician review required.",
        }

        fallback = json.dumps(fallback_obj, ensure_ascii=False, separators=(",", ":"))

        meta: Dict[str, Any] = {
            "attempts": attempts,
            "ctx_limit": self.ctx_limit,
            "max_new_tokens_requested": int(overrides.get("max_new_tokens") or preset.max_new_tokens),
            "usage": usage_all,
        }

        if debug:
            meta["previews"] = previews
            meta["token_debug"] = token_debug
            meta["last_unparsed_text_head"] = (last_clean or "")[:900]

        return fallback, meta


# -----------------------------
# FastAPI app (lifespan)
# -----------------------------
def create_app(server: ModelServer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        server.load()
        yield

    app = FastAPI(
        title="MedGemma QLoRA Local API",
        version="1.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # local-only dev convenience
        allow_credentials=False,
        allow_methods=["*"],  # includes OPTIONS
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return server.health()

    @app.post("/v1/analyze", response_model=AnalyzeResponse)
    async def analyze(req: AnalyzeRequest):
        if server.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet.")

        preset = PRESETS.get(req.preset)
        if preset is None:
            raise HTTPException(status_code=400, detail=f"Unknown preset: {req.preset}")

        note = (req.note or "").strip()
        if not note:
            raise HTTPException(status_code=400, detail="Empty note.")

        overrides = {
            "max_new_tokens": req.max_new_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "repetition_penalty": req.repetition_penalty,
        }

        t0 = _now_ms()

        async with server.gpu_sem:
            try:
                reply, meta = await asyncio.to_thread(
                    server.generate_json,
                    preset,
                    note,
                    overrides,
                    req.debug,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Generation failed: {repr(e)}")

        dt = _now_ms() - t0

        return AnalyzeResponse(
            preset=req.preset,
            reply=reply,
            meta=(dict(meta, latency_ms=dt) if req.debug else None),
        )

    return app


# -----------------------------
# Entrypoint
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter-dir", type=str, required=True)
    ap.add_argument("--processor-dir", type=str, default="")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--use-4bit", action="store_true")
    ap.add_argument("--hf-token", type=str, default="")
    ap.add_argument("--gpu-semaphore", type=int, default=1)
    ap.add_argument("--ctx-fallback", type=int, default=8192)
    args = ap.parse_args()

    hf_token = get_hf_token(args.hf_token)

    server = ModelServer(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        processor_dir=args.processor_dir.strip() or None,
        use_4bit=bool(args.use_4bit),
        device=args.device,
        hf_token=hf_token,
        gpu_semaphore=max(1, args.gpu_semaphore),
        ctx_fallback=args.ctx_fallback,
    )

    app = create_app(server)

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
