#!/usr/bin/env python
# medgemma_api.py
# ------------------------------------------------------------
# Local FastAPI for MedGemma 1.5 4B IT + QLoRA adapter
# - CUDA + 4-bit NF4 (bitsandbytes) when available
# - 3 presets: quick / normal / detailed
# - Correct Gemma-style inputs via apply_chat_template(tokenize=True, return_dict=True)
# - FIX: attention_mask/input_ids ALWAYS 2D
# - FIX: reply-empty bug:
#     * if decoded text is empty/whitespace, auto-retry with stronger settings
#     * optional ban of PAD/EOS from generation to avoid "all-special-token" output
# - FIX: pass-2 duplication:
#     * detect when preset output is already complete (esp. quick) and STOP (even if token cap was hit)
#     * safe overlap trimming + "already included" suppression
# - Optional debug: meta.previews + token previews
# ------------------------------------------------------------

import argparse
import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple
from fastapi.middleware.cors import CORSMiddleware

import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

from hf_auth import get_hf_token, try_with_token

DEFAULT_BASE_MODEL = "google/medgemma-1.5-4b-it"


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
    max_passes: int


PRESETS: Dict[str, Preset] = {
    "quick": Preset("quick", 220, 0.20, 0.90, 1.05, False, 150, 2),
    "normal": Preset("normal", 520, 0.40, 0.90, 1.08, True, 350, 3),
    "detailed": Preset("detailed", 1100, 0.55, 0.90, 1.10, True, 900, 4),
}


# -----------------------------
# Schemas
# -----------------------------
class AnalyzeRequest(BaseModel):
    preset: Literal["quick", "normal", "detailed"] = "normal"
    note: str = Field(..., description="Clinical note / case text to analyze.")

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


def build_system_prompt(p: Preset) -> str:
    return (
        "You are a clinical decision-support assistant for clinicians.\n"
        "Rules:\n"
        f"- Keep the answer under ~{p.max_words_hint} words unless absolutely necessary.\n"
        "- Be direct, actionable, and safety-conscious.\n"
        "- If key information is missing, state what you need.\n"
        "- Do NOT output hidden reasoning, chain-of-thought, or internal tags.\n"
        "- Never output '<unused94>thought'.\n"
        "- Output only the final answer.\n"
    )


def build_user_instruction(preset: str) -> str:
    if preset == "quick":
        return (
            "Provide:\n"
            "1) Top problems (max 5 bullets)\n"
            "2) Immediate next steps (max 5 bullets)\n"
            "3) Red flags (max 5 bullets)\n"
            "Keep it concise."
        )
    if preset == "normal":
        return (
            "Provide a structured clinical summary:\n"
            "- SOAP (S, O, A, P)\n"
            "- Task list\n"
            "- Red flags\n"
            "- Patient-friendly summary (2-4 sentences)\n"
            "Keep it practical."
        )
    return (
        "Provide a thorough clinical analysis:\n"
        "- Key problems + severity/urgency\n"
        "- Differential considerations\n"
        "- Recommended workup (tests/imaging/labs)\n"
        "- Treatment/management considerations\n"
        "- Safety / red flags\n"
        "- Patient-friendly summary\n"
        "Avoid overconfidence; state uncertainty clearly."
    )


def _messages_text_only(system_text: str, user_text: str, assistant_text: Optional[str] = None) -> List[Dict[str, Any]]:
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    if assistant_text is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
    return msgs


def _get_tokenizer(processor: Any):
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        raise RuntimeError("AutoProcessor did not expose tokenizer. Upgrade transformers.")
    return tok


def _sanitize_reply(text: str) -> str:
    text = (text or "")
    text = re.sub(r"<unused94>thought.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _supports_generate_kw(model: Any, kw: str) -> bool:
    try:
        import inspect
        return kw in inspect.signature(model.generate).parameters
    except Exception:
        return False


def _is_bullet(line: str) -> bool:
    s = (line or "").strip()
    return bool(s) and (s.startswith("*") or s.startswith("-") or s.startswith("•"))


def _quick_complete(text: str) -> bool:
    """
    True if the quick format appears complete:
      - has Top Problems, Immediate Next Steps, Red Flags
      - and at least 1 bullet under each section
    """
    t = text or ""
    lines = t.splitlines()
    headings = ["top problems", "immediate next steps", "red flags"]

    def has_heading(h: str) -> bool:
        return any(h in (ln or "").lower() for ln in lines)

    def bullets_after(h: str) -> bool:
        # find first heading line
        idx = None
        for i, ln in enumerate(lines):
            if h in (ln or "").lower():
                idx = i
                break
        if idx is None:
            return False

        # scan next ~25 lines until next heading; find any bullet
        for j in range(idx + 1, min(len(lines), idx + 26)):
            lj = (lines[j] or "").lower()
            if any(h2 in lj for h2 in headings) and j != idx:
                break
            if _is_bullet(lines[j]):
                return True
        return False

    if not all(has_heading(h) for h in headings):
        return False
    if not all(bullets_after(h) for h in headings):
        return False
    return True


def _trim_overlap(prev: str, new: str, max_window: int = 900, min_k: int = 40) -> str:
    """
    Removes repeated prefix of `new` that overlaps with suffix of `prev`.
    Very safe: only trims if overlap is at least `min_k` chars.
    """
    prev = prev or ""
    new = new or ""
    if not prev or not new:
        return new

    tail = prev[-max_window:]
    head = new[:max_window]
    max_k = min(len(tail), len(head))
    for k in range(max_k, min_k - 1, -1):
        if tail.endswith(head[:k]):
            return new[k:]
    return new


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

        # generation bans
        self.bad_words_ids: List[List[int]] = []

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
                bnb_4bit_compute_dtype=compute_dtype if compute_dtype in (torch.float16, torch.bfloat16) else torch.float16,
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

        self.model = PeftModel.from_pretrained(self.model, self.adapter_dir, is_trainable=False)
        self.model.eval()

        # enable cache for inference speed
        try:
            if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = True
        except Exception:
            pass
        try:
            if hasattr(self.model, "generation_config") and hasattr(self.model.generation_config, "use_cache"):
                self.model.generation_config.use_cache = True
        except Exception:
            pass

        self.ctx_limit = _estimate_ctx_limit(self.model, fallback=self.ctx_fallback)

        # Build bad_words_ids to avoid "empty decode" (PAD/EOS spam) on retries
        self.bad_words_ids = []
        if self.tok.eos_token_id is not None:
            self.bad_words_ids.append([int(self.tok.eos_token_id)])
        if self.tok.pad_token_id is not None:
            self.bad_words_ids.append([int(self.tok.pad_token_id)])

        # Also ban "<unused94>thought" if it exists
        try:
            tid = self.tok.convert_tokens_to_ids("<unused94>thought")
            if isinstance(tid, int) and tid >= 0 and tid != self.tok.unk_token_id:
                self.bad_words_ids.append([tid])
        except Exception:
            pass

    def health(self) -> Dict[str, Any]:
        return {
            "ok": self.model is not None,
            "cuda_available": torch.cuda.is_available(),
            "gpu": get_gpu_name(),
            "device_requested": self.device,
            "is_loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", None) if self.model is not None else None,
            "ctx_limit_est": self.ctx_limit,
            "base_model": self.base_model,
            "adapter_dir": self.adapter_dir,
            "processor_dir": self.processor_dir,
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
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        do_sample: bool,
        min_new_tokens: int,
        ban_specials: bool,
        debug: bool,
    ) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
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

        if ban_specials and self.bad_words_ids:
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
            toks = self.tok.convert_ids_to_tokens(gen_ids[:50].tolist())
            dbg["first_tokens"] = toks
            dbg["decoded_raw_unsafe"] = self.tok.decode(gen_ids[:200], skip_special_tokens=False)

        return text, usage, dbg

    def generate_with_continuations(
        self,
        preset: Preset,
        note: str,
        overrides: Dict[str, Any],
        debug: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        max_new = int(overrides.get("max_new_tokens") or preset.max_new_tokens)
        temperature = float(overrides.get("temperature") if overrides.get("temperature") is not None else preset.temperature)
        top_p = float(overrides.get("top_p") if overrides.get("top_p") is not None else preset.top_p)
        repetition_penalty = float(
            overrides.get("repetition_penalty") if overrides.get("repetition_penalty") is not None else preset.repetition_penalty
        )

        do_sample = preset.do_sample
        if overrides.get("temperature") is not None:
            do_sample = temperature > 0.0

        system_text = build_system_prompt(preset)
        user_text = build_user_instruction(preset.name) + "\n\nCASE:\n" + note.strip()

        stitched = ""
        passes = 0
        usage_all = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        previews: List[str] = []
        token_debug: List[Dict[str, Any]] = []

        def record(tag: str, txt: str):
            if debug:
                previews.append(f"{tag}: {repr((txt or '')[:400])}")

        def is_empty(s: str) -> bool:
            return (s or "").strip() == ""

        base_min_new = 32 if preset.name == "quick" else 64

        while passes < preset.max_passes:
            # ✅ stop BEFORE creating pass2 if quick is already complete
            if preset.name == "quick" and _quick_complete(stitched):
                break

            passes += 1

            if passes == 1:
                msgs = _messages_text_only(system_text, user_text)
            else:
                # Stronger continuation instruction: ONLY missing text; if complete, output nothing
                msgs = _messages_text_only(system_text, user_text, assistant_text=stitched.strip())
                msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Continue ONLY with missing content. "
                                    "Do NOT repeat any headings or bullets already present above. "
                                    "If everything is already complete, output nothing."
                                ),
                            }
                        ],
                    }
                )

            # Attempt #1
            chunk, usage, dbg = self._generate_once(
                messages=msgs,
                max_new_tokens=max_new,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                min_new_tokens=base_min_new,
                ban_specials=False,
                debug=debug,
            )
            record(f"pass{passes}_decoded", chunk)
            if debug:
                token_debug.append({"pass": passes, "attempt": 1, **dbg})

            # If empty decode, retry with bans + sampling
            if is_empty(chunk):
                retry_msgs = msgs + [
                    {"role": "user", "content": [{"type": "text", "text": "Do NOT return an empty answer. Provide the requested content now."}]}
                ]
                chunk2, usage2, dbg2 = self._generate_once(
                    messages=retry_msgs,
                    max_new_tokens=max_new,
                    temperature=max(0.35, temperature),
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    min_new_tokens=max(128, base_min_new),
                    ban_specials=True,  # crucial on retry
                    debug=debug,
                )
                record(f"pass{passes}_retry_decoded", chunk2)
                if debug:
                    token_debug.append({"pass": passes, "attempt": 2, **dbg2})

                for k in usage_all:
                    usage_all[k] += int(usage.get(k, 0)) + int(usage2.get(k, 0))

                chunk_to_add = chunk2 or ""
            else:
                for k in usage_all:
                    usage_all[k] += int(usage.get(k, 0))
                chunk_to_add = chunk or ""

            chunk_to_add = _sanitize_reply(chunk_to_add)

            # ✅ dedupe: if chunk is already contained, skip it
            if chunk_to_add.strip() and chunk_to_add.strip() in stitched:
                chunk_to_add = ""

            # ✅ dedupe: overlap trim
            chunk_to_add = _trim_overlap(stitched, chunk_to_add)

            # ✅ quick-specific: if we already have complete quick output, ignore further repeated restarts
            if preset.name == "quick" and _quick_complete(stitched) and chunk_to_add.strip().lower().startswith(("**top problems", "top problems")):
                chunk_to_add = ""

            if chunk_to_add.strip():
                stitched = (stitched + "\n" + chunk_to_add).strip()

            stitched = _sanitize_reply(stitched)

            # ✅ quick-specific: stop immediately when complete
            if preset.name == "quick" and _quick_complete(stitched):
                break

            # For non-quick, if model stopped early (didn't use full budget), assume it's done
            if preset.name != "quick":
                if usage.get("completion_tokens", 0) < max_new and not is_empty(stitched):
                    break

            # Stabilize later passes
            temperature = max(0.15, temperature * 0.85)
            do_sample = do_sample and temperature > 0.0

        final_text = _sanitize_reply(stitched)
        if final_text.strip() == "":
            final_text = "No visible text was generated. Try preset='normal' or set temperature=0.6."

        meta: Dict[str, Any] = {
            "passes": passes,
            "ctx_limit": self.ctx_limit,
            "max_new_tokens_requested": int(overrides.get("max_new_tokens") or preset.max_new_tokens),
            "usage": usage_all,
        }
        if debug:
            meta["previews"] = previews
            meta["token_debug"] = token_debug

        return final_text, meta


# -----------------------------
# FastAPI app (lifespan)
# -----------------------------
def create_app(server: ModelServer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        server.load()
        yield

    app = FastAPI(title="MedGemma QLoRA Local API", version="1.0.5", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # local-only dev convenience
        allow_credentials=False,
        allow_methods=["*"],          # includes OPTIONS
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
                    server.generate_with_continuations,
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
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=True)


if __name__ == "__main__":
    main()
