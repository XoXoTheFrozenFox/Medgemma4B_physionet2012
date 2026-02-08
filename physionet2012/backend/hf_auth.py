# hf_auth.py
import os
import inspect
from typing import Any, Callable


def get_hf_token(cli_token: str = "") -> str:
    """
    Priority:
      1) CLI --hf-token
      2) env HF_TOKEN
      3) env HUGGINGFACEHUB_API_TOKEN
      4) empty
    """
    cli_token = (cli_token or "").strip()
    if cli_token:
        return cli_token

    for k in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        v = (os.getenv(k) or "").strip()
        if v:
            return v
    return ""


def try_with_token(fn: Callable[..., Any], *args, token: str = "", **kwargs) -> Any:
    """
    Calls HF loaders with token if supported.
    Handles both `token=` (newer) and `use_auth_token=` (older).
    """
    token = (token or "").strip()
    if not token:
        return fn(*args, **kwargs)

    try:
        sig = inspect.signature(fn)
        if "token" in sig.parameters:
            return fn(*args, token=token, **kwargs)
        if "use_auth_token" in sig.parameters:
            return fn(*args, use_auth_token=token, **kwargs)
    except Exception:
        pass

    # fallback: try token kw, then no token
    try:
        return fn(*args, token=token, **kwargs)
    except TypeError:
        return fn(*args, **kwargs)
