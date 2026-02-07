"""Hugging Face gated model auth WITHOUT `huggingface-cli login`.

Recommended:
- Create a READ token on Hugging Face
- Set an env var:
    HF_TOKEN=...
  or:
    HUGGINGFACE_HUB_TOKEN=...

Then Transformers will use that token to download gated weights.
"""

import os
from typing import Optional

def get_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    if cli_token and cli_token.strip():
        return cli_token.strip()
    for k in ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"]:
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
    return None

def try_with_token(fn, *args, token: Optional[str] = None, **kwargs):
    """Call a from_pretrained-like function with token compatibility handling."""
    if not token:
        return fn(*args, **kwargs)
    try:
        return fn(*args, token=token, **kwargs)
    except TypeError:
        # Older transformers used `use_auth_token`
        return fn(*args, use_auth_token=token, **kwargs)
