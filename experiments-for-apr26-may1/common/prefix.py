"""Tokenizer helpers for building y-conditioned prompts and generation seeds.

- get_tokenizer(name): cached AutoTokenizer.
- make_seed(y, tok, n): take first n tokens of y, decode back to string. Used
  to seed the teacher's generation (variant V3: prompt = (x,y,o) + seed,
  y_star = seed + completion).
"""

from transformers import AutoTokenizer


_tokenizer_cache = {}


def get_tokenizer(model_name: str):
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    return _tokenizer_cache[model_name]


def make_seed(y: str, tokenizer, n_tokens: int = 7) -> str:
    """First n tokens of y, decoded back to text. Returns '' if y is empty."""
    if not y:
        return ""
    ids = tokenizer.encode(y, add_special_tokens=False)
    n = min(n_tokens, len(ids))
    if n == 0:
        return ""
    return tokenizer.decode(ids[:n], skip_special_tokens=True)
