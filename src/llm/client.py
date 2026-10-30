"""LLM-клиент (OpenAI-совместимый API из .env) + кэш ответов по sha1 промпта."""
import hashlib
import json
import time
from pathlib import Path

import tiktoken

from utils import DATA, ENCODEING, load_env

CACHE = DATA / "llm_cache.json"
ENC = tiktoken.get_encoding(ENCODEING)


def client():
    from openai import OpenAI
    env = load_env()
    return OpenAI(base_url=env["LLM_BASE_URL"], api_key=env["LLM_API_KEY"],
                  max_retries=0), env["LLM_MODEL"]


def load_cache() -> dict:
    return json.load(open(CACHE, encoding="utf-8")) if CACHE.exists() else {}


def save_cache(cache: dict) -> None:
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def ask(cl, model, prompt: str, key: str, cache: dict) -> tuple[str, float]:
    """(text, latency_s); кэш по ключу — повторный прогон ~0 токенов."""
    if key in cache:
        return cache[key]["out"], 0.0
    t0 = time.time()
    r = cl.chat.completions.create(model=model, temperature=0,
                                   messages=[{"role": "user", "content": prompt}])
    out = (r.choices[0].message.content or "").strip()
    dt = round(time.time() - t0, 2)
    cache[key] = {"out": out, "prompt_tok": len(ENC.encode(prompt)),
                  "out_tok": len(ENC.encode(out))}
    return out, dt


def key_for(prefix: str, text: str) -> str:
    return prefix + hashlib.sha1(text.encode()).hexdigest()


def cache_tokens(cache: dict) -> int:
    return sum(v["prompt_tok"] + v["out_tok"] for v in cache.values())
