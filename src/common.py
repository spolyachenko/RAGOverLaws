"""Общее для всех КТ: пути, загрузка корпуса, токены."""
import json
from pathlib import Path

import pandas as pd
import tiktoken

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
GOLD_DIR = DATA / "gold"

CORPUS_PATH = DATA / "corpus.jsonl"
GOLD_FULL = GOLD_DIR / "gold_full.json"
GOLD_VAL = GOLD_DIR / "gold_val.json"
GOLD_TEST = GOLD_DIR / "gold_test.json"
BASELINE_OUT = GOLD_DIR / "baseline_no_rag.json"

WIKI_API = "https://bindingofisaacrebirth.wiki.gg/api.php"
ENCODEING = "cl100k_base"
RANDOM_STATE = 42
MIN_DOCS, MIN_TOKENS = 500, 200_000


def load_corpus(path: Path = CORPUS_PATH) -> pd.DataFrame:
    df = pd.DataFrame(json.loads(l) for l in open(path, encoding="utf-8"))
    df["text"] = df["text"].astype(str)
    return df


def with_tokens(df: pd.DataFrame) -> pd.DataFrame:
    enc = tiktoken.get_encoding(ENCODEING)
    df = df.copy()
    df["n_tokens"] = df["text"].map(lambda t: len(enc.encode(t)))
    return df


def assert_minima(df: pd.DataFrame) -> None:
    assert len(df) >= MIN_DOCS, f"минимум {MIN_DOCS} документов, сейчас {len(df)}"
    assert int(df["n_tokens"].sum()) >= MIN_TOKENS, f"минимум {MIN_TOKENS} токенов"


def load_env(path: Path = ROOT / ".env") -> dict:
    """Мини-загрузчик .env (KEY=VALUE, # комментарии)."""
    env = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env
