"""Предобработка корпуса — одна функция на стадию.

raw        — wikitext как есть (data/corpus.jsonl)
markup     — шаблоны {{...}}/HTML: оставить внутренний текст
links      — [[A|B]] -> B, [[A]] -> A, Category/File удалены
normalize  — NFKC, пробелы/переносы
dedup      — exact + MinHash (Jaccard >= 0.7), оставляем самую длинную

Главные функции: run() — полный прогон на диск; load_stage(name) — стадия в памяти.
Артефакт: data/postprocess_corpus.jsonl (статы стадий — в return и stdout).
Запуск: python src/corpus/clean.py
"""
import re
import sys
import unicodedata

import pandas as pd
from datasketch import MinHash, MinHashLSH

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from utils import CORPUS_PATH, DATA  # noqa: E402


def _clean(df: pd.DataFrame, fn) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].map(fn)
    df = df[df["text"].str.len() >= 20].reset_index(drop=True)
    return df


# ---------------------------------------------------------------- стадии

def raw(df: pd.DataFrame) -> pd.DataFrame:
    """Корпус как есть."""
    return _clean(df, lambda t: t)


def markup(df: pd.DataFrame) -> pd.DataFrame:
    """Шаблоны и HTML: оставить внутренний текст, выбросить имена шаблонов и ключи infobox'а."""
    def fn(text: str) -> str:
        t = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
        t = re.sub(r"</?(u|b|i|center|div|p|strong|em)\b[^>]*>", "", t, flags=re.I)
        t = re.sub(r"\{\{dlcalt?\|[^}]*\}\}", "", t)
        t = re.sub(r"\{\{dlc-\}\}", "", t)
        t = re.sub(r"\{\{(r|ip|floor|hearts?|heart|dlc|icon|img|flag)\|([^}]*)\}\}", r"\2", t)
        t = re.sub(r"\{\{[a-zA-Z][^{}]*\}\}", "", t)   # прочие простые {{name|...}}
        t = re.sub(r"\{\{|\}\}", "", t)                  # вложенные хвосты
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t
    return _clean(df, fn)


def links(df: pd.DataFrame) -> pd.DataFrame:
    """[[A|B]] -> B, [[A]] -> A; Category/File/... удалены."""
    def fn(text: str) -> str:
        t = re.sub(r"\[\[(?:Category|File|File:|Image|Image:|Portal|Help|Main)[^\]]*\]\]", "", text)
        t = re.sub(r"\[\[[^\|\]]*\|([^\]]+)\]\]", r"\1", t)
        t = re.sub(r"\[\[([^\]]+)\]\]", r"\1", t)
        return t
    return _clean(df, fn)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """NFKC, NBSP/табы/переносы, отступы строк."""
    def fn(text: str) -> str:
        t = unicodedata.normalize("NFKC", text)
        t = t.replace("\u00a0", " ")
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r" ?\n ?", "\n", t)
        return t.strip()
    return _clean(df, fn)


def dedup(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Exact-дубли + MinHash-LSH 0.7 (в кластере оставляем самую длинную)."""
    before = len(df)
    df = df.drop_duplicates(subset="text", keep="last").reset_index(drop=True)
    exact = before - len(df)
    mhs = []
    for t in df["text"]:
        mh = MinHash(num_perm=128)
        for w in re.findall(r"\w+", t.lower())[:2000]:
            mh.update(w.encode())
        mhs.append(mh)
    lsh = MinHashLSH(threshold=0.7)
    for i, mh in enumerate(mhs):
        lsh.insert(str(i), mh)
    keep = set(range(len(df)))
    for i in range(len(df)):
        for j in lsh.query(mhs[i]):
            j = int(j)
            if j <= i:
                continue  # пару обрабатываем один раз, по меньшему индексу
            if len(df["text"].iloc[i]) >= len(df["text"].iloc[j]):
                keep.discard(j)
            else:
                keep.discard(i)
    near = len(df) - len(keep)
    df = df.iloc[sorted(keep)].reset_index(drop=True)
    return df, {"before": before, "exact_removed": exact, "near_removed": near, "after": len(df)}


# ---------------------------------------------------------------- статы

def stage_junk(df: pd.DataFrame) -> float:
    """Доля «мусорных» токенов: скобки/равно/разметка, числа, URL."""
    toks = pd.concat([pd.Series(t.split()) for t in df["text"]])
    if toks.empty:
        return 0.0
    junk = toks.map(lambda t: bool(re.search(r"[{}\[\]|=<>_*#]", t)) or t.isdigit()
                    or bool(re.match(r"https?://", t)))
    return float(junk.mean())


STAGES = ["raw", "markup", "links", "normalize"]
STAGE_FNS = {"raw": raw, "markup": markup, "links": links, "normalize": normalize}


def _stage_stats(df: pd.DataFrame) -> dict:
    return {"docs": len(df), "junk_share": round(stage_junk(df), 4)}


# ---------------------------------------------------------------- главные

def run() -> dict:
    """Полный прогон raw..dedup. На диск: data/postprocess_corpus.jsonl."""
    df = pd.read_json(CORPUS_PATH, lines=True)
    stats = {}
    for name in STAGES:
        df = STAGE_FNS[name](df)
        stats[name] = _stage_stats(df)
        print(f"{name:10s} docs={len(df):5d}  junk={stats[name]['junk_share']:.2%}")
    df, dd = dedup(df)
    stats["dedup"] = {**dd, "junk_share": round(stage_junk(df), 4)}
    print(f"{'dedup':10s} docs={dd['after']:5d}  exact -{dd['exact_removed']}, "
          f"near -{dd['near_removed']}  junk={stats['dedup']['junk_share']:.2%}")
    DATA.mkdir(exist_ok=True)
    df.to_json(DATA / "postprocess_corpus.jsonl", orient="records", lines=True, force_ascii=False)
    return stats


def load_stage(name: str) -> pd.DataFrame:
    """Стадия корпуса: dedup — с диска; raw..normalize — в памяти (на диск не пишутся)."""
    if name == "dedup":
        return pd.read_json(DATA / "postprocess_corpus.jsonl", lines=True)
    if name not in STAGE_FNS:
        raise ValueError(name)
    df = pd.read_json(CORPUS_PATH, lines=True)
    for n in STAGES:
        df = STAGE_FNS[n](df)
        if n == name:
            return df


if __name__ == "__main__":
    run()
