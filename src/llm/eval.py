"""LLM-чанки: метрики original vs summary vs HyDE.

BM25 на полном индексе dedup/token (12 067 чанков), 100 чанков подмножества заменены
на summary/HyDE-текст; конвенция — retrieval.bm25 (топ-40 -> уникальные доки).
Запуск: python src/llm/eval.py
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chunking.chunkers import chunk  # noqa: E402
from utils import DATA, GOLD_VAL  # noqa: E402
from retrieval.metrics import _dedup_docs, _gold_titles, metrics  # noqa: E402
from llm.client import CACHE, cache_tokens, load_cache  # noqa: E402

OUT = DATA / "llm_chunks.jsonl"


def eval_llm(k_list=(5, 10), verbose=True) -> dict:
    """BM25-метрики: original vs summary vs HyDE (100 чанков индекса заменены)."""
    ck = pd.read_json(OUT, lines=True)
    base = chunk("dedup", "token")
    # позиционный маппинг (doc, pos) -> строка base-индекса
    pos_in_base, cnt = {}, {}
    for i, d in enumerate(base["doc"]):
        cnt[d] = cnt.get(d, 0)
        pos_in_base[(d, cnt[d])] = i
        cnt[d] += 1
    g = _gold_titles(GOLD_VAL)
    queries = list(g["question"])

    def bm25_texts(texts) -> dict:
        bm = BM25Okapi([re.findall(r"\w+", t.lower()) for t in texts])
        out = {}
        for q in queries:
            top = bm.get_scores(re.findall(r"\w+", q.lower())).argsort()[::-1][:40]
            out[q] = _dedup_docs(base["doc"].iloc[top].tolist())
        return out

    texts_sum, texts_hyde = list(base["text"]), list(base["text"])
    n_repl = 0
    for r in ck.itertuples():
        i = pos_in_base.get((r.doc, r.pos))
        if i is None:
            continue
        texts_sum[i] = r.summary
        texts_hyde[i] = r.hyde
        n_repl += 1
    cache = load_cache()
    topks = {"original": bm25_texts(list(base["text"])),
             "summarized": bm25_texts(texts_sum),
             "hyde": bm25_texts(texts_hyde)}
    out = {"chunks_replaced": n_repl,
           "llm_total_tokens": cache_tokens(cache),
           **{name: metrics(g, topk, k_list) for name, topk in topks.items()}}
    # by-type: проверка гипотезы ТЗ «суммаризация хуже на factual, лучше на abstract»
    by_type = {}
    for t in ("factual", "composite"):
        sub = g[g.type == t]
        by_type[t] = {name: metrics(sub, topk, k_list) for name, topk in topks.items()}
    out["by_type"] = by_type
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    eval_llm()
