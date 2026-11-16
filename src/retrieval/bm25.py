"""BM25-ретрив (rank_bm25) + метрики на gold100.

Конвенция: топ-40 чанков -> уникальные документы в порядке ранга.
Запуск: python src/retrieval/bm25.py eval <стадия> <чанкер>
  стадии: raw markup links normalize dedup
  чанкеры: token sentence semantic neural
"""
import json
import re
import sys
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import GOLD_VAL  # noqa: E402
from retrieval.metrics import _dedup_docs, _gold_titles, metrics  # noqa: E402


def bm25_topk(ck, queries: list[str], pool: int = 40) -> dict[str, list[str]]:
    """BM25: топ-pool чанков -> уникальные документы в порядке ранга."""
    bm = BM25Okapi([re.findall(r"\w+", t.lower()) for t in ck["text"]])
    out = {}
    for q in queries:
        scores = bm.get_scores(re.findall(r"\w+", q.lower()))
        top = scores.argsort()[::-1][:pool]
        out[q] = _dedup_docs(ck["doc"].iloc[top].tolist())
    return out


def eval_pair(stage: str, strategy: str, k_list=(5, 10), verbose=True, **kw) -> dict:
    """BM25-ретрив по gold100 + Hit@k/Recall@k + статы чанков."""
    from chunking.chunkers import chunk  # lazy: chunkers импортирует bm25_topk
    ck = chunk(stage, strategy, **kw)
    g = _gold_titles(GOLD_VAL)
    queries = g["question"].tolist()
    t0 = time.time()
    topk = bm25_topk(ck, queries)
    rows = {f"{stage}/{strategy}/bm25 ({time.time() - t0:.0f}s)": {
        "chunks": int(len(ck)),
        "mean_tok": int(ck.n_tokens.mean()), "med_tok": int(ck.n_tokens.median()),
        "std_tok": int(ck.n_tokens.std()),
        "share<50": round(float((ck.n_tokens < 50).mean()), 4),
        "share>500": round(float((ck.n_tokens > 500).mean()), 4),
        **metrics(g, topk, k_list),
    }}
    if verbose:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
    return rows


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "eval":
        eval_pair(sys.argv[2], sys.argv[3])
