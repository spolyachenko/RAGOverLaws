"""KT3: hybrid — RRF и weighted linear combination (alpha-грид).

Слияние на уровне чанков: top-POOL от BM25 (лучшие k1/b из bm25_grid) и
top-POOL от dense (лучшая модель) -> слияние -> уникальные документы -> метрики.
- RRF: score(d) = sum(1/(60 + rank_i(d))), k=60 (стандарт, без подбора весов);
- weighted: minmax-нормализация скоров по запросу, score = alpha*dense + (1-alpha)*bm25,
  alpha = 0..1 шаг 0.1.

Вопрос ТЗ: при каких типах вопросов hybrid выигрывает у каждого компонента.
Запуск: python src/retrieval/hybrid.py [dense_model]
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA, GOLD_VAL  # noqa: E402
from retrieval.metrics import _gold_titles, _dedup_docs, metrics, by_type_metrics  # noqa: E402

POOL = 40
RRF_K = 60


def _bm25_scores(ck, k1, b):
    bm = BM25Okapi([re.findall(r"\w+", t.lower()) for t in ck["text"]], k1=k1, b=b)
    return bm


def run(dense_model: str, best_bm25: dict) -> dict:
    ck = pd.read_json(DATA / "chunks.jsonl", lines=True)
    g = _gold_titles(GOLD_VAL)
    queries = g["question"].tolist()
    c_emb = np.load(DATA / f"emb_{dense_model}.npy")
    from retrieval.embed_models import MODELS, _model
    pre = MODELS[dense_model][1]
    q_emb = _model(dense_model)[0].encode([pre + q for q in queries], batch_size=8)
    d_scores = q_emb @ c_emb.T

    bm = _bm25_scores(ck, best_bm25["k1"], best_bm25["b"])
    b_scores = np.array([bm.get_scores(re.findall(r"\w+", q.lower())) for q in queries])

    d_top = d_scores.argsort(axis=1)[:, ::-1][:, :POOL]
    b_top = b_scores.argsort(axis=1)[:, ::-1][:, :POOL]

    def fuse(score_fn) -> dict[str, list[str]]:
        out = {}
        for i, q in enumerate(queries):
            s = score_fn(i)
            top = s.argsort()[::-1][:POOL * 2]  # уже слитые чанки; берём побольше
            out[q] = _dedup_docs(ck["doc"].iloc[top].tolist())
        return out

    def rrf_score(i):
        s = np.zeros(len(ck))
        for rank, j in enumerate(d_top[i]):
            s[j] += 1.0 / (RRF_K + rank + 1)
        for rank, j in enumerate(b_top[i]):
            s[j] += 1.0 / (RRF_K + rank + 1)
        return s

    def mm(x):  # minmax по запросу (строке)
        lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
        return (x - lo) / np.where(hi - lo == 0, 1.0, hi - lo)

    d_mm, b_mm = mm(d_scores), mm(b_scores)

    t0 = time.time()
    rrf_topk = fuse(rrf_score)
    rrf_secs = time.time() - t0
    rrf_m = metrics(g, rrf_topk)
    dense_topk = {q: _dedup_docs(ck["doc"].iloc[d_top[i][:POOL]].tolist())
                  for i, q in enumerate(queries)}
    bm25_topk = {q: _dedup_docs(ck["doc"].iloc[b_top[i][:POOL]].tolist())
                 for i, q in enumerate(queries)}
    alphas = []
    for a in np.arange(0.0, 1.01, 0.1):
        w_topk = fuse(lambda i, a=a: a * d_mm[i] + (1 - a) * b_mm[i])
        m = metrics(g, w_topk)
        alphas.append({"alpha": round(float(a), 1), **m})
        print(json.dumps({"method": "weighted", **alphas[-1]}, ensure_ascii=False), flush=True)
    best_w = max(alphas, key=lambda r: (r["hit@5"], r["ndcg@5"]))

    # латенси одного запроса: fusion на 1 запрос (CPU, numpy)
    t1 = time.perf_counter()
    rrf_score(0)
    rrf_q_ms = (time.perf_counter() - t1) * 1e3

    return {
        "dense_model": dense_model,
        "bm25": {"params": best_bm25, "metrics": metrics(g, bm25_topk),
                 "by_type": by_type_metrics(g, bm25_topk)},
        "dense": {"metrics": metrics(g, dense_topk),
                  "by_type": by_type_metrics(g, dense_topk)},
        "rrf": {"k": RRF_K, "metrics": rrf_m, "secs_all_queries": round(rrf_secs, 1),
                "query_ms": round(rrf_q_ms, 1),
                "by_type": by_type_metrics(g, rrf_topk)},
        "weighted": {"grid": alphas, "best": best_w,
                     "by_type": by_type_metrics(
                         g, fuse(lambda i: best_w["alpha"] * d_mm[i]
                                 + (1 - best_w["alpha"]) * b_mm[i]))},
    }


if __name__ == "__main__":
    dense_model = sys.argv[1] if len(sys.argv) > 1 else "bge-m3"
    # лучшие bm25-параметры передаём из bm25_grid (data/bm25_grid.json)
    bp = DATA / "bm25_grid.json"
    best_bm25 = json.load(open(bp, encoding="utf-8"))["best"] if bp.exists() \
        else {"k1": 1.5, "b": 0.75}
    print(json.dumps(run(dense_model, best_bm25), ensure_ascii=False))
