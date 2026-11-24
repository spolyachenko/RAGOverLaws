"""KT3: BM25-грид параметров k1 x b + время индексирования/запроса.

Границы интервалов (обоснование в отчёте):
- k1 (насыщение TF): 0.5..2.0 — ниже 0.5 TF почти не насыщается (повтор слова
  даёт почти полный вес сразу), выше 2.0 разницу между 1 и 10 вхождениями
  почти не видно; дефолт Elasticsearch/Okapi — 1.2/1.5.
- b (нормировка длины): 0.25..1.0 — 0 = без нормировки (длинные чанки не
  штрафуются), 1 = полная (вес ~ 1/длина); дефолт ES/Okapi — 0.75.

Запуск: python src/retrieval/bm25_grid.py
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import GOLD_VAL  # noqa: E402
from retrieval.metrics import _gold_titles, _dedup_docs, metrics, by_type_metrics  # noqa: E402

K1S = (0.5, 1.0, 1.5, 2.0)
BS = (0.25, 0.5, 0.75, 1.0)
POOL = 40


def bm25_grid(ck, k1s=K1S, bs=BS) -> list[dict]:
    g = _gold_titles(GOLD_VAL)
    queries = g["question"].tolist()
    docs = [re.findall(r"\w+", t.lower()) for t in ck["text"]]
    rows = []
    for k1 in k1s:
        for b in bs:
            t0 = time.time()
            bm = BM25Okapi(docs, k1=k1, b=b)
            idx_secs = time.time() - t0
            lats = []
            topk = {}
            for q in queries:
                qt = re.findall(r"\w+", q.lower())
                t1 = time.perf_counter()
                scores = bm.get_scores(qt)
                top = np.argpartition(scores, -POOL)[-POOL:]
                top = top[scores[top].argsort()[::-1]]
                lats.append((time.perf_counter() - t1) * 1e3)
                topk[q] = _dedup_docs(ck["doc"].iloc[top].tolist())
            lats.sort()
            row = {"k1": k1, "b": b,
                   "index_secs": round(idx_secs, 1),
                   "query_ms_median": round(lats[len(lats) // 2], 1),
                   "query_ms_p95": round(lats[int(len(lats) * 0.95)], 1),
                   **metrics(g, topk)}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    # best: по hit@5, при равенстве — по ndcg@5
    rows.sort(key=lambda r: (r["hit@5"], r["ndcg@5"]), reverse=True)
    best = rows[0]
    out = {"grid": rows, "best": best,
           "best_by_type": by_type_metrics(g, _topk_for(best, ck, queries))}
    return out


def _topk_for(best, ck, queries):
    bm = BM25Okapi([re.findall(r"\w+", t.lower()) for t in ck["text"]],
                   k1=best["k1"], b=best["b"])
    out = {}
    for q in queries:
        s = bm.get_scores(re.findall(r"\w+", q.lower()))
        top = np.argpartition(s, -POOL)[-POOL:]
        top = top[s[top].argsort()[::-1]]
        out[q] = _dedup_docs(ck["doc"].iloc[top].tolist())
    return out


if __name__ == "__main__":
    from chunking.chunkers import chunk
    from utils import DATA
    ck = chunk("dedup", "token")
    out = bm25_grid(ck)
    (DATA / "bm25_grid.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print("=== BEST ===")
    print(json.dumps(out["best"], ensure_ascii=False))
    print("=== BEST BY TYPE ===")
    print(json.dumps(out["best_by_type"], ensure_ascii=False))
