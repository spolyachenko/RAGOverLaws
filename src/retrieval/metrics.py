"""Метрики ретрива: Hit@k / Recall@k / Precision@k / NDCG@k / MAP@k / MRR (multi-doc).

Конвенция: topk — question -> список УНИКАЛЬНЫХ doc-названий в порядке ранга
(выбирается из топ-40 чанков, см. bm25_topk / dense_topk).
"""
import json

import pandas as pd


def _gold_titles(gold_path) -> pd.DataFrame:
    """gold -> DataFrame с question/titles; остаются только вопросы с doc_ids."""
    g = pd.DataFrame(json.load(open(gold_path, encoding="utf-8")))
    g = g[g.doc_ids.map(len) > 0].copy()  # только с ground-truth документами
    g["titles"] = g.doc_ids.map(lambda ds: [d.split("://", 1)[1] if "://" in d else d for d in ds])
    return g


def metrics(queries: pd.DataFrame, topk: dict[str, list[str]], k_list=(1, 5, 10)) -> dict:
    """Метрики для question -> список уникальных doc-названий в порядке ранга.

    hit@k — хотя бы один релевантный doc в топ-k; recall@k — доля релевантных в топ-k;
    precision@k — доля топ-k из релевантных; ndcg@k — бинарный DCG (дисконт log2(i+2));
    map@k — среднее по запросам AP@k; mrr — reciprocal rank первого релевантного doc.
    """
    import math

    n = len(queries)
    agg = {f"hit@{k}": 0 for k in k_list}
    sums = {f"{m}@{k}": 0.0 for k in k_list for m in ("recall", "precision", "ndcg", "map")}
    mrr = 0.0
    for q, titles in zip(queries["question"], queries["titles"]):
        tgt = set(titles)
        rel_full = [d in tgt for d in topk.get(q, [])]
        first_rel = next((i for i, r in enumerate(rel_full) if r), None)
        if first_rel is not None:
            mrr += 1.0 / (first_rel + 1)
        for k in k_list:
            rel = rel_full[:k]
            agg[f"hit@{k}"] += int(any(rel))
            sums[f"recall@{k}"] += sum(rel) / len(tgt)
            sums[f"precision@{k}"] += sum(rel) / k
            dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(tgt), k)))
            sums[f"ndcg@{k}"] += dcg / idcg if idcg else 0.0
            hits_k = 0
            ap = 0.0
            for i, r in enumerate(rel):
                if r:
                    hits_k += 1
                    ap += hits_k / (i + 1)
            sums[f"map@{k}"] += ap / min(len(tgt), k) if min(len(tgt), k) else 0.0
    out = {f"hit@{k}": round(agg[f"hit@{k}"] / n, 4) for k in k_list}
    for key, v in sums.items():
        out[key] = round(v / n, 4)
    out["mrr"] = round(mrr / n, 4)
    return out


def by_type_metrics(queries: pd.DataFrame, topk: dict[str, list[str]], k_list=(1, 5, 10)) -> dict:
    """Тот же metrics, но в разрезе по queries['type'] (factual/composite/abstract)."""
    out = {}
    for t in dict.fromkeys(queries["type"]):
        sub = queries[queries["type"] == t].reset_index(drop=True)
        out[t] = {"n": int(len(sub)), **metrics(sub, topk, k_list)}
    return out


def _dedup_docs(docs: list[str]) -> list[str]:
    seen, out = set(), []
    for d in docs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out
