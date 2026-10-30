"""Метрики ретрива: Hit@k / Recall@k (multi-doc) + загрузка gold-запросов.

Конвенция: topk — question -> список УНИКАЛЬНЫХ doc-названий в порядке ранга
(выбирается из топ-40 чанков, см. bm25_topk / dense_topk).
"""
import json

import pandas as pd


def _gold_titles(gold_path) -> pd.DataFrame:
    """gold -> DataFrame с question/titles; остаются только вопросы с doc_ids."""
    g = pd.DataFrame(json.load(open(gold_path, encoding="utf-8")))
    g = g[g.doc_ids.map(len) > 0].copy()  # только с ground-truth документами
    g["titles"] = g.doc_ids.map(lambda ds: [d.split("wiki://", 1)[1] for d in ds])
    return g


def metrics(queries: pd.DataFrame, topk: dict[str, list[str]], k_list=(5, 10)) -> dict:
    """Метрики для question -> список уникальных doc-названий в порядке ранга.

    hit@k — хотя бы один релевантный doc в топ-k; recall@k — доля релевантных в топ-k;
    precision@k — доля топ-k из релевантных; ndcg@k — бинарный DCG (дисконт log2(i+2)).
    """
    import math

    out = {}
    for k in k_list:
        hits, recs, precs, ndcgs = 0, 0.0, 0.0, 0.0
        for q, titles in zip(queries["question"], queries["titles"]):
            got = topk.get(q, [])[:k]
            tgt = set(titles)
            rel = [d in tgt for d in got]
            hits += int(any(rel))
            recs += sum(rel) / len(tgt)
            precs += sum(rel) / k
            dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(tgt), k)))
            ndcgs += dcg / idcg
        out["hit@%d" % k] = round(hits / len(queries), 4)
        out["recall@%d" % k] = round(recs / len(queries), 4)
        out["precision@%d" % k] = round(precs / len(queries), 4)
        out["ndcg@%d" % k] = round(ndcgs / len(queries), 4)
    return out


def _dedup_docs(docs: list[str]) -> list[str]:
    seen, out = set(), []
    for d in docs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out
