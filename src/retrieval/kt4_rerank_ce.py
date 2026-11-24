"""KT4: cross-encoder реранкинг (BAAI/bge-reranker-v2-m3).

Выбор модели: семейство M3 — то же, что эмбеддер baseline (bge-m3), мультиязычная
включая RU, top-MTEB в reranking; 570M — реалистичный размер для GPU-инференса.

Эксперименты (gold100 val, конвенция: top-40 чанков -> уникальные документы):
1. pool: dense top-20/50/100 -> CE-score -> top-40 -> метрики + латенси CE;
2. порог: CE score-порог (pool=100) -> кривая P@5/R@5;
3. сценарии: pool из dense top-100 vs из hybrid(RRF dense+BM25) top-100 -> CE.

Запуск (GPU): python src/retrieval/kt4_rerank_ce.py
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA, GOLD_VAL  # noqa: E402
from retrieval.metrics import _gold_titles, _dedup_docs, metrics, by_type_metrics  # noqa: E402
from retrieval.hybrid import RRF_K  # noqa: E402

CE_MODEL = "BAAI/bge-reranker-v2-m3"
POOLS = (20, 50, 100)


def dense_top100(queries, c_emb):
    from retrieval.embed_models import _model
    m = _model("bge-m3")[0]
    q_emb = m.encode(queries, batch_size=8)
    return (q_emb @ c_emb.T).argsort(axis=1)[:, ::-1][:, :max(POOLS)]


def hybrid_topn(queries, ck, dense100, n):
    grid = json.load(open(DATA / "bm25_grid.json", encoding="utf-8"))
    best = grid["best"]
    from rank_bm25 import BM25Okapi
    bm = BM25Okapi([re.findall(r"\w+", t.lower()) for t in ck["text"]],
                   k1=best["k1"], b=best["b"])
    out = np.empty((len(queries), n), dtype=int)
    for i, q in enumerate(queries):
        s_b = bm.get_scores(re.findall(r"\w+", q.lower()))
        s = np.zeros(len(ck))
        for r, j in enumerate(dense100[i][:n]):
            s[j] += 1.0 / (RRF_K + r + 1)
        for r, j in enumerate(s_b.argsort()[:n][::-1]):
            s[j] += 1.0 / (RRF_K + r + 1)
        out[i] = s.argsort()[::-1][:n]
    return out


def ce_score(model, tok, queries, pairs, texts, batch=64):
    """[n_q, n_pool] логитов релевантности + latency scoring-а по query (мс)."""
    import torch
    lats, out = [], np.empty((len(queries), pairs.shape[1]))
    for i, q in enumerate(queries):
        pairs_q = [(q, texts[j]) for j in pairs[i]]
        t0 = time.perf_counter()
        for b0 in range(0, len(pairs_q), batch):
            b = tok([p[0] + "\n" + p[1] for p in pairs_q[b0:b0 + batch]],
                    truncation=True, max_length=512, padding=True, return_tensors="pt")
            b = {k: v.cuda() for k, v in b.items()}
            with torch.inference_mode():
                s = model(**b).logits.squeeze(-1).cpu().float()
            out[i, b0:b0 + batch] = s.numpy()
        lats.append((time.perf_counter() - t0) * 1e3)
    return out, lats


def reranked_topk(pairs, sc, k=40):
    """Каждую строку pool переставляет по убыванию CE-score; берёт первые k."""
    order = np.argsort(sc, axis=1)[:, ::-1]
    return pairs[np.arange(pairs.shape[0])[:, None], order][:, :k]


def run():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ck = pd.read_json(DATA / "chunks.jsonl", lines=True)
    c_emb = np.load(DATA / "emb_bge-m3.npy")
    g = _gold_titles(GOLD_VAL)
    queries = g["question"].tolist()
    texts = ck["text"].tolist()
    out = {"model": CE_MODEL, "pools": {}, "threshold": {}, "scenarios": {}}

    tok = AutoTokenizer.from_pretrained(CE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(CE_MODEL).cuda().half()
    model.eval()

    d100 = dense_top100(queries, c_emb)

    for n in POOLS:
        pairs, sc, lats = d100[:, :n], *ce_score(model, tok, queries, d100[:, :n], texts)
        ls = sorted(lats)
        topk = {q: _dedup_docs(ck["doc"].iloc[reranked_topk(pairs, sc)[i]].tolist())
                for i, q in enumerate(queries)}
        out["pools"][str(n)] = {"metrics": metrics(g, topk),
                                "by_type_h5": {t: v["hit@5"] for t, v in
                                               by_type_metrics(g, topk).items()},
                                "ce_ms_median": round(ls[len(ls) // 2], 1),
                                "ce_ms_p95": round(ls[int(len(ls) * 0.95)], 1)}
        print(json.dumps({"pool": n, **out["pools"][str(n)]["metrics"]}, ensure_ascii=False),
              flush=True)

    # порог: pool=100; score каждого чанка в pool; порог отбраковывает хвост
    pairs100, sc100, _ = d100[:, :100], *ce_score(model, tok, queries, d100[:, :100], texts)
    score_by_idx = {i: {int(j): float(s) for j, s in zip(pairs100[i], sc100[i])}
                    for i in range(len(queries))}
    curve = []
    for t in np.quantile(sc100, np.arange(0.05, 1.0, 0.05)):
        topk = {}
        for i, q in enumerate(queries):
            order = reranked_topk(pairs100[i:i + 1], sc100[i:i + 1])[0]
            keep = [int(j) for j in order if score_by_idx[i][int(j)] >= t][:40]
            topk[q] = _dedup_docs(ck["doc"].iloc[keep].tolist())
        m = metrics(g, topk, k_list=(5,))
        f1 = 2 * m["precision@5"] * m["recall@5"] / max(1e-9, m["precision@5"] + m["recall@5"])
        curve.append({"thr": round(float(t), 3), **m, "f1@5": round(f1, 4)})
    best_t = max(curve, key=lambda r: r["f1@5"])
    out["threshold"] = {"curve": curve, "best": best_t}
    _threshold_plot(curve)
    print(json.dumps({"threshold_best": best_t}, ensure_ascii=False), flush=True)

    # сценарии: dense top-100 vs hybrid top-100 как pool для CE
    h100 = hybrid_topn(queries, ck, d100, 100)
    for name, src in (("dense", d100), ("hybrid", h100)):
        sc, _ = ce_score(model, tok, queries, src, texts)
        topk = {q: _dedup_docs(ck["doc"].iloc[reranked_topk(src[i:i + 1], sc[i:i + 1])[0]].tolist())
                for i, q in enumerate(queries)}
        out["scenarios"][name] = metrics(g, topk)
        print(json.dumps({"scenario": name, **out["scenarios"][name]}, ensure_ascii=False),
              flush=True)
    return out


def _threshold_plot(curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    t = [c["thr"] for c in curve]
    ax.plot(t, [c["precision@5"] for c in curve], "o-", label="P@5")
    ax.plot(t, [c["recall@5"] for c in curve], "s-", label="R@5")
    ax.plot(t, [c["f1@5"] for c in curve], "^--", label="F1@5")
    ax.set_xlabel("CE score threshold")
    ax.set_title("CE threshold: P@5 / R@5 / F1@5 (pool=100)")
    ax.legend()
    ax.grid(alpha=0.3)
    p = Path("reports/kt4/ce_threshold.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    print(f"plot -> {p}")


if __name__ == "__main__":
    out = run()
    (DATA / "kt4_ce.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print("saved data/kt4_ce.json")
