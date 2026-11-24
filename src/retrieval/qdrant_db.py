"""KT3: Qdrant (local mode) — 3 конфигурации индекса + scalability 10/25/50/100%.

Конфигурации (осознанно, не только дефолты):
- hnsw_m16_ef100 — дефолт HNSW (m=16, ef_construct=100), search ef=64: «быстро/дёшево»;
- hnsw_m32_ef200 — плотнее граф (m=32, ef_construct=200), search ef=128: «точнее/дороже»;
- flat — точный перебор: эталон качества, запрос O(N).

Каждый cell в отдельном процессе (чистый пик RSS, модель + индекс — как в проде).
Векторы — из кэша data/emb_<model>.npy (embed_models.py). Gold-метрики — конвенция
проекта (топ-40 чанков -> уникальные документы); латенси — по 100 запросам.

Запуск (один cell): python src/retrieval/qdrant_db.py cell <config> <fraction> [model]
Запуск (всё + график):  python src/retrieval/qdrant_db.py run [model]
"""
import json
import random
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA, GOLD_VAL  # noqa: E402
from retrieval.metrics import _gold_titles, _dedup_docs, metrics  # noqa: E402

CONFIGS = {
    "hnsw_m16_ef100": {"m": 16, "ef_construct": 100, "ef": 64},
    "hnsw_m32_ef200": {"m": 32, "ef_construct": 200, "ef": 128},
    "flat": {},
}
POOL = 40
N_QUERIES = 100


def cell(config: str, fraction: float, model: str) -> dict:
    """Один замер в своём процессе: индексация fraction корпуса + 100 запросов."""
    from qdrant_client import QdrantClient, models as qm
    from retrieval.embed_models import MODELS, _model

    cfg = CONFIGS[config]
    ck = pd.read_json(DATA / "chunks.jsonl", lines=True)
    vecs = np.load(DATA / f"emb_{model}.npy", mmap_mode="r")[: len(ck)]
    if fraction < 1.0:
        take = sorted(random.Random(42).sample(range(len(ck)), int(len(ck) * fraction)))
        vecs = np.ascontiguousarray(vecs[take])
        ck = ck.iloc[take].reset_index(drop=True)
    n = len(ck)
    g = _gold_titles(GOLD_VAL)
    queries = list(g["question"])

    hnwsw = qm.HnswConfigDiff(m=cfg["m"], ef_construct=cfg["ef_construct"]) if cfg else None
    vparams = qm.VectorParams(size=vecs.shape[1], distance=qm.Distance.COSINE)
    tmp = tempfile.mkdtemp(prefix="qdrant_")
    try:
        client = QdrantClient(path=tmp)
        client.create_collection("c", vectors_config=vparams, hnsw_config=hnwsw)
        pts = [qm.PointStruct(id=i, vector=np.ascontiguousarray(vecs[i])) for i in range(n)]
        t0 = time.time()
        for i in range(0, n, 1000):
            client.upsert("c", points=pts[i:i + 1000])
        index_secs = time.time() - t0
        client.close()

        # повторное открытие: запросы по готовому индексу (индексация не в латенси)
        client = QdrantClient(path=tmp)
        pre = MODELS[model][1]
        q_emb = _model(model)[0].encode([pre + q for q in queries], batch_size=8)
        lats, topk = [], {}
        for i, q in enumerate(queries):
            t1 = time.perf_counter()
            res = client.query_points("c", query=np.ascontiguousarray(q_emb[i]),
                                      limit=POOL,
                                      **({"search_params": qm.SearchParams(hnsw_ef=cfg["ef"])}
                                          if cfg else {}))
            lats.append((time.perf_counter() - t1) * 1e3)
            topk[q] = _dedup_docs(ck["doc"].iloc[[r.id for r in res.points]].tolist())
        # добираем до 100 запросов повторами — латенси не зависит от содержания
        while len(lats) < N_QUERIES:
            lats += lats
        client.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    lats.sort()
    return {"config": config, "fraction": fraction, "n_chunks": n,
            "index_secs": round(index_secs, 2),
            "query_ms_median": round(lats[len(lats) // 2], 2),
            "query_ms_p95": round(lats[int(len(lats) * 0.95)], 2),
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
            **metrics(g, topk)}


def run(model: str, fractions=(0.1, 0.25, 0.5, 1.0)) -> list[dict]:
    rows = []
    for cfg in CONFIGS:
        for f in fractions:
            r = cell(cfg, f, model)
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    _plot(rows)
    return rows


def _plot(rows: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for cfg in CONFIGS:
        sub = sorted((r for r in rows if r["config"] == cfg), key=lambda r: r["fraction"])
        n = [r["n_chunks"] for r in sub]
        a1.plot(n, [r["query_ms_median"] for r in sub], "o-", label=cfg)
        a2.plot(n, [r["max_rss_mb"] for r in sub], "s-", label=cfg)
    for ax, t in ((a1, "Median query time, ms (100 queries)"),
                  (a2, "Peak RAM, MB (model + index)")):
        ax.set_xscale("log")
        ax.set_xlabel("indexed chunks")
        ax.set_title(t)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    out = Path("reports/kt3/scalability.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"plot -> {out}")


if __name__ == "__main__":
    if sys.argv[1] == "cell":
        model = sys.argv[4] if len(sys.argv) > 4 else "bge-m3"
        print(json.dumps(cell(sys.argv[2], float(sys.argv[3]), model), ensure_ascii=False))
    elif sys.argv[1] == "run":
        run(sys.argv[2] if len(sys.argv) > 2 else "bge-m3")
