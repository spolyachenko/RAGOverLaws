"""KT3: сравнение трёх эмбеддинг-моделей разного размера (MTEB RU).

Точки trade-off: лёгкая (<100M) / средняя / тяжёлая (топ MTEB).
Для каждой: размер (МБ), время эмбеддинга корпуса (dedup/token512, 67 022 чанка),
время одного запроса, метрики ретрива на gold100 val (конвенция: топ-40 чанков
-> уникальные документы). Эмбеддинги кэшируются в data/emb_<mname>.npy.

Запуск: python src/retrieval/embed_models.py [e5-small bge-m3 gte-7B ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA, GOLD_VAL  # noqa: E402
from retrieval.metrics import _gold_titles, _dedup_docs, metrics, by_type_metrics  # noqa: E402

# (hf-модель, префикс запроса, префикс чанка). E5 требует prefixes; BGE-M3 и GTE — нет.
MODELS = {
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "bge-m3": ("BAAI/bge-m3", "", ""),
    "gte-7B": ("Alibaba-NLP/gte-Qwen2-7B-instruct", "", ""),
    "gte-1.5B": ("Alibaba-NLP/gte-Qwen2-1.5B-instruct", "", ""),
}
POOL = 40


def _model(mname: str):
    from sentence_transformers import SentenceTransformer
    hf, _, _ = MODELS[mname]
    m = SentenceTransformer(hf, device="cuda").half()
    return m, hf


def model_size_mb(hf: str) -> int:
    from huggingface_hub import snapshot_download
    d = snapshot_download(hf)
    p = Path(d)
    return int(sum(f.stat().st_size for f in list(p.glob("**/*.safetensors"))
                   + list(p.glob("**/*.bin"))) // 1_048_576)


def embed_model(mname: str, ck, queries: list[str], k_list=(1, 5, 10)) -> dict:
    """Эмбеддинг корпуса + gold-метрики; векторы кэшируются (data/emb_<mname>.npy)."""
    m, hf = _model(mname)
    q_pre, p_pre = MODELS[mname][1], MODELS[mname][2]
    g = _gold_titles(GOLD_VAL)

    emb_path = DATA / f"emb_{mname}.npy"
    t0 = time.time()
    if emb_path.exists():
        c_emb = np.load(emb_path)
        idx_secs = None
    else:
        c_emb = m.encode([p_pre + t for t in ck["text"]], batch_size=64,
                         show_progress_bar=True)
        np.save(emb_path, c_emb)
        idx_secs = time.time() - t0
    q_emb = m.encode([q_pre + q for q in queries], batch_size=8)
    per_query_ms = [0.0] * len(queries)
    for i, q in enumerate(queries):  # честное время ОДНОГО запроса (по очереди)
        t1 = time.perf_counter()
        m.encode([q_pre + q])
        per_query_ms[i] = (time.perf_counter() - t1) * 1e3

    idx = (q_emb @ c_emb.T).argsort(axis=1)[:, ::-1]
    topk = {q: _dedup_docs(ck["doc"].iloc[idx[i][:POOL]].tolist()) for i, q in enumerate(queries)}
    ms = sorted(per_query_ms)
    dim = int(c_emb.shape[1])
    return {"model": mname, "hf": hf, "dim": dim,
            "size_mb": model_size_mb(hf),
            "corpus_embed_secs": int(idx_secs) if idx_secs else None,
            "query_ms_median": round(ms[len(ms) // 2], 1),
            "query_ms_p95": round(ms[int(len(ms) * 0.95)], 1),
            "metrics": metrics(g, topk, k_list),
            "by_type": by_type_metrics(g, topk, k_list),
            "topk": topk,
            "chunk_idx": idx[:, :POOL]}  # для hybrid; topk/chunk_idx не сериализуются


if __name__ == "__main__":
    from chunking.chunkers import chunk
    ck = chunk("dedup", "token")
    g = _gold_titles(GOLD_VAL)
    queries = g["question"].tolist()
    names = sys.argv[1:] or list(MODELS)
    for mname in names:
        r = embed_model(mname, ck, queries)
        r.pop("topk")
        r.pop("chunk_idx")
        print(json.dumps(r, ensure_ascii=False))
