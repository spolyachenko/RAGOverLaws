"""Dense-ретрив (BGE-M3, fp16/CUDA) и ParentDocumentRetriever.

ParentDocumentRetriever: документ -> родители TokenChunker(p) -> в каждом родителе
дети TokenChunker(c). Индексируются ДЕТИ (dense); топ-k детей мапится на документы
двумя схемами: plain (документы напрямую) и parent_child (через уникальных родителей).
Метрики — та же конвенция, что у BM25 (уникальные документы, gold100).

Запуск: python src/retrieval/dense.py eval <стадия> <чанкер> | python src/retrieval/dense.py parent
"""
import json
import sys
import time
from pathlib import Path

from chonkie import TokenChunker
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.clean import load_stage  # noqa: E402
from utils import GOLD_VAL  # noqa: E402
from chunking.parent import build_parents  # noqa: E402
from retrieval.metrics import _dedup_docs, _gold_titles, metrics  # noqa: E402

MODEL = "BAAI/bge-m3"
COMBOS = [(1024, 256), (1536, 512), (2048, 256)]

_model = None


def _m():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL, device="cuda").half()
        print(f"BGE-M3 загружен ({time.strftime('%H:%M:%S')})", flush=True)
    return _model


def dense_topk(ck, queries: list[str], pool: int = 40) -> dict[str, list[str]]:
    """BGE-M3: топ-pool чанков -> уникальные документы в порядке ранга."""
    model = _m()
    q_emb = model.encode(queries, batch_size=8)
    c_emb = model.encode(list(ck["text"]), batch_size=64, show_progress_bar=True)
    idx = (q_emb @ c_emb.T).argsort(axis=1)[:, ::-1]
    return {q: _dedup_docs(ck["doc"].iloc[idx[i][:pool]].tolist()) for i, q in enumerate(queries)}


def dense_eval(stage: str, strategy: str, k_list=(5, 10), verbose=True, **kw) -> dict:
    """Dense-метрики (BGE-M3) для чанков stage/strategy."""
    from chunking.chunkers import chunk  # lazy: chunkers импортирует bm25_topk
    ck = chunk(stage, strategy, **kw)
    g = _gold_titles(GOLD_VAL)
    queries = g["question"].tolist()
    t0 = time.time()
    topk = dense_topk(ck, queries)
    rows = {f"{stage}/{strategy}/dense ({time.time() - t0:.0f}s)": {
        "chunks": int(len(ck)),
        **metrics(g, topk, k_list),
    }}
    if verbose:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
    return rows


def parent_child_eval(parent: int, child: int, k_list=(5, 10)) -> dict:
    """ParentDocumentRetriever: поиск детей -> plain vs parent/child метрики."""
    model = _m()
    df = load_stage("dedup")
    g = _gold_titles(GOLD_VAL)
    queries = list(g["question"])
    t0 = time.time()
    parents = build_parents(df, parent, child)
    child_texts, child_doc, child_pidx = [], [], []
    for row in parents:
        child_texts += row["c"]
        child_doc += [row["doc"]] * len(row["c"])
        child_pidx += [row["pidx"]] * len(row["c"])
    c_emb = model.encode(child_texts, batch_size=64, show_progress_bar=False)
    idx = (model.encode(queries, batch_size=8) @ c_emb.T).argsort(axis=1)[:, ::-1]
    plain, pc = {}, {}
    for i, q in enumerate(queries):
        pl, pcr, seen_p = [], [], set()
        for j in idx[i][:300]:
            d = child_doc[j]
            if d not in pl:
                pl.append(d)
            key = (d, child_pidx[j])
            if key not in seen_p:
                seen_p.add(key)
                if d not in pcr:
                    pcr.append(d)
        plain[q], pc[q] = pl, pcr
    return {"parent": parent, "child": child, "children": len(child_texts),
            "parents": len(parents), "secs": int(time.time() - t0),
            "plain": metrics(g, plain, k_list),
            "parent_child": metrics(g, pc, k_list)}


def parent_child_grid(combos=COMBOS) -> list[dict]:
    """Все комбинации parent/child (одна загрузка модели)."""
    rows = [parent_child_eval(p, c) for p, c in combos]
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    return rows


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "eval":
        dense_eval(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] == "parent":
        parent_child_grid()
