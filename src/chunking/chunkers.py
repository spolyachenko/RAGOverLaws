"""Чанкинг (chonkie): фабрика чанкеров + отдельная работа с SemanticChunker.

Стратегии: token (512) | sentence (512) | semantic (SemanticChunker, thr 0.8/1024) | neural (distilbert)
Артефакт: data/chunks.jsonl (dedup + token512 — главная линия).
Запуск: python src/chunking/chunkers.py ablation   # semantic: пороги × эмбеддеры
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.clean import load_stage  # noqa: E402
from utils import ENCODEING, GOLD_VAL, DATA  # noqa: E402
from retrieval.bm25 import bm25_topk  # noqa: E402
from retrieval.metrics import _gold_titles, metrics  # noqa: E402

ENC = tiktoken.get_encoding(ENCODEING)


def semantic_chunker(threshold: float = 0.8, model: str = "minishlab/potion-base-32M"):
    """SemanticChunker (chonkie): границы — падение cosine-сходства соседних предложений.

    ВНИМАНИЕ: модель передавать обёрткой SentenceTransformerEmbeddings —
    аргумент model= в SemanticChunker молча игнорируется (падает в **kwargs).
    """
    from chonkie import SemanticChunker
    from chonkie.embeddings import SentenceTransformerEmbeddings
    from sentence_transformers import SentenceTransformer

    if not hasattr(semantic_chunker, "_models"):
        semantic_chunker._models = {}
    if model not in semantic_chunker._models:
        semantic_chunker._models[model] = SentenceTransformer(model)
    enc = semantic_chunker._models[model]
    return SemanticChunker(embedding_model=SentenceTransformerEmbeddings(enc),
                           threshold=threshold, chunk_size=1024)


def token_chunker(strategy: str):
    """Фабрика простых чанкеров: token | sentence | neural."""
    from chonkie import NeuralChunker, SentenceChunker, TokenChunker

    if strategy == "token":
        return TokenChunker(chunk_size=512)
    if strategy == "sentence":
        return SentenceChunker(chunk_size=512)
    if strategy == "neural":
        return NeuralChunker(device_map="cuda")
    raise ValueError(strategy)


def _chunk_texts(df: pd.DataFrame, ch) -> pd.DataFrame:
    rows = []
    for title, chunks in zip(df["title"], ch.chunk_batch(list(df["text"]))):
        rows += [{"doc": title, "idx": len(rows), "text": c.text} for c in chunks]
    ck = pd.DataFrame(rows)
    ck["n_tokens"] = ck["text"].map(lambda t: len(ENC.encode(t)))
    return ck


def chunk(stage: str, strategy: str, force=False, **kw) -> pd.DataFrame:
    """Чанки stage-корпуса стратегией. Кэш на диск — только dedup/token: data/chunks.jsonl."""
    out = DATA / "chunks.jsonl"
    to_disk = (stage, strategy) == ("dedup", "token")
    if to_disk and out.exists() and not force:
        return pd.read_json(out, lines=True)
    df = load_stage(stage)
    ch = semantic_chunker(**kw) if strategy == "semantic" else token_chunker(strategy)
    t0 = time.time()
    ck = _chunk_texts(df, ch)
    if to_disk:
        ck.to_json(out, orient="records", lines=True, force_ascii=False)
    print(f"chunked {stage}/{strategy}: {len(df)} docs -> {len(ck)} chunks "
          f"({time.time() - t0:.1f}s)")
    return ck


def ablation(stage: str = "dedup",
             thresholds=(0.4, 0.5, 0.6, 0.8),
             models=("minishlab/potion-base-32M", "BAAI/bge-m3")):
    """Отдельная работа с SemanticChunker: сетка порог × эмбеддер, BM25-метрики на gold100."""
    g = _gold_titles(GOLD_VAL)
    queries = list(g["question"])
    df = load_stage(stage)
    rows = []
    for mname in models:
        for thr in thresholds:
            t0 = time.time()
            ck = _chunk_texts(df, semantic_chunker(threshold=thr, model=mname))
            topk = bm25_topk(ck, queries)
            rows.append({"model": mname.split("/")[-1], "threshold": thr,
                         "chunks": int(len(ck)), "mean_tok": int(ck.n_tokens.mean()),
                         "secs": int(time.time() - t0), **metrics(g, topk)})
            print(json.dumps(rows[-1], ensure_ascii=False))
    return rows


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "ablation":
        ablation()
