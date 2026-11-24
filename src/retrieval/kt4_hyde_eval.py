"""KT4: HyDE-поиск на dense baseline (ноут, numpy).

Ищем по эмбеддингу гипотетического документа (kt4_hyde_emb.npy, с 153) вместо
эмбеддинга вопроса. Метрики vs baseline (emb вопроса) — overall и by type.
Запуск (ноут): python src/retrieval/kt4_hyde_eval.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA, GOLD_VAL  # noqa: E402
from retrieval.metrics import _gold_titles, _dedup_docs, metrics, by_type_metrics  # noqa: E402

POOL = 40


def main():
    g = _gold_titles(GOLD_VAL)
    queries = g["question"].tolist()
    ck = pd.read_json(DATA / "chunks.jsonl", lines=True)
    c_emb = np.load(DATA / "emb_bge-m3.npy", mmap_mode="r")
    q_emb = np.load(DATA / "kt4_q_emb.npy")
    h_emb = np.load(DATA / "kt4_hyde_emb.npy")
    hyde = json.load(open(DATA / "kt4_hyde.json", encoding="utf-8"))

    def topk(emb):
        idx = (emb @ c_emb.T).argsort(axis=1)[:, ::-1][:, :POOL]
        return {q: _dedup_docs(ck["doc"].iloc[idx[i]].tolist()) for i, q in enumerate(queries)}

    base, hyde_topk = topk(q_emb), topk(h_emb)
    out = {
        "baseline": {"metrics": metrics(g, base),
                     "by_type_h5": {t: v["hit@5"] for t, v in by_type_metrics(g, base).items()}},
        "hyde": {"metrics": metrics(g, hyde_topk),
                 "by_type_h5": {t: v["hit@5"] for t, v in by_type_metrics(g, hyde_topk).items()}},
    }
    print(json.dumps(out, ensure_ascii=False))
    (DATA / "kt4_hyde_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
