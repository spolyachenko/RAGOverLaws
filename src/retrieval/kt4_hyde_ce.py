"""KT4: HyDE + CE (153, GPU) — комбинация для итоговой таблицы.

Pool top-100 по эмбеддингу HyDE-документа -> CE bge-reranker-v2-m3 -> top-40.
Запуск (после kt4_rerank_ce.py): python src/retrieval/kt4_hyde_ce.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA, GOLD_VAL  # noqa: E402
from retrieval.metrics import _gold_titles, _dedup_docs, metrics, by_type_metrics  # noqa: E402
from retrieval.kt4_rerank_ce import CE_MODEL, ce_score, reranked_topk  # noqa: E402


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ck = pd.read_json(DATA / "chunks.jsonl", lines=True)
    texts = ck["text"].tolist()
    g = _gold_titles(GOLD_VAL)
    queries = g["question"].tolist()
    hyde = json.load(open(DATA / "kt4_hyde.json", encoding="utf-8"))
    h_emb = np.load(DATA / "kt4_hyde_emb.npy")

    d100 = (h_emb @ np.load(DATA / "emb_bge-m3.npy", mmap_mode="r").T
            ).argsort(axis=1)[:, ::-1][:, :100]
    tok = AutoTokenizer.from_pretrained(CE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(CE_MODEL).cuda().half()
    model.eval()
    sc, lats = ce_score(model, tok, queries, d100, texts)
    ls = sorted(lats)
    topk = {q: _dedup_docs(ck["doc"].iloc[reranked_topk(d100[i:i + 1], sc[i:i + 1])[0]].tolist())
            for i, q in enumerate(queries)}
    out = {"pool": 100, "metrics": metrics(g, topk),
           "by_type_h5": {t: v["hit@5"] for t, v in by_type_metrics(g, topk).items()},
           "ce_ms_median": round(ls[len(ls) // 2], 1)}
    print(json.dumps(out, ensure_ascii=False))
    (DATA / "kt4_hyde_ce.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
