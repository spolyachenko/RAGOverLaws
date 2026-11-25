"""KT5: top-5 чанков на 100 вопросов из двух конфигураций (153, GPU).

- baseline (КТ3): bge-m3, прямой поиск по эмбеддингу вопроса;
- best (КТ4): HyDE-эмбеддинг -> dense top-100 -> CE bge-reranker-v2-m3 -> top-5.

Артефакт: data/kt5_top5.json {question: {baseline: [5 idx], best: [5 idx]}}
Запуск: python src/retrieval/kt5_retrieve.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA, GOLD_FULL  # noqa: E402
from retrieval.embed_models import _model  # noqa: E402
from retrieval.kt4_rerank_ce import CE_MODEL, ce_score  # noqa: E402


def main():
    queries = [q["question"] for q in json.load(open(GOLD_FULL, encoding="utf-8"))]
    hyde = json.load(open(DATA / "kt5_hyde100.json", encoding="utf-8"))
    m = _model("bge-m3")[0]
    q_emb = m.encode(queries, batch_size=8)
    h_emb = m.encode([hyde[q] for q in queries], batch_size=8)
    c_emb = np.load(DATA / "emb_bge-m3.npy")

    base5 = (q_emb @ c_emb.T).argsort(axis=1)[:, ::-1][:, :5]

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ck = pd.read_json(DATA / "chunks.jsonl", lines=True)
    texts = ck["text"].tolist()
    d100 = (h_emb @ c_emb.T).argsort(axis=1)[:, ::-1][:, :100]
    tok = AutoTokenizer.from_pretrained(CE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(CE_MODEL).cuda().half()
    model.eval()
    sc, _ = ce_score(model, tok, queries, d100, texts)
    best5 = np.empty((len(queries), 5), dtype=int)
    for i in range(len(queries)):
        order = np.argsort(sc[i])[::-1][:5]
        best5[i] = d100[i][order]

    out = {q: {"baseline": [int(j) for j in base5[i]],
               "best": [int(j) for j in best5[i]]} for i, q in enumerate(queries)}
    (DATA / "kt5_top5.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(f"saved data/kt5_top5.json ({len(out)} q)")


if __name__ == "__main__":
    main()
