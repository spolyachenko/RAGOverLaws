"""KT4: эмбеддинги запросов + HyDE-документов (153, bge-m3) -> data/kt4_*_emb.npy."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA, GOLD_VAL  # noqa: E402
from retrieval.metrics import _gold_titles  # noqa: E402
from retrieval.embed_models import _model  # noqa: E402


def main():
    m = _model("bge-m3")[0]
    queries = _gold_titles(GOLD_VAL)["question"].tolist()
    hyde = json.load(open(DATA / "kt4_hyde.json", encoding="utf-8"))
    q_emb = m.encode(queries, batch_size=8)
    h_emb = m.encode([hyde[q]["doc"] for q in queries], batch_size=8)
    np.save(DATA / "kt4_q_emb.npy", q_emb)
    np.save(DATA / "kt4_hyde_emb.npy", h_emb)
    print(f"q_emb {q_emb.shape} -> kt4_q_emb.npy; h_emb {h_emb.shape} -> kt4_hyde_emb.npy")


if __name__ == "__main__":
    main()
