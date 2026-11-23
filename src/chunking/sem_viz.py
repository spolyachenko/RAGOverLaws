"""Визуализация семантического чанкинга: одна глава, пороги 0.4/0.6/0.8.

Косинусное сходство соседних предложений; границы чанка — провалы ниже порога.
Вывод: reports/kt2/semantic_thresholds.png
Запуск: python src/chunking/sem_viz.py
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.clean import load_stage  # noqa: E402
from utils import REPORTS  # noqa: E402

MODEL = "minishlab/potion-base-32M"
DOCS = ["Гражданский кодекс РФ/Глава 6", "Семейный кодекс РФ/Глава 13"]
OUTS = REPORTS / "kt2" / "semantic_thresholds.png"
THRS = [0.4, 0.6, 0.8]


def sentences(t: str) -> list[str]:
    t = re.sub(r"\s+", " ", t).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ])", t)
    return [p for p in parts if len(p) > 20]


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    m = SentenceTransformer(MODEL)
    fig, axes = plt.subplots(len(DOCS), 1, figsize=(12, 4.2 * len(DOCS)), sharex=True)
    df = load_stage("dedup").set_index("title")
    for ax, title in zip(axes, DOCS):
        sents = sentences(df.loc[title, "text"])[:400]
        emb = m.encode(sents, normalize_embeddings=True)
        sims = (emb[:-1] * emb[1:]).sum(1)
        x = np.arange(1, len(sims) + 1)
        ax.plot(x, sims, lw=0.8, color="#444")
        for thr, c in zip(THRS, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            cuts = x[sims < thr]
            for cx in cuts:
                ax.axvline(cx, color=c, alpha=0.35, lw=1)
            ax.plot([], [], color=c, lw=2, label=f"threshold {thr}  ({len(cuts)} cuts)")
        ax.set_ylim(0, 1)
        ax.set_title(f"{title}: {len(sents)} sentences, boundaries by cosine threshold")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_ylabel("cosine")
    axes[-1].set_xlabel("sentence index")
    OUTS.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTS, dpi=150)
    print("saved:", OUTS)


if __name__ == "__main__":
    main()
