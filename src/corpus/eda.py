"""EDA корпуса: распределение длин (гистограмма), доля мусорных токенов, дубликаты (MinHash).

Результат: reports/kt1/lengths_hist.png + сводка в stdout.
Запуск: python src/corpus/eda.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import REPORTS, assert_minima, load_corpus, with_tokens  # noqa: E402

OUT = REPORTS / "kt1"

URL_RE = re.compile(r"https?://\S+|\b\w+\.\w{2,}\S*")
EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]")


def _junk_stats(text):
    toks = text.split()
    s = {"total": len(toks), "emoji": 0, "url": 0, "num": 0, "punct": 0}
    for t in toks:
        if EMOJI_RE.search(t):
            s["emoji"] += 1
        elif URL_RE.fullmatch(t):
            s["url"] += 1
        elif t.isdigit():
            s["num"] += 1
        elif not re.search(r"[\w\d]", t):
            s["punct"] += 1
    return s


def _make_mh(text):
    mh = MinHash(num_perm=128)
    for w in re.findall(r"\w+", text.lower()):
        mh.update(w.encode())
    return mh


def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    corpus = with_tokens(load_corpus())
    assert_minima(corpus)

    vals = corpus["n_tokens"]
    bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 45)  # лог-бины: равная ширина в log-шкале
    vals.hist(bins=bins, figsize=(10, 4))
    plt.axvline(50, color="r", ls="--", label="<50 токенов")
    plt.axvline(2000, color="orange", ls="--", label=">2000 токенов")
    plt.xscale("log")
    plt.xlabel("длина, токены (log)")
    plt.ylabel("число документов")
    plt.title("Распределение длин документов")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "lengths_hist.png", dpi=150)

    junk = pd.DataFrame([_junk_stats(t) for t in corpus["text"]])
    tot = junk["total"].sum()
    jfrac = (junk[["emoji", "url", "num", "punct"]].sum(axis=1) / junk["total"].replace(0, 1))

    mhs = [_make_mh(t) for t in corpus["text"]]
    lsh = MinHashLSH(threshold=0.7)
    for i, mh in enumerate(mhs):
        lsh.insert(str(i), mh)
    near = sum(1 for i in range(len(mhs)) if any(int(j) != i for j in lsh.query(mhs[i])))

    summary = {
        "docs": int(len(corpus)), "tokens": int(corpus["n_tokens"].sum()),
        "mean_tokens": round(float(corpus["n_tokens"].mean())),
        "median_tokens": float(corpus["n_tokens"].median()),
        "short_lt50": int((corpus["n_tokens"] < 50).sum()),
        "long_gt2000": int((corpus["n_tokens"] > 2000).sum()),
        "junk_share": {k: round(float(junk[k].sum() / tot), 4) for k in ["emoji", "url", "num", "punct"]},
        "docs_gt20pct_junk": int((jfrac > 0.2).sum()),
        "exact_duplicates": int(corpus["text"].duplicated().sum()),
        "near_duplicate_docs": near, "near_dup_threshold": 0.7,
        "smallest": corpus.nsmallest(4, "n_tokens")[["title", "n_tokens"]].to_dict("records"),
        "largest": corpus.nlargest(5, "n_tokens")[["title", "n_tokens"]].to_dict("records"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


if __name__ == "__main__":
    run()
