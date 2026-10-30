"""Золотой набор: валидация (40–60, типы, doc_ids в корпусе, answer_contexts).

Запуск: python src/gold/check.py
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import CORPUS_PATH, GOLD_FULL  # noqa: E402

TYPES = {"factual", "composite", "abstract", "negative"}


def check():
    corpus_ids = {json.loads(l)["id"] for l in open(CORPUS_PATH, encoding="utf-8")}
    gold = json.load(open(GOLD_FULL, encoding="utf-8"))
    assert 40 <= len(gold) <= 60, f"нужно 40-60, сейчас {len(gold)}"
    for g in gold:
        assert g["type"] in TYPES, g
        assert all(d in corpus_ids for d in g["doc_ids"]), g["id"]
        assert (not g["answer_contexts"]) == (g["type"] in {"negative", "abstract"}), g["id"]
    print("примеров:", len(gold), dict(collections.Counter(g["type"] for g in gold)))
    print("валидация ОК")
    return gold


if __name__ == "__main__":
    check()
