"""Золотой набор: валидация (40–60, типы, doc_ids в корпусе, answer_contexts дословные).

Правила (правовой корпус): negative — без контекстов/doc_ids; остальные типы —
1+ дословная выдержка ≤250 символов (substring текста дока после нормализации ссылок)
и 1+ doc_id (composite — 2+).

Запуск: python src/gold/check.py
"""
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import CORPUS_PATH, GOLD_FULL  # noqa: E402

TYPES = {"factual", "composite", "abstract", "negative"}


def _norm(s: str) -> str:
    s = re.sub(r"\[\[([^\|\]]*)\|([^\]]*)\]\]", r"\2", s)
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    return s


def check():
    corpus = {json.loads(l)["id"]: _norm(json.loads(l)["text"]) for l in open(CORPUS_PATH, encoding="utf-8")}
    gold = json.load(open(GOLD_FULL, encoding="utf-8"))
    assert 40 <= len(gold) <= 60, f"нужно 40-60, сейчас {len(gold)}"
    for g in gold:
        assert g["type"] in TYPES, g
        if g["type"] == "negative":
            assert not g["answer_contexts"] and not g["doc_ids"], g["id"]
            continue
        assert g["doc_ids"] and all(d in corpus for d in g["doc_ids"]), g["id"]
        if g["type"] == "composite":
            assert len(g["doc_ids"]) >= 2, g["id"]
        assert g["answer_contexts"], g["id"]
        for ctx in g["answer_contexts"]:
            assert len(ctx) <= 250, (g["id"], len(ctx))
            assert any(_norm(ctx) in text for text in (corpus[d] for d in g["doc_ids"])), (g["id"], ctx[:60])
    print("примеров:", len(gold), dict(collections.Counter(g["type"] for g in gold)))
    print("валидация ОК")
    return gold


if __name__ == "__main__":
    check()
