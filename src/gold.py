"""Золотой набор: валидация (check) и validation/test split.

Запуск: python src/gold.py [check|split|all]
"""
import collections
import json
import sys

from sklearn.model_selection import train_test_split

from common import CORPUS_PATH, GOLD_FULL, GOLD_TEST, GOLD_VAL, RANDOM_STATE

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


def split():
    gold = json.load(open(GOLD_FULL, encoding="utf-8"))
    val, test = train_test_split(gold, test_size=0.3, random_state=RANDOM_STATE,
                                 stratify=[g["type"] for g in gold])
    for path, part in [(GOLD_VAL, val), (GOLD_TEST, test)]:
        path.write_text(json.dumps(part, ensure_ascii=False, indent=1), encoding="utf-8")
    print("validation:", dict(collections.Counter(g["type"] for g in val)))
    print("test:      ", dict(collections.Counter(g["type"] for g in test)))
    assert set(g["type"] for g in val) == TYPES, "во всех частях все типы"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("check", "all"):
        check()
    if cmd in ("split", "all"):
        split()
