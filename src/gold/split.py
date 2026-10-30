"""Золотой набор: стратифицированный validation/test split (42/18).

Запуск: python src/gold/split.py
"""
import collections
import json
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import GOLD_FULL, GOLD_TEST, GOLD_VAL, RANDOM_STATE  # noqa: E402
from gold.check import TYPES  # noqa: E402


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
    split()
