"""Пайплайн корпуса по порядку: fetch → eda → gold(check+split) → baseline.

Запуск: python src/main.py [--fetch]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import eda, fetch  # noqa: E402
from evals import baseline  # noqa: E402
from gold import check as gold_check  # noqa: E402
from gold import split as gold_split  # noqa: E402
from utils import CORPUS_PATH  # noqa: E402


def main():
    if "--fetch" in sys.argv or not CORPUS_PATH.exists():
        print("\n=== fetch ===", flush=True)
        fetch.fetch_corpus()
    else:
        print("data/corpus.jsonl уже есть, выгрузка пропущена (--fetch чтобы повторить)")
    print("\n=== eda ===", flush=True)
    eda.run()
    print("\n=== check-gold ===", flush=True)
    gold_check.check()
    print("\n=== split ===", flush=True)
    gold_split.split()
    print("\n=== baseline ===", flush=True)
    baseline.run()
    print("Готово. Выводы: reports/, data/gold/")


if __name__ == "__main__":
    main()
