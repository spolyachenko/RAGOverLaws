"""Пайплайн корпуса по порядку: fetch → eda → baseline.

Запуск: python src/main.py [--fetch]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import eda, fetch  # noqa: E402
from evals import baseline  # noqa: E402
from utils import CORPUS_PATH  # noqa: E402


def main():
    if "--fetch" in sys.argv or not CORPUS_PATH.exists():
        print("\n=== fetch ===", flush=True)
        fetch.fetch_corpus()
    else:
        print("data/corpus.jsonl уже есть, выгрузка пропущена (--fetch чтобы повторить)")
    print("\n=== eda ===", flush=True)
    eda.run()
    print("\n=== baseline (gold100, без RAG) ===", flush=True)
    baseline.run()
    print("Готово. Выводы: reports/, data/gold/")


if __name__ == "__main__":
    main()
