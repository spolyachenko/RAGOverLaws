"""Весь КТ1 по порядку: fetch → eda → gold(check+split) → baseline.

Запуск: python src/run_kt1.py [--fetch]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import CORPUS_PATH  # noqa: E402
import baseline  # noqa: E402
import eda  # noqa: E402
import fetch_wiki  # noqa: E402
import gold  # noqa: E402


def step(name, fn):
    print(f"\n=== {name} ===", flush=True)
    fn()


def main():
    if "--fetch" in sys.argv or not CORPUS_PATH.exists():
        step("fetch", fetch_wiki.run)
    else:
        print("data/corpus.jsonl уже есть, выгрузка пропущена (--fetch чтобы повторить)")
    step("eda", eda.run)
    step("check-gold", gold.check)
    step("split", gold.split)
    step("baseline", baseline.run)
    print("\nГотово. Выводы: reports/kt1/, data/gold/")


if __name__ == "__main__":
    main()
