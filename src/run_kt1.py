"""Весь КТ1 по порядку: fetch → eda → gold(check+split) → baseline.

Запуск: python src/run_kt1.py [--fetch]
"""
import sys

from common import CORPUS_PATH
import baseline
import eda
import fetch_wiki
import gold


def main():
    if "--fetch" in sys.argv or not CORPUS_PATH.exists():
        print("\n=== fetch ===", flush=True)
        fetch_wiki.run()
    else:
        print("data/corpus.jsonl уже есть, выгрузка пропущена (--fetch чтобы повторить)")
    print("\n=== eda ===", flush=True)
    eda.run()
    print("\n=== check-gold ===", flush=True)
    gold.check()
    print("\n=== split ===", flush=True)
    gold.split()
    print("\n=== baseline ===", flush=True)
    baseline.run()
    print("\nГотово. Выводы: reports/kt1/, data/gold/")


if __name__ == "__main__":
    main()
