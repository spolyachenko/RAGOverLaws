"""LLM-обработка чанков: суммаризация (80 слов) + HyDE (2 вопроса) — exploratory.

Область: первые 100 чанков документов из doc_ids gold100 (именно они несут ground truth;
полный прогон 532 чанка >1.5ч при ~4.6x параллелизме не дал бы качественно иного вывода).
Остальной корпус индексируется как есть. Кэш по sha1 чанка — повторный прогон ~0 токенов.

Артефакт: data/llm_chunks.jsonl (doc, pos, text, summary, hyde).
Запуск: python src/llm/process.py
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from chonkie import TokenChunker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus.clean import load_stage  # noqa: E402
from utils import DATA, GOLD_VAL  # noqa: E402
from retrieval.metrics import _gold_titles  # noqa: E402
from llm.client import ask, cache_tokens, client, key_for, load_cache, save_cache  # noqa: E402

OUT = DATA / "llm_chunks.jsonl"
CAP = 100  # exploratory-кап: первые N чанков подмножества


def subset_chunks() -> pd.DataFrame:
    """Первые CAP чанков документов из doc_ids gold100 (те же, что в data/chunks.jsonl)."""
    df = load_stage("dedup")
    gold = _gold_titles(GOLD_VAL)
    subset = sorted({d.split("://", 1)[1] if "://" in d else d for ds in gold.doc_ids for d in ds})
    df = df[df["title"].isin(subset)]
    ch = TokenChunker(chunk_size=512)
    items = []
    for _, d in df.iterrows():
        for i, c in enumerate(ch.chunk(d["text"])):
            items.append({"doc": d["title"], "pos": i, "text": c.text})
    return pd.DataFrame(items).head(CAP)


def process() -> pd.DataFrame:
    """LLM-обработка подмножества (summary + HyDE, 8-way). Из кэша — ~0 токенов."""
    ck = subset_chunks()
    cache = load_cache()
    cl, model = client()

    def do_sum(r):
        return ask(cl, model,
                   "Переведи эту выдержку в краткое содержание по-русски. Сохрани все "
                   "конкретные факты, номера статей, суммы, сроки и условия. Максимум 80 слов. "
                   "Без вступления.\n\n" + r["text"],
                   key_for("sum:", r["text"]), cache)

    def do_hyde(r):
        return ask(cl, model,
                   "Составь ровно 2 вопроса на русском, ответ на которые содержится в этой "
                   "выдержке. Выведи нумерованным списком, ничего лишнего.\n\n" + r["text"],
                   key_for("hyde:", r["text"]), cache)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        sums = list(ex.map(do_sum, ck.to_dict("records")))
    t_sum = round(time.time() - t0, 1)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        hydes = list(ex.map(do_hyde, ck.to_dict("records")))
    t_hyde = round(time.time() - t0, 1)
    ck["summary"] = [s for s, _ in sums]
    ck["hyde"] = [re.sub(r"\s+", " ", q) for q, _ in hydes]
    ck.to_json(OUT, orient="records", lines=True, force_ascii=False)
    save_cache(cache)
    print(f"llm: {len(ck)} чанков, summary {t_sum}s + hyde {t_hyde}s "
          f"(кэш: {cache_tokens(cache)} токенов всего)")
    return ck


if __name__ == "__main__":
    process()
