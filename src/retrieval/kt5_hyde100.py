"""KT5: HyDE-документы на ВСЕ 100 вопросов (val 56 в кэше, test 44 новые).

Тот же промпт и префикс кэша, что в КТ4 (kt4hyde:) — val не генерируется заново.
Артефакт: data/kt5_hyde100.json
Запуск (ноут): python src/retrieval/kt5_hyde100.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA, GOLD_FULL  # noqa: E402
from llm.client import ask, client, key_for, load_cache, save_cache  # noqa: E402
from retrieval.kt4_hyde import PROMPT  # noqa: E402


def main():
    queries = [q["question"] for q in json.load(open(GOLD_FULL, encoding="utf-8"))]
    cl, model = client()
    cache = load_cache()
    out, t0 = {}, time.time()
    fresh = 0
    for i, q in enumerate(queries):
        doc, dt = ask(cl, model, PROMPT.format(q=q), key_for("kt4hyde:", q), cache)
        if dt > 0:
            fresh += 1
        out[q] = doc
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(queries)} fresh={fresh} ({time.time() - t0:.0f}s)", flush=True)
    save_cache(cache)
    (DATA / "kt5_hyde100.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(f"done: {len(out)} docs, fresh {fresh}")


if __name__ == "__main__":
    main()
