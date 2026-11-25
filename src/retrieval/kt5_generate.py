"""KT5: генерация ответов, 2 конфигурации контекста (ноут, LLM .239).

Baseline retrieval — top-5 чанков из КТ3 (bge-m3, прямой поиск).
Best retrieval — top-5 чанков из КТ4 (HyDE + dense + CE pool 20).
Промпт и модель ЗАФИКСИРОВАНЫ (не меняются в течение КТ5):
  модель Qwen/Qwen3.8-27B-FP8, temp=0, thinking off.
Артефакт: data/kt5_answers.json
Запуск: python src/retrieval/kt5_generate.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA, GOLD_FULL  # noqa: E402
from llm.client import ask, client, key_for, load_cache, save_cache  # noqa: E402

SYSTEM = ("Ты — помощник по вопросам российского законодательства. "
          "Отвечай ТОЛЬКО на основании предоставленного контекста (фрагменты законов). "
          "Если ответа в контексте нет — ответь ровно: «Не знаю». "
          "Не используй знания вне контекста. Отвечай сжато, 2–6 предложений, "
          "указывая источник (закон, статья), если он есть в контексте.")


def _context(ck_doc, ck_text, idx):
    parts = []
    for i, j in enumerate(idx, 1):
        parts.append(f"[{i}] ({ck_doc[j]})\n{ck_text[j]}")
    return "\n\n".join(parts)


def main():
    gold = json.load(open(GOLD_FULL, encoding="utf-8"))
    import pandas as pd
    ck = pd.read_json(DATA / "chunks.jsonl", lines=True)
    top5 = json.load(open(DATA / "kt5_top5.json", encoding="utf-8"))
    ck_doc, ck_text = ck["doc"].tolist(), ck["text"].tolist()
    cl, model = client()
    cache = load_cache()
    out = {}
    t0 = time.time()
    n = 0
    for g in gold:
        q = g["question"]
        out[q] = {"type": g["type"], "doc_ids": g.get("doc_ids", [])}
        for cfg in ("baseline", "best"):
            idx = top5[q][cfg]
            ctx = _context(ck_doc, ck_text, idx)
            prompt = SYSTEM + "\n\nКонтекст:\n" + ctx + f"\n\nВопрос: {q}"
            key = key_for(f"kt5gen_{cfg}:", q)
            ans, dt = ask(cl, model, prompt, key, cache)
            out[q][cfg] = {"answer": ans, "context_idx": idx,
                           "prompt_tok": cache[key]["prompt_tok"],
                           "out_tok": cache[key]["out_tok"], "latency_s": dt}
            n += 1
            if n % 20 == 0:
                print(f"{n}/200 ({time.time() - t0:.0f}s)", flush=True)
    save_cache(cache)
    (DATA / "kt5_answers.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print("saved data/kt5_answers.json")


if __name__ == "__main__":
    main()
