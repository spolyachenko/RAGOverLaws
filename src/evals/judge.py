"""Оценка корректности ответов baseline (без RAG) по gold100.

LLM-судья сравнивает каждый ответ с эталонными фрагментами (answer_contexts)
и ставит verdict: верно / частично / неверно. Для negative эталон — констатация
отсутствия закона в корпусе.

Вывод: data/gold/judge_baseline.json + сводка в stdout.
Запуск: python src/evals/judge.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.client import ask, client, load_cache, save_cache  # noqa: E402
from utils import GOLD_DIR  # noqa: E402

GOLD = GOLD_DIR / "gold100.json"
BASE = GOLD_DIR / "baseline_no_rag_val.json"
OUT = GOLD_DIR / "judge_baseline.json"

P = """Оцени корректность ответа модели на вопрос по эталонным фрагментам (выдержки из официальных документов РФ).

Вопрос: {q}

Эталон: {gt}

Ответ модели: {a}

Правила:
- если ответ содержит факты, противоречащие эталону (неверные номера статей/законов, суммы, сроки, правила), — "неверно";
- если суть верна, но есть неточности, выдуманные детали или пропущены ключевые факты эталона, — "частично";
- если ответ полностью соответствует эталону, — "верно".
Ответь только JSON: {{"verdict": "верно|частично|неверно", "reason": "<одна строка>"}}"""


def main():
    gold = json.load(open(GOLD, encoding="utf-8"))
    answers = {x["id"]: x for x in json.load(open(BASE, encoding="utf-8"))}
    cl, model = client()
    cache = load_cache()
    rows = []
    for g in gold:
        a = answers[g["id"]]["answer"]
        if g["type"] == "negative":
            gt = ("Этот вопрос о законе/механизме, которого НЕТ в корпусе документов; "
                  "корректный ответ указывает, что такого закона/механизма/статьи не существует, "
                  "а ответ, описывающий несуществующее как существующее, — неверен.")
        else:
            gt = "\n".join(g["answer_contexts"])
        prompt = P.format(q=g["question"], gt=gt, a=a)
        out, _ = ask(cl, model, prompt, f"judge:{g['id']}", cache)
        m = re.search(r"\{.*\}", out, re.S)
        try:
            v = json.loads(m.group(0))
            verdict, reason = v["verdict"], v.get("reason", "")
        except Exception:
            verdict, reason = "не распознано", out[:200]
        rows.append({"id": g["id"], "type": g["type"], "verdict": verdict, "reason": reason})
        print(f"[{g['id']}] ({g['type']}) {verdict}: {reason[:90]}", flush=True)
    save_cache(cache)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    by = {}
    for r in rows:
        a, b = by.get((r["type"], r["verdict"]), (0, 0))
        by[(r["type"], r["verdict"])] = (a + 1, b + 1)
    for t in ["factual", "composite", "abstract", "negative"]:
        v = by.get((t, "верно"), (0, 0))[0]
        p = by.get((t, "частично"), (0, 0))[0]
        w = by.get((t, "неверно"), (0, 0))[0]
        n = sum(by.get((t, x), (0, 0))[0] for x in ["верно", "частично", "неверно", "не распознано"])
        print(f"{t:10s}: верно {v}/{n}, частично {p}, неверно {w}")
    tot = sum(r["type"] != "negative" for r in rows)
    good = sum(r["type"] != "negative" and r["verdict"] == "верно" for r in rows)
    print(f"итого (без negative): верно {good}/{tot}")


if __name__ == "__main__":
    main()
