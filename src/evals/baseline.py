"""Baseline без RAG: ответы LLM на test-вопросы БЕЗ контекста («а нужен ли RAG?»).

Модель берётся из .env (локальная Ollama по умолчанию, OpenAI-совместимый API):
    LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
Если API недоступен — грузит предвычисленный data/gold/baseline_no_rag.json.

Запуск: python src/evals/baseline.py
"""
import json
import os
import sys
import time

from utils import BASELINE_OUT, GOLD_TEST, GOLD_VAL, load_env

PROMPT = "Отвечай кратко на русском. Если не знаешь ответа — явно скажи «не знаю».\nВопрос: {q}"


def _client():
    from openai import OpenAI

    env = load_env()
    base_url = os.environ.get("LLM_BASE_URL", env.get("LLM_BASE_URL", "http://localhost:11434/v1"))
    api_key = os.environ.get("LLM_API_KEY", env.get("LLM_API_KEY", "ollama"))
    model = os.environ.get("LLM_MODEL", env.get("LLM_MODEL", "qwen2.5:7b"))
    return OpenAI(base_url=base_url, api_key=api_key), model


def run():
    results = None
    try:
        client, model = _client()
        # ТЗ: baseline — на отложенной выборке ВАЛИДАЦИИ; по умолчанию val, test — аргументом
        split = sys.argv[1] if len(sys.argv) > 1 else "val"
        gold_file = GOLD_TEST if split == "test" else GOLD_VAL
        out_file = BASELINE_OUT if split == "test" else BASELINE_OUT.parent / f"baseline_no_rag_{split}.json"
        test = json.load(open(gold_file, encoding="utf-8"))
        results = []
        for g in test:
            r = None
            for a in range(3):  # ретрай на вопрос: один 500 не рвёт весь прогон
                t0 = time.time()
                try:
                    r = client.chat.completions.create(model=model, temperature=0,
                                                       messages=[
                                                            {"role": "system", "content": "Отвечай на русском, опираясь только на собственные знания, без обращения к документам."},
                                                            {"role": "user", "content": PROMPT.format(q=g["question"])}])
                    break
                except Exception:
                    if a == 2:
                        raise
                    time.sleep(3)
            results.append({"id": g["id"], "type": g["type"], "question": g["question"],
                            "answer": r.choices[0].message.content,
                            "latency_s": round(time.time() - t0, 2)})
            print(f"[{g['id']}] ({g['type']}) {g['question'][:60]} -> {results[-1]['answer'][:80]}", flush=True)
        out_file.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"baseline {split} (модель {model}) сохранён в {out_file}")
    except Exception as e:
        print(f"LLM недоступен ({type(e).__name__}), грузим предвычисленный baseline")

    if not results:
        results = json.load(open(BASELINE_OUT, encoding="utf-8"))
        print(f"Загружен предвычисленный baseline: {len(results)} ответов из {BASELINE_OUT}")
    print(f"Средний latency: {sum(r['latency_s'] for r in results) / len(results):.2f}s")
    admit = sum(1 for r in results if "не знаю" in r["answer"].lower())
    print(f"Модель без RAG признала нерешимыми {admit}/{len(results)} вопросов")
    for t in ["factual", "composite", "abstract", "negative"]:
        sub = [r for r in results if r["type"] == t]
        if sub:
            print(f"  {t:10s}: {sum(1 for r in sub if 'не знаю' in r['answer'].lower())}/{len(sub)} «не знаю»")


if __name__ == "__main__":
    run()
