"""KT4: HyDE — гипотетический документ-ответ на запрос.

Выбор метода (из HyDE / multi-query / step-back): HyDE. Корпус — статьи законов,
gold-вопросы — фактические/композитные с номерами ФЗ и статьями. Step-back обобщает
и теряет точные лексемы (номер ФЗ), multi-query стоит 3–5 LLM-вызовов на запрос
без явного выигрыша на фактических. HyDE = 1 вызов, и в KT2 показал +0.018 hit@5
на BM25; здесь проверяем на dense baseline (bge-m3), где HyDE может помочь сильнее
(эмбеддинг «статьи закона» ближе к чанкам, чем эмбеддинг вопроса).

LLM: Qwen3.8-27B-FP8 (shared .239, только запросы). Токены/latency — из кэша.
Артефакт: data/kt4_hyde.json {question: {doc, prompt_tok, out_tok, latency_s}}.

Запуск (ноут): python src/retrieval/kt4_hyde.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA, GOLD_VAL  # noqa: E402
from retrieval.metrics import _gold_titles  # noqa: E402
from llm.client import ask, client, key_for, load_cache, save_cache  # noqa: E402

PROMPT = ("Ты — эксперт по российскому законодательству. На вопрос: «{q}» напиши "
          "гипотетический фрагмент закона, который мог бы на него отвечать. "
          "3–6 предложений, в стиле текста закона (статья, пункт, абзац). "
          "Только текст, без пояснений и без слова «ответ».")


def main():
    queries = _gold_titles(GOLD_VAL)["question"].tolist()
    cl, model = client()
    cache = load_cache()
    out = {}
    t0 = time.time()
    for i, q in enumerate(queries):
        doc, dt = ask(cl, model, PROMPT.format(q=q), key_for("kt4hyde:", q), cache)
        out[q] = {"doc": doc, "latency_s": dt}
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(queries)} ({time.time() - t0:.0f}s)", flush=True)
    save_cache(cache)
    for q in out:
        c = cache[key_for("kt4hyde:", q)]
        out[q]["prompt_tok"], out[q]["out_tok"] = c["prompt_tok"], c["out_tok"]
    (DATA / "kt4_hyde.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    lat = sorted(v["latency_s"] for v in out.values() if v["latency_s"])
    pt = sum(v["prompt_tok"] for v in out.values()) / len(out)
    ot = sum(v["out_tok"] for v in out.values()) / len(out)
    print(f"hyde: {len(out)} доков, latency med {lat[len(lat) // 2]:.1f}s, "
          f"tokens/запрос {pt:.0f}+{ot:.0f} (prompt+out)")


if __name__ == "__main__":
    main()
