"""Composite-вопросы из верифицированных фактов: пара фактов из разных документов ->
один составной вопрос (LLM перефразирует), answer_contexts — обе дословные выдержки.

Вход:  data/gold/legal_gold_verified.json
Выход: data/gold/legal_composite.json (готовые composite-кандидаты)
Запуск: python src/gold/compose.py
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import GOLD_DIR  # noqa: E402
from llm.client import client, load_cache, save_cache, key_for  # noqa: E402
from gold.generate import _ask_safe  # noqa: E402

VER = GOLD_DIR / "legal_gold_verified.json"
OUT = GOLD_DIR / "legal_composite.json"

P_COMP_Q = """Даны два отдельных вопроса о разных правовых документах РФ:
1) {q1} (документ: {t1})
2) {q2} (документ: {t2})

Составь ОДИН составной вопрос на русском, на который надо ответить, опираясь на ОБА
документа (форма: «Что/Как ... по первому документу и ... по второму?» — естественный деловой вопрос,
а не спаянные два вопроса). Выведи только текст вопроса, ничего лишнего.
/no_think"""


def main():
    ver = json.loads(VER.read_text(encoding="utf-8"))
    corpus = {}
    import json as j
    for line in open("data/law_corpus.jsonl", encoding="utf-8"):
        r = j.loads(line)
        corpus[r["id"]] = r["title"]

    facts = ver["factual"]["ok"]
    # до 2 фактов на документ, жадный подбор пар из разных документов
    import itertools
    by_doc = {}
    for f in facts:
        by_doc.setdefault(f["doc_ids"][0], []).append(f)
    pool = [f for d in sorted(by_doc) for f in by_doc[d][:2]]
    pairs, used = [], set()
    for i, j in itertools.combinations(range(len(pool)), 2):
        if i in used or j in used:
            continue
        if pool[i]["doc_ids"][0] != pool[j]["doc_ids"][0]:
            pairs.append((pool[i], pool[j]))
            used.add(i)
            used.add(j)
        if len(pairs) >= 18:
            break
    print(f"pool={len(pool)} docs={len(by_doc)} pairs={len(pairs)}", flush=True)

    cl, model = client()
    cache = load_cache()

    def mk(pair):
        a, b = pair
        raw = _ask_safe(cl, model, P_COMP_Q.format(q1=a["question"], t1=corpus.get(a["doc_ids"][0], "?"),
                                                   q2=b["question"], t2=corpus.get(b["doc_ids"][0], "?")),
                        key_for("comp_q:", a["question"] + b["question"]), cache)
        return {"question": raw.strip().strip('"').strip(),
                "answer_contexts": [a["answer_context"], b["answer_context"]],
                "doc_ids": [a["doc_ids"][0], b["doc_ids"][0]]}

    with ThreadPoolExecutor(max_workers=8) as ex:
        out = [c for c in ex.map(mk, pairs) if c["question"]]
    save_cache(cache)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("written:", len(out), "->", OUT, flush=True)


if __name__ == "__main__":
    main()
