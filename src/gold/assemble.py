"""Сборка gold_full.json (60 вопросов) из верифицированных кандидатов.

Состав: factual 24 + composite 13 + abstract 13 + negative 10 (6 верифицированных
+ 4 ручных, grep-проверенных). Исключаются документы, не относящиеся к РФ (Дурасов).
Вход:  data/gold/legal_gold_verified.json, data/gold/legal_composite.json
Выход: data/gold/gold_full.json (id, question, type, answer_contexts, doc_ids)
Запуск: python src/gold/assemble.py
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import GOLD_DIR, DATA  # noqa: E402

VER = GOLD_DIR / "legal_gold_verified.json"
COMP = GOLD_DIR / "legal_composite.json"
OUT = GOLD_DIR / "gold_full.json"

EXCLUDE_TITLE = re.compile(r"Дурасов", re.I)
META_Q = re.compile(r"дата редакции|ред\. \d|какая дата (редакции|указан)|какое название страницы", re.I)

MANUAL_NEG = [
    "Каков порядок регистрации и переоформления оружия в соответствии с Федеральным законом «Об оружии»?",
    "Какие полномочия предусмотрены для Росреестра Федеральным законом «О государственной регистрации недвижимости»?",
    "Каков порядок выдачи лицензий на образовательную деятельность в соответствии с Федеральным законом «Об образовании в Российской Федерации»?",
    "Каковы основания для временного ограничения деятельности организации согласно Федеральному закону «О некоммерческих организациях»?",
    "Какие основания прекращения трудового договора предусмотрены Федеральным законом «О занятости населения в Российской Федерации»?",
]


def main():
    ver = json.loads(VER.read_text(encoding="utf-8"))
    comp = json.loads(COMP.read_text(encoding="utf-8")) if COMP.exists() else []

    corpus = {}
    for line in open(DATA / "corpus.jsonl", encoding="utf-8"):
        r = json.loads(line)
        corpus[r["id"]] = r
    hay = "\n".join(r["text"] for r in corpus.values()).lower()

    rng = random.Random(42)
    ok = lambda d: not EXCLUDE_TITLE.search(corpus[d]["title"]) if d in corpus else False

    # factual: без мета-вопросов, приоритет по информативности ответа
    facts = [f for f in ver["factual"]["ok"] if ok(f["doc_ids"][0])]
    substantive = [f for f in facts if not META_Q.search(f["question"]) and len(f["answer_context"]) > 40]
    filler = [f for f in facts if f not in substantive and not META_Q.search(f["question"])]
    rng.shuffle(substantive)
    rng.shuffle(filler)
    n_fact = 24
    pick_f = (substantive + filler)[:n_fact]
    while len(pick_f) < n_fact:  # если не хватило — досыпать из мета
        pick_f += [f for f in facts if f not in pick_f][:n_fact - len(pick_f)]
    print(f"factual: {len(facts)} -> {len(pick_f)} (субст. {len(substantive)})", flush=True)

    comps = [c for c in comp if ok(c["doc_ids"][0]) and ok(c["doc_ids"][1])]
    rng.shuffle(comps)
    pick_c = comps[:13]
    print(f"composite: {len(comps)} -> {len(pick_c)}", flush=True)

    absr = [f for f in ver["abstract"]["ok"] if ok(f["doc_ids"][0])]
    rng.shuffle(absr)
    pick_a = absr[:13]
    print(f"abstract: {len(absr)} -> {len(pick_a)}", flush=True)

    negs = [f["question"] for f in ver["negative"]["ok"]]
    for q in MANUAL_NEG:
        words = {w for w in re.findall(r"[А-ЯЁ][а-яё]{5,}", q)
                 if w.lower() not in ("федеральным", "федерального", "федеральная",
                                      "российской", "федерации", "соответствии", "предусмотрены",
                                      "предусмотрен", "порядок", "каков", "какие", "полномочия")}
        if any(w.lower() in hay for w in words):
            print(f"  manual neg ОТКЛОНЕН (слово в корпусе): {q[:60]}", flush=True)
            continue
        negs.append(q)
    pick_n = negs[:10]
    print(f"negative: {len(negs)} -> {len(pick_n)}", flush=True)

    out = []
    i = 1
    for kind, items, has_ctx in (("factual", pick_f, True), ("composite", pick_c, True),
                                 ("abstract", pick_a, True), ("negative", pick_n, False)):
        for it in items:
            if kind == "negative":
                out.append({"id": i, "question": it, "type": kind,
                            "answer_contexts": [], "doc_ids": []})
            else:
                ctx = it.get("answer_contexts") or [it["answer_context"]]
                out.append({"id": i, "question": it["question"], "type": kind,
                            "answer_contexts": ctx, "doc_ids": it["doc_ids"]})
            i += 1
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written: {len(out)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
