"""Финализация gold v2: DRAGOn-фильтр, верификация, сборка 60 вопросов.

doc_ids берутся из ИСТОЧНИКА (мы сами отдавали модели фрагмент), а не из
модельного doc_title. Verbatim: выдержка обязана дословно встречаться в
источнике (нормализация: [[..]], пробелы, тире); длина 30–250 (250–320 → trim).

Вход:  data/gold/gold2_candidates.json, data/gold/gold2_negative.json
Выход: data/gold/gold_full.json (60) + отчёт
Запуск: python src/gold/finalize2.py
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import GOLD_DIR, DATA  # noqa: E402

CAND = GOLD_DIR / "gold2_candidates.json"
NEGC = GOLD_DIR / "gold2_negative.json"
CORPUS = DATA / "corpus.jsonl"
OUT = GOLD_DIR / "gold_full.json"

BANNED = re.compile(
    r"как называ|какой номер|какую дату|в каком номере|какова дата|дата редакции|"
    r"какое название|в каком году (принят|издан|опубликован)", re.I)
# именованная сущность: цифры (статья/номер/год/сумма), «…» или юридический объект
ENTITY = re.compile(r"\d|«|кодекс|зако|стать|глав|Указ|постановлен|регламент|"
                    r"ФНС|суд|Государственн|Совет", re.I)
STOP_NEG = {"федеральным", "федерального", "федеральная", "федеральном", "кодекса",
            "кодексу", "кодекс", "законе", "закона", "закон", "российской", "федерации",
            "республики", "гражданский", "гражданским", "уголовный", "уголовном",
            "нормативный", "правовой", "акты", "документ", "документах", "качестве",
            "федеральный", "федеральному", "россии", "соответствии", "порядок",
            "порядка", "условия", "условий", "основания", "оснований", "полномочия",
            "требования", "процедура", "ответственности", "субъекта", "субъектов",
            "управления", "деятельности", "права", "правил", "запрещается",
            "выполнение", "лицензии", "лицензирования", "регистрации"}


def _norm(s):
    s = re.sub(r"\[\[([^\|\]]*)\|([^\]]*)\]\]", r"\2", s)
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def _quality(q):
    return (not BANNED.search(q)) and bool(ENTITY.search(q)) and 15 <= len(q) <= 200


def main():
    cand = json.loads(CAND.read_text(encoding="utf-8"))
    corpus = {}
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            corpus[r["title"]] = r
    id2text = {r["id"]: _norm(r["text"]) for r in corpus.values()}
    titles_all = "\n".join(corpus.keys())

    def verbatim(ctx, doc_id):
        if not ctx or len(ctx) < 30:
            return None
        ctxn = _norm(ctx)
        if len(ctxn) > 250:
            if len(ctxn) > 320:
                return None
            ctx = ctx[:250]
            ctxn = _norm(ctx)
        return ctx if ctxn in id2text[doc_id] else None

    # --- single-doc типы: doc_id из источника ---
    def take_single(kind, n_ok):
        ok, seen = [], set()
        for it in cand[kind]:
            q = it.get("question", "")
            if not _quality(q) or q in seen:
                continue
            ctx = verbatim(it.get("answer_context", ""), it.get("source_doc"))
            if not ctx:
                continue
            if kind == "multistep":
                desc = verbatim(it.get("description", ""), it.get("source_doc"))
                if not desc:
                    continue
            seen.add(q)
            ok.append((q, ctx, it.get("source_doc")))
        return ok

    facts = take_single("factual", 24)
    mults = take_single("multiple", 13)
    # multistep нужен description — отдельно
    multi_ok, seen_m = [], set()
    for it in cand["multistep"]:
        q = it.get("question", "")
        if not _quality(q) or q in seen_m:
            continue
        ctx = verbatim(it.get("answer_context", ""), it.get("source_doc"))
        desc = verbatim(it.get("description", ""), it.get("source_doc"))
        if not (ctx and desc):
            continue
        seen_m.add(q)
        multi_ok.append((q, ctx, desc, it.get("source_doc")))
    print(f"verified: factual={len(facts)} multistep={len(multi_ok)} multiple={len(mults)}")

    # --- cond: пара не сохранилась в кандидате — ищем документы по дословным выдержкам ---
    doclist = [(did, t) for did, t in id2text.items()]
    conds, seen = [], set()
    for it in cand["cond"]:
        q = it.get("question", "")
        # у условного вопроса сущность — в ответе, не в вопросе: фильтр мягче
        if BANNED.search(q) or not (20 <= len(q) <= 200) or q in seen:
            continue
        c1, c2 = it.get("answer_context_1", ""), it.get("answer_context_2", "")
        if not (30 <= len(c1) <= 320 and 30 <= len(c2) <= 320):
            continue
        c1n, c2n = _norm(c1), _norm(c2)
        hits1 = [did for did, t in doclist if c1n in t]
        hits2 = [did for did, t in doclist if c2n in t]
        if not hits1 or not hits2:
            continue
        d1, d2 = hits1[0], hits2[0]
        if d1 == d2:
            continue
        seen.add(q)
        conds.append((q, c1[:250], d1, c2[:250], d2))
    print(f"cond verified: {len(conds)}")

    # --- negative: 10 вопросов о законах, отсутствующих в корпусе (естественный стиль);
    # grep-чистота: слова с заглавной (не из STOP) не должны встречаться в корпусе ---
    hay = _norm("\n".join(r["text"] for r in corpus.values())).lower()
    negs = []
    for q in json.loads(NEGC.read_text(encoding="utf-8")) if NEGC.exists() else []:
        if not (15 <= len(q) <= 200) or BANNED.search(q):
            continue
        words = {w for w in re.findall(r"[А-ЯЁ][а-яё]{5,}", q) if w.lower() not in STOP_NEG}
        if any(w.lower() in hay for w in words):
            continue
        if any(f"№ {n}-ФЗ" in titles_all for n in re.findall(r"(\d{1,4})-ФЗ", q)):
            continue
        negs.append(q)
    print(f"negative verified: {len(negs)}")

    # --- разнообразие: жадный отбор с лимитами по началам ---
    def diverse(items, n, first_max=3, pair_max=2):
        rng = random.Random(42)
        pool = list(items)
        rng.shuffle(pool)
        picked, f_cnt, p_cnt = [], Counter(), Counter()
        for it in pool:
            if len(picked) >= n:
                break
            w = it[0].lower().split()
            f, p = w[0], " ".join(w[:2])
            if f_cnt[f] >= first_max or p_cnt[p] >= pair_max:
                continue
            picked.append(it)
            f_cnt[f] += 1
            p_cnt[p] += 1
        for it in pool:
            if len(picked) >= n:
                break
            if it not in picked:
                picked.append(it)
        return picked

    pick_f = diverse(facts, 24)
    pick_m = diverse(multi_ok, 7)
    pick_c = diverse(conds, 6)
    pick_a = diverse(mults, 13)
    pick_n = negs[:10]
    print(f"picked: factual={len(pick_f)} multistep={len(pick_m)} "
          f"cond={len(pick_c)} multiple={len(pick_a)} negative={len(pick_n)}")

    out, i = [], 1
    for q, ctx, did in pick_f:
        out.append({"id": i, "question": q, "type": "factual", "subtype": "simple",
                    "answer_contexts": [ctx], "doc_ids": [did]}); i += 1
    for q, ctx, desc, did in pick_m:
        out.append({"id": i, "question": q, "type": "composite", "subtype": "multistep",
                    "answer_contexts": [ctx], "answer_description": desc,
                    "doc_ids": [did]}); i += 1
    for q, c1, d1, c2, d2 in pick_c:
        out.append({"id": i, "question": q, "type": "composite", "subtype": "conditional",
                    "answer_contexts": [c1, c2], "doc_ids": [d1, d2]}); i += 1
    for q, ctx, did in pick_a:
        out.append({"id": i, "question": q, "type": "abstract", "subtype": "multiple",
                    "answer_contexts": [ctx], "doc_ids": [did]}); i += 1
    for q in pick_n:
        out.append({"id": i, "question": q, "type": "negative", "subtype": "negative",
                    "answer_contexts": [], "doc_ids": []}); i += 1

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"written: {len(out)} -> {OUT}")
    print("types:", dict(Counter(x["type"] for x in out)))
    print("subtypes:", dict(Counter(x["subtype"] for x in out)))


if __name__ == "__main__":
    main()
