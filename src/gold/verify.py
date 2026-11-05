"""Верификация кандидатов в золотой набор: дословность, doc_ids, negatives.

Вход: data/gold/legal_gold_candidates.json (из generate.py)
Выход: data/gold/legal_gold_verified.json + отчёт (сколько провело каждую проверку)

Запуск: python src/gold/verify.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import GOLD_DIR, DATA  # noqa: E402

CAND = GOLD_DIR / "legal_gold_candidates.json"
VER = GOLD_DIR / "legal_gold_verified.json"
CORPUS = DATA / "corpus.jsonl"


def _norm(s: str) -> str:
    """Минимальная нормализация для substring-проверки (wikitext-артефакты)."""
    s = re.sub(r"\[\[([^\|\]]*)\|([^\]]*)\]\]", r"\2", s)
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    return s


def main():
    cand = json.loads(CAND.read_text(encoding="utf-8"))
    corpus = {}
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            corpus[r["title"]] = r

    title2id = {t: r["id"] for t, r in corpus.items()}
    norm_cache = {}

    def norm_text(title):
        if title not in norm_cache:
            norm_cache[title] = _norm(corpus[title]["text"])
        return norm_cache[title]

    def verbatim(ctx, title):
        return bool(ctx) and len(ctx) <= 250 and _norm(ctx) in norm_text(title)

    out = {}
    for kind in ("factual", "abstract"):
        ok, bad = [], 0
        for it in cand[kind]:
            t = it.get("doc_title")
            if t not in title2id:
                bad += 1
                continue
            if not verbatim(it.get("answer_context", ""), t):
                bad += 1
                continue
            it2 = dict(it)
            it2["doc_ids"] = [title2id[t]]
            ok.append(it2)
        out[kind] = {"ok": ok, "rejected": bad}
        print(f"{kind}: ok={len(ok)} rejected={bad}", flush=True)

    ok, bad = [], 0
    for it in cand["composite"]:
        t1, t2 = it.get("doc_title_1"), it.get("doc_title_2")
        if t1 not in title2id or t2 not in title2id:
            bad += 1
            continue
        if not (verbatim(it.get("answer_context_1", ""), t1)
                and verbatim(it.get("answer_context_2", ""), t2)):
            bad += 1
            continue
        it2 = dict(it)
        it2["answer_contexts"] = [it["answer_context_1"], it["answer_context_2"]]
        it2["doc_ids"] = [title2id[t1], title2id[t2]]
        ok.append(it2)
    out["composite"] = {"ok": ok, "rejected": bad}
    print(f"composite: ok={len(out['composite']['ok'])} rejected={bad}", flush=True)

    # negative: отличительные слова вопроса не должны встречаться в тексте корпуса (grep-check)
    hay = "\n".join(r["text"] for r in corpus.values()).lower()
    STOP = {"федеральным", "федерального", "федеральная", "федеральном", "кодекса",
            "кодексу", "кодекс", "законе", "закона", "закон", "российской", "федерации",
            "республики", "гражданский", "гражданским", "уголовный", "уголовном",
            "нормативный", "правовой", "акты", "документ", "документах", "качестве"}
    neg_ok, neg_bad = [], 0
    corpus_titles = "\n".join(corpus.keys())
    for it in cand["negative"]:
        q = it["question"]
        words = {w for w in re.findall(r"[А-ЯЁ][а-яё]{5,}", q) if w.lower() not in STOP}
        if any(w.lower() in hay for w in words):
            # объект вопроса упоминается в корпусе — как negative не годится
            neg_bad += 1
            continue
        # номер закона в вопросе не должен существовать в корпусе (иначе negative ложный)
        lawnums = re.findall(r"(\d{1,4})-ФЗ", q)
        if any(f"№ {n}-ФЗ" in corpus_titles for n in lawnums):
            neg_bad += 1
            continue
        neg_ok.append(it)
    out["negative"] = {"ok": neg_ok, "rejected": neg_bad}
    print(f"negative: ok={len(neg_ok)} rejected={neg_bad}", flush=True)

    VER.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("written:", VER, flush=True)


if __name__ == "__main__":
    main()
