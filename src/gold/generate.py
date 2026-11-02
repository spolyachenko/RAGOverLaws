"""Генерация кандидатов в золотой набор (русские правовые документы).

LLM-прогон: по образцам документов — фактические и абстрактные вопросы с дословными
выдержками-ответами; по парам страниц — composite; по списку заголовков — negative.
Результат: data/gold/legal_gold_candidates.json (черновая, до верификации).

Запуск: python src/gold/generate.py
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import ENCODEING, DATA, GOLD_DIR  # noqa: E402
from llm.client import client, ask, load_cache, save_cache, key_for  # noqa: E402

CAND_PATH = GOLD_DIR / "legal_gold_candidates.json"
CORPUS = DATA / "law_corpus.jsonl"

P_FACT = """Ты — составитель тестового набора вопросов к базе правовых документов РФ.
Ниже — фрагменты страниц «{titles}» (wikisource).

Придумай {n} фактических вопросов на русском языке, ответ на каждый — конкретный
факт (сумма, срок, число, перечень) из приведённого текста. Требования:
1) ответ на вопрос — дословная выдержка из приведённого текста, не длиннее 250 символов;
2) вопрос должен однозначно определяться этой выдержкой;
3) укажи страницу (title), на которой находится ответ.

Ответ — строго JSON-массив без пояснений:
[{{"question": "...", "answer_context": "<дословная выдержка ≤250 символов>", "doc_title": "..."}}]
"""

P_ABSTRACT = """Ты — составитель тестового набора вопросов к базе правовых документов РФ.
Ниже — фрагменты страниц «{titles}».

Придумай {n} обобщённых (абстрактных) вопросов на русском, требующих осмысления нормы,
а не простого поиска числа: «Как регулируется...?», «Каковы последствия...?», «В каких
случаях...?». Ответ на каждый вопрос должен опираться на приведённый текст; укажи
дословную выдержку-основание (≤250 символов) и страницу (title).

Ответ — строго JSON-массив без пояснений:
[{{"question": "...", "answer_context": "<дословная выдержка ≤250 символов>", "doc_title": "..."}}]
"""

P_COMP = """Ты — составитель тестового набора вопросов к базе правовых документов РФ.
Ниже — фрагменты ДВУХ разных страниц: «{t1}» и «{t2}».

Придумай {n} вопросов на русском, ответ на которые требует связать информацию с ОБЕИХ
страниц (сравнить нормы, условия, последствия из разных документов). Для каждого вопроса
укажи дословную выдержку-основание из каждой страницы (≤250 символов) и оба title.

Ответ — строго JSON-массив без пояснений:
[{{"question": "...", "answer_context_1": "...", "doc_title_1": "...",
  "answer_context_2": "...", "doc_title_2": "..."}}]
"""

P_NEG = """Ты — составитель тестового набора вопросов к базе правовых документов РФ.
База СОДЕРЖИТ только следующие документы:
{titles}

Придумай {n} вопросов на русском языке о российском законодательстве, которые НЕЛЬЗЯ
ответить, имея только эти документы (например, о документах, которых нет в списке:
иные кодексы, иные законы, подзаконные акты, законы других стран). Вопросы должны
выглядеть правдоподобно и быть сформулированы как обычные деловые вопросы.

Ответ — строго JSON-массив строк: ["...", "..."]
"""


def _load_corpus():
    rows = []
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _sample_docs(rows, n=36):
    """Длинные, содержательные страницы: 1/2 кодексы, 1/2 законы."""
    codes = [r for r in rows if "кодекс" in r["title"].lower()]
    laws = [r for r in rows if "кодекс" not in r["title"].lower() and len(r["text"]) > 3000]
    pick = (codes[:18] + laws[:18])[:n]
    if len(pick) < n:
        rest = [r for r in rows if r not in pick and len(r["text"]) > 3000]
        pick += rest[:n - len(pick)]
    return pick


def _doc_text(r, max_chars=6000, window=None):
    """Короткая страница — целиком; длинная — одно окно (window=None -> начало)."""
    t = r["text"]
    if len(t) <= max_chars:
        return t
    if window is None:
        return t[:max_chars]
    start = min(window * max_chars, len(t) - max_chars)
    return t[start:start + max_chars]


def _ask_safe(cl, model, prompt, key, cache, tries=3):
    """ask с ретраями: одна повисшая заявка не роняет весь прогон."""
    import time
    for a in range(tries):
        try:
            raw, _ = ask(cl, model, prompt, key, cache)
            return raw
        except Exception as e:
            if a == tries - 1:
                print(f"  SKIP {key[:20]}: {type(e).__name__}", flush=True)
                return ""
            time.sleep(10)
    return ""


def _parse_json(txt):
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt.startswith("json"):
            txt = txt[4:]
    i, j = txt.find("["), txt.rfind("]")
    return json.loads(txt[i:j + 1]) if i >= 0 else []


def generate():
    rows = _load_corpus()
    print("corpus:", len(rows), flush=True)
    cl, model = client()
    cache = load_cache()

    docs = _sample_docs(rows)
    out = {"factual": [], "abstract": [], "composite": [], "negative": []}

    def fact_task(doc, w, kind, n):
        frag = _doc_text(doc, window=w)
        prompt = (P_FACT if kind == "factual" else P_ABSTRACT).format(titles=doc["title"], n=n)
        raw = _ask_safe(cl, model, prompt + "\n\n" + frag + "\n/no_think",
                        key_for(f"gold_{kind}:{doc['id']}:{w}:{n}", frag), cache)
        items = []
        for it in _parse_json(raw):
            it["source_doc"] = doc["id"]
            items.append(it)
        return kind, items

    tasks = []
    for r in docs:
        windows = (0, 1) if len(r["text"]) > 2 * 6000 else (0,)
        for w in windows:
            tasks.append((r, w, "factual", 2))
            tasks.append((r, w, "abstract", 1))
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for kind, items in ex.map(lambda t: fact_task(*t), tasks):
            out[kind] += items
            done += 1
            if done % 20 == 0:
                save_cache(cache)
                print(f"  fact+abs: {done}/{len(tasks)}", flush=True)
    save_cache(cache)
    print(f"fact+abs: {len(out['factual'])}+{len(out['abstract'])}", flush=True)

    # composite: пары соседних документов (кодекс-кодекс, кодекс-закон)
    def comp_task(pair):
        a, b = pair
        raw = _ask_safe(cl, model, P_COMP.format(t1=a["title"], t2=b["title"], n=2)
                  + "\n\n=== " + a["title"] + " ===\n" + _doc_text(a, 6000)
                  + "\n\n=== " + b["title"] + " ===\n" + _doc_text(b, 6000) + "\n/no_think",
                  key_for("gold_comp:", a["id"] + "|" + b["id"]), cache)
        items = []
        for it in _parse_json(raw):
            it["source_docs"] = [a["id"], b["id"]]
            items.append(it)
        return items

    pairs = [(docs[i], docs[i + 1]) for i in range(0, len(docs) - 1, 2)][:12]
    with ThreadPoolExecutor(max_workers=8) as ex:
        out["composite"] = [it for items in ex.map(comp_task, pairs) for it in items]
    save_cache(cache)
    print(f"composite: {len(out['composite'])}", flush=True)

    # negative: по списку заголовков
    titles = "\n".join(f"- {r['title']}" for r in rows[:200])
    raw = _ask_safe(cl, model, P_NEG.format(titles=titles, n=10) + "\n/no_think",
                 key_for("gold_neg:", "all"), cache)
    save_cache(cache)
    out["negative"] = [{"question": q} for q in _parse_json(raw)]

    CAND_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAND_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in out.items():
        print(k, len(v), flush=True)
    print("written:", CAND_PATH, flush=True)


if __name__ == "__main__":
    generate()
