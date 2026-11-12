"""Генерация gold-набора v2: естественные вопросы по критериям DRAGOn.

Типы:
- factual   (simple): один факт, естественная формулировка, НЕ «как называется
                     глава N» / «какой номер» — номер сам по себе не вопрос.
- multistep : промежуточная сущность в вопросе НЕ названа, а описана через другой
              факт из корпуса (многошаговый).
- cond      : в вопросе два факта из (возможно) разных документов, общей сущностью
              является ОТВЕТ (условный).
- multiple  : ответ — набор (перечень оснований/документов/случаев) из конкретного
              фрагмента (множественный).
- negative  : закон/кодекс, отсутствующие в корпусе.

Запреты для всех: шаблонные конструкции, повторяющиеся начала, вопросы без
именованных сущностей (статья/номер/дата/орган), ответы, не зависящие от
конкретного фрагмента (знает любая малая модель без контекста).

Вход:  data/corpus.jsonl
Выход: data/gold/gold2_candidates.json
Запуск: python src/gold/generate2.py
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import ENCODEING, DATA, GOLD_DIR  # noqa: E402
from llm.client import client, ask, load_cache, save_cache, key_for  # noqa: E402

CAND_PATH = GOLD_DIR / "gold2_candidates.json"
CORPUS = DATA / "corpus.jsonl"

COMMON = """Ты — составитель тестового набора вопросов к базе правовых документов РФ
для RAG-системы. КРИТИЧЕСКИ ВАЖНО: вопросы должны звучать как вопросы реальных
людей, а не как поисковый запрос и не как шаблон.

ЗАПРЕЩЕНО (такие вопросы отбрасываются):
- «Как называется статья/глава N…», «Какой номер…», «В каком номере СЗ РФ…»,
  «Какова дата редакции…» — где номер/название и есть суть вопроса;
- одинаковые конструкции; вопрос, на который можно ответить, не читая документ
  (общие знания любой малой модели);
- расплывчатые «Как регулируется …?», «Что такое …?» без привязки к конкретной
  норме;
- вопросы без именованной сущности (номер статьи, номер/год закона, орган,
  конкретная сумма или срок).

ОБЯЗАТЕЛЬНО:
- у каждого вопроса есть именованная сущность (статья, закон с номером/датой,
  орган, сумма, срок);
- ответ на вопрос нельзя дать точно, не прочитав конкретный фрагмент документа
  (зависит от действующей редакции: точные сроки, суммы, перечни);
- вопрос понятен без уточнений, намерение однозначное;
- стиль живой, разный от вопроса к вопросу: переставляй порядок слов, меняй
  вопросительные слова, иногда начинай с обстоятельства, иногда с вопроса,
  допустимы разговорно-деловые обороты («Что будет, если…», «Можно ли…»,
  «К кому обращаться…»).

ПРАВИЛА ОТВЕТА:
- answer_context — ДОСЛОВНАЯ выдержка из приведённого текста, ≤250 символов,
  которая является (или однозначно содержит) ответом;
- doc_title — точный заголовок страницы из приведённого текста.

Формат ответа — строго JSON-массив, без пояснений."""

P_FACT = COMMON + """

Задача: придумать {n} ПРОСТЫХ вопроса (один факт в ответе: сумма, срок, число,
кратность, орган, перечень-однозначение). Каждый вопрос — о СУЩНОСТИ нормы, а не
о её имени/номере. Примеры ДОБРОго: «Какой размер ущерба по ст. 217.1 УК РФ
считается крупным?», «Что грозит за нарушение требований промышленной
безопасности по ч. 2 ст. 215.1 УК РФ?», «Сколько времени даётся на представление
документов в налоговый орган при досмотре?». Примеры ПЛОХОго: «Как называется
глава 9 закона № 209-ФЗ?», «В каком номере Собрания законодательства опубликована
Конституция?».

[{{"question": "...", "answer_context": "...", "doc_title": "..."}}]
"""

P_MULTISTEP = COMMON + """

Задача: придумать {n} МНОГОШАГОВЫХ вопроса. Промежуточная сущность (закон,
орган, статья, действие) в вопросе НЕ НАЗВАНА — она описана через другой факт
из того же текста, и ответить можно, только связав два фрагмента.
Пример: вместо «Какой штраф по ст. 217.1 УК РФ?» — «Какой штраф установлен тем
положениями УК РФ, которыми введено деяние, за которое впервые предусмотрена
наказание… за нарушение правил безопасности при ведении горных работ?» (статья
названа через факт о ней). Второй шаг — поиск описываемой сущности, первый —
ответ о ней. Оба факта должны быть в приведённом тексте; answer_context —
фрагмент с ОТВЕТОМ (о сущности), description — фактом-описателем (тоже
дословно, ≤250 символов).

[{{"question": "...", "answer_context": "...", "description": "...", "doc_title": "..."}}]
"""

P_COND = COMMON + """

Задача: по ДВУМ фрагментам придумать {n} УСЛОВНЫХ вопроса. В вопросе
используются ДВА факта (по одному из каждого фрагмента), а ответом является
ОБЩАЯ для обоих сущность (орган, закон, срок, сумма, понятие). Общая сущность
в вопросе НЕ называется — только её свойства из обоих фактов.
Пример: факт A из закона X: «…органы прокуратуры…», факт B из кодекса Y:
«…прокуратура осуществляет надзор…»; вопрос: «Кто и следствием руководит, и
надзор за исполнением законов осуществляет?» — ответ: прокуратура.
answer_context_1 / answer_context_2 — дословные выдержки из каждого фрагмента
(≤250 символов), doc_title_1 / doc_title_2 — заголовки.

[{{"question": "...", "answer_context_1": "...", "doc_title_1": "...",
  "answer_context_2": "...", "doc_title_2": "..."}}]
"""

P_MULTIPLE = COMMON + """

Задача: придумать {n} МНОЖЕСТВЕННЫХ вопроса, где ответ — ПЕРЕЧЕНЬ (набор
оснований, документов, случаев, полномочий, действий) из конкретного фрагмента.
Формулируй естественно: «Что нужно сделать, чтобы…», «В каких случаях …
не допускается», «Какие документы … обязан хранить …». Перечень в тексте должен
быть компактным (3–7 пунктов), чтобы поместился в выдержку ≤250 символов.

[{{"question": "...", "answer_context": "...", "doc_title": "..."}}]
"""

P_NEG = """Ты — составитель тестового набора к базе правовых документов РФ.
База содержит ТОЛЬКО документы из списка ниже.

Придумай {n} естественных вопросов на русском о российском законодательстве,
которые НЕЛЬЗЯ ответить по этой базе, потому что нужного закона/кодекса в базе
НЕТ (например: о персональных данных, о защите конкуренции, об образовании,
о госслужбе, о недвижимости, об оружии, о лицензировании — но ТОЛЬКО те,
которых нет в списке). Вопросы — как у реальных людей: деловые или
разговорно-деловые, без канцелярита. Не упоминай слов, которые входят в
названия документов из списка.

Список документов в базе:
{titles}

Ответ — строго JSON-массив строк: ["...", "..."]
"""


def _load_corpus():
    with open(CORPUS, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def _sample_docs(rows, n=40):
    codes = [r for r in rows if "кодекс" in r["title"].lower()]
    laws = [r for r in rows if "кодекс" not in r["title"].lower() and len(r["text"]) > 3000]
    pick = (codes[:20] + laws[:20])[:n]
    if len(pick) < n:
        rest = [r for r in rows if r not in pick and len(r["text"]) > 3000]
        pick += rest[:n - len(pick)]
    return pick


def _doc_text(r, max_chars=6000, window=None):
    t = r["text"]
    if len(t) <= max_chars:
        return t
    if window is None:
        return t[:max_chars]
    start = min(window * max_chars, len(t) - max_chars)
    return t[start:start + max_chars]


def _ask_safe(cl, model, prompt, key, cache, tries=3):
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
    if i < 0:
        return []
    try:
        return json.loads(txt[i:j + 1])
    except json.JSONDecodeError:
        # попробуем довести до ближайшего валидного массива
        for end in range(j, i, -50):
            try:
                return json.loads(txt[i:end + 1] if txt[end] != "]" else txt[i:end + 1])
            except json.JSONDecodeError:
                continue
        return []


def generate():
    rows = _load_corpus()
    print("corpus:", len(rows), flush=True)
    cl, model = client()
    cache = load_cache()
    out = {"factual": [], "multistep": [], "cond": [], "multiple": [], "negative": []}

    docs = _sample_docs(rows)

    def single_task(doc, w, kind, n):
        frag = _doc_text(doc, window=w)
        p = {"factual": P_FACT, "multistep": P_MULTISTEP, "multiple": P_MULTIPLE}[kind]
        raw = _ask_safe(cl, model, p.format(n=n) + "\n\n" + frag + "\n/no_think",
                        key_for(f"g2_{kind}:{doc['id']}:{w}", frag), cache)
        items = [{"question": it.get("question", ""),
                  "answer_context": it.get("answer_context", ""),
                  "doc_title": it.get("doc_title", "")} for it in _parse_json(raw)]
        if kind == "multistep":
            for it2 in _parse_json(raw):
                pass
            items = []
            for it in _parse_json(raw):
                items.append({"question": it.get("question", ""),
                              "answer_context": it.get("answer_context", ""),
                              "description": it.get("description", ""),
                              "doc_title": it.get("doc_title", "")})
        for it in items:
            it["source_doc"] = doc["id"]
        return kind, items

    tasks = []
    for r in docs:
        windows = (0, 1) if len(r["text"]) > 12000 else (0,)
        for w in windows:
            tasks.append((r, w, "factual", 3))
            tasks.append((r, w, "multistep", 2))
            tasks.append((r, w, "multiple", 2))
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for kind, items in ex.map(lambda t: single_task(*t), tasks):
            out[kind] += items
            done += 1
            if done % 20 == 0:
                save_cache(cache)
                print(f"  single: {done}/{len(tasks)}", flush=True)
    save_cache(cache)
    print(f"factual={len(out['factual'])} multistep={len(out['multistep'])} "
          f"multiple={len(out['multiple'])}", flush=True)

    # cond: пары документов, общая сущность
    def cond_task(pair):
        a, b = pair
        prompt = (P_COND.format(n=2)
                  + "\n\n=== " + a["title"] + " ===\n" + _doc_text(a, 4000)
                  + "\n\n=== " + b["title"] + " ===\n" + _doc_text(b, 4000) + "\n/no_think")
        raw = _ask_safe(cl, model, prompt, key_for("g2_cond:", a["id"] + "|" + b["id"]), cache)
        items = []
        for it in _parse_json(raw):
            items.append({"question": it.get("question", ""),
                          "answer_context_1": it.get("answer_context_1", ""),
                          "doc_title_1": it.get("doc_title_1", ""),
                          "answer_context_2": it.get("answer_context_2", ""),
                          "doc_title_2": it.get("doc_title_2", "")})
        return items

    pairs = [(docs[i], docs[i + 1]) for i in range(0, len(docs) - 1, 2)][:16]
    with ThreadPoolExecutor(max_workers=8) as ex:
        out["cond"] = [it for items in ex.map(cond_task, pairs) for it in items]
    save_cache(cache)
    print(f"cond={len(out['cond'])}", flush=True)

    # negative
    titles = "\n".join(f"- {r['title']}" for r in rows[:300])
    raw = _ask_safe(cl, model, P_NEG.format(titles=titles, n=12) + "\n/no_think",
                    key_for("g2_neg:", "all"), cache)
    save_cache(cache)
    out["negative"] = [{"question": q} for q in _parse_json(raw)]
    print(f"negative={len(out['negative'])}", flush=True)

    CAND_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAND_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=4), encoding="utf-8")
    print("written:", CAND_PATH, flush=True)


if __name__ == "__main__":
    generate()
