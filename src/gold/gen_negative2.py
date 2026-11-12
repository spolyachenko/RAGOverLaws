"""Генерация negative-вопросов v2: законы, отсутствующие в корпусе, естественный стиль.

Вход:  data/corpus.jsonl (список заголовков)
Выход: data/gold/gold2_negative.json (массив строк, до grep-фильтра в finalize2)
Запуск: python src/gold/gen_negative2.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA, GOLD_DIR  # noqa: E402
from llm.client import client, ask, load_cache, save_cache, key_for  # noqa: E402

OUT = GOLD_DIR / "gold2_negative.json"
CORPUS = DATA / "corpus.jsonl"

P = """Ты — составитель тестового набора к базе правовых документов РФ.
База содержит ТОЛЬКО документы из списка ниже (кодексы РФ и часть федеральных
законов).

Придумай {n} естественных вопросов на русском языке о российском законодательстве,
на которые НЕЛЬЗЯ ответить, имея только эти документы, потому что нужного закона
в базе НЕТ. Каждый вопрос должен называть конкретный закон, которого нет в списке
(например: «О персональных данных», «О защите конкуренции», «Об образовании в РФ»,
«О государственной гражданской службе», «О лицензировании отдельных видов
деятельности», «Об оружии», «О государственной регистрации недвижимости»,
«О почтовой связи», «О видеопрограммах», «О страховании», «О банкротстве» — но
ТОЛЬКО если его нет в списке; перепроверь по списку).

Стиль — как у реальных людей, без канцелярита:
«Как оформить согласие на обработку персональных данных?»,
«Можно ли уволить судью без согласия суда?»,
«Что нужно для регистрации товарного знака?»
— разные конструкции, без повторов «Каков порядок…».

Список документов в базе:
{titles}

Ответ — строго JSON-массив строк, без пояснений.
"""


def main():
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    titles = "\n".join(f"- {r['title']}" for r in rows)
    cl, model = client()
    cache = load_cache()
    raw, _ = ask(cl, model, P.format(n=40, titles=titles) + "\n/no_think",
                 key_for("g2neg:", "v2"), cache)
    save_cache(cache)
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt.startswith("json"):
            txt = txt[4:]
    i, j = txt.find("["), txt.rfind("]")
    qs = json.loads(txt[i:j + 1])
    OUT.write_text(json.dumps(qs, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"written: {len(qs)} -> {OUT}")


if __name__ == "__main__":
    main()
