# Корпус

В качестве корпуса взяли вики **The Binding of Isaac: Rebirth** (`bindingofisaacrebirth.wiki.gg`) — «вики-подобный
ресурс с шаблонами, разметкой и служебными страницами» из ТЗ.

Итог: **1 776 документов, 2 186 317 токенов** (tiktoken, cl100k_base). 

| Показатель | Значение |
|---|---|
| Документов | 1 776 |
| Суммарно токенов | 2 186 317 |
| Средняя длина, токенов | 1 231 |
| Медиана | 688 |
| Очень короткие (<50 токенов) | 4 (0.2%) |
| Очень длинные (>2000 токенов) | 246 (13.9%) |
| Эмодзи | 0.00% |
| URL | 0.65% |
| Числа | 1.39% |
| Спецсимволы / разметка | **11.46%** |
| Документов с >20% «мусора» | 301 (16.9%) |
| Точные дубликаты | 0 |
| Near-duplicates (MinHash, Jaccard ≥ 0.7) | **84 (4.7%)** |

![](lengths_hist.png){width=85%}

Самые длинные: «The Binding of Isaac: Repentance (Pre-Release)» — 31 331, «Co-op» — 27 527,
«Isaac's Last Will» — 22 599, «V1.06.J787» — 21 159, «Bestiary (Repentance)» — 14 212 токенов.
Самые короткие: «AB» — 13, «AB+» — 15, «Key Piece (Disambiguation)» — 35, «V1.06 Hotfix» — 47.

**Сложность и лексическое разнообразие.**
- Основной мусор — **разметка** (эмодзи 0%, спецсимволы 11.46%): в среднем ~56 пар
  `{{ }}` / `[[ ]]` на страницу. BM25 без очистки будет ранжировать по скобкам и ссылкам.
- Условные значения: `{{dlcalt|13|r=9}}` — в одном месте две цифры (Rebirth | Repentance) без
  явной привязки. Ретрив вернёт «правильный» чанк, но генерации будет неясно, какую цифру брать.
- Длинные метасводки: наивный чанкинг разнесёт секции Effects/Notes/Synergies по разным чанкам.
- Near-дубли (84): версии одних страниц («…(Pre-Release)» vs основная), bestiary-сводки, hotfix —
  топы выдачи заполнятся версиями одной статьи; нужна дедупликация (задача КТ 2/3).
- Disambiguation-страницы — ловушки BM25: много совпадений по имени, мало смысла в содержимом.

Вывод для КТ 2: агрессивная чистка именно разметки (развернуть `{{dlc|...}}`-условности и ссылки в
текст), секционный чанкинг длинных страниц, дедупликация near-dup.

# Золотой набор

## Схема записи

```json
{
  "id": 52,
  "question": "According to the wiki, what is Delirium's boss id and base HP, in which room does he appear, and from what chest does he drop?",
  "type": "factual",
  "answer_contexts": [" | id          = 412  | environment = <u>Boss</u><br>{{floor|The Void}} ..."],
  "doc_ids": ["wiki://Delirium"]
}
```

- `id` — порядковый номер примера (1–60);
- `question` — вопрос на английском (корпус англ.), в разных регистрах: короткие точечные и
  развёрнутые long-form («List every floor and room where…», «Compare X and Y based on their wiki pages…»);
- `type` — тип (таблица ниже);
- `doc_ids` — страницы корпуса, где ответ находится (префикс `wiki://` = заголовок страницы;
  для негативных и абстрактных — пусто);
- `answer_contexts` — **вербальные** цитаты (подстроки wikitext, ≤250 символов) из этих страниц;
  для негативных/абстрактных — пусто.

## Баланс типов

| Тип | Кол-во |
|---|---|
| фактический (factual) | 19 |
| составной (composite) | 18 |
| абстрактный (abstract) | 13 |
| негативный (negative) | 10 |

## Лексическое разнообразие

- **Опечатки/грамматические ошибки и мусор — 6 из 60 (10%)**: id 1 «How do unlock Cain?», id 2
  «What need to do…», id 3 «Hashy» (Hushy), id 22 «Compare it livse and Mom's Heart» (Lilith),
  id 23 «…the path ????», id 28 «Gread Mode … how much HPit have».
- **Короткие (3–5 слов) vs развёрнутые (до 34 слов)**: от «Who is Hush?» (id 4) до long-form
  (id 52–63, например 34 слова у id 59); медиана 9–10 слов.
- **Разговорные vs формальные**: «How do I unlock Eve?» (id 11) против «Which item pool do the
  Boss Rush items come from according to the two pages…?» (id 59).
- **Имена собственные в разных формах**: d6 / The D6 (id 51/14), Boss Rush / Boss Run (id 54/29),
  Hush / Hushy (id 16/17), Eve / Tainted Eve (id 8/20), Mom / Tainted Mom (id 38/47).
- **Глубоко доменные вопросы**: тайм-лимиты тайм-челленджей и completion marks (id 29),
  нотации версий DLC в ссылках (id 40), spindown-механика D6 (id 51).

## Примеры каждого типа

factual:

```json
{"id": 52, "question": "According to the wiki, what is Delirium's boss id and base HP, ...?",
 "type": "factual", "answer_contexts": [" | id = 412 | environment = Boss<br>The Void | base hp = 10000 ..."],
 "doc_ids": ["wiki://Delirium"]}
```

composite:

```json
{"id": 56, "question": "Compare Monstro and Delirium based on their wiki pages: base HP, boss id, where they spawn, and what drops from each of them.",
 "type": "composite",
 "answer_contexts": ["| name = Monstro | id = 20 ...", " | base hp = 250 | pool = Boss", " | id = 412 | environment = Boss<br>The Void ..."],
 "doc_ids": ["wiki://Monstro", "wiki://Delirium"]}
```

abstract:

```json
{"id": 60, "question": "Describe the overall structure of this wiki: what are the main categories of pages (characters, items, bosses, modes, meta), how are they organized, and how do pages cross-reference each other?",
 "type": "abstract", "answer_contexts": [], "doc_ids": []}
```

negative:

```json
{"id": 62, "question": "According to the wiki, which new items and trinkets did the Booster Pack #6 update add, and in which shop locations do they appear?",
 "type": "negative", "answer_contexts": [], "doc_ids": []}
```

## Разбиение val / test

`src/gold/split.py`: `train_test_split`, `test_size=0.3`, `random_state=42`, стратификация по типу.

| Тип | Validation | Test |
|---|---|---|
| factual | 13 | 6 |
| composite | 13 | 5 |
| abstract | 9 | 4 |
| negative | 7 | 3 |
| **Итого** | **42** | **18** |

# Baseline ответы без RAG

`src/evals/baseline.py`: модель отвечает на 18 test-вопросов **без добавленных контекстов**
(промпт: «Отвечай кратко на русском. Если не знаешь — явно скажи „не знаю“»). Модель — локальная LLM
(Qwen3.8-27B-NVFP4, OpenAI-совместимый API; адрес/ключ/модель в `.env`, шаблон — `.env.example`);
Baseline ответы лежат в `data/gold/baseline_no_rag.json`.

**Результат: 11 из 18 — «не знаю».**

| Тип | «Не знаю» |
|---|---|
| factual | 4/6 |
| composite | 2/5 |
| abstract | 2/4 |
| negative | 3/3 |

Остальные 7 — **параметрические знания модели о самой игре** (6 содержательных ответов и 1 пустой,
id 53), не привязанные к корпусу — нельзя проверить вербальность/источность (нет doc_ids, нет цитат
из страниц); Частично верные, частично нет.

# Воспроизведение

| Файл | Что |
|---|---|
| `data/corpus.jsonl` | корпус: 1 776 доков, 2.19M токенов (сырой wikitext) |
| `data/gold/gold_full.json` | gold: **60** вопросов (источник) |
| `data/gold/gold_val.json` / `gold_test.json` | split 42/18 (производные от full) |
| `data/gold/baseline_no_rag.json` | 18 ответов LLM без контекста (схема: id/type/question/answer/latency_s) |
| `reports/kt1/lengths_hist.png` | гистограмма длин (лог-бины, пороги 50/2000) |
| `src/` (main.py, corpus/, gold/, evals/, utils.py) | пайплайн |

```bash
pip install -r requirements.txt
python src/main.py            # eda → check-gold → split → baseline
python src/main.py --fetch    # + перескачать корпус (~3 мин)
# по шагам:
python src/corpus/fetch.py
python src/corpus/eda.py
python src/gold/check.py
python src/gold/split.py
python src/evals/baseline.py
```
