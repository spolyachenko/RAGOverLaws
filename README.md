# КТ 1 — Корпус и золотой набор (RAG-проект, курс LLM-инженер)

Сквозной RAG-проект (КТ1–КТ5). КТ 1: **корпус сырых данных + EDA + золотой набор + baseline без RAG**.

## Структура

```
src/common.py        — общее для всех КТ: пути, загрузка корпуса, токены, .env
src/fetch_wiki.py    — выгрузка всей вики TBOI:Rebirth (asyncio) → data/corpus.jsonl
src/eda.py           — EDA: гистограмма длин, мусорные токены, MinHash-дубли → reports/kt1/
src/gold.py          — валидация gold + validation/test split (python src/gold.py check|split)
src/baseline.py      — baseline LLM без RAG (локальная модель из .env)
src/run_kt1.py       — весь КТ1 по порядку
data/                — корпус и золотой набор
reports/kt1/         — отчёт: otchet_kt1.md + lengths_hist.png (вся EDA-статистика — в таблицах отчёта)
prompts/gold_prompt.md   — промпт генерации gold-набора (EN)
.env.example         — шаблон конфигурации LLM
requirements.txt
```

Следующие КТ: `src/kt2_*.py` … по аналогии, общее — в `src/common.py`.

| Данные | Что |
|---|---|
| `data/corpus.jsonl` | Корпус: **вся вики The Binding of Isaac: Rebirth** (1776 страниц, 2.19M токенов, сырой wikitext). 1661 служебная redirect-страница отбрасывается. |
| `data/gold/gold_full.json` | Золотой набор: **48** вопросов (v2: 15 факт / 15 составной / 10 абстрактных / 8 негативных), с `doc_ids` и вербальными цитатами. Сгенерирован по `prompts/gold_prompt.md`, верифицирован `build_gold_v2.py` против корпуса. |
| `data/gold/gold_val.json` / `gold_test.json` / `baseline_no_rag.json` | Регенерируются по требованию: `python src/gold.py split`, `python src/baseline.py` |

## Запуск

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # LLM: локальная (OpenAI-совместимый API)

python src/run_kt1.py             # весь КТ1
python src/run_kt1.py --fetch     # + перескачать корпус

# по шагам:
python src/fetch_wiki.py          # выгрузка вики
python src/eda.py                 # EDA
python src/gold.py check|split    # gold
python src/baseline.py            # baseline
```

## LLM (baseline)

Модель берётся из `.env` (OpenAI-совместимый API). Если API недоступен — baseline.py грузит
предвычисленный `data/gold/baseline_no_rag.json`.

## Источник корпуса

Целенаправленно «грязный» источник из ТЗ — вики-подобный ресурс. `fetch_wiki.py` берёт все страницы
namespace 0 вики `bindingofisaacrebirth.wiki.gg` через `api.php` (list=allpages + revisions по 50 за
запрос, 20 параллельных, asyncio), сохраняет сырой wikitext с шаблонами `{{...}}` и ссылками `[[...]]`.
Redirect-страницы `#REDIRECT` и страницы <40 символов отбрасываются как служебные.

## Золотой набор

В2: 48 вопросов, сгенерированы нормальной моделью по `prompts/gold_prompt.md` (дёрфт в
`gold_draft_v2.json`), верифицированы `build_gold_v2.py` против корпуса: цитаты — вербальные
подстрочники wikitext (assert), doc_ids — точные заголовки страниц, негативы — grep по
корпусу (2 негатива отклонены как "есть в корпусе": Slay the Spire, Celeste — см.
`gold_out_v2.json`). Баланс: 15/15/10/8. Три abstract-вопроса (id 33, 38, 40) поправлены вручную
(id=40 был невалиден: Talk-страницы в корпусе отсутствуют), split и baseline регенерируются скриптами.

## Выводы (кратко, подробно в `reports/kt1/otchet_kt1.md`)

- Средний документ 1231 токен, медиана 688; 246 страниц >2000 (метасводки), 4 страницы <50 (hotfix/аббрев).
- Основной шум — разметка `{{}}`/`[[ ]]` (спецсимволы 11.5%), а не эмодзи (0%).
- Точных дублей 0, near-dup (Jaccard ≥ 0.7) — 84 страницы (версии страниц, bestiary-сводки, hotfix).
- Baseline (локальная Qwen3.8-27B без контекста): «не знаю» на 14/15 test-вопросов (factual 4/4,
  composite 5/5, abstract 3/3, negative 2/3). RAG по вики обязателен.
