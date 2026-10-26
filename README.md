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
| `data/gold/gold_full.json` | Золотой набор: **60** вопросов (v2: 19 факт / 18 составной / 13 абстрактных / 10 негативных), вербальные цитаты из wikitext. Сгенерирован по `prompts/gold_prompt.md`, верифицирован `build_gold_v2.py` против корпуса. |
| `data/gold/gold_val.json` / `gold_test.json` / `baseline_no_rag.json` | Split 42/18 и baseline (18 ответов LLM без контекста) — регенерируются: `python src/gold.py split`, `python src/baseline.py` |

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

В2: 60 вопросов (лимит ТЗ 40–60). Пайплайн: модель по `prompts/gold_prompt.md` →
`gold_draft_v2.json` → автостроитель `build_gold_v2.py` (вербальные цитаты assert'ятся как
подстроки wikitext, doc_ids — точные заголовки, негативы — grep по корпусу; 2 негатива
отклонены как «есть в корпусе») → ручная правка 3 abstract (id 33/38/40) → 11 long-form
вопросов (id 52–63). Баланс: 19/18/13/10. Валидация: `python src/gold.py check`.

## Выводы (кратко, подробно в `reports/kt1/otchet_kt1.md`)

- Средний документ 1231 токен, медиана 688; 246 страниц >2000 (метасводки), 4 страницы <50 (hotfix/аббрев).
- Основной шум — разметка `{{}}`/`[[ ]]` (спецсимволы 11.5%), а не эмодзи (0%).
- Точных дублей 0, near-dup (Jaccard ≥ 0.7) — 84 страницы (версии страниц, bestiary-сводки, hotfix).
- Baseline (локальная Qwen3.8-27B без контекста): «не знаю» на 11/18 test-вопросов (factual 4/6,
  composite 2/5, abstract 2/4, negative 3/3); остальное — параметрические знания об игре, не
  привязанные к корпусу. RAG по вики обязателен.
