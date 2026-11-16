# RAG-проект (курс LLM-инженер, КТ1–КТ5)

Сквозной RAG по правовым документам РФ (ru.wikisource.org, сырой wikitext MediaWiki,
русский язык): корпус + золотой набор (КТ1) → предобработка/чанкинг/метаданные (КТ2) →
векторная БД, эмбеддинги и sparse/dense/hybrid (КТ3) → трансформация запросов и реранкинг (КТ4)
→ генерация и оценка (КТ5). Отчёты по каждому КТ — в `reports/`.

## Структура

```
data/
├── corpus.jsonl              # корпус (2 218 доков: 2 207 вики + 12 законов, добавленных под gold100)
├── postprocess_corpus.jsonl  # после предобработки (1 404 дока, dedup)
├── page_dates.json           # даты ревизий страниц (из дампа wikisource)
├── llm_chunks.jsonl          # LLM-чанки summary/HyDE (100)
├── llm_cache.json            # кэш LLM (повторный прогон ~0 токенов)
└── gold/
    ├── gold100.json              # золотой набор: 100 вопросов (factual 30, composite 30, abstract 20, negative 20)
    ├── gold100_val.json          # validation split (70): для подбора всех конфигураций
    └── gold100_test.json         # test split (30): только финальная оценка

src/
├── utils.py            # общее: пути, загрузка корпуса, токены, .env
├── main.py             # пайплайн корпуса: fetch → eda → baseline
├── corpus/
│   ├── fetch.py        # выгрузка вики (MediaWiki API + fetch_law_corpus из дампа) → data/corpus.jsonl
│   ├── eda.py          # EDA: длины, мусорные токены, дубли
│   └── clean.py        # предобработка: raw → markup → links → normalize → dedup
├── chunking/
│   ├── chunkers.py     # чанкеры: token/sentence/semantic/neural + semantic-ablation
│   ├── parent.py       # ParentDocumentRetriever: дерево документ → родители → дети
│   └── metadata.py     # метаданные чанков + сценарии (фильтр/prepend)
├── retrieval/
│   ├── metrics.py      # Hit/Recall@k (multi-doc) + gold-запросы
│   ├── bm25.py         # BM25 (rank_bm25)
│   └── dense.py        # BGE-M3 dense + parent/child (GPU)
├── llm/
│   ├── client.py       # OpenAI-совместимый клиент из .env + кэш по sha1
│   ├── process.py      # суммаризация + HyDE (8-way, кэш)
│   └── eval.py         # метрики original vs summary vs HyDE
└── evals/
    ├── baseline.py     # baseline LLM без RAG (по ТЗ — на validation)
    └── judge.py        # LLM-судья: корректность ответов по эталонным контекстам

reports/kt1/  # отчёт otchet_kt1.md (закрыт)
reports/kt2/  # отчёт otchet_kt2.md
.env          # локальная LLM (OpenAI-совместимый API); шаблон — .env.example
requirements.txt
```

## Запуск

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # локальная LLM (OpenAI-совместимый API)

# КТ1 (пайплайн корпуса)
python src/main.py            # весь пайплайн (fetch пропускается, если корпус есть; --fetch — перескачать)

# предобработка и ретрив (каждая таблица отчёта — своя команда; соответствие — reports/kt2/otchet_kt2.md §9)
python src/corpus/clean.py                               # → data/postprocess_corpus.jsonl
python src/retrieval/bm25.py eval <стадия> <чанкер>      # стадии: raw markup links normalize dedup
                                                         # чанкеры: token sentence semantic neural
python src/chunking/chunkers.py ablation                 # SemanticChunker: пороги × эмбеддеры
python src/retrieval/dense.py eval dedup token           # BGE-M3 dense (GPU)
python src/retrieval/dense.py parent                     # ParentDocumentRetriever, 3 комбинации (GPU)
python src/chunking/metadata.py                          # метаданные + сценарии A/B
python src/llm/process.py && python src/llm/eval.py      # LLM summary + HyDE (кэш — перезапуск дёшев)
```

Метрики ретрива — на validation split gold100 (56 вопросов с doc_ids;
конвенция: top-40 чанков → уникальные документы); test (24 с doc_ids) —
только финальная оценка, в подборе параметров не участвует.

gold100: все 80 не-негативных вопросов заполнены (factual 30, composite 30, abstract 20),
каждый answer_contexts — нормализованная дословная подстрока текста документа (30–250 символов),
doc_ids существуют в корпусе. 20 negative — намеренно без контекста.
