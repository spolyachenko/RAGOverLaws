# RAG-проект (курс LLM-инженер, КТ1–КТ5)

Сквозной RAG по вики The Binding of Isaac: Rebirth (сырой wikitext, MediaWiki):
корпус + золотой набор (КТ1) → предобработка/чанкинг/метаданные (КТ2) → векторная БД,
эмбеддинги и sparse/dense/hybrid (КТ3) → трансформация запросов и реранкинг (КТ4)
→ генерация и оценка (КТ5). Отчёты по каждому КТ — в `reports/`.

## Структура

```
data/
├── corpus.jsonl              # сырой корпус (1 776 доков, 2.19M токенов)
├── postprocess_corpus.jsonl  # после предобработки (1 734 дока)
├── chunks.jsonl              # чанки TokenChunker 512 (12 067)
├── llm_chunks.jsonl          # LLM-чанки summary/HyDE (100)
├── llm_cache.json            # кэш LLM (повторный прогон ~0 токенов)
└── gold/                     # gold_full/val/test.json, baseline_no_rag.json

src/
├── utils.py            # общее: пути, загрузка корпуса, токены, .env
├── main.py             # пайплайн корпуса: fetch → eda → gold → baseline
├── corpus/
│   ├── fetch.py        # выгрузка вики (MediaWiki API, asyncio) → data/corpus.jsonl
│   ├── eda.py          # EDA: длины, мусорные токены, дубли → reports/
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
├── gold/
│   ├── check.py        # валидация золотого набора
│   └── split.py        # стратифицированный validation/test split
└── evals/
    └── baseline.py     # baseline LLM без RAG

reports/kt1/  # отчёт otchet_kt1.md + lengths_hist.png (закрыт)
reports/kt2/  # отчёт otchet_kt2.md
notebooks/    # rag_project.ipynb
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

# золотой набор
python src/gold/check.py                                 # валидация
python src/gold/split.py                                 # validation/test split
```

Метрики ретрива — на validation split золотого набора (42 вопроса, multi-doc: 1–3 релевантных
документа на вопрос); test до финала проекта не используется.
