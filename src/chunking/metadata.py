"""Метаданные чанков (2 этапа: извлечение + применение), дедуп-конвенция retrieval.bm25.

Извлечение (dedup + token512):
  структурные: date (последняя ревизия страницы, MediaWiki API), doc_len_tok,
               chunk_pos, n_chunks_in_doc
  контентные:  keywords — top-5 TF-IDF (bigram, sublinear, рус. стоп-слова),
               language — эвристика по доле кириллицы (корпус однотипный: ru)
  качественные: noise — доля неалфавитных токенов

Применение (метрики на gold100: hit/recall/precision/ndcg + тайминги):
  A1) фильтр «свежесть» — только страницы с последней ревизией в последних 50% периода
      (пост-фильтр по ранжированной выдаче; precision/recall до/после)
  A2) фильтр noise — отбросить чанки с noise > 0.5
  B)  prepend «Title. Keywords: …» в текст чанка перед индексированием

Запуск: python src/chunking/metadata.py
Артефакт: stdout (JSON); даты страниц кэшируются в data/page_dates.json.
"""
import json
import re
import sys
import time

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
from chunking.chunkers import chunk, ENC  # noqa: E402
from corpus.clean import load_stage  # noqa: E402
from utils import DATA, GOLD_VAL  # noqa: E402
from retrieval.bm25 import bm25_topk  # noqa: E402
from retrieval.metrics import _dedup_docs, _gold_titles, metrics  # noqa: E402

# sklearn не знает русских стоп-слов — компактный список функциональной лексики
# + генерических правовых слов (они несут мало дискриминации в keywords)
RU_STOP = {"и", "в", "во", "не", "что", "он", "она", "они", "это", "как", "по", "но",
           "а", "то", "так", "же", "из", "для", "о", "об", "при", "от", "к", "у", "на",
           "был", "была", "было", "были", "есть", "быть", "его", "ее", "её", "их",
           "государство", "федерация", "российской", "россии", "федеральный",
           "закона", "законе", "закон", "законов", "статья", "статьи", "пункт",
           "пункта", "пункте", "части", "раздел", "глава", "кодекс", "федеральных",
           "принят", "принята", "подписан", "подписана", "вступил", "вступает",
           "действие", "внесены", "внесение", "изменения", "дополнения", "редакции",
           "ред", "установлен", "установлена", "предусмотрен", "предусмотрена",
           "предусмотрено", "предусмотрены", "случае", "случаях", "целях", "порядке",
           "основание", "основания", "оснований", "гражданин", "граждане", "организация",
           "организации", "юридическое", "лицо", "лицу", "лиц", "должностное"}


def noise_score(text: str) -> float:
    toks = text.split()
    if not toks:
        return 0.0
    alpha = sum(1 for t in toks if re.search(r"[A-Za-zА-Яа-яЁё]", t))
    return 1.0 - alpha / len(toks)


def _language(text: str) -> str:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return "other"
    cyr = sum(1 for c in alpha if "\u0400" <= c <= "\u04FF") / len(alpha)
    return "ru" if cyr >= 0.5 else ("en" if cyr <= 0.1 else "other")


def _load_dates() -> dict:
    p = DATA / "page_dates.json"
    if not p.exists():
        from corpus.fetch import fetch_page_dates
        return fetch_page_dates(p)
    return json.loads(p.read_text(encoding="utf-8"))


def fields(stage: str = "dedup", strategy: str = "token"):
    """Чанки + все метаданные (структурные/контентные/качественные)."""
    t0 = time.time()
    ck = chunk(stage, strategy).copy()
    df = load_stage(stage)
    doc_len = {t: len(ENC.encode(x)) for t, x in zip(df["title"], df["text"])}
    lang = {t: _language(x) for t, x in zip(df["title"], df["text"])}
    dates = _load_dates()
    ck["doc_len_tok"] = ck["doc"].map(doc_len)
    ck["chunk_pos"] = ck.groupby("doc").cumcount()
    ck["n_chunks_in_doc"] = ck.groupby("doc")["doc"].transform("size")
    ck["date"] = ck["doc"].map(dates).fillna("")
    ck["language"] = ck["doc"].map(lang)
    ck["content_type"] = "article"  # единая вики: тип контента гомогенен
    vec = TfidfVectorizer(sublinear_tf=True, stop_words=list(RU_STOP),
                          max_features=20000, ngram_range=(1, 2))
    mat = vec.fit_transform(ck["text"])
    terms = vec.get_feature_names_out()
    ck["keywords"] = [", ".join(terms[j] for j in row.nonzero()[0][:5]) for row in mat]
    ck["noise"] = ck["text"].map(noise_score)
    ck.attrs["extract_secs"] = round(time.time() - t0, 1)
    return ck


def _bm25(toks_list):
    t0 = time.time()
    bm = BM25Okapi(toks_list)
    return bm, round(time.time() - t0, 1)


def _rank(bm, query_toks, ck_docs, keep=None, pool=40):
    """Топ-pool чанков (опционально только из keep-доков) -> уникальные доки."""
    s = bm.get_scores(query_toks)
    order = s.argsort()[::-1]
    if keep is None:
        return _dedup_docs([ck_docs[j] for j in order[:pool]])
    out, seen, kept = [], set(), 0
    for j in order:
        d = ck_docs[j]
        if d in keep:
            kept += 1
            if d not in seen:
                seen.add(d)
                out.append(d)
        if kept >= pool:
            break
    return out


def eval_metadata(stage: str = "dedup", strategy: str = "token", k_list=(5, 10),
                  verbose=True) -> dict:
    ck = fields(stage, strategy)
    extract_secs = ck.attrs["extract_secs"]
    g = _gold_titles(GOLD_VAL)
    queries = list(g["question"])
    qtoks = {q: re.findall(r"\w+", q.lower()) for q in queries}
    ck_docs = ck["doc"].tolist()

    # baseline
    bm, build_b = _bm25([re.findall(r"\w+", t.lower()) for t in ck["text"]])
    lat, out = [], {}
    for q in queries:
        t0 = time.perf_counter()
        out[q] = _rank(bm, qtoks[q], ck_docs)
        lat.append((time.perf_counter() - t0) * 1e3)
    base = metrics(g, out, k_list)

    # A1) свежесть: страницы с датой в последних 50% периода
    dated = sorted((d for d in ck["date"].drop_duplicates() if d))
    cut = dated[len(dated) // 2]
    fresh = set(ck[ck["date"] >= cut]["doc"].drop_duplicates())
    lat_f, out_f = [], {}
    for q in queries:
        t0 = time.perf_counter()
        out_f[q] = _rank(bm, qtoks[q], ck_docs, keep=fresh)
        lat_f.append((time.perf_counter() - t0) * 1e3)
    fresh_res = {"cutoff_date": cut, "docs_kept": len(fresh),
                 "before": base, "after": metrics(g, out_f, k_list)}

    # A2) noise
    low = ck["noise"].to_numpy()
    keep_n = set(ck_docs[j] for j in range(len(ck_docs)) if low[j] <= 0.5)
    lat_n, out_n = [], {}
    for q in queries:
        t0 = time.perf_counter()
        out_n[q] = _rank(bm, qtoks[q], ck_docs, keep=keep_n)
        lat_n.append((time.perf_counter() - t0) * 1e3)
    noise_res = {"before": base, "after": metrics(g, out_n, k_list)}

    # B) prepend
    prep = ck["doc"] + ". Keywords: " + ck["keywords"] + ". " + ck["text"]
    bm2, build_p = _bm25([re.findall(r"\w+", t.lower()) for t in prep])
    lat_p, out_p = [], {}
    for q in queries:
        t0 = time.perf_counter()
        out_p[q] = _rank(bm2, qtoks[q], ck_docs)
        lat_p.append((time.perf_counter() - t0) * 1e3)
    prepend_res = {"build_secs": build_p, "before": base, "after": metrics(g, out_p, k_list)}

    res = {
        "n_chunks": int(len(ck)),
        "extract_secs": extract_secs,
        "noise_mean": round(float(ck.noise.mean()), 4),
        "noise_gt05_share": round(float((ck.noise > 0.5).mean()), 4),
        "language_dist": ck.groupby("language")["doc"].nunique().to_dict(),
        "date_range": [ck.loc[ck["date"] != "", "date"].min(), ck.loc[ck["date"] != "", "date"].max()],
        "bm25_build_secs": build_b,
        "query_ms": {"baseline": round(float(sorted(lat)[len(lat) // 2]), 2),
                     "fresh_filter": round(float(sorted(lat_f)[len(lat_f) // 2]), 2),
                     "noise_filter": round(float(sorted(lat_n)[len(lat_n) // 2]), 2),
                     "prepend": round(float(sorted(lat_p)[len(lat_p) // 2]), 2)},
        "A1_fresh": fresh_res,
        "A2_noise": noise_res,
        "B_prepend": prepend_res,
    }
    if verbose:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    return res


if __name__ == "__main__":
    eval_metadata()
