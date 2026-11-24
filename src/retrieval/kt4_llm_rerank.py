"""KT4: LLM-as-reranker (Qwen3.8-27B-FP8, shared .239 — только запросы).

Listwise (основной): 1 вызов — запрос + N чанков -> упорядоченный список номеров.
Pointwise (субвыборка 10 вопросов): N вызовов — балл 0-2 на пару (запрос, чанк).
Pool: dense top-5/10/20 (bge-m3, эмбеддинги запросов — data/kt4_q_emb.npy с 153).
Детерминированность: 20 вопросов, top-10, два прогона при temp=0 -> agreement.
Конвенция: после реранка top-40 чанков -> уникальные документы (как в КТ3).

Артефакт: data/kt4_llm_rerank.json
Запуск (ноут): python src/retrieval/kt4_llm_rerank.py
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATA, GOLD_VAL  # noqa: E402
from retrieval.metrics import _gold_titles, _dedup_docs, metrics, by_type_metrics  # noqa: E402
from llm.client import ask, client, key_for, load_cache, save_cache  # noqa: E402

CHUNK_CHARS = 450
POOL_SIZES = (5, 10, 20)
DET_NQ = 20
PT_NQ = 10


def _topn(q_emb, c_emb):
    s = q_emb @ c_emb.T
    return s.argsort(axis=1)[:, ::-1][:, :max(POOL_SIZES)]


LIST_PROMPT = ("Ты — судья релевантности юридических текстов. Вопрос: {q}\n\n"
               "Чанки корпуса:\n{chunks}\n\n"
               "Расставь чанки по убыванию релевантности к вопросу. "
               "Ответ: только номера чанков через запятую, от самого релевантного "
               "(пример: 3, 1, 7, 5, 2, 4, 6, 8, 9, 10).")

PT_PROMPT = ("Вопрос: {q}\nЧанк: {c}\n\nНасколько чанк релевантен вопросу: "
             "0 = не релевантен, 1 = косвенно связан, 2 = прямо отвечает. "
             "Ответ: одно число 0, 1 или 2.")


def _chunks_str(chunk_ids, texts):
    return "\n".join(f"[{i + 1}] {texts[j][:CHUNK_CHARS]}"
                     for i, j in enumerate(chunk_ids))


def _parse_listwise(out, n):
    nums = [int(x) for x in re.findall(r"\d+", out)]
    valid = [x - 1 for x in nums if 1 <= x <= n]
    seen, order = set(), []
    for x in valid:
        if x not in seen:
            seen.add(x)
            order.append(x)
    while len(order) < n:  # недобор — добираем в исходном порядке
        order.append(next(i for i in range(n) if i not in seen))
        seen.add(order[-1])
    return order[:n], len(valid) >= n  # (порядок, is_wellformed)


def main():
    g = _gold_titles(GOLD_VAL)
    queries = g["question"].tolist()
    ck = pd.read_json(DATA / "chunks.jsonl", lines=True)
    texts = ck["text"].tolist()
    c_emb = np.load(DATA / "emb_bge-m3.npy", mmap_mode="r")
    q_emb = np.load(DATA / "kt4_q_emb.npy")
    cl, model = client()
    cache = load_cache()
    dmax = _topn(q_emb, c_emb)

    out = {"pools": {}, "determinism": {}, "pointwise": {}}

    # 1. listwise pools
    for n in POOL_SIZES:
        lats, topk, bad, ptoks, otoks = [], {}, 0, 0, 0
        for i, q in enumerate(queries):
            ids = dmax[i, :n]
            prompt = LIST_PROMPT.format(q=q, chunks=_chunks_str(ids, texts))
            res, dt = ask(cl, model, prompt, key_for(f"kt4lr{n}:", q), cache)
            order, ok = _parse_listwise(res, n)
            bad += int(not ok)
            ptoks += cache[key_for(f"kt4lr{n}:", q)]["prompt_tok"]
            otoks += cache[key_for(f"kt4lr{n}:", q)]["out_tok"]
            lats.append(dt)
            topk[q] = _dedup_docs(ck["doc"].iloc[ids[order][:40]].tolist())
        ls = sorted(l for l in lats if l > 0) or [0.0]
        out["pools"][str(n)] = {"metrics": metrics(g, topk),
                                "by_type_h5": {t: v["hit@5"] for t, v in
                                               by_type_metrics(g, topk).items()},
                                "llm_ms_median": round(ls[len(ls) // 2] * 1e3, 0),
                                "llm_ms_p95": round(ls[int(len(ls) * 0.95)] * 1e3, 0),
                                "tokens_per_query": round((ptoks + otoks) / len(queries), 0),
                                "malformed": bad}
        print(json.dumps({"listwise_pool": n, **out["pools"][str(n)]["metrics"]},
                         ensure_ascii=False), flush=True)

    # 2. детерминированность: 20 вопросов, top-10, два прогона (temp=0)
    agree_top1, agree_full = 0, 0
    for i, q in enumerate(queries[:DET_NQ]):
        ids = dmax[i, :10]
        prompt = LIST_PROMPT.format(q=q, chunks=_chunks_str(ids, texts))
        r1, _ = ask(cl, model, prompt, key_for("kt4lr10d1:", q), cache)
        r2, _ = ask(cl, model, prompt, key_for("kt4lr10d2:", q), cache)
        o1, _ = _parse_listwise(r1, 10)
        o2, _ = _parse_listwise(r2, 10)
        agree_top1 += int(o1[0] == o2[0])
        agree_full += int(o1 == o2)
        print(f"det {i + 1}/{DET_NQ}: top1={'=' if o1[0] == o2[0] else 'x'} "
              f"full={'=' if o1 == o2 else 'x'}", flush=True)
    out["determinism"] = {"n": DET_NQ, "top1_agreement": round(agree_top1 / DET_NQ, 3),
                          "full_list_agreement": round(agree_full / DET_NQ, 3)}

    # 3. pointwise на 10 вопросах (10 чанков каждый)
    lats, pw, lw = [], {}, {}
    for i, q in enumerate(queries[:PT_NQ]):
        ids = dmax[i, :10]
        scores = []
        for j in ids:
            prompt = PT_PROMPT.format(q=q, c=texts[j][:CHUNK_CHARS])
            res, dt = ask(cl, model, prompt, key_for(f"kt4pt:", q + str(j)), cache)
            scores.append(int(re.search(r"[012]", res).group())
                          if re.search(r"[012]", res) else 0)
            lats.append(dt)
        order = sorted(range(10), key=lambda k: -scores[k])
        pw[q] = _dedup_docs(ck["doc"].iloc[ids[order][:40]].tolist())
        res, _ = ask(cl, model, LIST_PROMPT.format(q=q, chunks=_chunks_str(ids, texts)),
                     key_for("kt4lr10:", q), cache)
        o, _ = _parse_listwise(res, 10)
        lw[q] = _dedup_docs(ck["doc"].iloc[ids[o][:40]].tolist())
    sub = g[g["question"].isin(queries[:PT_NQ])]
    ls = sorted(l for l in lats if l > 0) or [0.0]
    out["pointwise"] = {"metrics": metrics(sub, pw),
                        "listwise_same10": metrics(sub, lw),
                        "llm_ms_median_pt": round(ls[len(ls) // 2] * 1e3, 0)}
    # 4. HyDE + listwise top-10 (пул по эмбеддингу HyDE-документа)
    hp = DATA / "kt4_hyde_emb.npy"
    if hp.exists():
        h_emb = np.load(hp)
        dh = (h_emb @ c_emb.T).argsort(axis=1)[:, ::-1][:, :10]
        lats, topk, bad = [], {}, 0
        for i, q in enumerate(queries):
            ids = dh[i]
            prompt = LIST_PROMPT.format(q=q, chunks=_chunks_str(ids, texts))
            res, dt = ask(cl, model, prompt, key_for("kt4lhr:", q), cache)
            order, ok = _parse_listwise(res, 10)
            bad += int(not ok)
            lats.append(dt)
            topk[q] = _dedup_docs(ck["doc"].iloc[ids[order][:40]].tolist())
        ls = sorted(l for l in lats if l > 0) or [0.0]
        out["hyde_listwise10"] = {"metrics": metrics(g, topk),
                                   "by_type_h5": {t: v["hit@5"] for t, v in
                                                  by_type_metrics(g, topk).items()},
                                   "llm_ms_median": round(ls[len(ls) // 2] * 1e3, 0),
                                   "malformed": bad}
        print(json.dumps({"hyde_listwise10": out["hyde_listwise10"]["metrics"]},
                         ensure_ascii=False), flush=True)
    save_cache(cache)
    (DATA / "kt4_llm_rerank.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved data/kt4_llm_rerank.json")


if __name__ == "__main__":
    main()
