"""Выгрузка всей вики TBOI:Rebirth (MediaWiki API) → data/corpus.jsonl.

Все страницы namespace 0: list=allpages + revisions по 50 за запрос, 20 параллельных.
Redirect-страницы (#REDIRECT) и страницы <40 символов отбрасываются как служебные.

Запуск: python src/corpus/fetch.py
"""
import asyncio
import json
import sys
import urllib.parse
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import CORPUS_PATH  # noqa: E402

WIKI_API = "https://bindingofisaacrebirth.wiki.gg/api.php"
HDRS = {"User-Agent": "gt1.0 (research)"}


async def _fetch_json(s, url, tries=4):
    for a in range(tries):
        try:
            async with s.get(url) as r:
                if r.status == 429:
                    await asyncio.sleep(2 + a * 2)
                    continue
                d = await r.json()
                if "query" in d:
                    return d
                await asyncio.sleep(1 + a)
        except Exception:
            await asyncio.sleep(1 + a)
    raise RuntimeError("gave up: " + url[:100])


def _slot_text(slot):
    return slot.get("content", slot.get("*", "")) or ""  # MediaWiki: "content" или "*"


async def _list_all_pages(s):
    titles, cont = [], None
    while True:
        url = f"{WIKI_API}?action=query&list=allpages&aplimit=500&apnamespace=0&format=json"
        if cont:
            url += "&apcontinue=" + urllib.parse.quote(cont)
        d = await _fetch_json(s, url)
        titles += [p["title"] for p in d["query"]["allpages"]]
        cont = d.get("continue", {}).get("apcontinue")
        if not cont:
            return titles
        await asyncio.sleep(0.2)


async def _content_batch(s, chunk):
    url = (f"{WIKI_API}?action=query&format=json&prop=revisions&rvprop=content&rvslots=main"
           f"&titles=" + "|".join(urllib.parse.quote(t) for t in chunk))
    d = await _fetch_json(s, url)
    out = {}
    for p in d["query"]["pages"].values():
        try:
            out[p["title"]] = _slot_text(p["revisions"][0]["slots"]["main"])
        except (KeyError, IndexError):
            out[p["title"]] = ""
    return out


def fetch_corpus():
    """Все страницы вики -> data/corpus.jsonl (id, title, text, source)."""
    async def _run():
        async with aiohttp.ClientSession(headers=HDRS, timeout=aiohttp.ClientTimeout(total=120)) as s:
            titles = await _list_all_pages(s)
            print("pages:", len(titles), flush=True)
            chunks = [titles[i:i + 50] for i in range(0, len(titles), 50)]
            results = []
            for k in range(0, len(chunks), 20):  # 20 параллельных
                results.extend(await asyncio.gather(*(_content_batch(s, c) for c in chunks[k:k + 20])))
                print(f"  {min(k + 20, len(chunks))}/{len(chunks)}", flush=True)
        content = {}
        for r in results:
            content.update(r)
        rows = [{"id": f"wiki://{t}", "title": t, "text": content.get(t, ""),
                 "source": "bindingofisaacrebirth.wiki.gg"} for t in titles]
        return [r for r in rows if not r["text"].startswith("#REDIRECT") and len(r["text"]) >= 40]

    rows = asyncio.run(_run())
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("written:", len(rows), "->", CORPUS_PATH)


async def _dates_batch(s, chunk):
    url = (f"{WIKI_API}?action=query&format=json&prop=revisions&rvprop=timestamp"
           f"&titles=" + "|".join(urllib.parse.quote(t) for t in chunk))
    d = await _fetch_json(s, url)
    out = {}
    for p in d["query"]["pages"].values():
        try:
            out[p["title"]] = p["revisions"][0]["timestamp"]
        except (KeyError, IndexError):
            out[p["title"]] = ""
    return out


def fetch_page_dates(out_path: Path | None = None) -> dict:
    """Дата последней ревизии каждой страницы -> {title: 'YYYY-MM-DD HH:MM:SS'} (кэш на диск)."""
    from utils import DATA
    out_path = out_path or (DATA / "page_dates.json")
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    async def _run():
        async with aiohttp.ClientSession(headers=HDRS, timeout=aiohttp.ClientTimeout(total=120)) as s:
            titles = await _list_all_pages(s)
            chunks = [titles[i:i + 50] for i in range(0, len(titles), 50)]
            results = []
            for k in range(0, len(chunks), 20):
                results.extend(await asyncio.gather(*(_dates_batch(s, c) for c in chunks[k:k + 20])))
        dates = {}
        for r in results:
            dates.update(r)
        return {t: d for t, d in dates.items() if d}
    dates = asyncio.run(_run())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dates, ensure_ascii=False), encoding="utf-8")
    print("page dates:", len(dates), "->", out_path)
    return dates


if __name__ == "__main__":
    if "--dates" in sys.argv:
        fetch_page_dates()
    else:
        fetch_corpus()
