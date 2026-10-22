"""Выгрузка всех страниц вики TBOI:Rebirth → data/corpus.jsonl.

Запуск: python src/fetch_wiki.py
"""
import asyncio
import json
import urllib.parse

import aiohttp

from common import CORPUS_PATH, WIKI_API

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


def run():
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
        # служебные redirect-страницы и пустые отбрасываем
        return [r for r in rows if not r["text"].startswith("#REDIRECT") and len(r["text"]) >= 40]

    rows = asyncio.run(_run())
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("written:", len(rows), "->", CORPUS_PATH)


if __name__ == "__main__":
    run()
