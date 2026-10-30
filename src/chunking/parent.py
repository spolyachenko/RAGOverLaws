"""ParentDocumentRetriever: дерево чанков документ -> родители -> дети.

родители — TokenChunker(parent), в каждом родителе дети — TokenChunker(child);
индексируются дети, топ-k детей мапится на родителей/документы (см. retrieval.dense).
"""
from chonkie import TokenChunker


def build_parents(df, p: int, c: int) -> list[dict]:
    """doc-корпус (title, text) -> [(doc, pidx, parent_text, [child_texts])]"""
    tp, tc = TokenChunker(chunk_size=p), TokenChunker(chunk_size=c)
    out = []
    for _, d in df.iterrows():
        for i, pc in enumerate(tp.chunk(d["text"])):
            children = [cc.text for cc in tc.chunk(pc.text)] or [pc.text]
            out.append({"doc": d["title"], "pidx": i, "p": pc.text, "c": children})
    return out
