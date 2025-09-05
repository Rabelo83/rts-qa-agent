"""
retriever.py
Builds and serves a simple TF-IDF search index over your site's chunks (from ingest.py).

Outputs:
- data/index.pkl         (pickled dict: {"vectorizer", "matrix"})
- data/catalog.jsonl     (one JSON per line with: idx, url, title, chunk_id, text)

APIs:
- build_index_from_chunks(chunks_path, index_path, catalog_path) -> dict (stats)
- load_index(index_path, catalog_path) -> Retriever
- Retriever.search(query, k=5) -> List[dict] with score, url, title, text, chunk_id, snippet
"""

import os
import re
import json
import pickle
from dataclasses import dataclass
from typing import List, Dict, Iterable, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = "data"
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")
INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")
CATALOG_PATH = os.path.join(DATA_DIR, "catalog.jsonl")


# ------------------------- Utilities -------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _write_jsonl(path: str, records: Iterable[Dict]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n

def _short_snippet(text: str, query: str, max_chars: int = 320) -> str:
    """Pick 1–2 sentences that best match the query terms, for display."""
    sents = _SENT_SPLIT.split(text)
    if not sents:
        return text[:max_chars]
    q_terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    if not q_terms:
        return " ".join(sents[:2])[:max_chars]
    scores = []
    for s in sents:
        low = s.lower()
        score = sum(low.count(t) for t in q_terms)
        scores.append((score, s))
    scores.sort(key=lambda x: x[0], reverse=True)
    best = " ".join(s for _, s in scores[:2]).strip()
    return (best or " ".join(sents[:2]))[:max_chars]


# ------------------------- Core Classes -------------------------

@dataclass
class SearchResult:
    score: float
    url: str
    title: str
    chunk_id: str
    text: str
    snippet: str

class Retriever:
    def __init__(self, vectorizer: TfidfVectorizer, matrix, catalog: List[Dict]):
        self.vectorizer = vectorizer
        self.matrix = matrix  # csr_matrix (n_chunks x vocab)
        self.catalog = catalog

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Return top-k results with metadata + snippet."""
        if not query or not query.strip():
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]  # shape: (n_chunks,)
        if k <= 0:
            k = 5
        # Top-k indices
        top_idx = np.argpartition(-sims, kth=min(k, len(sims)-1))[:k]
        # Sort by score desc
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        results: List[Dict] = []
        for i in top_idx:
            meta = self.catalog[i]
            results.append({
                "score": float(sims[i]),
                "url": meta.get("url"),
                "title": meta.get("title"),
                "chunk_id": meta.get("chunk_id"),
                "text": meta.get("text"),
                "snippet": _short_snippet(meta.get("text", ""), query),
            })
        return results


# ------------------------- Build / Load -------------------------

def build_index_from_chunks(
    chunks_path: str = CHUNKS_PATH,
    index_path: str = INDEX_PATH,
    catalog_path: str = CATALOG_PATH,
    max_features: Optional[int] = 50000,
) -> Dict:
    """
    Build TF-IDF index from chunks.jsonl and persist index + catalog.
    Returns stats dict.
    """
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    # Load chunks
    texts: List[str] = []
    catalog: List[Dict] = []
    for idx, rec in enumerate(_read_jsonl(chunks_path)):
        # Keep text + basic metadata
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        texts.append(text)
        catalog.append({
            "idx": idx,  # original line index (not strictly needed)
            "url": rec.get("url"),
            "title": rec.get("title"),
            "chunk_id": rec.get("chunk_id"),
            "text": text,
        })

    if not texts:
        raise ValueError("No text chunks loaded; cannot build index.")

    # Vectorize
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)  # csr_matrix

    # Persist index
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump(
            {
                "vectorizer": vectorizer,
                "shape": matrix.shape,
                # Store in compressed sparse format to keep file smaller
                "matrix": matrix,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Persist catalog
    _write_jsonl(catalog_path, catalog)

    return {
        "chunks": len(texts),
        "vocab": len(vectorizer.vocabulary_),
        "matrix_shape": list(matrix.shape),
        "index_path": index_path,
        "catalog_path": catalog_path,
    }


def load_index(
    index_path: str = INDEX_PATH,
    catalog_path: str = CATALOG_PATH,
) -> Retriever:
    """Load TF-IDF index and catalog into a Retriever instance."""
    if not (os.path.exists(index_path) and os.path.exists(catalog_path)):
        raise FileNotFoundError(
            f"Missing index or catalog. Expected:\n - {index_path}\n - {catalog_path}"
        )

    with open(index_path, "rb") as f:
        blob = pickle.load(f)

    vectorizer: TfidfVectorizer = blob["vectorizer"]
    matrix = blob["matrix"]

    catalog: List[Dict] = []
    for rec in _read_jsonl(catalog_path):
        catalog.append(rec)

    return Retriever(vectorizer=vectorizer, matrix=matrix, catalog=catalog)


# ------------------------- CLI (optional for quick tests) -------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TF-IDF retriever over site chunks.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build index from chunks.jsonl")
    p_build.add_argument("--chunks", default=CHUNKS_PATH)
    p_build.add_argument("--index", default=INDEX_PATH)
    p_build.add_argument("--catalog", default=CATALOG_PATH)

    p_query = sub.add_parser("query", help="Query the existing index")
    p_query.add_argument("text", help="search text")
    p_query.add_argument("--k", type=int, default=5)
    p_query.add_argument("--index", default=INDEX_PATH)
    p_query.add_argument("--catalog", default=CATALOG_PATH)

    args = parser.parse_args()

    if args.cmd == "build":
        stats = build_index_from_chunks(args.chunks, args.index, args.catalog)
        print("[build] stats:", stats)

    elif args.cmd == "query":
        r = load_index(args.index, args.catalog)
        hits = r.search(args.text, k=args.k)
        for h in hits:
            print(f"- {h['score']:.4f}  {h['title'] or ''}  {h['url']}")
            print(f"  snippet: {h['snippet']}\n")
