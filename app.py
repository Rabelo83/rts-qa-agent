import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

# Load .env (used locally; Render gets env from render.yaml)
load_dotenv()

# Crawl / data settings (used for info + index file paths)
SITE_SEED_URL = os.environ.get("SITE_SEED_URL", "https://alfredorabelo.com/")
SITE_ALLOWED_HOST = os.environ.get("SITE_ALLOWED_HOST", "alfredorabelo.com")
CRAWL_MAX_DEPTH = int(os.environ.get("CRAWL_MAX_DEPTH", "2"))
CACHE_DIR = os.environ.get("CACHE_DIR", ".site_cache")
PORT = int(os.environ.get("PORT", "8000"))

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")
CATALOG_PATH = os.path.join(DATA_DIR, "catalog.jsonl")

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ---------- Load retriever lazily ----------
_retriever = None
_load_error = None

def get_retriever():
    """Lazy-load the TF-IDF retriever when the first question arrives."""
    global _retriever, _load_error
    if _retriever is not None:
        return _retriever
    try:
        from retriever import load_index
        _retriever = load_index(index_path=INDEX_PATH, catalog_path=CATALOG_PATH)
    except Exception as e:
        _load_error = str(e)
        _retriever = None
    return _retriever

# ---------- Routes ----------
@app.get("/health")
def health():
    has_index = os.path.exists(INDEX_PATH) and os.path.exists(CATALOG_PATH)
    return {
        "ok": True,
        "seed": SITE_SEED_URL,
        "host": SITE_ALLOWED_HOST,
        "depth": CRAWL_MAX_DEPTH,
        "has_index": has_index,
        "load_error": _load_error,
    }

@app.get("/")
def home():
    return send_from_directory("static", "chat.html")

@app.post("/api/ask")
def api_ask():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"answer": "Please enter a question.", "sources": []}), 200

    r = get_retriever()
    if r is None:
        # Index missing or failed to load
        msg = (
            "I don't have a search index yet. "
            "Please run: python ingest.py --build-index"
        )
        if _load_error:
            msg += f"\n(Index load error: { _load_error })"
        return jsonify({"answer": msg, "sources": []}), 200

    hits = r.search(q, k=5)
    if not hits:
        return jsonify({
            "answer": "I couldn’t find anything for that. Try adding a route/stop/day (e.g., 'Route 1 weekday Reitz Union').",
            "sources": []
        }), 200

    # Build a concise answer from the top hit's snippet
    top = hits[0]
    answer = top["snippet"].strip()
    if not answer:
        # fallback to trimmed text if snippet is empty
        answer = (top.get("text") or "")[:320]

    # Include up to 3 sources (title + URL + score)
    sources = []
    for h in hits[:3]:
        sources.append({
            "title": h.get("title") or "",
            "url": h.get("url"),
            "score": round(h.get("score", 0.0), 4),
        })

    # Add a lightweight prefix to make it read nicely
    pretty_answer = (
        "Here’s what I found:\n\n"
        f"{answer}\n"
    )

    return jsonify({
        "answer": pretty_answer,
        "sources": sources
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
