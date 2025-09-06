import os
import re
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

# Load .env (used locally; Render gets env from dashboard)
load_dotenv()

# ---- Crawl / data settings (for info + paths) ----
SITE_SEED_URL = os.environ.get("SITE_SEED_URL", "https://alfredorabelo.com/")
SITE_ALLOWED_HOST = os.environ.get("SITE_ALLOWED_HOST", "alfredorabelo.com")
CRAWL_MAX_DEPTH = int(os.environ.get("CRAWL_MAX_DEPTH", "2"))
CACHE_DIR = os.environ.get("CACHE_DIR", ".site_cache")
PORT = int(os.environ.get("PORT", "8000"))

# Absolute paths so Gunicorn / CWD don't confuse things
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")
CATALOG_PATH = os.path.join(DATA_DIR, "catalog.jsonl")

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ---------- Lazy index loader ----------
_retriever = None
_load_error = None
def get_retriever():
    """Lazy-load the TF-IDF retriever on first question."""
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

# ---------- Query helpers (simple, deterministic) ----------
DAY_KEYWORDS = ["weekday", "weekdays", "saturday", "sundays", "sunday", "reduced", "holiday", "holidays"]

def extract_entities(q: str):
    """Find route number, day words, and keyword tokens."""
    ql = q.lower()
    route = None
    m = re.search(r"\b(?:route|rt|bus)\s*([0-9]{1,3}[a-z]?)\b", ql)
    if m:
        route = m.group(1)

    day = None
    for d in DAY_KEYWORDS:
        if d in ql:
            day = d
            break

    # simple keyword list for light scoring
    terms = [t for t in re.findall(r"[a-z0-9]+", ql) if len(t) > 2]
    return route, day, terms

def rank_hits(query: str, hits, k: int = 3):
    """Boost results that match route/day/terms; return up to k unique-URL hits."""
    route, day, terms = extract_entities(query)
    ranked = []
    for h in hits:
        score = float(h.get("score", 0.0))
        textlow  = (h.get("text") or "").lower()
        titlelow = (h.get("title") or "").lower()
        urllow   = (h.get("url") or "").lower()

        # Prefer matches that clearly reference the asked route/day
        if route and (route in textlow or route in titlelow or route in urllow):
            score += 0.20
        if day and (day in textlow or day in titlelow):
            score += 0.15

        # Light keyword coverage bonus
        term_hits = sum(1 for t in terms if t in textlow)
        score += 0.02 * term_hits

        ranked.append((score, h))

    # sort by adjusted score desc
    ranked.sort(key=lambda x: x[0], reverse=True)

    # keep top unique URLs
    uniq, out = set(), []
    for sc, h in ranked:
        url = h.get("url")
        if not url or url in uniq:
            continue
        uniq.add(url)
        hh = dict(h)
        hh["score"] = sc
        out.append(hh)
        if len(out) >= k:
            break
    return out

def nice_sources(hits):
    out = []
    for h in hits:
        out.append({
            "title": (h.get("title") or "").strip()[:120] or "(Untitled)",
            "url": h.get("url"),
            "score": round(float(h.get("score", 0.0)), 4),
        })
    return out

def build_answer(hits, query):
    """Concise, readable answer built from top snippets."""
    parts = []
    for h in hits:
        title = (h.get("title") or "").strip()[:120] or "(Untitled)"
        snippet = (h.get("snippet") or "").strip()
        if not snippet:
            snippet = (h.get("text") or "").strip()[:320]
        parts.append(f"• {title}\n  {snippet}")
    intro = "Here’s what I found (most relevant first):\n\n"
    return intro + "\n\n".join(parts)

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
        msg = "I don't have a search index yet. Please run: python ingest.py --build-index"
        if _load_error:
            msg += f"\n(Index load error: { _load_error })"
        return jsonify({"answer": msg, "sources": []}), 200

    raw_hits = r.search(q, k=8)  # fetch more then re-rank
    top = rank_hits(q, raw_hits, k=3)

    if not top:
        return jsonify({
            "answer": "I couldn’t find anything for that. Try adding a route/stop/day (e.g., 'Route 1 weekday Reitz Union').",
            "sources": []
        }), 200

    answer = build_answer(top, q)
    sources = nice_sources(top)

    return jsonify({
        "answer": answer,
        "sources": sources
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
