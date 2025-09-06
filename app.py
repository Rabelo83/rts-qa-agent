import os
import re
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

# Load .env (used locally; Render uses dashboard env)
load_dotenv()

# ---- Crawl / data settings ----
SITE_SEED_URL = os.environ.get("SITE_SEED_URL", "https://alfredorabelo.com/")
SITE_ALLOWED_HOST = os.environ.get("SITE_ALLOWED_HOST", "alfredorabelo.com")
CRAWL_MAX_DEPTH = int(os.environ.get("CRAWL_MAX_DEPTH", "2"))
CACHE_DIR = os.environ.get("CACHE_DIR", ".site_cache")
PORT = int(os.environ.get("PORT", "8000"))

# Absolute paths (avoid CWD confusion)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")
CATALOG_PATH = os.path.join(DATA_DIR, "catalog.jsonl")
TIMETABLES_PATH = os.path.join(DATA_DIR, "timetables.json")

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ---------- Lazy loaders ----------
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

# Timetables cache
_tt_cache = None

def get_timetables():
    """Load timetables.json once."""
    global _tt_cache
    if _tt_cache is not None:
        return _tt_cache
    path = TIMETABLES_PATH
    if not os.path.exists(path):
        _tt_cache = []
        return _tt_cache
    import json
    with open(path, "r", encoding="utf-8") as f:
        _tt_cache = json.load(f)
    return _tt_cache

# ---------- Query helpers ----------
DAY_KEYWORDS = ["weekday", "weekdays", "saturday", "sunday", "sundays", "reduced", "holiday", "holidays"]
DAY_CANON = {
    "weekday": "weekday", "weekdays": "weekday", "week days": "weekday",
    "saturday": "saturday", "saturdays": "saturday",
    "sunday": "sunday", "sundays": "sunday",
    "reduced": "reduced", "reduced service": "reduced",
    "holiday": "holiday", "holidays": "holiday",
}
DAY_ORDER = ["weekday", "saturday", "sunday", "reduced", "holiday", "unknown"]

def canon_day(s: str | None):
    if not s:
        return None
    s = s.lower().strip()
    return DAY_CANON.get(s, s) if s else None

def extract_entities(q: str):
    """Find route number, a day label if present, and terms for light scoring."""
    ql = q.lower()
    route = None
    m = re.search(r"\b(?:route|rt|bus)\s*([0-9]{1,3}[a-z]?)\b", ql)
    if m:
        route = m.group(1)

    day = None
    for d in DAY_KEYWORDS:
        if d in ql:
            day = canon_day(d)
            break

    terms = [t for t in re.findall(r"[a-z0-9]+", ql) if len(t) > 2]
    return route, day, terms

# For fuzzy stop matching, reuse helpers from timetables.py
try:
    from timetables import normalize_stop, token_similarity
except Exception:
    # Minimal fallbacks in case the module isn't present for some reason
    def normalize_stop(s: str) -> str:
        return re.sub(r"[^a-z0-9\s]", "", (s or "").lower()).strip()
    def token_similarity(a: str, b: str) -> float:
        ta = set(normalize_stop(a).split())
        tb = set(normalize_stop(b).split())
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        return inter / max(1, min(len(ta), len(tb)))

def best_timetable_match(query: str):
    """
    If the user asked about a route + stop (+ optional day),
    return a dict: {route, stop, day, times, url, title} or None if no good match.
    """
    route, day, _terms = extract_entities(query)
    if not route:
        return None  # timetable extraction requires a route anchor

    timetables = get_timetables()
    if not timetables:
        return None

    # Narrow by route first
    candidates = [r for r in timetables if (r.get("route") or "").lower() == route.lower()]
    if not candidates:
        return None

    # If the question likely contains a stop name, we fuzzy-match by comparing
    # the whole question text to each candidate's stop field.
    # (You can enhance this by detecting quoted phrases or "at <stop>" patterns.)
    best = None
    best_score = 0.0
    for rec in candidates:
        stop_name = rec.get("stop") or ""
        score = token_similarity(query, stop_name)
        # Light bonus if the stop_name appears verbatim in the question
        if normalize_stop(stop_name) and normalize_stop(stop_name) in normalize_stop(query):
            score += 0.25
        if score > best_score:
            best_score = score
            best = rec

    # Require a reasonable similarity to avoid bad matches
    if not best or best_score < 0.40:
        return None

    # Day selection: if question specifies a day, prefer it; otherwise prefer weekday > saturday > sunday > reduced > holiday
    best_stop = best.get("stop")
    stop_group = [r for r in candidates if (r.get("stop") or "") == best_stop]

    chosen = None
    if day:
        for r in stop_group:
            if canon_day(r.get("day")) == day:
                chosen = r
                break

    if not chosen:
        # pick the best available by our DAY_ORDER preference
        for d in DAY_ORDER:
            for r in stop_group:
                if canon_day(r.get("day")) == d:
                    chosen = r
                    break
            if chosen:
                break

    return chosen

# ---------- Ranking + answer (fallback to snippets) ----------
def rank_hits(query: str, hits, k: int = 3):
    route, day, terms = extract_entities(query)
    ranked = []
    for h in hits:
        score = float(h.get("score", 0.0))
        textlow  = (h.get("text") or "").lower()
        titlelow = (h.get("title") or "").lower()
        urllow   = (h.get("url") or "").lower()
        if route and (route in textlow or route in titlelow or route in urllow):
            score += 0.20
        if day and (day in textlow or day in titlelow):
            score += 0.15
        term_hits = sum(1 for t in terms if t in textlow)
        score += 0.02 * term_hits
        ranked.append((score, h))
    ranked.sort(key=lambda x: x[0], reverse=True)
    uniq, out = set(), []
    for sc, h in ranked:
        url = h.get("url")
        if not url or url in uniq:
            continue
        uniq.add(url)
        hh = dict(h); hh["score"] = sc
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
    has_tt = os.path.exists(TIMETABLES_PATH)
    return {
        "ok": True,
        "seed": SITE_SEED_URL,
        "host": SITE_ALLOWED_HOST,
        "depth": CRAWL_MAX_DEPTH,
        "has_index": has_index,
        "has_timetables": has_tt,
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

    # 1) Timetable path (preferred if a good route+stop match is found)
    match = best_timetable_match(q)
    if match and (match.get("times") or []):
        times = ", ".join(match["times"])
        day = canon_day(match.get("day")) or "unknown"
        route = (match.get("route") or "").upper()
        stop = (match.get("stop") or "").title()
        title = (match.get("title") or "(Untitled)").strip()
        url = match.get("url")

        answer = (
            f"**Route {route} — {stop} — {day.title()} times**\n"
            f"{times}"
        )
        sources = [{"title": title, "url": url, "score": 1.0}]
        return jsonify({"answer": answer, "sources": sources}), 200

    # 2) Fallback: snippet search from TF-IDF
    r = get_retriever()
    if r is None:
        msg = "I don't have a search index yet. Please run: python ingest.py --build-index"
        if _load_error:
            msg += f"\n(Index load error: { _load_error })"
        return jsonify({"answer": msg, "sources": []}), 200

    raw_hits = r.search(q, k=8)
    top = rank_hits(q, raw_hits, k=3)
    if not top:
        return jsonify({
            "answer": "I couldn’t find anything for that. Try adding a route/stop/day (e.g., 'Route 1 weekday Reitz Union').",
            "sources": []
        }), 200

    answer = build_answer(top, q)
    sources = nice_sources(top)
    return jsonify({"answer": answer, "sources": sources}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
