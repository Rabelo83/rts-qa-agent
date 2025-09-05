import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

# Load .env (used later on Render / local)
load_dotenv()

# Basic settings (Step 1 scope; used in Step 3)
SITE_SEED_URL = os.environ.get("SITE_SEED_URL", "https://alfredorabelo.com/")
SITE_ALLOWED_HOST = os.environ.get("SITE_ALLOWED_HOST", "alfredorabelo.com")
CRAWL_MAX_DEPTH = int(os.environ.get("CRAWL_MAX_DEPTH", "2"))
CACHE_DIR = os.environ.get("CACHE_DIR", ".site_cache")
PORT = int(os.environ.get("PORT", "8000"))

app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.get("/health")
def health():
    """Simple health check (Render uses this)."""
    return {
        "ok": True,
        "seed": SITE_SEED_URL,
        "host": SITE_ALLOWED_HOST,
        "depth": CRAWL_MAX_DEPTH,
    }


@app.get("/")
def home():
    """Serve the minimal chat UI (we'll add the file next)."""
    return send_from_directory("static", "chat.html")


@app.post("/api/ask")
def api_ask():
    """
    STEP 2: Stub endpoint.
    In Step 3 we'll wire this to a retriever that searches your site's content.
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"answer": "Please enter a question.", "sources": []}), 200

    # Placeholder until we add the crawler/index (Step 3)
    return jsonify({
        "answer": (
            "MVP is running! I don't have an index yet.\n"
            "Next: we'll crawl your site, build a TF-IDF index, and answer from relevant pages."
        ),
        "echo": question,
        "nextSteps": [
            "Add ingest.py and retrieval in Step 3",
            "Build TF-IDF index from your site",
            "Return answers + source links"
        ],
        "sources": []
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
