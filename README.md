# RTS Q&A Agent (Step 2 – Project Skeleton)

A tiny Flask web app + chat UI that will answer questions **from your website**.  
This step gives you a Render-ready skeleton with a working `/api/ask` stub.  
**No LLM yet** — Step 3 will add the crawler + TF-IDF search index.

---

## ✨ What’s included (Step 2)
- `app.py` — Flask app with `/`, `/health`, and a stub `/api/ask`
- `static/chat.html` — minimal chat UI (vanilla JS)
- `requirements.txt` — Python dependencies
- `render.yaml` — Render deploy config (free tier friendly)
- `.env.example` — environment variables template (used locally)
- `.gitignore` — ignores caches, env files, and future index artifacts

Planned for **Step 3**:
- `ingest.py` — crawl site → clean HTML → chunk → save data
- `retriever.py` — TF-IDF index + retrieval for answers

---

## 📁 Project structure
