"""
ingest.py
Crawl your site (scoped to one host), extract clean text from HTML,
split into chunks, and save them to data/chunks.jsonl

NEW: Extract timetable rows (stop -> list of times) into data/timetables.json

Optional: if retriever.py is available and you pass --build-index,
this script will also build the TF-IDF index on disk.
"""

import os
import re
import time
import json
import queue
import argparse
import urllib.parse as urlparse
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple

import requests
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
from timetables import extract_timetables_from_html  # NEW

# ---- Settings from environment (configured in .env / Render dashboard) ----
SITE_SEED_URL = os.environ.get("SITE_SEED_URL", "https://alfredorabelo.com/").strip()
SITE_ALLOWED_HOST = os.environ.get("SITE_ALLOWED_HOST", "alfredorabelo.com").strip()
CRAWL_MAX_DEPTH = int(os.environ.get("CRAWL_MAX_DEPTH", "2"))
CACHE_DIR = os.environ.get("CACHE_DIR", ".site_cache")
RATE_LIMIT = float(os.environ.get("RATE_LIMIT", "1.0"))  # seconds between requests
RESPECT_ROBOTS = os.environ.get("RESPECT_ROBOTS", "true").lower() != "false"  # set to false to diagnose

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")
TIMETABLES_PATH = os.path.join(DATA_DIR, "timetables.json")  # NEW

# Accept both bare and www host variants
_allowed_hosts = {SITE_ALLOWED_HOST.lower()}
if SITE_ALLOWED_HOST.lower().startswith("www."):
    _allowed_hosts.add(SITE_ALLOWED_HOST.lower()[4:])
else:
    _allowed_hosts.add("www." + SITE_ALLOWED_HOST.lower())

HEADERS = {
    # friendlier UA; some sites block generic bots
    "User-Agent": "Mozilla/5.0 (compatible; RTS-QA-Agent/1.0)"
}

# ------------------------- Helpers -------------------------

def norm_url(u: str) -> str:
    """Normalize a URL: strip fragments, keep scheme/host/path; drop querystrings for now."""
    if not u:
        return ""
    parsed = urlparse.urlparse(u)
    if not parsed.scheme:
        # relative URL -> make absolute based on seed
        u = urlparse.urljoin(SITE_SEED_URL, u)
        parsed = urlparse.urlparse(u)

    host = (parsed.hostname or "").lower()
    if host not in _allowed_hosts:
        return ""

    # Remove fragments and query strings for stability
    cleaned = parsed._replace(query="", fragment="").geturl()
    return cleaned

def is_html_response(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    return "text/html" in ctype or "application/xhtml+xml" in ctype

def load_robots_txt(seed_url: str) -> RobotFileParser:
    parsed = urlparse.urlparse(seed_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        rp = RobotFileParser()
        rp.parse(["User-agent: *", "Allow: /"])
    return rp

def allowed_by_robots(rp: RobotFileParser, url: str) -> bool:
    if not RESPECT_ROBOTS:
        return True
    return rp.can_fetch(HEADERS["User-Agent"], url)

def clean_html_to_text(html: str) -> Tuple[str, str]:
    """Return (title, cleaned_text) from HTML."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "iframe", "svg", "canvas"]):
        tag.decompose()
    for sel in ["header", "footer", "nav", "form", "aside"]:
        for t in soup.select(sel):
            t.decompose()

    main = soup.select_one("main") or soup.body or soup
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    else:
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            title = h1.get_text(strip=True)

    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text

@dataclass
class PageDoc:
    url: str
    title: str
    text: str

# ------------------------- Crawler -------------------------

def crawl(seed_url: str, max_depth: int, rate_limit: float) -> Tuple[List[PageDoc], List[Dict]]:
    """Return (docs, timetable_records)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    seen: Set[str] = set()
    docs: List[PageDoc] = []
    tt_records: List[Dict] = []  # NEW

    rp = load_robots_txt(seed_url)

    q = queue.Queue()
    seed_norm = norm_url(seed_url)
    if seed_norm:
        q.put((seed_norm, 0))
        seen.add(seed_norm)
    else:
        print(f"[ingest] WARNING: seed URL rejected by host filter: {seed_url}")

    while not q.empty():
        url, depth = q.get()
        if not url:
            continue
        if depth > max_depth:
            continue
        if not allowed_by_robots(rp, url):
            print(f"[ingest] robots disallowed: {url}")
            continue

        try:
            time.sleep(rate_limit)
            resp = requests.get(url, headers=HEADERS, timeout=15)
        except requests.RequestException as e:
            print(f"[ingest] request error: {url} -> {e}")
            continue

        if resp.status_code != 200 or not is_html_response(resp):
            print(f"[ingest] skipped non-HTML or status {resp.status_code}: {url}")
            continue

        # Clean text for search index
        title, text = clean_html_to_text(resp.text)
        if text:
            docs.append(PageDoc(url=url, title=title, text=text))

        # Extract timetable rows from the raw HTML (NEW)
        try:
            rows = extract_timetables_from_html(resp.text, url=url, title_text=title)
            if rows:
                tt_records.extend(rows)
        except Exception as e:
            # Don't let parsing issues stop the crawl
            print(f"[timetable] parse error on {url}: {e}")

        # Enqueue links if we haven't exceeded depth
        if depth < max_depth:
            soup = BeautifulSoup(resp.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = a.get("href")
                if not href:
                    continue
                if href.startswith(("mailto:", "tel:")):
                    continue
                nu = norm_url(href)
                if not nu or nu in seen:
                    continue
                seen.add(nu)
                q.put((nu, depth + 1))

    return docs, tt_records

# ------------------------- Save Chunks & Timetables -------------------------

def save_chunks(docs: List[PageDoc], path: str) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            for i, ch in enumerate(_chunk_text(d.text)):
                rec = {
                    "url": d.url,
                    "title": d.title,
                    "chunk_id": f"{d.url}#chunk{i}",
                    "text": ch
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
    return count

def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    out, n, i = [], len(text), 0
    while i < n:
        j = min(i + chunk_size, n)
        out.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return out

def _dedupe_timetables(records: List[Dict]) -> List[Dict]:
    """Merge times for identical (route, day, stop)."""
    merged: Dict[Tuple[str, str, str], Dict] = {}
    for r in records:
        key = (r.get("route", ""), r.get("day", ""), (r.get("stop") or "").strip().lower())
        if key not in merged:
            merged[key] = {
                "route": key[0], "day": key[1], "stop": key[2],
                "times": [], "url": r.get("url"), "title": r.get("title"), "direction": r.get("direction", "")
            }
        # merge times uniquely
        for t in r.get("times", []):
            if t not in merged[key]["times"]:
                merged[key]["times"].append(t)
    # sort times lexicographically (works for HH:MM)
    for v in merged.values():
        v["times"].sort()
    return list(merged.values())

def save_timetables(records: List[Dict], path: str) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean = _dedupe_timetables(records)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    return len(clean)

# ------------------------- Main -------------------------

def main(build_index: bool):
    print(f"[ingest] Seed: {SITE_SEED_URL}  Host: {SITE_ALLOWED_HOST}  Depth: {CRAWL_MAX_DEPTH}")
    print(f"[ingest] Respect robots: {RESPECT_ROBOTS}  Rate limit: {RATE_LIMIT}s")
    print(f"[ingest] Output: {CHUNKS_PATH}")
    docs, tt = crawl(SITE_SEED_URL, CRAWL_MAX_DEPTH, RATE_LIMIT)
    print(f"[ingest] Crawled pages: {len(docs)}")
    n_chunks = save_chunks(docs, CHUNKS_PATH)
    print(f"[ingest] Wrote chunks: {n_chunks} -> {CHUNKS_PATH}")

    n_tt = save_timetables(tt, TIMETABLES_PATH)
    print(f"[ingest] Timetable rows saved: {n_tt} -> {TIMETABLES_PATH}")

    if build_index:
        if n_chunks == 0:
            print("[ingest] WARNING: 0 chunks written; skipping index build.")
        else:
            try:
                import retriever  # already in your repo
                retriever.build_index_from_chunks(
                    chunks_path=CHUNKS_PATH,
                    index_path=os.path.join(DATA_DIR, "index.pkl"),
                    catalog_path=os.path.join(DATA_DIR, "catalog.jsonl"),
                )
                print("[ingest] Index built successfully.")
            except ImportError:
                print("[ingest] retriever.py not found yet. Skipping index build.")
            except Exception as e:
                print(f"[ingest] ERROR building index: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl site and build chunks (and optionally index).")
    parser.add_argument("--build-index", action="store_true", help="Also build TF-IDF index (requires retriever.py).")
    args = parser.parse_args()
    main(build_index=args.build_index)
