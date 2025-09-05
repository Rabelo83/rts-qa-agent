"""
ingest.py
Crawl your site (scoped to one host), extract clean text from HTML,
split into chunks, and save them to data/chunks.jsonl

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


# ---- Settings from environment (configured in .env / render.yaml) ----
SITE_SEED_URL = os.environ.get("SITE_SEED_URL", "https://alfredorabelo.com/").strip()
SITE_ALLOWED_HOST = os.environ.get("SITE_ALLOWED_HOST", "alfredorabelo.com").strip()
CRAWL_MAX_DEPTH = int(os.environ.get("CRAWL_MAX_DEPTH", "2"))
CACHE_DIR = os.environ.get("CACHE_DIR", ".site_cache")
RATE_LIMIT = float(os.environ.get("RATE_LIMIT", "1.0"))  # seconds between requests

DATA_DIR = "data"
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")

HEADERS = {
    "User-Agent": "RTS-QA-Agent/1.0 (+https://example.com)"
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

    # Must stay within allowed host
    if parsed.hostname and parsed.hostname.lower() != SITE_ALLOWED_HOST.lower():
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
        # If robots fetch fails, be conservative: allow crawling the seed only.
        rp = RobotFileParser()
        rp.parse(["User-agent: *", "Allow: /"])
    return rp


def allowed_by_robots(rp: RobotFileParser, url: str) -> bool:
    return rp.can_fetch(HEADERS["User-Agent"], url)


def clean_html_to_text(html: str) -> Tuple[str, str]:
    """Return (title, cleaned_text) from HTML."""
    soup = BeautifulSoup(html, "lxml")

    # Remove obvious non-content elements
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "canvas"]):
        tag.decompose()
    for sel in ["header", "footer", "nav", "form", "aside"]:
        for t in soup.select(sel):
            t.decompose()

    # Try to keep main content if the site uses <main>
    main = soup.select_one("main") or soup.body or soup

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    else:
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            title = h1.get_text(strip=True)

    text = main.get_text(separator="\n", strip=True)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Simple character-based chunking with overlap."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


@dataclass
class PageDoc:
    url: str
    title: str
    text: str


# ------------------------- Crawler -------------------------

def crawl(seed_url: str, max_depth: int, rate_limit: float) -> List[PageDoc]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    seen: Set[str] = set()
    docs: List[PageDoc] = []

    rp = load_robots_txt(seed_url)

    q = queue.Queue()
    q.put((seed_url, 0))
    seen.add(norm_url(seed_url))

    while not q.empty():
        url, depth = q.get()
        if not url:
            continue
        if depth > max_depth:
            continue
        if not allowed_by_robots(rp, url):
            continue

        try:
            time.sleep(rate_limit)
            resp = requests.get(url, headers=HEADERS, timeout=15)
        except requests.RequestException:
            continue

        if resp.status_code != 200 or not is_html_response(resp):
            continue

        title, text = clean_html_to_text(resp.text)
        if text:
            docs.append(PageDoc(url=url, title=title, text=text))

        # Enqueue links if we haven't exceeded depth
        if depth < max_depth:
            soup = BeautifulSoup(resp.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = a.get("href")
                if not href:
                    continue
                if href.startswith("mailto:") or href.startswith("tel:"):
                    continue
                nu = norm_url(href)
                if not nu or nu in seen:
                    continue
                seen.add(nu)
                q.put((nu, depth + 1))

    return docs


# ------------------------- Save Chunks -------------------------

def save_chunks(docs: List[PageDoc], path: str) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            chunks = chunk_text(d.text)
            for i, ch in enumerate(chunks):
                rec = {
                    "url": d.url,
                    "title": d.title,
                    "chunk_id": f"{d.url}#chunk{i}",
                    "text": ch
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
    return count


# ------------------------- Main -------------------------

def main(build_index: bool):
    print(f"[ingest] Seed: {SITE_SEED_URL}  Host: {SITE_ALLOWED_HOST}  Depth: {CRAWL_MAX_DEPTH}")
    print(f"[ingest] Rate limit: {RATE_LIMIT}s  Output: {CHUNKS_PATH}")

    docs = crawl(SITE_SEED_URL, CRAWL_MAX_DEPTH, RATE_LIMIT)
    print(f"[ingest] Crawled pages: {len(docs)}")

    n_chunks = save_chunks(docs, CHUNKS_PATH)
    print(f"[ingest] Wrote chunks: {n_chunks} -> {CHUNKS_PATH}")

    if build_index:
        try:
            import retriever  # we will add this file next
            retriever.build_index_from_chunks(
                chunks_path=CHUNKS_PATH,
                index_path=os.path.join(DATA_DIR, "index.pkl"),
                catalog_path=os.path.join(DATA_DIR, "catalog.jsonl"),
            )
            print("[ingest] Index built successfully.")
        except ImportError:
            print("[ingest] retriever.py not found yet. Skipping index build.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl site and build chunks (and optionally index).")
    parser.add_argument("--build-index", action="store_true", help="Also build TF-IDF index (requires retriever.py).")
    args = parser.parse_args()
    main(build_index=args.build_index)
