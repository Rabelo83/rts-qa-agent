"""
timetables.py
Extracts structured timetable rows (stop → list of times) from HTML pages.

Output entries look like:
{
  "route": "1",
  "day": "weekday",           # one of: weekday|saturday|sunday|reduced|holiday|unknown
  "stop": "reitz union",
  "times": ["6:40","6:50","7:16", ...],
  "url": "https://...",
  "title": "Route 1 ...",
  "direction": "optional text if detectable"
}

Heuristics:
- Route number is taken from <title> or the first H1/H2 containing "Route X".
- Day is inferred from the nearest previous heading (H1–H4) that mentions Weekday/Saturday/Sunday/Reduced/Holiday.
- Each <table> is scanned: we treat the *first cell* of a row as the stop label
  and the remaining cells as potential times (HH:MM or H:MM, 24h or 12h without am/pm).
- If a row yields >= 3 time values and the stop cell has letters, we keep it.
"""

import re
from typing import List, Dict, Optional
from bs4 import BeautifulSoup

TIME_RE = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b")  # 6:05, 06:05, 18:45
ROUTE_RE = re.compile(r"\broute\s*([0-9]{1,3}[a-z]?)\b", re.IGNORECASE)

DAY_CANON = {
    "weekday": "weekday", "weekdays": "weekday", "week days": "weekday",
    "saturday": "saturday", "saturdays": "saturday",
    "sunday": "sunday", "sundays": "sunday",
    "reduced": "reduced", "reduced service": "reduced",
    "holiday": "holiday", "holidays": "holiday"
}
DAY_RE = re.compile(r"\b(weekday(?:s)?|week\s*days|saturdays?|sundays?|reduced(?:\s+service)?|holidays?)\b", re.IGNORECASE)

def _canon_day(text: str) -> Optional[str]:
    if not text:
        return None
    m = DAY_RE.search(text)
    if not m:
        return None
    key = m.group(1).lower().strip()
    return DAY_CANON.get(key, None)

def _extract_route(title_text: str, soup: BeautifulSoup) -> Optional[str]:
    if title_text:
        m = ROUTE_RE.search(title_text)
        if m:
            return m.group(1)
    # fallback: look at headings
    for tag in soup.find_all(["h1", "h2"]):
        t = (tag.get_text(" ", strip=True) or "")
        m = ROUTE_RE.search(t)
        if m:
            return m.group(1)
    return None

def _nearest_day_for_table(tbl) -> Optional[str]:
    # Walk backwards among previous siblings to find a heading with a day word
    prev = tbl
    for _ in range(12):  # small look-back window
        prev = prev.find_previous_sibling()
        if not prev:
            break
        if prev.name and prev.name.lower() in ("h1", "h2", "h3", "h4", "p", "div"):
            txt = prev.get_text(" ", strip=True) if hasattr(prev, "get_text") else ""
            day = _canon_day(txt)
            if day:
                return day
    return None

def _times_from_cells(cells_text: List[str]) -> List[str]:
    times: List[str] = []
    for cell in cells_text:
        for t in TIME_RE.findall(cell):
            if t not in times:
                times.append(t)
    return times

def _parse_table(tbl, route: Optional[str], url: str, page_title: str, default_day: Optional[str]) -> List[Dict]:
    records: List[Dict] = []
    rows = tbl.find_all("tr")
    if not rows or len(rows) < 2:
        return records

    # Nearest day label above this table (fallback: default_day)
    day = _nearest_day_for_table(tbl) or default_day or "unknown"

    for r in rows[1:]:
        cells = r.find_all(["td", "th"])
        if not cells or len(cells) < 2:
            continue
        # First cell presumed stop label
        stop = (cells[0].get_text(" ", strip=True) or "").strip()
        if not stop or not any(c.isalpha() for c in stop.lower()):
            continue

        rest_text = [(c.get_text(" ", strip=True) or "") for c in cells[1:]]
        times = _times_from_cells(rest_text)
        if len(times) >= 3:
            records.append({
                "route": (route or ""),
                "day": day,
                "stop": stop.lower(),
                "times": times,
                "url": url,
                "title": page_title or "",
                "direction": ""  # could be inferred later if needed
            })
    return records

def extract_timetables_from_html(html: str, url: str, title_text: str) -> List[Dict]:
    soup = BeautifulSoup(html, "lxml")
    # Route from title/headings
    route = _extract_route(title_text, soup)
    # Page-level default day if page title hints it
    default_day = _canon_day(title_text or "")

    out: List[Dict] = []
    for tbl in soup.find_all("table"):
        out.extend(_parse_table(tbl, route=route, url=url, page_title=title_text, default_day=default_day))
    return out

# --- simple fuzzy matcher helpers (used by app.py) ---

def normalize_stop(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", (s or "").lower()).strip()

def token_similarity(a: str, b: str) -> float:
    ta = set(normalize_stop(a).split())
    tb = set(normalize_stop(b).split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / max(1, min(len(ta), len(tb)))
