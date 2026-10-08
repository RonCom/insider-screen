"""Download and parse SEC litigation releases into DuckDB.

Release pages live at /enforcement-litigation/litigation-releases/lr-<number>,
including releases from before the 2024 site redesign (e.g. lr-22991 from 2011).
Numbers are walked sequentially; the start number for a date is found by bisection.

Usage:
    uv run python -m insider_screen.sec_releases --since 2016-01-01 --db data/releases.duckdb
"""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime

import duckdb
import pandas as pd
from bs4 import BeautifulSoup

from insider_screen.db import RELEASES
from insider_screen.http import PoliteClient

BASE = "https://www.sec.gov/enforcement-litigation/litigation-releases/lr-{n}"

HEADER_RE = re.compile(
    r"Litigation Release No\.\s*(\d{4,6})\s*(?:/\s*([A-Z][a-z]+\.?\s+\d{1,2},\s+\d{4}))?"
)
DATE_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4}"
)
INSIDER_RE = re.compile(
    r"insider trading|material,?\s+non-?public|non-?public information"
    r"|in advance of the .{0,80}announcement|ahead of (?:the|an) .{0,40}announce"
    r"|\btipp(?:ed|ee|ees|er|ing)\b",
    re.IGNORECASE,
)
FLAG_PATTERNS = {
    "mentions_finra": re.compile(r"Financial Industry Regulatory Authority|\bFINRA\b"),
    "mentions_detection_center": re.compile(r"Analysis and Detection Center", re.IGNORECASE),
    "mentions_market_abuse_unit": re.compile(r"Market Abuse Unit", re.IGNORECASE),
    "mentions_options": re.compile(r"\b(call|put)\s+options?\b|\boptions?\s+contracts?\b", re.IGNORECASE),
}
FOOTER_RE = re.compile(r"Last Reviewed or Updated", re.IGNORECASE)


@dataclass
class Release:
    lr_no: int
    url: str
    release_date: date | None
    respondents: str
    text: str
    is_insider_candidate: bool
    mentions_finra: bool
    mentions_detection_center: bool
    mentions_market_abuse_unit: bool
    mentions_options: bool


def _parse_date(s: str) -> date | None:
    s = re.sub(r"\s+", " ", s.replace(".", "")).strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def parse_release(html: str | bytes, lr_no: int, url: str = "") -> Release | None:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    h1 = soup.find("h1")
    respondents = h1.get_text(" ", strip=True) if h1 else ""
    root = soup.find("main") or soup.body or soup
    text = root.get_text("\n", strip=True)

    m = HEADER_RE.search(text)
    if not m:
        return None
    text = text[m.start():]
    footer = FOOTER_RE.search(text)
    if footer:
        text = text[: footer.start()].rstrip()

    release_date = _parse_date(m.group(2)) if m.group(2) else None
    if release_date is None:
        dm = DATE_RE.search(text[:300])
        release_date = _parse_date(dm.group(0)) if dm else None
    if release_date is None:
        t = soup.find("time")
        if t and t.get("datetime"):
            release_date = date.fromisoformat(t["datetime"][:10])

    flags = {k: bool(p.search(text)) for k, p in FLAG_PATTERNS.items()}
    return Release(
        lr_no=int(m.group(1)) if m.group(1) else lr_no,
        url=url or BASE.format(n=lr_no),
        release_date=release_date,
        respondents=respondents,
        text=text,
        is_insider_candidate=bool(INSIDER_RE.search(text)),
        **flags,
    )


def fetch(client: PoliteClient, n: int) -> Release | None:
    url = BASE.format(n=n)
    status, body = client.get(url)
    if status != 200:
        return None
    return parse_release(body, n, url)


def _nearest(client: PoliteClient, n: int, max_step: int = 25) -> Release | None:
    """Release at n, or the next existing number above it (numbers have gaps)."""
    for k in range(n, n + max_step):
        r = fetch(client, k)
        if r and r.release_date:
            return r
    return None


def find_start(client: PoliteClient, since: date, lo: int = 20000, hi: int = 27500) -> int:
    """Smallest release number dated on or after `since`, by bisection."""
    while lo < hi:
        mid = (lo + hi) // 2
        r = _nearest(client, mid)
        if r is None or r.release_date >= since:
            hi = mid
        else:
            lo = r.lr_no + 1
    return lo


def crawl(client: PoliteClient, start: int, stop_after_misses: int = 40) -> list[Release]:
    """Walk upward from `start` until `stop_after_misses` consecutive numbers are missing."""
    out, misses, n = [], 0, start
    while misses < stop_after_misses:
        r = fetch(client, n)
        if r is None:
            misses += 1
        else:
            misses = 0
            out.append(r)
            if len(out) % 200 == 0:
                print(f"  LR-{n}: {r.release_date}  ({len(out)} parsed)")
        n += 1
    return out


def save(releases: list[Release], db_path: str) -> None:
    df = pd.DataFrame([asdict(r) for r in releases])
    con = duckdb.connect(db_path)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.register("df", df)
    con.execute("CREATE OR REPLACE TABLE raw.sec_litigation_releases AS SELECT * FROM df")
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2016-01-01")
    ap.add_argument("--start", type=int, help="Skip bisection and start at this release number")
    ap.add_argument("--db", default=RELEASES)
    ap.add_argument("--cache", default="data/cache/sec")
    args = ap.parse_args()

    client = PoliteClient(cache_dir=args.cache)
    start = args.start or find_start(client, date.fromisoformat(args.since))
    print(f"Starting at LR-{start}")
    releases = crawl(client, start)
    save(releases, args.db)
    n_ins = sum(r.is_insider_candidate for r in releases)
    print(f"Saved {len(releases)} releases ({n_ins} insider-trading candidates) to {args.db}")


if __name__ == "__main__":
    main()
