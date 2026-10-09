"""Ticker-to-CIK map with dates, from Massive's (formerly Polygon) reference data, free plan.

Prices (Alpaca, FINRA) are keyed by the ticker as traded on each date; events are keyed by CIK. Tickers
get reused after a delisting and change when companies rename, so the map is dated:

1. download: every US stock ticker Massive lists, active and delisted (/v3/reference/tickers, market
   stocks), into raw.massive_tickers. About 30-50 pages of 1,000 at 5 requests a minute on the free plan
   (10-15 minutes). Each page is saved as it arrives; a rerun resumes from the last page.
2. build: ref.ticker_cik, one row per (ticker, holder) with valid_from and valid_to. A ticker held by
   several companies over time is split at each holder's delisting date: a holder's range ends on its
   delisting date and the next holder's starts the day after. Ranges are approximate at the edges
   (Massive gives delisting dates, not listing dates), which is why lookups go by day 0 inside a range.
3. coverage: share of events in data/edgar.duckdb with a ticker valid on day 0, by event type and year.

Settings (.env): MASSIVE_API_KEY; MASSIVE_API_URL (default https://api.massive.com; the older
https://api.polygon.io is tried if that host doesn't answer).

Usage:
    uv run python -m insider_screen.tickers download
    uv run python -m insider_screen.tickers build
    uv run python -m insider_screen.tickers coverage
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import duckdb
import httpx
import pandas as pd

from insider_screen.db import EDGAR, REFERENCE

BASE_URLS = [os.environ.get("MASSIVE_API_URL", "https://api.massive.com"), "https://api.polygon.io"]
PER_MINUTE = 5  # free plan
FIELDS = ["ticker", "name", "cik", "type", "active", "primary_exchange", "composite_figi", "share_class_figi",
          "delisted_utc", "last_updated_utc"]
# security types kept in the map: common stock, ADRs, and the like; funds, warrants, units, rights are left out
STOCK_TYPES = ("CS", "ADRC", "ADRP", "ADRS", "OS", "NYRS", "GDR")


class Massive:
    def __init__(self, key: str, transport: httpx.BaseTransport | None = None, per_minute: float = PER_MINUTE,
                 base_urls: list[str] | None = None) -> None:
        self.client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=60, transport=transport)
        self.key = key
        self.min_interval = 60.0 / per_minute
        self._last = 0.0
        self.base_urls = base_urls or BASE_URLS
        self.base: str | None = None

    def _wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()

    def get(self, path_or_url: str, params: dict | None = None) -> dict:
        """GET a path (on the first base URL that answers) or a full next_url. Waits out rate limits."""
        if path_or_url.startswith("http"):
            urls = [path_or_url]
        elif self.base:
            urls = [self.base + path_or_url]
        else:
            urls = [b + path_or_url for b in self.base_urls]
        last_err: Exception | None = None
        for url in urls:
            for attempt in range(6):
                self._wait()
                try:
                    r = self.client.get(url, params=params)
                except httpx.TransportError as err:  # host down or unknown: try the next base URL
                    last_err = err
                    break
                if r.status_code == 429:
                    time.sleep(min(60, 15 * (attempt + 1)))
                    continue
                if r.status_code in (401, 403):
                    raise SystemExit(f"Massive rejected the key ({r.status_code}: {r.text[:150]}). "
                                     "Check MASSIVE_API_KEY in .env.")
                r.raise_for_status()
                if not path_or_url.startswith("http") and self.base is None:
                    self.base = url[: len(url) - len(path_or_url)]
                return r.json()
        raise RuntimeError(f"no Massive host answered for {path_or_url}: {last_err}")


def _setup(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.execute(f"""CREATE TABLE IF NOT EXISTS raw.massive_tickers (
        {', '.join(f'{f} VARCHAR' if f != 'active' else 'active BOOLEAN' for f in FIELDS)}, fetched_at TIMESTAMP)""")
    con.execute("""CREATE TABLE IF NOT EXISTS raw.massive_progress (
        active BOOLEAN PRIMARY KEY, next_url VARCHAR, done BOOLEAN, pages INTEGER)""")


def download(api: Massive, db: str = REFERENCE) -> None:
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db)
    _setup(con)
    for active in (True, False):
        row = con.execute("SELECT next_url, done, pages FROM raw.massive_progress WHERE active = ?", [active]).fetchone()
        if row and row[1]:
            print(f"active={active}: already downloaded ({row[2]} pages)")
            continue
        next_url, pages = (row[0], row[2]) if row else (None, 0)
        print(f"active={active}: {'resuming after page ' + str(pages) if pages else 'starting'}")
        while True:
            if next_url:
                body = api.get(next_url)
            else:
                body = api.get("/v3/reference/tickers", {"market": "stocks", "active": str(active).lower(),
                                                         "limit": 1000, "sort": "ticker", "order": "asc"})
            results = body.get("results") or []
            df = pd.DataFrame([{f: r.get(f) for f in FIELDS} for r in results], columns=FIELDS)
            df["active"] = df["active"].fillna(active).astype(bool)
            df["fetched_at"] = pd.Timestamp.now()
            next_url = body.get("next_url")
            pages += 1
            con.execute("BEGIN")
            if len(df):
                con.register("df", df)
                con.execute("INSERT INTO raw.massive_tickers SELECT * FROM df")
                con.unregister("df")
            con.execute("INSERT OR REPLACE INTO raw.massive_progress VALUES (?, ?, ?, ?)",
                        [active, next_url, next_url is None, pages])
            con.execute("COMMIT")
            print(f"  page {pages}: {len(df)} tickers ({df.ticker.iloc[0] if len(df) else '-'} .. "
                  f"{df.ticker.iloc[-1] if len(df) else '-'})")
            if not next_url:
                break
    n = con.execute("SELECT count(*), count(DISTINCT ticker), count(cik) FROM raw.massive_tickers").fetchone()
    con.close()
    print(f"{n[0]:,} rows, {n[1]:,} distinct tickers, {n[2]:,} with a CIK -> {db}")


def build(db: str = REFERENCE) -> pd.DataFrame:
    """ref.ticker_cik from raw.massive_tickers: one row per (ticker, CIK) holding with its date range."""
    con = duckdb.connect(db)
    raw = con.execute(
        f"""SELECT DISTINCT ticker, TRY_CAST(cik AS BIGINT) AS cik, name, type, active, primary_exchange,
                   composite_figi, CAST(TRY_CAST(delisted_utc AS TIMESTAMP) AS DATE) AS delisted
            FROM raw.massive_tickers
            WHERE type IN ({', '.join(repr(t) for t in STOCK_TYPES)})""").df()
    out = ticker_ranges(raw)
    con.execute("CREATE SCHEMA IF NOT EXISTS ref")
    con.register("out", out)
    con.execute("CREATE OR REPLACE TABLE ref.ticker_cik AS SELECT * FROM out")
    con.unregister("out")
    stats = con.execute("""SELECT count(*) AS rows, count(DISTINCT ticker) AS tickers, count(DISTINCT cik) AS ciks,
                                  sum((cik IS NULL)::INT) AS rows_without_cik,
                                  count(*) - count(DISTINCT ticker) AS reused_ticker_rows
                           FROM ref.ticker_cik""").df()
    con.close()
    print(stats.to_string(index=False))
    return out


def ticker_ranges(raw: pd.DataFrame) -> pd.DataFrame:
    """Date ranges per ticker: holders ordered by delisting date (active last). A holder's range runs from
    the day after the previous holder's delisting to its own delisting (open-ended for the active one)."""
    rows = []
    raw = raw.copy()
    raw["delisted"] = pd.to_datetime(raw["delisted"]).dt.date
    for ticker, g in raw.groupby("ticker", sort=False):
        # one row per holder: a CIK (or a FIGI when the CIK is missing), latest delisting kept
        g = g.assign(holder=g.cik.astype("string").fillna("figi:" + g.composite_figi.astype("string").fillna("?")))
        g = (g.sort_values("delisted", na_position="last")
              .groupby("holder", sort=False).tail(1)
              .sort_values("delisted", na_position="last"))
        prev_end = None
        for r in g.itertuples(index=False):
            start = None if prev_end is None else prev_end + pd.Timedelta(days=1)
            rows.append({"ticker": ticker, "cik": None if pd.isna(r.cik) else int(r.cik), "name": r.name,
                         "type": r.type, "primary_exchange": r.primary_exchange, "composite_figi": r.composite_figi,
                         "active": bool(r.active), "valid_from": start,
                         "valid_to": None if pd.isna(r.delisted) else r.delisted})
            if not pd.isna(r.delisted):
                prev_end = r.delisted
    out = pd.DataFrame(rows, columns=["ticker", "cik", "name", "type", "primary_exchange", "composite_figi",
                                      "active", "valid_from", "valid_to"])
    out["cik"] = out["cik"].astype("Int64")
    return out


TICKER_ON_SQL = """
    SELECT e.*, t.ticker
    FROM {events} e
    LEFT JOIN {map} t
      ON t.cik = e.cik
     AND (t.valid_from IS NULL OR e.day0 >= t.valid_from)
     AND (t.valid_to IS NULL OR e.day0 <= t.valid_to)
    QUALIFY row_number() OVER (PARTITION BY e.event_id ORDER BY t.type = 'CS' DESC, t.valid_to NULLS LAST) = 1
"""


def coverage(db: str = REFERENCE, edgar_db: str = EDGAR) -> pd.DataFrame:
    """Share of events with a ticker valid on day 0, by event type and year."""
    con = duckdb.connect(db, read_only=True)
    con.execute(f"ATTACH '{Path(edgar_db).as_posix()}' AS edgar (READ_ONLY)")
    sql = TICKER_ON_SQL.format(events="edgar.events.announcements", map="ref.ticker_cik")
    df = con.execute(f"""SELECT event_type, year(day0) AS year, count(*) AS events,
                                count(ticker) AS with_ticker, round(count(ticker) / count(*), 3) AS share
                         FROM ({sql}) GROUP BY 1, 2 ORDER BY 1, 2""").df()
    con.close()
    print(df.to_string(index=False))
    tot = df.groupby("event_type")[["events", "with_ticker"]].sum()
    tot["share"] = (tot.with_ticker / tot.events).round(3)
    print("\n" + tot.to_string())
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--db", default=REFERENCE)
    b = sub.add_parser("build")
    b.add_argument("--db", default=REFERENCE)
    c = sub.add_parser("coverage")
    c.add_argument("--db", default=REFERENCE)
    c.add_argument("--edgar-db", default=EDGAR)
    a = ap.parse_args()
    if a.cmd == "download":
        key = os.environ.get("MASSIVE_API_KEY")
        if not key:
            raise SystemExit("Set MASSIVE_API_KEY in .env")
        download(Massive(key), a.db)
    elif a.cmd == "build":
        build(a.db)
    else:
        coverage(a.db, a.edgar_db)


if __name__ == "__main__":
    main()
