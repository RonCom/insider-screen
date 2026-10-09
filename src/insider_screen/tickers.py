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
3. coverage: share of events in data/edgar.duckdb with a ticker valid on day 0, by event type and year
   (map first, then the supplement from step 5).
4. check: error rate of the map. For a sample of events the map covers (split evenly across event types),
   the ticker printed next to the company's name in the 8-K's press release is compared with the map's;
   rows go to data/ticker_check.csv.
5. fill: for each company-year with events but no ticker, one 8-K press release supplies the ticker
   (ref.ticker_supplement). A ticker the map gives another company on that date is rejected. Resumable.

Settings (.env): MASSIVE_API_KEY; MASSIVE_API_URL (default https://api.massive.com; the older
https://api.polygon.io is tried if that host doesn't answer).

Usage:
    uv run python -m insider_screen.tickers download
    uv run python -m insider_screen.tickers build
    uv run python -m insider_screen.tickers coverage
    uv run python -m insider_screen.tickers check --n 200
    uv run python -m insider_screen.tickers fill --types acquisition_target
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import duckdb
import httpx
import pandas as pd

from insider_screen.db import EDGAR, FINRA, REFERENCE

BASE_URLS = [os.environ.get("MASSIVE_API_URL", "https://api.massive.com"), "https://api.polygon.io"]
PER_MINUTE = 5  # free plan
FIELDS = ["ticker", "name", "cik", "type", "active", "primary_exchange", "composite_figi", "share_class_figi",
          "delisted_utc", "last_updated_utc"]
# security types kept in the map: common stock, ADRs, and the like; funds, warrants, units, rights are left out
STOCK_TYPES = ("CS", "ADRC", "ADRP", "ADRS", "OS", "NYRS", "GDR")
# Older delisted tickers often have no type in Massive (6,742 of 23,492 delisted rows). Those with a CIK are
# kept as 'untyped' unless the name or ticker shows another kind of security.
NOT_STOCK_NAME_RE = (r"(?i)\b(warrants?|units?|rights?|preferred|pfd|depositary shares?|notes?|debentures?|"
                     r"etf|etn|fund|index|trust preferred|subordinated|senior|%)\b|%")


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
        f"""WITH r AS (
              SELECT DISTINCT ticker, TRY_CAST(cik AS BIGINT) AS cik, name, coalesce(type, 'untyped') AS type, active,
                     primary_exchange, composite_figi, CAST(TRY_CAST(delisted_utc AS TIMESTAMP) AS DATE) AS delisted
              FROM raw.massive_tickers)
            SELECT * FROM r
            WHERE type IN ({', '.join(repr(t) for t in STOCK_TYPES)})
               OR (type = 'untyped' AND cik IS NOT NULL
                   AND NOT regexp_matches(coalesce(name, ''), '{NOT_STOCK_NAME_RE}')
                   -- another ticker of the same company plus a warrant/unit/right suffix
                   AND NOT EXISTS (SELECT 1 FROM r b WHERE b.cik = r.cik AND b.ticker <> r.ticker
                                   AND regexp_matches(r.ticker, '^' || regexp_escape(b.ticker) || '[.-]?(W|WS|WT|U|UN|R|RT)$')))""").df()
    out = ticker_ranges(raw)
    con.execute("CREATE SCHEMA IF NOT EXISTS ref")
    con.register("out", out)
    con.execute("""CREATE OR REPLACE TABLE ref.ticker_cik AS
                   SELECT ticker, CAST(cik AS BIGINT) AS cik, name, type, primary_exchange, composite_figi,
                          CAST(active AS BOOLEAN) AS active,
                          CAST(valid_from AS DATE) AS valid_from, CAST(valid_to AS DATE) AS valid_to
                   FROM out""")
    con.unregister("out")
    stats = con.execute("""SELECT count(*) AS rows, count(DISTINCT ticker) AS tickers, count(DISTINCT cik) AS ciks,
                                  sum((cik IS NULL)::INT) AS rows_without_cik,
                                  count(*) - count(DISTINCT ticker) AS reused_ticker_rows,
                                  sum((NOT active)::INT) AS delisted_rows, count(valid_to) AS rows_with_end_date
                           FROM ref.ticker_cik""").df()
    by_list = con.execute("""SELECT active, count(*) AS rows, count(delisted_utc) AS with_delisting_date,
                                    min(delisted_utc) AS earliest_delisting, max(delisted_utc) AS latest_delisting
                             FROM raw.massive_tickers GROUP BY 1 ORDER BY 1""").df()
    con.close()
    print(stats.to_string(index=False))
    print("\nDownloaded, all security types:")
    print(by_list.to_string(index=False))
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
    QUALIFY row_number() OVER (PARTITION BY e.event_id
                               ORDER BY {volume} DESC, t.type = 'CS' DESC, t.valid_to NULLS LAST) = 1
"""
# When a company has several tickers valid on day 0 (another share class, a when-issued or preferred line,
# or an old and a new ticker with no start date known), the one that traded most in FINRA's files in day 0's
# month and the month before is the company's stock on that date.
VOLUME_SQL = """coalesce((SELECT sum(v.vol) FROM ticker_volume v WHERE v.symbol = t.ticker
                          AND v.month BETWEEN date_trunc('month', e.day0) - INTERVAL 1 MONTH
                                          AND date_trunc('month', e.day0)), 0)"""


def ticker_on_sql(con: duckdb.DuckDBPyConnection, events: str, map: str = "ref.ticker_cik") -> str:
    has_volume = con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'ticker_volume'").fetchone()[0]
    return TICKER_ON_SQL.format(events=events, map=map, volume=VOLUME_SQL if has_volume else "0")


def attach(con: duckdb.DuckDBPyConnection, edgar_db: str = EDGAR, finra_db: str = FINRA) -> bool:
    """Attach the EDGAR events read-only, and build ticker_volume (monthly FINRA volume per symbol) for
    choosing between a company's tickers. Returns False, with a warning, if FINRA isn't available."""
    con.execute(f"ATTACH '{Path(edgar_db).as_posix()}' AS edgar (READ_ONLY)")
    try:
        con.execute(f"ATTACH '{Path(finra_db).as_posix()}' AS finra (READ_ONLY)")
        con.execute("""CREATE TEMP TABLE ticker_volume AS
                       SELECT symbol, date_trunc('month', date) AS month, sum(total_volume) AS vol
                       FROM finra.raw.finra_short_daily GROUP BY ALL""")
        return True
    except (duckdb.IOException, duckdb.CatalogException) as err:
        print(f"Warning: no FINRA volume ({err}); a company's tickers are ranked without it")
        return False


def event_tickers_sql(con: duckdb.DuckDBPyConnection, events: str = "edgar.events.announcements") -> str:
    """SQL for events with their ticker on day 0: the map first, else the press-release supplement for
    that company and year (when the supplement table exists). Adds columns ticker and ticker_source."""
    mapped = ticker_on_sql(con, events)
    has_supp = con.execute("""SELECT count(*) FROM duckdb_tables()
                              WHERE schema_name = 'ref' AND table_name = 'ticker_supplement'""").fetchone()[0]
    if not has_supp:
        return f"SELECT m.*, CASE WHEN m.ticker IS NOT NULL THEN 'map' END AS ticker_source FROM ({mapped}) m"
    return f"""SELECT m.* EXCLUDE (ticker), coalesce(m.ticker, s.ticker) AS ticker,
                      CASE WHEN m.ticker IS NOT NULL THEN 'map' WHEN s.ticker IS NOT NULL THEN 'press_release' END
                        AS ticker_source
               FROM ({mapped}) m
               LEFT JOIN ref.ticker_supplement s
                 ON s.cik = m.cik AND s.year = year(m.day0) AND s.ticker IS NOT NULL AND NOT s.conflict"""


def coverage(db: str = REFERENCE, edgar_db: str = EDGAR, finra_db: str = FINRA) -> pd.DataFrame:
    """Share of events with a ticker valid on day 0, by event type and year."""
    con = duckdb.connect(db, read_only=True)
    attach(con, edgar_db, finra_db)
    sql = event_tickers_sql(con)
    df = con.execute(f"""SELECT event_type, year(day0) AS year, count(*) AS events,
                                count(ticker) AS with_ticker, round(count(ticker) / count(*), 3) AS share
                         FROM ({sql}) GROUP BY 1, 2 ORDER BY 1, 2""").df()
    con.close()
    print(df.to_string(index=False))
    tot = df.groupby("event_type")[["events", "with_ticker"]].sum()
    tot["share"] = (tot.with_ticker / tot.events).round(3)
    print("\n" + tot.to_string())
    return df


def _names_sql(con, alias: str = "c") -> str:
    """Current and former EDGAR names joined by "|" (former_names is missing from older edgar files)."""
    has = con.execute("""SELECT count(*) FROM duckdb_columns() WHERE database_name = 'edgar'
                         AND table_name = 'edgar_companies' AND column_name = 'former_names'""").fetchone()[0]
    if not has:
        return f"{alias}.name"
    return f"{alias}.name || CASE WHEN coalesce({alias}.former_names, '') <> '' THEN '|' || {alias}.former_names ELSE '' END"


_SECURITY_TAIL_RE = re.compile(r"\b(?:class [a-z] )?(?:common|ordinary|capital) (?:stock|shares)\b.*$|\badrs?\b.*$", re.I)


def _name_key(name: str | None) -> str:
    from insider_screen.match import normalize
    if not isinstance(name, str):  # a map row with no name comes back from pandas as NaN
        return ""
    return normalize(_SECURITY_TAIL_RE.sub("", name))


def names(db: str = REFERENCE, edgar_db: str = EDGAR, finra_db: str = FINRA) -> pd.DataFrame:
    """ref.ticker_supplement rows from names, no network: for each company-year still without a ticker,
    a map row whose name equals the company's current or former EDGAR name (after normalizing) and whose
    dates overlap the year's events. The map row may have no CIK (Massive leaves it off many delisted
    tickers) or a CIK that EDGAR doesn't know or knows under the same name (Versar's VSR carries
    another CIK). Taken only when exactly one ticker matches, no differently named company holds it, and
    no other company's events carry it, through the CIK join, between the year's first and last event
    (that rejects a parent's ticker for a subsidiary, e.g. DUK for Duke Energy Carolinas, whose former
    name is Duke Energy Corp, and another firm of the same short name, e.g. PHI Inc for PHI Group)."""
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA IF NOT EXISTS ref")
    con.execute("""CREATE TABLE IF NOT EXISTS ref.ticker_supplement (
        cik BIGINT, year INTEGER, ticker VARCHAR, conflict BOOLEAN, accession VARCHAR, how VARCHAR)""")
    attach(con, edgar_db, finra_db)
    # rerun from scratch, so a tightened rule also removes matches it no longer makes
    dropped = con.execute("DELETE FROM ref.ticker_supplement WHERE how LIKE 'name match%'").fetchone()[0]
    mapped = ticker_on_sql(con, "edgar.events.announcements")
    # each ticker's events through the CIK join: a ticker another company's events carry over the same
    # dates belongs to that company (a parent, an operating subsidiary, a different firm of the same name)
    taken: dict[str, list] = {}
    for t, cik, day in con.execute(f"""SELECT ticker, cik, CAST(day0 AS DATE) FROM ({mapped})
                                       WHERE ticker IS NOT NULL""").fetchall():
        taken.setdefault(t, []).append((pd.Timestamp(day), int(cik)))
    todo = con.execute(f"""
        WITH m AS ({mapped})
        SELECT m.cik, year(m.day0) AS year, {_names_sql(con)} AS names,
               CAST(min(m.day0) AS DATE) AS first_day0, CAST(max(m.day0) AS DATE) AS last_day0
        FROM m JOIN edgar.raw.edgar_companies c USING (cik)
        WHERE m.ticker IS NULL
          AND NOT EXISTS (SELECT 1 FROM ref.ticker_supplement s
                          WHERE s.cik = m.cik AND s.year = year(m.day0) AND s.ticker IS NOT NULL)
        GROUP BY 1, 2, 3""").df()
    edgar_keys = {int(cik): {_name_key(n) for n in str(ns).split("|")} for cik, ns in con.execute(
        f"SELECT c.cik, {_names_sql(con)} FROM edgar.raw.edgar_companies c").fetchall()}
    cands = con.execute("SELECT ticker, cik, name, valid_from, valid_to FROM ref.ticker_cik").df()
    by_name: dict[str, list] = {}
    for r in cands.itertuples(index=False):
        key = _name_key(r.name)
        if key:
            by_name.setdefault(key, []).append(r)

    def same_company(cik, key) -> bool:  # a map CIK EDGAR doesn't know, or knows under this name
        return pd.isna(cik) or int(cik) not in edgar_keys or key in edgar_keys[int(cik)]

    rows = []
    for r in todo.itertuples(index=False):
        lo, hi = r.first_day0, r.last_day0
        hits = {}
        for key in {_name_key(n) for n in r.names.split("|")} - {""}:
            for c in by_name.get(key, []):
                if not pd.isna(c.cik) and int(c.cik) == int(r.cik):
                    continue  # the CIK join already had this row; its dates don't fit
                if not same_company(c.cik, key):
                    continue
                if (pd.isna(c.valid_from) or c.valid_from <= hi) and (pd.isna(c.valid_to) or c.valid_to >= lo):
                    hits[c.ticker] = (c.name, c.cik)
        if len(hits) != 1:
            continue
        (ticker, (mname, mcik)), = hits.items()
        holders = cands[(cands.ticker == ticker) & cands.cik.notna()
                        & (cands.valid_from.isna() | (cands.valid_from <= hi))
                        & (cands.valid_to.isna() | (cands.valid_to >= lo))]
        if any(int(h) != int(r.cik) and not same_company(h, _name_key(mname)) for h in holders.cik):
            continue
        if any(pd.Timestamp(lo) <= day <= pd.Timestamp(hi) and cik != int(r.cik) for day, cik in taken.get(ticker, [])):
            continue
        how = (f"name match: '{mname}' in the map has no CIK" if pd.isna(mcik)
               else f"name match: '{mname}' in the map under CIK {int(mcik)}")
        rows.append((int(r.cik), int(r.year), ticker, False, None, how))
    for cik, year, *_ in rows:
        con.execute("DELETE FROM ref.ticker_supplement WHERE cik = ? AND year = ?", [cik, year])
    if rows:
        con.executemany("INSERT INTO ref.ticker_supplement VALUES (?, ?, ?, ?, ?, ?)", rows)
    con.close()
    print(f"{len(todo)} company-years without a ticker; {len(rows)} matched by name to a map row "
          f"({dropped} earlier name matches recomputed)")
    return pd.DataFrame(rows, columns=["cik", "year", "ticker", "conflict", "accession", "how"])


def _sec():
    from insider_screen.http import DEFAULT_USER_AGENT, PoliteClient
    return PoliteClient(cache_dir="data/cache/sec", user_agent=DEFAULT_USER_AGENT, max_per_second=5)


def _index_url(cik: int, accession: str) -> str:
    from insider_screen.press_release import INDEX_URL
    return INDEX_URL.format(cik=int(cik), folder=accession.replace("-", ""), acc=accession)


def _release_ticker(sec, cik: int, accession: str, company: str, filed=None) -> tuple[str | None, str]:
    """Ticker from the 8-K's press release, its body, or (with `filed`) the company's other filings that day."""
    from insider_screen.market_move import release_ticker
    # DuckDB hands list elements back as datetime, which can't be compared with the filing dates
    filed = None if filed is None or pd.isna(filed) else pd.Timestamp(filed).date()
    try:
        return release_ticker(sec, _index_url(cik, accession), filed, company)
    except Exception as err:  # one bad filing shouldn't stop the run
        return None, f"lookup failed: {err}"[:200]


def classify(map_ticker: str, release: str | None, cik: int, ranges: pd.DataFrame) -> str:
    if not release:
        return "no_ticker_in_release"
    if release == map_ticker:
        return "agree"
    if re.fullmatch(re.escape(map_ticker) + r"[.\-]?(?:U|UN|W|WS|WT)", release):
        return "units_or_warrants"  # a SPAC's release quotes its units; the map's common-share ticker is right
    holders = ranges[ranges.ticker == release]
    if holders.empty:
        return "release_ticker_not_in_map"
    if (holders.cik == cik).any():
        return "same_company_other_ticker"
    return "release_ticker_other_company"


def check(db: str = REFERENCE, edgar_db: str = EDGAR, n: int = 200, seed: int = 7,
          out: str = "data/ticker_check.csv", sec=None, finra_db: str = FINRA) -> pd.DataFrame:
    """Compare the map's ticker with the press release's for n sampled events (n/3 per event type)."""
    sec = sec or _sec()
    con = duckdb.connect(db, read_only=True)
    attach(con, edgar_db, finra_db)
    mapped = ticker_on_sql(con, "edgar.events.announcements")
    per_type = max(1, n // 3)
    sample = con.execute(f"""
        WITH m AS ({mapped})
        SELECT m.event_id, m.event_type, m.cik, c.name AS company, m.accession, m.day0, m.ticker AS map_ticker
        FROM m JOIN edgar.raw.edgar_companies c USING (cik)
        WHERE m.ticker IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY m.event_type ORDER BY hash(m.event_id || '{int(seed)}')) <= {per_type}
        ORDER BY m.event_type, m.day0""").df()
    ranges = con.execute("SELECT ticker, cik FROM ref.ticker_cik").df()
    con.close()
    print(f"Checking {len(sample)} events against their 8-K press releases")
    results = []
    for k, r in enumerate(sample.itertuples(index=False), 1):
        release, how = _release_ticker(sec, r.cik, r.accession, r.company)
        results.append({"release_ticker": release or "", "outcome": classify(r.map_ticker, release, r.cik, ranges),
                        "how": how})
        if k % 25 == 0:
            print(f"  {k}/{len(sample)}")
    df = pd.concat([sample, pd.DataFrame(results)], axis=1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    comparable = df[~df.outcome.isin(["no_ticker_in_release", "units_or_warrants"])]
    print("\n" + pd.crosstab(df.event_type, df.outcome, margins=True).to_string())
    if len(comparable):
        bad = comparable[comparable.outcome != "agree"]
        print(f"\nError rate: {len(bad)} of {len(comparable)} comparable events ({len(bad) / len(comparable):.1%})")
        if len(bad):
            print(bad[["event_type", "company", "day0", "map_ticker", "release_ticker", "outcome"]].to_string(index=False))
    print(f"\nAll rows: {out}")
    return df


PRIORITY = "CASE m.event_type WHEN 'earnings' THEN 0 WHEN 'acquisition_target' THEN 1 ELSE 2 END"


def fill(db: str = REFERENCE, edgar_db: str = EDGAR, types: list[str] | None = None, sec=None,
         limit: int | None = None, finra_db: str = FINRA, retry: bool = False) -> None:
    """ref.ticker_supplement: a press-release ticker for each company-year that has events without one.
    retry=True looks again at company-years that came back without a ticker (not at conflicts)."""
    sec = sec or _sec()
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA IF NOT EXISTS ref")
    con.execute("""CREATE TABLE IF NOT EXISTS ref.ticker_supplement (
        cik BIGINT, year INTEGER, ticker VARCHAR, conflict BOOLEAN, accession VARCHAR, how VARCHAR)""")
    if retry:
        n = con.execute("SELECT count(*) FROM ref.ticker_supplement WHERE ticker IS NULL").fetchone()[0]
        con.execute("DELETE FROM ref.ticker_supplement WHERE ticker IS NULL")
        print(f"Retrying {n} company-years that had no ticker")
    attach(con, edgar_db, finra_db)
    mapped = ticker_on_sql(con, "edgar.events.announcements")
    # --types picks company-years with such an event; every event's 8-K that year is a candidate
    type_filter = f"HAVING bool_or(m.event_type IN ({', '.join(repr(t) for t in types)}))" if types else ""
    # per company-year without a ticker: up to four 8-Ks, earnings first (their releases nearly always
    # quote the ticker), then deal announcements, then the rest; latest first within each
    todo = con.execute(f"""
        WITH m AS ({mapped})
        SELECT m.cik, year(m.day0) AS year, {_names_sql(con)} AS company,
               list(m.accession ORDER BY {PRIORITY}, m.day0 DESC)[1:4] AS accessions,
               list(CAST(m.accepted_et AS DATE) ORDER BY {PRIORITY}, m.day0 DESC)[1:4] AS filed,
               max(m.day0) AS last_day0
        FROM m JOIN edgar.raw.edgar_companies c USING (cik)
        WHERE m.ticker IS NULL
          AND NOT EXISTS (SELECT 1 FROM ref.ticker_supplement s WHERE s.cik = m.cik AND s.year = year(m.day0))
        GROUP BY 1, 2, 3 {type_filter} ORDER BY 2, 1""").df()
    if limit:
        todo = todo.head(limit)
    print(f"{len(todo)} company-years to look up")
    found = 0
    for k, r in enumerate(todo.itertuples(index=False), 1):
        ticker, how, acc = None, "no 8-K tried", None
        for acc, filed in zip(r.accessions, r.filed):
            ticker, how = _release_ticker(sec, r.cik, acc, r.company, filed)
            if ticker:
                break
        conflict = False
        if ticker:
            other = con.execute(
                """SELECT name FROM ref.ticker_cik WHERE ticker = ? AND cik IS DISTINCT FROM ?
                   AND (valid_from IS NULL OR valid_from <= ?) AND (valid_to IS NULL OR valid_to >= ?)""",
                [ticker, int(r.cik), r.last_day0, r.last_day0]).fetchone()
            if other:
                conflict, how = True, f"{how}; map gives {ticker} to {other[0]} on {r.last_day0:%Y-%m-%d}"
            else:
                found += 1
        con.execute("INSERT INTO ref.ticker_supplement VALUES (?, ?, ?, ?, ?, ?)",
                    [int(r.cik), int(r.year), ticker, conflict, acc, how[:300]])
        if k % 50 == 0 or k == len(todo):
            print(f"  {k}/{len(todo)}: {found} tickers found")
    con.close()
    print("Run coverage to see the effect.")


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
    k = sub.add_parser("check")
    k.add_argument("--db", default=REFERENCE)
    k.add_argument("--edgar-db", default=EDGAR)
    k.add_argument("--n", type=int, default=200)
    k.add_argument("--out", default="data/ticker_check.csv")
    f = sub.add_parser("fill")
    f.add_argument("--db", default=REFERENCE)
    f.add_argument("--edgar-db", default=EDGAR)
    f.add_argument("--types", nargs="*", help="event types, e.g. acquisition_target earnings")
    f.add_argument("--limit", type=int)
    f.add_argument("--retry", action="store_true", help="look again at company-years that had no ticker")
    n = sub.add_parser("names", help="match company-years without a ticker to map rows that have no CIK")
    n.add_argument("--db", default=REFERENCE)
    n.add_argument("--edgar-db", default=EDGAR)
    a = ap.parse_args()
    if a.cmd == "download":
        key = os.environ.get("MASSIVE_API_KEY")
        if not key:
            raise SystemExit("Set MASSIVE_API_KEY in .env")
        download(Massive(key), a.db)
    elif a.cmd == "build":
        build(a.db)
    elif a.cmd == "check":
        check(a.db, a.edgar_db, a.n, out=a.out)
    elif a.cmd == "fill":
        fill(a.db, a.edgar_db, a.types, limit=a.limit, retry=a.retry)
    elif a.cmd == "names":
        names(a.db, a.edgar_db)
    else:
        coverage(a.db, a.edgar_db)


if __name__ == "__main__":
    main()
