"""Daily stock bars from Alpaca's market data API (free plan, SIP feed).

The free plan returns history from January 2016, including tickers that have since been delisted
(checked 2026-10-08 with CELG, TWTR and ATVI). Bars are keyed by ticker as traded on each date, so
a reused ticker returns different companies over time; joining to events goes through a dated
ticker-to-CIK map.

Two copies are loaded: adjustment=raw (actual traded prices, for the $1 and market-cap filters) and
adjustment=all (split- and dividend-adjusted prices and split-adjusted volume, for returns and
abnormal volume).

Symbols come from the FINRA daily short-sale table, which lists every NMS stock with off-exchange
trading on each date, delisted ones included. SPY is always added for the market model.

Credentials: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY environment variables.

Usage:
    uv run python -m insider_screen.prices --start 2016-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import os
import re
import time

import duckdb
import httpx
import pandas as pd

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
MAX_PER_MINUTE = 190  # free plan allows 200
BATCH = 100
PAGE_LIMIT = 10000
ADJUSTMENTS = ["raw", "all"]
MARKET = "SPY"
# Alpaca writes share classes with a dot (BRK.B); FINRA files may use a space or slash.
VALID_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")


class Alpaca:
    def __init__(self, key: str, secret: str, transport: httpx.BaseTransport | None = None,
                 max_per_minute: int = MAX_PER_MINUTE) -> None:
        self.client = httpx.Client(
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=60, transport=transport)
        self.min_interval = 60.0 / max_per_minute
        self._last = 0.0

    def get(self, params: dict) -> httpx.Response:
        for attempt in range(6):
            gap = time.monotonic() - self._last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last = time.monotonic()
            try:
                r = self.client.get(BARS_URL, params=params)
            except httpx.TransportError:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            return r
        return r


def fetch_bars(api: Alpaca, symbols: list[str], start: str, end: str, adjustment: str) -> pd.DataFrame:
    """All daily bars for `symbols`, following next_page_token. Raises httpx.HTTPStatusError on 4xx."""
    rows, token = [], None
    while True:
        params = {"symbols": ",".join(symbols), "timeframe": "1Day", "start": start, "end": end,
                  "adjustment": adjustment, "feed": "sip", "limit": PAGE_LIMIT}
        if token:
            params["page_token"] = token
        r = api.get(params)
        r.raise_for_status()
        body = r.json()
        for sym, bars in (body.get("bars") or {}).items():
            for b in bars:
                rows.append((sym, b["t"][:10], b["o"], b["h"], b["l"], b["c"], b["v"], b.get("n"), b.get("vw")))
        token = body.get("next_page_token")
        if not token:
            break
    df = pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "volume", "trades", "vwap"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def fetch_batch(api: Alpaca, symbols: list[str], start: str, end: str, adjustment: str
                ) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Fetch a batch; when Alpaca rejects it (one bad symbol fails the whole request), split it in
    half until the bad symbols are isolated. Returns (bars, [(symbol, error)])."""
    try:
        return fetch_bars(api, symbols, start, end, adjustment), []
    except httpx.HTTPStatusError as err:
        if err.response.status_code not in (400, 422):
            raise
        if len(symbols) == 1:
            return pd.DataFrame(), [(symbols[0], err.response.text[:200])]
        mid = len(symbols) // 2
        a, ea = fetch_batch(api, symbols[:mid], start, end, adjustment)
        b, eb = fetch_batch(api, symbols[mid:], start, end, adjustment)
        return pd.concat([a, b], ignore_index=True), ea + eb


def finra_symbols(con: duckdb.DuckDBPyConnection) -> list[str]:
    try:
        syms = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM raw.finra_short_daily").fetchall()]
    except duckdb.CatalogException as err:
        raise SystemExit("raw.finra_short_daily not found; run `insider_screen.shortsale daily` first") from err
    return syms


def load(api: Alpaca, db: str, start: str, end: str, symbols: list[str] | None = None) -> None:
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.execute(
        """CREATE TABLE IF NOT EXISTS raw.alpaca_bars_daily (
             symbol VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
             volume DOUBLE, trades BIGINT, vwap DOUBLE, adjustment VARCHAR)""")
    con.execute(
        """CREATE TABLE IF NOT EXISTS raw.alpaca_symbols_done (
             symbol VARCHAR, adjustment VARCHAR, n_bars BIGINT, error VARCHAR)""")
    if symbols is None:
        symbols = finra_symbols(con)
    symbols = sorted({s.strip().upper().replace("/", ".").replace(" ", ".") for s in symbols} | {MARKET})
    skipped = [s for s in symbols if not VALID_SYMBOL.match(s)]
    symbols = [s for s in symbols if VALID_SYMBOL.match(s)]
    if skipped:
        print(f"{len(skipped)} symbols skipped for characters Alpaca doesn't use, e.g. {skipped[:5]}")

    for adj in ADJUSTMENTS:
        done = {r[0] for r in con.execute(
            "SELECT symbol FROM raw.alpaca_symbols_done WHERE adjustment = ?", [adj]).fetchall()}
        todo = [s for s in symbols if s not in done]
        print(f"adjustment={adj}: {len(todo)} of {len(symbols)} symbols to load")
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            df, errors = fetch_batch(api, batch, start, end, adj)
            counts = df.groupby("symbol").size().to_dict() if not df.empty else {}
            bad = dict(errors)
            done_rows = pd.DataFrame(
                [(s, adj, int(counts.get(s, 0)), bad.get(s)) for s in batch],
                columns=["symbol", "adjustment", "n_bars", "error"])
            con.execute("BEGIN")
            if not df.empty:
                df["adjustment"] = adj
                con.register("df", df)
                con.execute("INSERT INTO raw.alpaca_bars_daily SELECT * FROM df")
                con.unregister("df")
            con.register("d", done_rows)
            con.execute("INSERT INTO raw.alpaca_symbols_done SELECT * FROM d")
            con.unregister("d")
            con.execute("COMMIT")
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)}: {len(df):,} bars"
                  + (f", {len(errors)} rejected" if errors else ""))
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/insider.duckdb")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--symbols", help="Comma-separated symbols instead of the FINRA list (for a test run)")
    a = ap.parse_args()
    key, secret = os.environ.get("ALPACA_API_KEY_ID"), os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY")
    syms = a.symbols.split(",") if a.symbols else None
    load(Alpaca(key, secret), a.db, a.start, a.end, syms)


if __name__ == "__main__":
    main()
