"""FINRA off-exchange short-sale data.

Daily Short Sale Volume Files (one per reporting facility per trade date, named
<FACILITY>shvol<YYYYMMDD>.txt). Layout since 2011-02-28, per FINRA's file layout guide:
Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market, with a header row and a trailer row.
- From 2018-08-01: CNMS (consolidated TRF + ADF file for NMS stocks).
- Before that: FNSQ (Nasdaq TRF) and FNYX (NYSE TRF) summed by symbol. FNQC (Nasdaq TRF Chicago)
  starts 2018-09-10, after the consolidated file, so it's never needed.

Monthly Short Sale Transaction Files (trade-level) at cdn.finra.org/equity/regsho/monthly/, listed
on FINRA's data catalog for 2009 through the latest month. Large facilities are split into parts:
August 2026 has FNSQsh202608_1.zip to _4.zip (about 1 GB each). The old regsho.finra.org links
redirect to the catalog page. ADF files are empty after January 2015 and are skipped.

Usage:
    uv run python -m insider_screen.shortsale probe
    uv run python -m insider_screen.shortsale daily --start 2015-01-01 --end 2025-12-31
    uv run python -m insider_screen.shortsale monthly --start 2015-01 --end 2025-12
"""

from __future__ import annotations

import argparse
import io
import re
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import duckdb
import exchange_calendars as xc
import httpx
import pandas as pd

from insider_screen.http import DEFAULT_USER_AGENT, PoliteClient

DAILY_URL = "https://cdn.finra.org/equity/regsho/daily/{fac}shvol{d:%Y%m%d}.txt"
MONTHLY_BASE = "https://cdn.finra.org/equity/regsho/monthly/"
MAX_PARTS = 9
CNMS_START = date(2018, 8, 1)
PRE_CNMS = ["FNSQ", "FNYX"]
MONTHLY_FACILITIES = ["FNSQ", "FNYX", "FNQC"]


def facilities_for(d: date) -> list[str]:
    return ["CNMS"] if d >= CNMS_START else PRE_CNMS


def parse_daily(text: str) -> pd.DataFrame:
    """Parse one daily file; drops the trailer row and tolerates the pre-2011 header names."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame(columns=["date", "symbol", "short_volume", "short_exempt_volume", "total_volume"])
    header = [h.strip().lower().replace(" ", "") for h in lines[0].split("|")]
    rows = [ln.split("|") for ln in lines[1:] if ln.count("|") >= 3]
    if not rows:  # header and trailer only: nothing reported to this facility that day
        return pd.DataFrame(columns=["date", "symbol", "short_volume", "short_exempt_volume", "total_volume"])
    df = pd.DataFrame(rows).iloc[:, : len(header)]
    df.columns = header[: df.shape[1]]
    rename = {"date": "date", "symbol": "symbol", "shortvolume": "short_volume",
              "shortexemptvolume": "short_exempt_volume", "totalvolume": "total_volume"}
    df = df.rename(columns=rename)
    missing = {"date", "symbol", "short_volume", "total_volume"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}; first line: {lines[0][:200]!r}")
    if "short_exempt_volume" not in df:
        df["short_exempt_volume"] = 0
    df = df[["date", "symbol", "short_volume", "short_exempt_volume", "total_volume"]]
    for c in ["short_volume", "short_exempt_volume", "total_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    return df.dropna(subset=["date", "total_volume"])


BAD_FILES: list[tuple[str, str]] = []


def fetch_day(client: PoliteClient, d: date) -> tuple[pd.DataFrame, list[str]]:
    frames, got = [], []
    for fac in facilities_for(d):
        url = DAILY_URL.format(fac=fac, d=d)
        status, body = client.get(url, store=False)
        if status == 200 and body:
            try:
                frames.append(parse_daily(body.decode("latin-1")))
                got.append(fac)
            except ValueError as err:
                BAD_FILES.append((url, str(err)))
                print(f"  skipped {url}: {err}")
    if not frames:
        return pd.DataFrame(), got
    df = pd.concat(frames)
    out = df.groupby(["date", "symbol"], as_index=False)[
        ["short_volume", "short_exempt_volume", "total_volume"]].sum()
    return out, got


def load_daily(client: PoliteClient, db: str, start: str, end: str) -> None:
    cal = xc.get_calendar("XNYS", start="2010-01-01")
    sessions = [s.date() for s in cal.sessions_in_range(start, end)]
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.execute(
        """CREATE TABLE IF NOT EXISTS raw.finra_short_daily (
             date DATE, symbol VARCHAR, short_volume BIGINT, short_exempt_volume BIGINT,
             total_volume BIGINT, facilities VARCHAR)"""
    )
    done = {r[0] for r in con.execute("SELECT DISTINCT date FROM raw.finra_short_daily").fetchall()}
    todo = [d for d in sessions if d not in done]
    print(f"{len(todo)} of {len(sessions)} sessions to load")
    missing = []
    for i, d in enumerate(todo, 1):
        df, got = fetch_day(client, d)
        if df.empty:
            missing.append(d)
            continue
        df["facilities"] = "|".join(got)
        con.register("df", df)
        con.execute("INSERT INTO raw.finra_short_daily SELECT * FROM df")
        con.unregister("df")
        if i % 100 == 0:
            print(f"  {d}: {i}/{len(todo)}")
    con.close()
    if missing:
        print(f"{len(missing)} sessions with no file, first few: {missing[:10]}")
    if BAD_FILES:
        print(f"{len(BAD_FILES)} files skipped for an unexpected layout; a day with one bad facility file "
              "is loaded from the others and listed here")


def find_parts(head, fac: str, y: int, m: int) -> list[str]:
    """URLs for one facility-month: the single file if present, else _1, _2, ... until one is missing.
    `head(url)` returns an HTTP status."""
    single = f"{MONTHLY_BASE}{fac}sh{y}{m:02d}.zip"
    if head(single) == 200:
        return [single]
    parts = []
    for k in range(1, MAX_PARTS + 1):
        url = f"{MONTHLY_BASE}{fac}sh{y}{m:02d}_{k}.zip"
        if head(url) != 200:
            break
        parts.append(url)
    return parts


def part_label(url: str) -> str:
    """'.../FNSQsh202608_1.zip' -> 'FNSQ_1'; '.../FNYXsh202608.zip' -> 'FNYX'."""
    m = re.match(r"([A-Z]{4})sh\d{6}(_\d+)?\.zip$", url.rsplit("/", 1)[1])
    return m.group(1) + (m.group(2) or "")


def _head_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": DEFAULT_USER_AGENT}, follow_redirects=False, timeout=30)


def probe(client: PoliteClient) -> None:
    for d in [date(2016, 3, 1), date(2018, 7, 31), date(2018, 8, 1), date(2021, 6, 1), date(2025, 6, 2)]:
        for fac in facilities_for(d):
            url = DAILY_URL.format(fac=fac, d=d)
            status, body = client.get(url, use_cache=False, store=False)
            print(f"{status} {len(body):>9,} bytes  {url}")
    with _head_client() as c:
        sizes = {}

        def head(url: str) -> int:
            r = c.head(url)
            sizes[url] = r.headers.get("content-length", "?")
            return r.status_code

        for fac, y, m in [("FNSQ", 2016, 3), ("FNYX", 2016, 3), ("FNSQ", 2021, 12), ("FNSQ", 2025, 12),
                          ("FNQC", 2025, 12)]:
            parts = find_parts(head, fac, y, m)
            shown = ", ".join(f"{u.rsplit('/', 1)[1]} ({int(sizes[u]) / 1e6:,.0f} MB)" if sizes[u] != "?"
                              else u.rsplit("/", 1)[1] for u in parts)
            print(f"{fac} {y}-{m:02d}: {len(parts)} file(s) {shown}")


def aggregate_monthly(lines: io.TextIOBase) -> pd.DataFrame:
    """Daily per-symbol trade counts and sizes from a monthly transaction file (header row required)."""
    header = [h.strip().lower() for h in next(lines).split("|")]
    need = {"symbol", "date", "size"}
    if not need <= set(header):
        raise ValueError(f"unexpected header {header}; expected columns including {sorted(need)}")
    idx = {h: i for i, h in enumerate(header)}
    chunks, buf = [], []
    for ln in lines:
        parts = ln.rstrip("\n").split("|")
        if len(parts) < len(header):
            continue
        buf.append((parts[idx["date"]], parts[idx["symbol"]], parts[idx["size"]]))
        if len(buf) >= 2_000_000:
            chunks.append(_agg(buf))
            buf = []
    if buf:
        chunks.append(_agg(buf))
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks)
    return df.groupby(["date", "symbol"], as_index=False).agg(
        short_trades=("short_trades", "sum"), short_shares=("short_shares", "sum"),
        small_trades=("small_trades", "sum"))


def aggregate_zip(zip_path: Path, work_dir: Path) -> pd.DataFrame:
    """Same output as aggregate_monthly, computed by DuckDB on the unzipped text file.
    A 1 GB part unzips to several GB; DuckDB reads it in parallel, far faster than line-by-line Python."""
    with zipfile.ZipFile(zip_path) as zf:
        member = zf.namelist()[0]
        txt = Path(zf.extract(member, work_dir))
    try:
        con = duckdb.connect()
        src = f"read_csv('{txt.as_posix()}', delim='|', header=true, all_varchar=true, ignore_errors=true)"
        cols = {c.lower(): c for c in con.execute(f"SELECT * FROM {src} LIMIT 0").df().columns}
        need = {"symbol", "date", "size"}
        if not need <= set(cols):
            raise ValueError(f"unexpected header {list(cols)}; expected columns including {sorted(need)}")
        df = con.execute(f"""
            SELECT strptime("{cols['date']}", '%Y%m%d')::DATE AS date, "{cols['symbol']}" AS symbol,
                   count(*) AS short_trades, sum(sz) AS short_shares, sum((sz <= 100)::INT) AS small_trades
            FROM (SELECT *, TRY_CAST("{cols['size']}" AS BIGINT) AS sz FROM {src})
            WHERE sz IS NOT NULL AND regexp_full_match("{cols['date']}", '\\d{{8}}')
            GROUP BY 1, 2""").df()
        con.close()
        return df
    finally:
        txt.unlink(missing_ok=True)


def _agg(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "symbol", "size"])
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df = df.dropna(subset=["size"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["small"] = df["size"] <= 100
    return df.groupby(["date", "symbol"], as_index=False).agg(
        short_trades=("size", "size"), short_shares=("size", "sum"), small_trades=("small", "sum"))


def load_monthly(db: str, start: str, end: str, keep_zips: bool = False) -> None:
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.execute(
        """CREATE TABLE IF NOT EXISTS raw.finra_short_trades_daily (
             date DATE, symbol VARCHAR, short_trades BIGINT, short_shares BIGINT, small_trades BIGINT,
             facility VARCHAR, file_month VARCHAR)"""
    )
    done = {(r[0], r[1]) for r in con.execute(
        "SELECT DISTINCT facility, file_month FROM raw.finra_short_trades_daily").fetchall()}
    zdir = Path("data/finra_monthly")
    zdir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": DEFAULT_USER_AGENT}, follow_redirects=False, timeout=None) as c:
        head = lambda url: c.head(url).status_code  # noqa: E731
        while (y, m) <= (ey, em):
            month = f"{y}-{m:02d}"
            urls = [u for fac in MONTHLY_FACILITIES for u in find_parts(head, fac, y, m)]
            if not urls:
                print(f"  {month}: no files found")
            for url in urls:
                fac = part_label(url)
                if (fac, month) in done:
                    continue
                with tempfile.NamedTemporaryFile(dir=zdir, suffix=".zip", delete=False) as tmp:
                    with c.stream("GET", url) as r:
                        status = r.status_code
                        if status == 200:
                            for chunk in r.iter_bytes(1 << 20):
                                tmp.write(chunk)
                path = Path(tmp.name)
                if status != 200 or path.stat().st_size < 1024:  # placeholder zips (~200 bytes) hold no trades
                    if status != 200:
                        print(f"  {status} {url}")
                    path.unlink()
                    continue
                df = aggregate_zip(path, zdir)
                df["facility"], df["file_month"] = fac, month
                con.register("df", df)
                con.execute("INSERT INTO raw.finra_short_trades_daily SELECT * FROM df")
                con.unregister("df")
                print(f"  {month} {fac}: {len(df):,} symbol-days")
                if keep_zips:
                    path.rename(zdir / url.rsplit("/", 1)[1])
                else:
                    path.unlink()
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    d = sub.add_parser("daily")
    d.add_argument("--db", default="data/insider.duckdb")
    d.add_argument("--start", default="2015-01-01")
    d.add_argument("--end", default="2025-12-31")
    mo = sub.add_parser("monthly")
    mo.add_argument("--db", default="data/insider.duckdb")
    mo.add_argument("--start", default="2015-01")
    mo.add_argument("--end", default="2025-12")
    mo.add_argument("--keep-zips", action="store_true")
    a = ap.parse_args()
    client = PoliteClient(cache_dir="data/cache/finra", max_per_second=2.0)
    if a.cmd == "probe":
        probe(client)
    elif a.cmd == "daily":
        load_daily(client, a.db, a.start, a.end)
    else:
        load_monthly(a.db, a.start, a.end, a.keep_zips)


if __name__ == "__main__":
    main()
