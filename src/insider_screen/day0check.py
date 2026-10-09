"""Day-0 check for acquisition targets (spec, "Events"): compare day 0 from the 8-K acceptance time
with day 0 from the press release that announced the deal, for 50 sampled events.

If more than 5 of the 50 differ by a trading session or more, the spec switches day 0 to the
press-release time.

Usage:
    uv run python -m insider_screen.day0check sample      # writes data/day0_check.csv
    uv run python -m insider_screen.day0check fill        # looks up press-release times (press_release.py)
    uv run python -m insider_screen.day0check score
    uv run python -m insider_screen.day0check market      # first market move per event (market_move.py)
    uv run python -m insider_screen.day0check score --column market_move_et
    uv run python -m insider_screen.day0check daily       # the daily-bar day-0 rule (day0_rule.py) on the sample

In the CSV, fill press_release_et with the release's date and Eastern time ("2021-06-16 07:00"),
press_release_source with the URL you took it from, and notes as needed. Leave press_release_et
blank for an event you couldn't resolve; it is reported, not scored.

`fill` does the lookup for rows whose press_release_et is blank and saves after each row, so it can be
stopped and rerun. Its notes start with "auto:" and name the exhibit and how the time was read; rows it
couldn't resolve get the reason and a search link. Check a few filled rows against their pages too.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import exchange_calendars as xc
import pandas as pd

from insider_screen import press_release
from insider_screen.db import EDGAR
from insider_screen.edgar import day0
from insider_screen.http import DEFAULT_USER_AGENT, PoliteClient

INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{acc}-index.htm"
N = 50
MAX_DIFFERING = 5


def sample(db: str, out: str, n: int = N, seed: int = 42, start: str = "2017-01-01",
           end: str = "2025-12-31") -> pd.DataFrame:
    con = duckdb.connect(db, read_only=True)
    df = con.execute(
        f"""SELECT a.event_id, a.cik, c.name AS company, a.accession, a.accepted_et, a.day0, a.evidence
            FROM events.announcements a JOIN raw.edgar_companies c USING (cik)
            WHERE a.event_type = 'acquisition_target' AND a.day0_basis = 'index_header'
              AND a.day0 BETWEEN ? AND ?
            ORDER BY hash(a.accession || '{int(seed)}') LIMIT {int(n)}""",
        [start, end]).df()
    con.close()
    df["filing_index"] = [INDEX_URL.format(cik=int(c), folder=a.replace("-", ""), acc=a)
                          for c, a in zip(df.cik, df.accession)]
    df["press_release_et"] = ""
    df["press_release_source"] = ""
    df["notes"] = ""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} acquisition targets to {out}")
    return df


def fill(csv: str, sec=None, web=None, limit: int | None = None) -> pd.DataFrame:
    """Fill press_release_et, press_release_source and notes for rows left blank. Saves after each row."""
    sec = sec or PoliteClient(cache_dir="data/cache/sec", user_agent=DEFAULT_USER_AGENT, max_per_second=5)
    # newswires and search engines: give up fast on a blocked or hanging site (one retry, 15 s timeout)
    web = web or PoliteClient(cache_dir="data/cache/press", user_agent=press_release.BROWSER_UA,
                              max_per_second=1.0, max_retries=1, timeout=15.0)
    if hasattr(web, "client"):  # headers a browser sends; some bot filters stall requests without them
        web.client.headers.update(press_release.BROWSER_HEADERS)
    df = pd.read_csv(csv, dtype=str, encoding="utf-8-sig").fillna("")
    todo = df.index[df.press_release_et.str.strip() == ""]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} of {len(df)} rows to look up")
    started = time.monotonic()
    for n, i in enumerate(todo, 1):
        try:
            accepted = df.at[i, "accepted_et"] if "accepted_et" in df else ""
            filed = pd.Timestamp(accepted).date() if accepted else None
            found = press_release.find_release(sec, web, df.at[i, "filing_index"], filed)
        except Exception as err:  # one bad page shouldn't stop the run
            found = {"press_release_et": "", "press_release_source": "", "notes": f"lookup failed: {err}"[:300]}
        for k, v in found.items():
            df.at[i, k] = v
        df.to_csv(csv, index=False, encoding="utf-8-sig")
        left = (time.monotonic() - started) / n * (len(todo) - n)
        print(f"  {n}/{len(todo)} {df.at[i, 'company'][:40]}: {found['press_release_et'] or 'not found'}"
              f"  (about {left / 60:.0f} min left)")
    done = (df.press_release_et.str.strip() != "").sum()
    print(f"{done} of {len(df)} rows have a press-release time; the rest have a note and a search link")
    return df


def market(csv: str, api=None, sec=None, cal=None) -> pd.DataFrame:
    """Fill ticker, market_move_et and market_move_note for rows without a market_move_et. Saves per row."""
    import os

    from insider_screen import market_move
    from insider_screen.prices import Alpaca
    if api is None:
        key, secret = os.environ.get("ALPACA_API_KEY_ID"), os.environ.get("ALPACA_API_SECRET_KEY")
        if not key or not secret:
            raise SystemExit("Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY (in .env)")
        api = Alpaca(key, secret)
    sec = sec or PoliteClient(cache_dir="data/cache/sec", user_agent=DEFAULT_USER_AGENT, max_per_second=5)
    cal = cal or xc.get_calendar("XNYS", start="2015-01-01")
    df = pd.read_csv(csv, dtype=str, encoding="utf-8-sig").fillna("")
    for col in ("ticker", "market_move_et", "market_move_note"):
        if col not in df:
            df[col] = ""
    todo = df.index[df.market_move_et.str.strip() == ""]
    print(f"{len(todo)} of {len(df)} rows to check")
    for n, i in enumerate(todo, 1):
        row = df.loc[i]
        filed = pd.Timestamp(row.accepted_et).date() if row.get("accepted_et", "") else None
        try:
            ticker, how = (row.ticker, "ticker from the CSV") if row.ticker.strip() else \
                market_move.release_ticker(sec, row.filing_index, filed, row.company)
            if not ticker:
                t, note = None, how
            else:
                t, note = market_move.locate(api, ticker.strip(), pd.Timestamp(row.day0).date(), cal)
                note = f"{note}; {how}"
        except Exception as err:  # one bad row shouldn't stop the run
            ticker, t, note = row.ticker, None, f"lookup failed: {err}"[:300]
        df.at[i, "ticker"] = ticker or ""
        df.at[i, "market_move_et"] = f"{t:%Y-%m-%d %H:%M}" if t is not None else ""
        df.at[i, "market_move_note"] = note
        df.to_csv(csv, index=False, encoding="utf-8-sig")
        print(f"  {n}/{len(todo)} {row.company[:40]} ({ticker or '?'}): {df.at[i, 'market_move_et'] or 'none'}")
    done = (df.market_move_et.str.strip() != "").sum()
    print(f"{done} of {len(df)} rows have a market move; score with --column market_move_et")
    return df


def daily(csv: str, api=None, cal=None) -> pd.DataFrame:
    """Apply the daily-bar day-0 rule to each row with a ticker; write day0_daily, day0_daily_basis,
    day0_daily_note, and compare with the minute-bar check where it found a move."""
    import os

    from insider_screen import day0_rule
    from insider_screen.prices import Alpaca, fetch_bars
    if api is None:
        key, secret = os.environ.get("ALPACA_API_KEY_ID"), os.environ.get("ALPACA_API_SECRET_KEY")
        if not key or not secret:
            raise SystemExit("Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY (in .env)")
        api = Alpaca(key, secret)
    cal = cal or xc.get_calendar("XNYS", start="2015-01-01")
    df = pd.read_csv(csv, dtype=str, encoding="utf-8-sig").fillna("")
    for col in ("ticker", "market_move_et", "day0_daily", "day0_daily_basis", "day0_daily_note", "prior_moves"):
        if col not in df:
            df[col] = ""
    for i, row in df.iterrows():
        t = row.ticker.strip()
        if not t:
            df.loc[i, ["day0_daily", "day0_daily_basis", "day0_daily_note", "prior_moves"]] = ["", "", "no ticker", ""]
            continue
        s = pd.Timestamp(row.day0).date()
        start = (pd.Timestamp(s) - pd.Timedelta(days=420)).date().isoformat()
        try:
            bars = fetch_bars(api, [t, "SPY"], start, s.isoformat(), "all")
            accepted = pd.Timestamp(row.accepted_et) if row.get("accepted_et", "") else None
            r = day0_rule.choose_day0(day0_rule.daily_frame(bars, t), s, accepted)
            df.loc[i, ["day0_daily", "day0_daily_basis", "day0_daily_note", "prior_moves"]] = [
                str(r.day0), r.basis, r.note, "|".join(map(str, r.prior_moves))]
        except Exception as err:  # one bad row shouldn't stop the run
            df.loc[i, ["day0_daily", "day0_daily_basis", "day0_daily_note"]] = ["", "", f"failed: {err}"[:200]]
        print(f"  {row.company[:40]} ({t}): {df.at[i, 'day0_daily_basis'] or df.at[i, 'day0_daily_note']} "
              f"{df.at[i, 'day0_daily']}")
    df.to_csv(csv, index=False, encoding="utf-8-sig")

    done = df[df.day0_daily != ""]
    print(f"\n{len(done)} of {len(df)} rows have a daily-rule day 0")
    print(done.day0_daily_basis.value_counts().to_string())
    flagged = done[done.prior_moves != ""]
    if len(flagged):
        print(f"\n{len(flagged)} with announcement-sized moves before day 0, left in the pre-event window:")
        print(flagged[["company", "accepted_et", "day0_daily", "prior_moves"]].to_string(index=False))
    moved = done[done.market_move_et != ""]
    if len(moved):
        mm_day0 = day0(pd.to_datetime(moved.market_move_et), cal).dt.date.astype(str)
        agree = (mm_day0.values == moved.day0_daily.values)
        print(f"\nAgainst the minute-bar check ({len(moved)} rows with a market move): {agree.sum()} agree")
        out = moved.assign(minute_day0=mm_day0.values, agree=agree)
        print(out[["company", "day0", "minute_day0", "day0_daily", "day0_daily_basis", "agree"]].to_string(index=False))
    return df


def score(csv: str, column: str = "press_release_et") -> pd.DataFrame:
    df = pd.read_csv(csv, dtype=str, encoding="utf-8-sig").fillna("")
    if column not in df:
        raise SystemExit(f"{csv} has no column {column}")
    filled = df[df[column].str.strip() != ""].copy()
    pr = pd.to_datetime(filled[column].str.strip(), errors="coerce")
    bad = filled[pr.isna()]
    if len(bad):
        print(f"Unreadable {column} (use 'YYYY-MM-DD HH:MM'): {bad.event_id.tolist()}")
    filled = filled[pr.notna()]
    pr = pr[pr.notna()]
    cal = xc.get_calendar("XNYS", start="2010-01-01")
    sessions = cal.sessions
    filled["pr_day0"] = day0(pr, cal).values
    filled["day0"] = pd.to_datetime(filled.day0)
    filled["session_diff"] = [int(sessions.searchsorted(a)) - int(sessions.searchsorted(b))
                              for a, b in zip(filled.day0, filled.pr_day0)]
    differing = filled[filled.session_diff != 0]
    print(f"{len(filled)} of {len(df)} events checked; {len(df) - len(filled)} left blank")
    print(f"{len(differing)} differ by a trading session or more (threshold: more than {MAX_DIFFERING})")
    if len(differing):
        print(differing[["event_id", "company", "accepted_et", column, "session_diff"]].to_string(index=False))
    if len(differing) > MAX_DIFFERING:
        print(f"FAIL: {len(differing)} differ, more than {MAX_DIFFERING}; blank rows can't change this. "
              "Switch day 0 for acquisition targets to when the news reached the market.")
    elif len(filled) < min(N, len(df)):
        print(f"Verdict pending: {N - len(filled)} more events needed for the spec's 50.")
    elif len(differing) > MAX_DIFFERING:
        print("FAIL: switch day 0 for acquisition targets to the press-release time.")
    else:
        print("PASS: keep the 8-K acceptance time as day 0.")
    return filled


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--db", default=EDGAR)
    s.add_argument("--out", default="data/day0_check.csv")
    s.add_argument("--n", type=int, default=N)
    f = sub.add_parser("fill")
    f.add_argument("--csv", default="data/day0_check.csv")
    f.add_argument("--limit", type=int, help="Look up only this many blank rows (for a quick test)")
    m = sub.add_parser("market")
    m.add_argument("--csv", default="data/day0_check.csv")
    dl = sub.add_parser("daily")
    dl.add_argument("--csv", default="data/day0_check.csv")
    c = sub.add_parser("score")
    c.add_argument("--csv", default="data/day0_check.csv")
    c.add_argument("--column", default="press_release_et", help="press_release_et or market_move_et")
    a = ap.parse_args()
    if a.cmd == "sample":
        sample(a.db, a.out, a.n)
    elif a.cmd == "fill":
        fill(a.csv, limit=a.limit)
    elif a.cmd == "market":
        market(a.csv)
    elif a.cmd == "daily":
        daily(a.csv)
    else:
        score(a.csv, a.column)


if __name__ == "__main__":
    main()
