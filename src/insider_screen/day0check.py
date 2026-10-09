"""Day-0 check for acquisition targets (spec, "Events"): compare day 0 from the 8-K acceptance time
with day 0 from the press release that announced the deal, for 50 sampled events.

If more than 5 of the 50 differ by a trading session or more, the spec switches day 0 to the
press-release time.

Usage:
    uv run python -m insider_screen.day0check sample      # writes data/day0_check.csv
    uv run python -m insider_screen.day0check fill        # looks up press-release times (press_release.py)
    uv run python -m insider_screen.day0check score

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


def score(csv: str) -> pd.DataFrame:
    df = pd.read_csv(csv, dtype=str, encoding="utf-8-sig").fillna("")
    filled = df[df.press_release_et.str.strip() != ""].copy()
    pr = pd.to_datetime(filled.press_release_et.str.strip(), errors="coerce")
    bad = filled[pr.isna()]
    if len(bad):
        print(f"Unreadable press_release_et (use 'YYYY-MM-DD HH:MM'): {bad.event_id.tolist()}")
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
        print(differing[["event_id", "company", "accepted_et", "press_release_et", "session_diff"]].to_string(index=False))
    if len(filled) < min(N, len(df)):
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
    c = sub.add_parser("score")
    c.add_argument("--csv", default="data/day0_check.csv")
    a = ap.parse_args()
    if a.cmd == "sample":
        sample(a.db, a.out, a.n)
    elif a.cmd == "fill":
        fill(a.csv, limit=a.limit)
    else:
        score(a.csv)


if __name__ == "__main__":
    main()
