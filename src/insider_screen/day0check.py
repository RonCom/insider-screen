"""Day-0 check for acquisition targets (spec, "Events"): compare day 0 from the 8-K acceptance time
with day 0 from the press release that announced the deal, for 50 sampled events.

If more than 5 of the 50 differ by a trading session or more, the spec switches day 0 to the
press-release time.

Usage:
    uv run python -m insider_screen.day0check sample      # writes data/day0_check.csv
    uv run python -m insider_screen.day0check score

In the CSV, fill press_release_et with the release's date and Eastern time ("2021-06-16 07:00"),
press_release_source with the URL you took it from, and notes as needed. Leave press_release_et
blank for an event you couldn't resolve; it is reported, not scored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import exchange_calendars as xc
import pandas as pd

from insider_screen.db import EDGAR
from insider_screen.edgar import day0

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


def score(csv: str) -> pd.DataFrame:
    df = pd.read_csv(csv, dtype=str).fillna("")
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
    c = sub.add_parser("score")
    c.add_argument("--csv", default="data/day0_check.csv")
    a = ap.parse_args()
    if a.cmd == "sample":
        sample(a.db, a.out, a.n)
    else:
        score(a.csv)


if __name__ == "__main__":
    main()
