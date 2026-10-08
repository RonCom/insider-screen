"""Build the announcement event table from SEC EDGAR filing histories.

Source: the nightly bulk file of the Submissions API (one JSON per filer plus continuation
files), https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip. Each filing
row carries form type, 8-K item numbers and the acceptance timestamp, so no per-filing
requests are needed.

Event types (spec, "Events"):
- earnings: 8-K with Item 2.02.
- acquisition_target: the earliest 8-K with Item 1.01 filed within 120 days before the filer's
  first target-side merger filing (PREM14A, DEFM14A, SC 14D9, SC14D9C, SC 13E3). Only targets
  file these, so the rule separates targets from acquirers without opening exhibits.
- other_material_candidate: 8-K with Item 7.01 or 8.01 and neither 1.01 nor 2.02. The spec keeps
  these only if the day-0 abnormal return exceeds 10%, which needs prices (later step).

Timestamps: acceptanceDateTime ends in "Z" but is treated as US Eastern time. Check it with
`check-tz` before relying on day 0: earnings 8-Ks should cluster just after 16:00 and before 09:30.

Usage:
    uv run python -m insider_screen.edgar download
    uv run python -m insider_screen.edgar load
    uv run python -m insider_screen.edgar check-tz
    uv run python -m insider_screen.edgar events
"""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import time
from pathlib import Path

import duckdb
import exchange_calendars as xc
import pandas as pd

from insider_screen.db import EDGAR
from insider_screen.http import DEFAULT_USER_AGENT

BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
KEEP_FORMS = {
    "8-K", "8-K/A", "10-K", "10-Q", "PREM14A", "DEFM14A", "SC 14D9", "SC14D9C", "SC 13E3",
    "SC TO-T", "25-NSE", "15-12B",
}
TARGET_FORMS = {"PREM14A", "DEFM14A", "SC 14D9", "SC14D9C", "SC 13E3"}
TARGET_WINDOW_DAYS = 120
MARKET_CLOSE = time(16, 0)
FIELDS = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items"]


def download(dest: str = "data/edgar/submissions.zip") -> Path:
    """Stream the bulk file to disk (several GB; resumes are not supported, rerun to replace)."""
    import httpx

    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", BULK_URL, headers={"User-Agent": DEFAULT_USER_AGENT},
                      timeout=None, follow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(path, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done % (200 << 20) < (1 << 20):
                    print(f"  {done / total:.0%} of {total / 1e9:.1f} GB")
    return path


def _rows(block: dict, cik: int) -> list[tuple]:
    cols = [block.get(k, []) for k in FIELDS]
    out = []
    for acc, fdate, accepted, form, items in zip(*cols):
        if form in KEEP_FORMS:
            out.append((cik, acc, fdate, accepted, form, items or ""))
    return out


def parse_member(name: str, data: dict) -> tuple[dict | None, list[tuple]]:
    """A main file (CIK##########.json) has company fields plus filings.recent; a continuation
    file (CIK##########-submissions-NNN.json) holds the filing arrays at top level."""
    cik = int(Path(name).name[3:13])
    if "filings" in data:
        company = {
            "cik": cik,
            "name": data.get("name"),
            "entity_type": data.get("entityType"),
            "sic": data.get("sic"),
            "tickers": "|".join(data.get("tickers") or []),
            "exchanges": "|".join(x or "" for x in (data.get("exchanges") or [])),
            "former_names": "|".join(f.get("name", "") for f in data.get("formerNames") or []),
        }
        return company, _rows(data["filings"].get("recent", {}), cik)
    return None, _rows(data, cik)


def load(zip_path: str, db: str) -> None:
    companies, filings = [], []
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json") and Path(n).name.startswith("CIK")]
        for i, name in enumerate(names, 1):
            company, rows = parse_member(name, json.loads(zf.read(name)))
            if company:
                companies.append(company)
            filings.extend(rows)
            if i % 100_000 == 0:
                print(f"  {i}/{len(names)} files, {len(filings):,} filings kept")
    cdf = pd.DataFrame(companies)
    fdf = pd.DataFrame(filings, columns=["cik", "accession", "filing_date", "accepted_raw", "form", "items"])
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.register("cdf", cdf)
    con.register("fdf", fdf)
    con.execute("CREATE OR REPLACE TABLE raw.edgar_companies AS SELECT * FROM cdf")
    con.execute(
        """CREATE OR REPLACE TABLE raw.edgar_filings AS
           SELECT DISTINCT cik, accession, CAST(filing_date AS DATE) AS filing_date,
                  -- the trailing Z is ignored: values are read as US Eastern wall-clock time
                  TRY_CAST(replace(replace(accepted_raw, 'T', ' '), '.000Z', '') AS TIMESTAMP) AS accepted_et,
                  form, items
           FROM fdf"""
    )
    n = con.execute("SELECT count(*) FROM raw.edgar_filings").fetchone()[0]
    con.close()
    print(f"{len(cdf):,} filers, {n:,} filings kept -> {db}")


def check_tz(db: str) -> pd.DataFrame:
    """Hour-of-day histogram for earnings 8-Ks. Read as Eastern, expect peaks at 16-17 and 6-9."""
    con = duckdb.connect(db, read_only=True)
    df = con.execute(
        """SELECT hour(accepted_et) AS hour, count(*) AS n FROM raw.edgar_filings
           WHERE form = '8-K' AND items LIKE '%2.02%' AND year(accepted_et) BETWEEN 2016 AND 2025
           GROUP BY 1 ORDER BY 1"""
    ).df()
    con.close()
    df["share"] = (df.n / df.n.sum()).round(3)
    print(df.to_string(index=False))
    return df


def day0(accepted: pd.Series, cal: xc.ExchangeCalendar) -> pd.Series:
    """Trading session in which the news can first trade: same session if accepted before 16:00 ET
    on a session day (pre-open filings trade at that day's open), otherwise the next session."""
    sessions = cal.sessions
    out = []
    for ts in accepted:
        if pd.isna(ts):
            out.append(pd.NaT)
            continue
        d = pd.Timestamp(ts.date())
        if cal.is_session(d) and ts.time() < MARKET_CLOSE:
            out.append(d)
        else:
            idx = sessions.searchsorted(d, side="right")
            out.append(sessions[idx] if idx < len(sessions) else pd.NaT)
    return pd.Series(out, index=accepted.index, dtype="datetime64[ns]")


def build_events(db: str, start: str = "2016-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    con = duckdb.connect(db)
    universe = con.execute(
        """SELECT DISTINCT f.cik FROM raw.edgar_filings f JOIN raw.edgar_companies c USING (cik)
           WHERE f.form IN ('10-K', '10-Q') AND f.filing_date BETWEEN ? AND ?
             AND coalesce(c.entity_type, 'operating') = 'operating'""",
        [start, end],
    ).df()
    con.register("universe", universe)
    eightk = con.execute(
        """SELECT cik, accession, filing_date, accepted_et, items FROM raw.edgar_filings
           WHERE form = '8-K' AND filing_date BETWEEN ? AND ? AND cik IN (SELECT cik FROM universe)""",
        [start, end],
    ).df()
    target_forms = con.execute(
        f"""SELECT cik, form, filing_date FROM raw.edgar_filings
            WHERE form IN ({",".join("'" + f + "'" for f in TARGET_FORMS)})
              AND filing_date BETWEEN ? AND CAST(? AS DATE) + INTERVAL {TARGET_WINDOW_DAYS} DAY""",
        [start, end],
    ).df()

    items = eightk["items"].fillna("").str.split(",")
    has = lambda code: items.apply(lambda xs: code in [x.strip() for x in xs])  # noqa: E731
    eightk["i101"], eightk["i202"] = has("1.01"), has("2.02")
    eightk["i701_801"] = has("7.01") | has("8.01")

    events = []
    e = eightk[eightk.i202]
    events.append(e.assign(event_type="earnings", evidence=""))

    # first target-side filing per deal: collapse filings within the window into one deal per filer
    tf = target_forms.sort_values(["cik", "filing_date"])
    tf["gap"] = tf.groupby("cik").filing_date.diff().dt.days
    tf = tf[(tf.gap.isna()) | (tf.gap > TARGET_WINDOW_DAYS)]
    deals = eightk[eightk.i101].merge(tf, on="cik", suffixes=("", "_t"))
    lag = (pd.to_datetime(deals.filing_date_t) - pd.to_datetime(deals.filing_date)).dt.days
    deals = deals[(lag >= 0) & (lag <= TARGET_WINDOW_DAYS)]
    deals = deals.sort_values(["cik", "filing_date_t", "accepted_et"]).drop_duplicates(["cik", "filing_date_t"])
    events.append(deals.assign(
        event_type="acquisition_target",
        evidence=deals.form + " " + pd.to_datetime(deals.filing_date_t).dt.strftime("%Y-%m-%d"),
    ))

    o = eightk[eightk.i701_801 & ~eightk.i101 & ~eightk.i202]
    events.append(o.assign(event_type="other_material_candidate", evidence=""))

    ev = pd.concat(events, ignore_index=True)[
        ["cik", "accession", "filing_date", "accepted_et", "items", "event_type", "evidence"]
    ]
    cal = xc.get_calendar("XNYS", start="2015-01-01")
    ev["day0"] = day0(pd.to_datetime(ev.accepted_et), cal)
    ev = ev.drop_duplicates(["accession", "event_type"]).sort_values(["day0", "cik"])
    ev.insert(0, "event_id", ev.event_type.str[:3] + "-" + ev.accession)

    con.execute("CREATE SCHEMA IF NOT EXISTS events")
    con.register("ev", ev)
    con.execute("CREATE OR REPLACE TABLE events.announcements AS SELECT * FROM ev")
    summary = con.execute(
        """SELECT event_type, year(day0) AS year, count(*) AS n FROM events.announcements
           GROUP BY 1, 2 ORDER BY 1, 2"""
    ).df()
    con.close()
    print(summary.pivot(index="year", columns="event_type", values="n").to_string())
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--dest", default="data/edgar/submissions.zip")
    ld = sub.add_parser("load")
    ld.add_argument("--zip", default="data/edgar/submissions.zip")
    ld.add_argument("--db", default=EDGAR)
    t = sub.add_parser("check-tz")
    t.add_argument("--db", default=EDGAR)
    e = sub.add_parser("events")
    e.add_argument("--db", default=EDGAR)
    e.add_argument("--start", default="2016-01-01")
    e.add_argument("--end", default="2025-12-31")
    a = ap.parse_args()
    if a.cmd == "download":
        print(f"Saved {download(a.dest)}")
    elif a.cmd == "load":
        load(a.zip, a.db)
    elif a.cmd == "check-tz":
        check_tz(a.db)
    else:
        build_events(a.db, a.start, a.end)


if __name__ == "__main__":
    main()
