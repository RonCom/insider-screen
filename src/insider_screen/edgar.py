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

Timestamps: acceptanceDateTime ends in "Z", but `tz-sample` (a sample checked against each filing's
index header, which gives Eastern time) found it runs 0, 1 or 2 times the UTC offset ahead of Eastern
time, in both the recent block and the continuation files. `events` keeps the earliest day 0 that fits
EDGAR's hours and the filing date, so an ambiguous time never puts the announcement inside the
pre-event window. `exact-times` reads the index header for every event of one type (acquisition
targets by default); `events` then uses those exact times.

Usage:
    uv run python -m insider_screen.edgar download
    uv run python -m insider_screen.edgar load
    uv run python -m insider_screen.edgar check-tz
    uv run python -m insider_screen.edgar tz-sample
    uv run python -m insider_screen.edgar events
    uv run python -m insider_screen.edgar exact-times
    uv run python -m insider_screen.edgar events
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime, time
from functools import lru_cache
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


def _rows(block: dict, cik: int, src: str) -> list[tuple]:
    cols = [block.get(k, []) for k in FIELDS]
    out = []
    for acc, fdate, accepted, form, items in zip(*cols):
        if form in KEEP_FORMS:
            out.append((cik, acc, fdate, accepted, form, items or "", src))
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
        return company, _rows(data["filings"].get("recent", {}), cik, "recent")
    return None, _rows(data, cik, "file")


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
    fdf = pd.DataFrame(filings, columns=["cik", "accession", "filing_date", "accepted_raw", "form", "items", "src"])
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.register("cdf", cdf)
    con.register("fdf", fdf)
    con.execute("CREATE OR REPLACE TABLE raw.edgar_companies AS SELECT * FROM cdf")
    con.execute(
        """CREATE OR REPLACE TABLE raw.edgar_filings AS
           SELECT DISTINCT cik, accession, CAST(filing_date AS DATE) AS filing_date,
                  -- wall-clock value as given, trailing Z dropped; tz-sample decides how to read it per src
                  TRY_CAST(replace(replace(accepted_raw, 'T', ' '), '.000Z', '') AS TIMESTAMP) AS accepted_json,
                  form, items, src
           FROM fdf"""
    )
    n = con.execute("SELECT count(*) FROM raw.edgar_filings").fetchone()[0]
    con.close()
    print(f"{len(cdf):,} filers, {n:,} filings kept -> {db}")


def check_tz(db: str) -> pd.DataFrame:
    """Hour-of-day histogram for earnings 8-Ks. Read as Eastern, expect peaks at 16-17 and 6-9."""
    con = duckdb.connect(db, read_only=True)
    df = con.execute(
        """SELECT hour(accepted_json) AS hour, count(*) AS n FROM raw.edgar_filings
           WHERE form = '8-K' AND items LIKE '%2.02%' AND year(accepted_json) BETWEEN 2016 AND 2025
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


HEADER_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{acc}-index-headers.html"
ACCEPT_RE = re.compile(r"ACCEPTANCE-DATETIME>\s*(\d{14})")
ERAS = [("2016-01-01", "2018-12-31"), ("2019-01-01", "2021-12-31"), ("2022-01-01", "2025-12-31")]


def tz_sample(db: str, per_cell: int = 15, seed: int = 42) -> pd.DataFrame:
    """Compare the JSON timestamp with the ACCEPTANCE-DATETIME in each filing's index header (Eastern
    time) for a sample of 8-Ks per source (recent block vs continuation files) and era. Stores the
    offsets in raw.edgar_tz_check; build_events reads them to convert each source to Eastern time."""
    from insider_screen.http import PoliteClient

    con = duckdb.connect(db)
    parts = []
    for src in ("recent", "file"):
        for lo, hi in ERAS:
            parts.append(con.execute(
                f"""SELECT cik, accession, accepted_json, src, ? AS era FROM raw.edgar_filings
                    WHERE form = '8-K' AND src = ? AND filing_date BETWEEN ? AND ?
                    ORDER BY hash(accession || '{int(seed)}') LIMIT {int(per_cell)}""",
                [f"{lo[:4]}-{hi[:4]}", src, lo, hi]).df())
    sample = pd.concat(parts, ignore_index=True)
    client = PoliteClient(cache_dir="data/cache/sec")
    offsets = []
    for r in sample.itertuples(index=False):
        url = HEADER_URL.format(cik=int(r.cik), folder=r.accession.replace("-", ""), acc=r.accession)
        status, body = client.get(url)
        m = ACCEPT_RE.search(body.decode("latin-1")) if status == 200 else None
        if not m or pd.isna(r.accepted_json):
            offsets.append(None)
            continue
        header = pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%d%H%M%S"))
        offsets.append(round((pd.Timestamp(r.accepted_json) - header).total_seconds() / 3600, 2))
    sample["offset_hours"] = offsets
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.register("s", sample)
    con.execute("CREATE OR REPLACE TABLE raw.edgar_tz_check AS SELECT * FROM s")
    con.close()
    table = sample.groupby(["src", "era"]).offset_hours.agg(
        lambda x: ", ".join(f"{k:g}h x{v}" for k, v in x.value_counts(dropna=False).sort_index().items()))
    print("JSON time minus header time (Eastern), by source and era:")
    print(table.to_string())
    return sample


EDGAR_OPEN, EDGAR_CLOSE = time(6, 0), time(22, 0)
FILING_CUTOFF = time(17, 30)  # accepted after 17:30 ET -> filing date is the next business day


@lru_cache(maxsize=None)
def _offset_for_date(d) -> int:
    noon = pd.Timestamp(d) + pd.Timedelta(hours=12)
    return int(-noon.tz_localize("America/New_York").utcoffset().total_seconds() // 3600)


def _ny_offset_hours(ts: pd.Timestamp) -> int:
    """Hours Eastern time is behind UTC on that date: 4 (daylight) or 5 (standard)."""
    return _offset_for_date(ts.date())


def _day0_one(ts: pd.Timestamp, sessions: pd.DatetimeIndex, session_set: set) -> pd.Timestamp:
    d = pd.Timestamp(ts.date())
    if d in session_set and ts.time() < MARKET_CLOSE:
        return d
    i = sessions.searchsorted(d, side="right")
    return sessions[i] if i < len(sessions) else pd.NaT


def _filing_date_for(et: pd.Timestamp, sessions: pd.DatetimeIndex) -> pd.Timestamp:
    d = pd.Timestamp(et.date())
    if et.time() <= FILING_CUTOFF and d in sessions:
        return d
    return sessions[sessions.searchsorted(d, side="right")]


def resolve_times(json_ts: pd.Series, filing_date: pd.Series, cal: xc.ExchangeCalendar) -> pd.DataFrame:
    """Candidate Eastern times for each filing: JSON time minus 0, 1 or 2 times the UTC offset (the
    tz-sample check found all three). Candidates outside EDGAR hours (06:00-22:00 ET) or inconsistent
    with the filing date are dropped. Returns the earliest surviving candidate, its day 0, and how many
    distinct day-0 values survived. Taking the earliest keeps the announcement out of the pre-event
    window when the time is ambiguous; it can cost the last pre-event day."""
    sessions = cal.sessions
    session_set = set(sessions)
    rows = []
    for j, f in zip(pd.to_datetime(json_ts), pd.to_datetime(filing_date)):
        if pd.isna(j):
            rows.append((pd.NaT, pd.NaT, 0))
            continue
        off = _ny_offset_hours(j)
        cands = [j - pd.Timedelta(hours=k * off) for k in (0, 1, 2)]
        ok = [c for c in cands if EDGAR_OPEN <= c.time() <= EDGAR_CLOSE]
        both = [c for c in ok if pd.isna(f) or _filing_date_for(c, sessions) == pd.Timestamp(f)]
        keep = both or ok or cands
        d0 = [_day0_one(c, sessions, session_set) for c in keep]
        i = min(range(len(keep)), key=lambda k: d0[k])
        rows.append((keep[i], d0[i], len(set(d0))))
    return pd.DataFrame(rows, columns=["accepted_et", "day0", "day0_candidates"], index=json_ts.index)


def fetch_exact_times(db: str, event_type: str = "acquisition_target") -> None:
    """Acceptance time from each event filing's index header (Eastern time) into raw.edgar_exact_times.
    Rerun `events` afterwards to use them. Already-fetched filings are skipped."""
    from insider_screen.http import PoliteClient

    con = duckdb.connect(db)
    con.execute("CREATE TABLE IF NOT EXISTS raw.edgar_exact_times (accession VARCHAR, cik BIGINT, accepted_et TIMESTAMP)")
    todo = con.execute(
        """SELECT DISTINCT cik, accession FROM events.announcements WHERE event_type = ?
           AND accession NOT IN (SELECT accession FROM raw.edgar_exact_times)""", [event_type]).fetchall()
    print(f"{len(todo)} {event_type} filings to look up")
    client = PoliteClient(cache_dir="data/cache/sec")
    missing = 0
    for i, (cik, acc) in enumerate(todo, 1):
        status, body = client.get(HEADER_URL.format(cik=int(cik), folder=acc.replace("-", ""), acc=acc))
        m = ACCEPT_RE.search(body.decode("latin-1")) if status == 200 else None
        if not m:
            missing += 1
            continue
        con.execute("INSERT INTO raw.edgar_exact_times VALUES (?, ?, ?)",
                    [acc, int(cik), datetime.strptime(m.group(1), "%Y%m%d%H%M%S")])
        if i % 500 == 0:
            print(f"  {i}/{len(todo)}")
    con.close()
    print(f"Done; {missing} headers not found")


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
        """SELECT cik, accession, filing_date, accepted_json, src, items FROM raw.edgar_filings
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
    deals = deals.sort_values(["cik", "filing_date_t", "accepted_json"]).drop_duplicates(["cik", "filing_date_t"])
    events.append(deals.assign(
        event_type="acquisition_target",
        evidence=deals.form + " " + pd.to_datetime(deals.filing_date_t).dt.strftime("%Y-%m-%d"),
    ))

    o = eightk[eightk.i701_801 & ~eightk.i101 & ~eightk.i202]
    events.append(o.assign(event_type="other_material_candidate", evidence=""))

    ev = pd.concat(events, ignore_index=True)[
        ["cik", "accession", "filing_date", "accepted_json", "items", "event_type", "evidence"]
    ]
    cal = xc.get_calendar("XNYS", start="2015-01-01")
    res = resolve_times(ev.accepted_json, ev.filing_date, cal)
    ev["accepted_et"], ev["day0"], ev["day0_candidates"] = res.accepted_et, res.day0, res.day0_candidates
    ev["day0_basis"] = "earliest_candidate"
    try:
        exact = con.execute("SELECT accession, accepted_et AS header_et FROM raw.edgar_exact_times").df()
    except duckdb.CatalogException:
        exact = pd.DataFrame(columns=["accession", "header_et"])
    if len(exact):
        ev = ev.merge(exact, on="accession", how="left")
        has = ev.header_et.notna()
        ev.loc[has, "accepted_et"] = pd.to_datetime(ev.loc[has, "header_et"])
        ev.loc[has, "day0"] = day0(pd.to_datetime(ev.loc[has, "header_et"]), cal).values
        ev.loc[has, "day0_candidates"] = 1
        ev.loc[has, "day0_basis"] = "index_header"
        ev = ev.drop(columns="header_et")
    ev = ev.drop_duplicates(["accession", "event_type"]).sort_values(["day0", "cik"])
    ev.insert(0, "event_id", ev.event_type.str[:3] + "-" + ev.accession)

    con.execute("CREATE SCHEMA IF NOT EXISTS events")
    con.register("ev", ev)
    con.execute("CREATE OR REPLACE TABLE events.announcements AS SELECT * FROM ev")
    summary = con.execute(
        """SELECT event_type, year(day0) AS year, count(*) AS n FROM events.announcements
           GROUP BY 1, 2 ORDER BY 1, 2"""
    ).df()
    basis = con.execute(
        """SELECT event_type, day0_basis, (day0_candidates > 1) AS ambiguous, count(*) AS n
           FROM events.announcements GROUP BY ALL ORDER BY ALL""").df()
    con.close()
    print(summary.pivot(index="year", columns="event_type", values="n").to_string())
    print("\nDay 0 basis (ambiguous = more than one possible day 0; the earliest is used):")
    print(basis.to_string(index=False))
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
    ts = sub.add_parser("tz-sample")
    ts.add_argument("--db", default=EDGAR)
    xt = sub.add_parser("exact-times")
    xt.add_argument("--db", default=EDGAR)
    xt.add_argument("--event-type", default="acquisition_target")
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
    elif a.cmd == "tz-sample":
        tz_sample(a.db)
    elif a.cmd == "exact-times":
        fetch_exact_times(a.db, a.event_type)
    else:
        build_events(a.db, a.start, a.end)


if __name__ == "__main__":
    main()
