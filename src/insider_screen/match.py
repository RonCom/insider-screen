"""Match traded events extracted from SEC releases to announcement events (spec, "Labels" step 3).

1. Issuer name -> CIK: exact match on a normalized name (current or former EDGAR name), else the
   best fuzzy match (rapidfuzz token_set_ratio) at or above MIN_SCORE among filers with events.
2. CIK + announcement date -> event: the event for that CIK whose day 0 is within +-3 trading
   sessions of the announcement date, preferring the same event type, then the smallest gap.
3. No verified announcement date but a verified last trade date: the first event for that CIK with
   day 0 after the last trade and within TRADE_WINDOW_DAYS calendar days, preferring the same type.
Unmatched releases are kept with a reason so the miss rate can be reported.

Usage:
    uv run python -m insider_screen.match --db data/releases.duckdb --edgar-db data/edgar.duckdb --model qwen3:8b
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb
import exchange_calendars as xc
import pandas as pd
from rapidfuzz import fuzz, process

from insider_screen.db import EDGAR, RELEASES
from insider_screen.extract import DEFAULT_MODEL, TABLE, model_key

MIN_SCORE = 92
MAX_SESSIONS = 3
TRADE_WINDOW_DAYS = 30
SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "holdings",
    "holding", "group", "lp", "llc", "sa", "nv", "ag", "the", "de", "class", "a", "b",
}


def normalize(name: str | None) -> str:
    if not name:
        return ""
    s = name.lower()
    s = re.sub(r"/[a-z]{2,3}/?", " ", s)          # EDGAR state tags: "APPLE INC /CA/"
    s = re.sub(r"\bd/?b/?a\b.*$", " ", s)          # "doing business as" tails
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    words = [w for w in s.split() if w not in SUFFIXES]
    return " ".join(words)


def company_lookup(companies: pd.DataFrame) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for cik, name, former in companies[["cik", "name", "former_names"]].itertuples(index=False):
        for n in [name, *(former or "").split("|")]:
            key = normalize(n)
            if key and key not in lookup:
                lookup[key] = int(cik)
    return lookup


def resolve_cik(issuer: str, lookup: dict[str, int], keys: list[str]) -> tuple[int | None, str, float]:
    key = normalize(issuer)
    if not key:
        return None, "no_name", 0.0
    if key in lookup:
        return lookup[key], "exact", 100.0
    hit = process.extractOne(key, keys, scorer=fuzz.token_set_ratio)
    if hit and hit[1] >= MIN_SCORE:
        return lookup[hit[0]], "fuzzy", float(hit[1])
    return None, "no_company", float(hit[1]) if hit else 0.0


TYPE_MAP = {"acquisition_target": "acquisition_target", "earnings": "earnings"}


def match(traded: pd.DataFrame, companies: pd.DataFrame, events: pd.DataFrame,
          cal: xc.ExchangeCalendar) -> pd.DataFrame:
    with_events = companies[companies.cik.isin(events.cik.unique())]
    lookup = company_lookup(with_events)
    keys = list(lookup)
    sessions = cal.sessions
    pos = {d: i for i, d in enumerate(sessions)}
    ev_by_cik = {cik: g for cik, g in events.groupby("cik")}

    def session_index(d: pd.Timestamp) -> int:
        return int(sessions.searchsorted(d))  # non-session dates map to the next session

    rows = []
    for t in traded.itertuples(index=False):
        cik, method, score = resolve_cik(t.issuer_name, lookup, keys)
        base = {"lr_no": t.lr_no, "issuer_name": t.issuer_name, "announcement_date": t.announcement_date,
                "release_event_type": t.event_type, "cik": cik, "name_method": method, "name_score": score,
                "event_id": None, "event_type": None, "session_gap": None, "date_source": None, "reason": None}
        if cik is None:
            rows.append({**base, "reason": method})
            continue
        cands = ev_by_cik.get(cik)
        if cands is None:
            rows.append({**base, "reason": "no_event_for_cik"})
            continue
        want = TYPE_MAP.get(t.event_type)
        if pd.isna(t.announcement_date):
            last = getattr(t, "last_trade_date", None)
            if last is None or pd.isna(last):
                rows.append({**base, "reason": "no_dates"})
                continue
            last = pd.Timestamp(last)
            after = cands[(cands.day0 > last) & (cands.day0 <= last + pd.Timedelta(days=TRADE_WINDOW_DAYS))]
            if after.empty:
                rows.append({**base, "reason": "no_event_after_trades"})
                continue
            after = after.assign(type_ok=(after.event_type == want) if want else False)
            best = after.sort_values(["type_ok", "day0"], ascending=[False, True]).iloc[0]
            rows.append({**base, "event_id": best.event_id, "event_type": best.event_type,
                         "session_gap": None, "date_source": "last_trade_date"})
            continue
        ai = session_index(pd.Timestamp(t.announcement_date))
        cands = cands.assign(gap=[pos[pd.Timestamp(d)] - ai for d in cands.day0])
        cands = cands[cands.gap.abs() <= MAX_SESSIONS]
        if cands.empty:
            rows.append({**base, "reason": "no_event_in_window"})
            continue
        cands = cands.assign(type_ok=(cands.event_type == want) if want else False,
                             absgap=cands.gap.abs())
        best = cands.sort_values(["type_ok", "absgap"], ascending=[False, True]).iloc[0]
        rows.append({**base, "event_id": best.event_id, "event_type": best.event_type,
                     "session_gap": int(best.gap), "date_source": "announcement_date"})
    return pd.DataFrame(rows)


def run(db: str, model: str, edgar_db: str = EDGAR) -> pd.DataFrame:
    con = duckdb.connect(db)
    con.execute(f"ATTACH '{Path(edgar_db).as_posix()}' AS edgar (READ_ONLY)")
    traded = con.execute(
        f"""SELECT DISTINCT lr_no, issuer_name, announcement_date, last_trade_date, event_type FROM {TABLE}
           WHERE model = ? AND is_insider_trading_case""",
        [model_key(model)],
    ).df()
    companies = con.execute("SELECT cik, name, former_names FROM edgar.raw.edgar_companies").df()
    events = con.execute("SELECT event_id, cik, event_type, day0 FROM edgar.events.announcements").df()
    cal = xc.get_calendar("XNYS", start="2005-01-01")
    out = match(traded, companies, events, cal)
    con.execute("CREATE SCHEMA IF NOT EXISTS labels")
    con.register("out", out)
    con.execute("CREATE OR REPLACE TABLE labels.release_event_matches AS SELECT * FROM out")
    con.execute(
        """CREATE OR REPLACE TABLE labels.charged_events AS
           SELECT a.*, m.event_id IS NOT NULL AS is_charged, m.lr_numbers
           FROM edgar.events.announcements a
           LEFT JOIN (SELECT event_id, string_agg(DISTINCT CAST(lr_no AS VARCHAR), '|') AS lr_numbers
                      FROM labels.release_event_matches WHERE event_id IS NOT NULL GROUP BY 1) m
           USING (event_id)"""
    )
    con.close()
    summary = out.assign(outcome=out.reason.fillna("matched")).outcome.value_counts()
    print(summary.to_string())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=RELEASES)
    ap.add_argument("--edgar-db", default=EDGAR)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    run(a.db, a.model, a.edgar_db)


if __name__ == "__main__":
    main()
