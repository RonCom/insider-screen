"""Day-0 check from the market: the first minute the target's stock moved on the news (day0check market).

Newswire pages can't be read by script (their searches block or return nothing), so the check in the
spec's change log uses the market's reaction: for each sampled acquisition target, the first minute,
in extended hours or the regular session, from two sessions before day 0 through day 0, where
- the price is at least THRESHOLD (or 3 daily standard deviations, if larger) from the previous close,
- that minute's volume is at least VOLUME_X times the median regular-session minute volume of the
  baseline sessions before the window, and
- the median price of the next PERSIST minutes stays at least half the threshold away, same direction.
That minute marks when the news reached the market. Its trading session is compared with day 0 from
the 8-K; the scoring rule is the spec's (more than 5 of 50 a session or more apart -> switch).

The ticker comes from the press release in the 8-K ("(NYSE: SIX)"), matched to the company's name;
a ticker typed into the CSV is kept. Bars are Alpaca's free minute bars (SIP feed, from 2016).
"""

from __future__ import annotations

import re
from datetime import date, time

import exchange_calendars as xc
import pandas as pd
from rapidfuzz import fuzz

from insider_screen import press_release
from insider_screen.match import normalize
from insider_screen.prices import Alpaca

EASTERN = "America/New_York"
THRESHOLD = 0.05
VOLUME_X = 5
PERSIST = 15
BASELINE_SESSIONS = 10
WINDOW_SESSIONS = 2  # sessions before day 0 searched for an earlier move
OPEN, CLOSE = time(9, 30), time(16, 0)

# "(NYSE: SIX)", "(Nasdaq: FIT)", "(NASDAQ Capital Market: XXX)", "(NYSE American: XXX)", "(NasdaqGS: XXX)".
# The exchange name is matched in any case; the ticker only in capitals.
TICKER_RE = re.compile(
    r"\(?\b(?:NYSE|NASDAQ|AMEX)(?:[\s-]*(?:American|Arca|MKT|Global|Select|Capital|Stock|Market|GS|GM|CM)){0,4}"
    r"\s*:\s*\"?(?-i:([A-Z]{1,5}(?:\.[A-Z])?))\b", re.I)


def ticker_from_text(text: str, company: str) -> tuple[str | None, str]:
    """The ticker given in a press release for `company`. Each "(EXCHANGE: TICKER)" is scored by how well
    the 150 characters before it match the company's name; returns (ticker, how it was chosen).
    `company` may hold several names joined by "|" (current and former EDGAR names): EDGAR shows today's
    name, and a release from before a rename (Westar Energy, now Evergy Kansas Central) uses the old one."""
    hits = [(m.group(1).upper(), text[max(0, m.start() - 150):m.start()]) for m in TICKER_RE.finditer(text[:8000])]
    names = [n for n in (normalize(x) for x in company.split("|")) if n]
    shown = company.split("|")[0]
    if not hits:
        return None, "no ticker in the press release"
    if not names:
        return None, f"tickers {sorted({t for t, _ in hits})} but no company name to match"
    scored = sorted(((max(fuzz.partial_ratio(n, normalize(before)) for n in names), t) for t, before in hits),
                    reverse=True)
    best, ticker = scored[0]
    if best >= 80:
        return ticker, f"ticker next to the company name (match {best:.0f})"
    return None, f"tickers {sorted({t for t, _ in hits})} but none next to '{shown}'"


def fetch_minutes(api: Alpaca, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Minute bars (all hours) between two Eastern times, as a frame indexed by Eastern time."""
    rows, token = [], None
    while True:
        params = {"symbols": symbol, "timeframe": "1Min", "start": start.tz_convert("UTC").isoformat(),
                  "end": end.tz_convert("UTC").isoformat(), "adjustment": "raw", "feed": "sip", "limit": 10000}
        if token:
            params["page_token"] = token
        r = api.get(params)
        r.raise_for_status()
        body = r.json()
        rows += (body.get("bars") or {}).get(symbol, [])
        token = body.get("next_page_token")
        if not token:
            break
    if not rows:
        return pd.DataFrame(columns=["c", "v"])
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df.t, utc=True).dt.tz_convert(EASTERN)
    return df[["c", "v"]].astype(float).sort_index()


def first_move(bars: pd.DataFrame, sessions: list[date], day0: date) -> tuple[pd.Timestamp | None, str]:
    """First minute that meets the rules above, searching from the open of extended trading
    WINDOW_SESSIONS sessions before day 0 through day 0's after-hours. `sessions` are the trading days
    covered by `bars`, in order, ending with day 0."""
    if bars.empty:
        return None, "no minute bars"
    i0 = sessions.index(day0)
    window_start = sessions[max(0, i0 - WINDOW_SESSIONS)]
    base_days = set(sessions[:max(0, i0 - WINDOW_SESSIONS)])
    times = bars.index.time
    regular = (times >= OPEN) & (times < CLOSE)
    days = bars.index.date
    closes = bars[regular].groupby(days[regular]).c.last()
    base_vol = bars[regular & pd.Series(days, index=bars.index).isin(base_days).values].v.median()
    base_closes = closes[[d in base_days for d in closes.index]]
    if pd.isna(base_vol) or len(base_closes) < 3:
        return None, "too little trading before the window for a baseline"
    thr = max(THRESHOLD, 3 * base_closes.pct_change().std())
    prev_session = {d: sessions[k - 1] for k, d in enumerate(sessions) if k}
    win = bars[(days >= window_start) & (days <= day0)]
    for k, (t, row) in enumerate(win.iterrows()):
        ref_day = t.date() if t.time() >= CLOSE else prev_session.get(t.date())
        ref = closes.get(ref_day)
        if ref is None or ref <= 0:
            continue
        ret = row.c / ref - 1
        if abs(ret) < thr or row.v < VOLUME_X * base_vol:
            continue
        after = win.c.iloc[k + 1:k + 1 + PERSIST] / ref - 1
        if len(after) and (after.median() * (1 if ret > 0 else -1)) >= thr / 2:
            return t, (f"{ret:+.1%} from the {ref_day} close on {row.v / base_vol:.0f}x median minute volume; "
                       f"threshold {thr:.1%}")
    biggest = (win.c / win.index.map(lambda t: closes.get(
        t.date() if t.time() >= CLOSE else prev_session.get(t.date()), float("nan"))) - 1).abs().max()
    return None, f"no move met the rules (largest {biggest:.1%} vs threshold {thr:.1%})"


def locate(api: Alpaca, symbol: str, day0: date, cal: xc.ExchangeCalendar) -> tuple[pd.Timestamp | None, str]:
    sessions = [s.date() for s in cal.sessions_window(pd.Timestamp(day0), -(BASELINE_SESSIONS + WINDOW_SESSIONS + 1))]
    start = pd.Timestamp(f"{sessions[0]} 04:00", tz=EASTERN)
    end = pd.Timestamp(f"{day0} 20:00", tz=EASTERN)
    bars = fetch_minutes(api, symbol, start, end)
    return first_move(bars, sessions, day0)


def release_ticker(sec, index_url: str, filed: date | None, company: str) -> tuple[str | None, str]:
    """Ticker from the 8-K's press release (or a same-day filing's), for `company`."""
    status, body = press_release._get(sec, index_url)
    if status != 200:
        return None, f"filing index returned {status}"
    docs = [u for u in [press_release.exhibit_url(body, index_url)] if u]
    if filed is not None:
        cik, acc = press_release._accession_cik(index_url)
        docs += press_release.related_documents(sec, cik, acc, filed)
    docs += [u for u in [press_release.primary_doc_url(body, index_url)] if u and u not in docs]
    why = "no press release in this 8-K or same-day filings"
    for doc in docs:
        status, ex = press_release._get(sec, doc)
        if status != 200:
            continue
        ticker, reason = ticker_from_text(press_release.exhibit_text(ex), company)
        if ticker:
            return ticker, f"{reason}; {doc}"
        if reason != "no ticker in the press release" or why.startswith("no press release"):
            why = reason  # keep the most informative reason across documents
    return None, why
