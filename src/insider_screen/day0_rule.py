"""Day 0 for acquisition targets from daily bars (spec change log, 2026-10-09).

The day-0 check failed: in 6 of 16 sampled targets the deal news moved the stock a session before the
8-K's day 0. In every one, the 8-K was accepted after the close, and the move was in the session of that
same calendar day (news before or during the session, or after the prior close). The rule fixes exactly
that, and nothing else, because any other move before the 8-K may be the pre-announcement trading the
screen measures:

- abnormal return AR = stock return - SPY return, per session; baseline sessions -250 to -31 before S
  (the 8-K day 0) give the AR's typical size and the median volume. The typical size is a robust SD
  (1.4826 x median absolute deviation), so a few shock days in the baseline year don't raise the bar;
- a session qualifies as an announcement-sized move if either
    |AR| >= max(THRESHOLD, SIGMAS x sd) on volume >= VOLUME_X x median, or
    |AR| >= THRESHOLD on volume >= HEAVY_VOLUME_X x median (a small deal premium on a volatile stock:
    Paragon 28, +8.7% on 24x volume against an 11% 3-SD bar),
  and the cumulative AR from it through S keeps the same sign and at least half the bar it cleared
  (not a one-day spike);
- day 0 moves to the acceptance day's session only if the 8-K was accepted after the close (so S is the
  next session) and that session qualifies. It moves one session at most;
- qualifying sessions earlier in the LOOKBACK, or before an 8-K accepted during or before the session,
  never move day 0. They're returned in `prior_moves` and flagged, and stay inside the pre-event window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd

LOOKBACK = 2  # sessions before S searched for announcement-sized moves (flagged unless the shift applies)
THRESHOLD = 0.05
SIGMAS = 3
VOLUME_X = 3
HEAVY_VOLUME_X = 10
BASELINE = (250, 31)  # sessions before S: from S-250 through S-31
MIN_BASELINE = 60
MARKET_CLOSE = time(16, 0)


@dataclass
class Day0:
    day0: date
    basis: str  # late_8k_shift | 8k_confirmed | 8k_no_move | 8k_no_data | no_ticker
    shift: int  # sessions moved earlier (0 or 1)
    note: str
    prior_moves: list[date] = field(default_factory=list)  # announcement-sized moves before day 0, not used


def robust_sd(x: pd.Series) -> float:
    """1.4826 x median absolute deviation: the SD for normal data, not inflated by a few outliers."""
    return float(1.4826 * (x - x.median()).abs().median())


def choose_day0(daily: pd.DataFrame, s: date, accepted: pd.Timestamp | None = None) -> Day0:
    """`daily` is indexed by session date (ascending) with columns close, volume, spy_close and covers at
    least BASELINE[0] sessions before `s` through `s`. `accepted` is the 8-K's acceptance time (Eastern)."""
    d = daily.sort_index()
    if s not in d.index:
        return Day0(s, "8k_no_data", 0, f"no bar for {s}")
    ar = d.close.pct_change() - d.spy_close.pct_change()
    i = d.index.get_loc(s)
    base = slice(max(0, i - BASELINE[0]), max(0, i - BASELINE[1] + 1))
    base_ar, base_vol = ar.iloc[base].dropna(), d.volume.iloc[base]
    if len(base_ar) < MIN_BASELINE:
        return Day0(s, "8k_no_data", 0, f"{len(base_ar)} baseline sessions (need {MIN_BASELINE})")
    thr = max(THRESHOLD, SIGMAS * robust_sd(base_ar))
    med_vol = base_vol.median()

    def qualifies(j: int) -> tuple[bool, str]:
        a = ar.iloc[j]
        vr = d.volume.iloc[j] / med_vol if med_vol else np.nan
        cum = (1 + ar.iloc[j:i + 1]).prod() - 1
        if pd.isna(a) or pd.isna(vr):
            return False, f"{d.index[j]}: no data"
        if abs(a) >= thr and vr >= VOLUME_X:
            bar = thr
        elif abs(a) >= THRESHOLD and vr >= HEAVY_VOLUME_X:
            bar = THRESHOLD
        else:
            return False, ""
        ok = np.sign(cum) == np.sign(a) and abs(cum) >= bar / 2
        return ok, f"{d.index[j]}: AR {a:+.1%} on {vr:.0f}x volume, {cum:+.1%} through {s}"

    moves = {d.index[j]: qualifies(j) for j in range(max(0, i - LOOKBACK), i + 1)}
    found = [day for day, (ok, _) in moves.items() if ok]
    acc_day = accepted.date() if accepted is not None and not pd.isna(accepted) else None
    late = (acc_day is not None and acc_day in d.index and accepted.time() >= MARKET_CLOSE
            and i > 0 and d.index[i - 1] == acc_day)
    if late and acc_day in found:
        prior = [x for x in found if x < acc_day]
        return Day0(acc_day, "late_8k_shift", 1,
                    f"8-K accepted {accepted:%Y-%m-%d %H:%M}, after the close; {moves[acc_day][1]}; "
                    f"threshold {thr:.1%}", prior)
    prior = [x for x in found if x < s]
    flag = (f"; announcement-sized moves before day 0 kept in the window: "
            f"{'; '.join(moves[x][1] for x in prior)}") if prior else ""
    if s in found:
        return Day0(s, "8k_confirmed", 0, f"{moves[s][1]}; threshold {thr:.1%}{flag}", prior)
    return Day0(s, "8k_no_move", 0, f"no announcement-sized move on {s}; threshold {thr:.1%}{flag}", prior)


def daily_frame(bars: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Join one stock's adjusted daily bars with SPY's (rows from prices.fetch_bars: symbol, date, close, volume)."""
    stock = bars[bars.symbol == ticker].set_index("date")[["close", "volume"]]
    spy = bars[bars.symbol == "SPY"].set_index("date")[["close"]].rename(columns={"close": "spy_close"})
    return stock.join(spy, how="inner").sort_index()


SPOT_CHECK_N = 20


def apply(reference_db: str, edgar_db: str, prices_db: str, finra_db: str,
          sample_out: str = "data/day0_shift_check.csv", dev_csv: str = "data/day0_check.csv",
          n: int = SPOT_CHECK_N, seed: int = 11) -> pd.DataFrame:
    """Run the rule on every acquisition target with a ticker, from the local daily bars (all
    adjustments), and write events.target_day0 to the EDGAR file. Then draw `n` shifted events that
    weren't in the development sample for the spot check, to `sample_out`."""
    import duckdb

    from insider_screen import tickers
    from insider_screen.press_release import INDEX_URL

    con = duckdb.connect(reference_db, read_only=True)
    tickers.attach(con, edgar_db, finra_db)
    events = con.execute(f"""
        SELECT e.event_id, e.cik, c.name AS company, e.accession, e.accepted_et, CAST(e.day0 AS DATE) AS day0_8k,
               e.ticker, e.ticker_source
        FROM ({tickers.event_tickers_sql(con)}) e JOIN edgar.raw.edgar_companies c USING (cik)
        WHERE e.event_type = 'acquisition_target'""").df()
    has_audit = con.execute("""SELECT count(*) FROM duckdb_tables() WHERE database_name = 'edgar'
                               AND schema_name = 'events' AND table_name = 'target_audit'""").fetchone()[0]
    if has_audit:  # only events the target audit confirms as takeovers of the filer
        keep = {r[0] for r in con.execute(
            "SELECT event_id FROM edgar.events.target_audit WHERE in_target_set").fetchall()}
        print(f"Target audit: {len(keep)} of {len(events)} target events kept")
        events = events[events.event_id.isin(keep)]
    con.close()
    with_ticker = events[events.ticker.notna()]
    con = duckdb.connect(prices_db, read_only=True)
    con.register("syms", pd.DataFrame({"symbol": sorted(set(with_ticker.ticker) | {"SPY"})}))
    bars = con.execute("""SELECT b.symbol, b.date, b.close, b.volume FROM raw.alpaca_bars_daily b
                          JOIN syms USING (symbol) WHERE b.adjustment = 'all' ORDER BY b.symbol, b.date""").df()
    con.close()
    bars["date"] = pd.to_datetime(bars.date).dt.date
    spy = bars[bars.symbol == "SPY"].set_index("date")[["close"]].rename(columns={"close": "spy_close"})
    by_symbol = {s: g.set_index("date")[["close", "volume"]] for s, g in bars.groupby("symbol")}

    rows = []
    for e in events.itertuples(index=False):
        s = pd.Timestamp(e.day0_8k).date()
        if e.ticker is None or pd.isna(e.ticker):
            res = Day0(s, "no_ticker", 0, "no ticker for the company on day 0")
        elif e.ticker not in by_symbol:
            res = Day0(s, "8k_no_data", 0, f"no daily bars for {e.ticker}")
        else:
            d = by_symbol[e.ticker].join(spy, how="inner").sort_index()
            d = d[d.index <= s]
            accepted = pd.Timestamp(e.accepted_et) if not pd.isna(e.accepted_et) else None
            res = choose_day0(d, s, accepted)
        rows.append({"event_id": e.event_id, "ticker": e.ticker, "ticker_source": e.ticker_source,
                     "day0_8k": s, "day0": res.day0, "basis": res.basis, "shift": res.shift, "note": res.note,
                     "prior_moves": "|".join(map(str, res.prior_moves))})
    out = pd.DataFrame(rows)
    con = duckdb.connect(edgar_db)
    con.execute("CREATE SCHEMA IF NOT EXISTS events")
    con.register("out", out)
    con.execute("CREATE OR REPLACE TABLE events.target_day0 AS SELECT * FROM out")
    con.close()

    print(f"{len(out)} acquisition targets:")
    print(out.basis.value_counts().to_string())
    print(f"{(out.prior_moves != '').sum()} with announcement-sized moves before day 0, kept in the window")

    dev = set()
    if Path(dev_csv).exists():
        dev = set(pd.read_csv(dev_csv, dtype=str).get("event_id", pd.Series(dtype=str)))
    shifted = out[(out.basis == "late_8k_shift") & ~out.event_id.isin(dev)].merge(
        events[["event_id", "company", "cik", "accession", "accepted_et"]], on="event_id")
    pick = shifted.sample(n=min(n, len(shifted)), random_state=seed).copy()
    pick["filing_index"] = [INDEX_URL.format(cik=int(c), folder=a.replace("-", ""), acc=a)
                            for c, a in zip(pick.cik, pick.accession)]
    pick["release_date"] = ""
    pick["verdict"] = ""
    pick["notes"] = ""
    cols = ["event_id", "company", "ticker", "accepted_et", "day0_8k", "day0", "note", "filing_index",
            "release_date", "verdict", "notes"]
    Path(sample_out).parent.mkdir(parents=True, exist_ok=True)
    pick[cols].to_csv(sample_out, index=False, encoding="utf-8-sig")
    print(f"\nSpot check: {len(pick)} of {len(shifted)} shifted events (not in the development sample) -> "
          f"{sample_out}")
    return out


def main() -> None:
    import argparse

    from insider_screen.db import EDGAR, FINRA, PRICES, REFERENCE
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("apply", help="day 0 for every acquisition target, plus the shift spot-check sample")
    a.add_argument("--n", type=int, default=SPOT_CHECK_N)
    args = ap.parse_args()
    apply(REFERENCE, EDGAR, PRICES, FINRA, n=args.n)


if __name__ == "__main__":
    main()
