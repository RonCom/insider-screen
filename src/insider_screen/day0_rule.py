"""Day 0 for acquisition targets from daily bars (spec change log, 2026-10-09).

The day-0 check failed: in 6 of 16 sampled targets the deal news moved the stock a session before the
8-K's day 0 (8-Ks filed after the close on announcement day, or the day after an after-hours release).
So day 0 starts from the 8-K session S and looks back LOOKBACK sessions for the announcement:

- abnormal return AR = stock return - SPY return, per session;
- baseline: sessions -250 to -31 before S (the spec's baseline window) give the AR standard deviation
  and the median volume;
- a session S-k (k = LOOKBACK..0) is the announcement if |AR| >= max(THRESHOLD, SIGMAS x sd), volume
  >= VOLUME_X x median, and the cumulative AR from that session through S keeps the same sign and at
  least half the threshold (the move held until the 8-K, so it wasn't a one-day spike);
- day 0 is the earliest such session; if none qualifies, it stays S.

The lookback is short and the bar is an announcement-sized jump on heavy volume, so drift from
pre-announcement trading (what the screen measures) doesn't move day 0. The 8-K day 0 is kept beside
the new one so results can be checked both ways.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

LOOKBACK = 2
THRESHOLD = 0.05
SIGMAS = 3
VOLUME_X = 3
BASELINE = (250, 31)  # sessions before S: from S-250 through S-31
MIN_BASELINE = 60


@dataclass
class Day0:
    day0: date
    basis: str  # announcement_move | 8k_confirmed | 8k_no_move | 8k_no_data
    shift: int  # sessions moved earlier (0 if none)
    note: str


def choose_day0(daily: pd.DataFrame, s: date) -> Day0:
    """`daily` is indexed by session date (ascending) with columns close, volume, spy_close and covers at
    least BASELINE[0] sessions before `s` through `s`."""
    d = daily.sort_index()
    if s not in d.index:
        return Day0(s, "8k_no_data", 0, f"no bar for {s}")
    ar = d.close.pct_change() - d.spy_close.pct_change()
    i = d.index.get_loc(s)
    base = slice(max(0, i - BASELINE[0]), max(0, i - BASELINE[1] + 1))
    base_ar, base_vol = ar.iloc[base].dropna(), d.volume.iloc[base]
    if len(base_ar) < MIN_BASELINE:
        return Day0(s, "8k_no_data", 0, f"{len(base_ar)} baseline sessions (need {MIN_BASELINE})")
    thr = max(THRESHOLD, SIGMAS * base_ar.std())
    med_vol = base_vol.median()
    for k in range(min(LOOKBACK, i), -1, -1):
        j = i - k
        a, vr = ar.iloc[j], d.volume.iloc[j] / med_vol if med_vol else np.nan
        cum = (1 + ar.iloc[j:i + 1]).prod() - 1
        if pd.isna(a) or abs(a) < thr or not vr >= VOLUME_X or np.sign(cum) != np.sign(a) or abs(cum) < thr / 2:
            continue
        note = f"AR {a:+.1%} on {vr:.0f}x volume, {cum:+.1%} through {s}; threshold {thr:.1%}"
        if k == 0:
            return Day0(s, "8k_confirmed", 0, note)
        return Day0(d.index[j], "announcement_move", k, note)
    return Day0(s, "8k_no_move", 0, f"no session from {d.index[i - min(LOOKBACK, i)]} to {s} met the rule; "
                                    f"threshold {thr:.1%}")


def daily_frame(bars: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Join one stock's adjusted daily bars with SPY's (rows from prices.fetch_bars: symbol, date, close, volume)."""
    stock = bars[bars.symbol == ticker].set_index("date")[["close", "volume"]]
    spy = bars[bars.symbol == "SPY"].set_index("date")[["close"]].rename(columns={"close": "spy_close"})
    return stock.join(spy, how="inner").sort_index()
