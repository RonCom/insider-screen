"""Day 0 for acquisition targets from daily bars (spec change log, 2026-10-09).

The day-0 check failed: in 6 of 16 sampled targets the deal news moved the stock a session before the
8-K's day 0. In every one, the 8-K was accepted after the close, and the move was in the session of that
same calendar day (news before or during the session, or after the prior close). The rule fixes exactly
that, and nothing else, because any other move before the 8-K may be the pre-announcement trading the
screen measures:

- abnormal return AR = stock return - SPY return, per session; baseline sessions -250 to -31 before S
  (the 8-K day 0) give the AR's typical size and the median volume. The typical size is a robust SD
  (1.4826 x median absolute deviation), so a few shock days in the baseline year don't raise the bar;
- a session qualifies as an announcement-sized move if |AR| >= max(THRESHOLD, SIGMAS x sd), volume
  >= VOLUME_X x median, and the cumulative AR from it through S keeps the same sign and at least half
  the threshold (not a one-day spike);
- day 0 moves to the acceptance day's session only if the 8-K was accepted after the close (so S is the
  next session) and that session qualifies. It moves one session at most;
- qualifying sessions earlier in the LOOKBACK, or before an 8-K accepted during or before the session,
  never move day 0. They're returned in `prior_moves` and flagged, and stay inside the pre-event window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time

import numpy as np
import pandas as pd

LOOKBACK = 2  # sessions before S searched for announcement-sized moves (flagged unless the shift applies)
THRESHOLD = 0.05
SIGMAS = 3
VOLUME_X = 3
BASELINE = (250, 31)  # sessions before S: from S-250 through S-31
MIN_BASELINE = 60
MARKET_CLOSE = time(16, 0)


@dataclass
class Day0:
    day0: date
    basis: str  # late_8k_shift | 8k_confirmed | 8k_no_move | 8k_no_data
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
        ok = not pd.isna(a) and abs(a) >= thr and vr >= VOLUME_X and np.sign(cum) == np.sign(a) and abs(cum) >= thr / 2
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
