from datetime import date

import exchange_calendars as xc
import httpx
import numpy as np
import pandas as pd

from insider_screen import day0check
from insider_screen import market_move as mm
from insider_screen.prices import Alpaca

CAL = xc.get_calendar("XNYS", start="2023-01-01")
DAY0 = date(2023, 11, 2)
SESSIONS = [s.date() for s in CAL.sessions_window(pd.Timestamp(DAY0), -13)]


def bars(jump_at: str | None = None, to: float = 25.0, vol: float = 20000):
    """Flat $20 stock with small wiggles: regular minutes at volume 1000, a few pre-market minutes at 50.
    From `jump_at` on, the price is `to`, the jump minute trading `vol`."""
    rng = np.random.default_rng(0)
    idx = []
    for d in SESSIONS:
        idx += list(pd.date_range(f"{d} 07:00", f"{d} 07:30", freq="5min", tz=mm.EASTERN))
        idx += list(pd.date_range(f"{d} 09:30", f"{d} 15:59", freq="1min", tz=mm.EASTERN))
        idx += list(pd.date_range(f"{d} 16:30", f"{d} 17:30", freq="5min", tz=mm.EASTERN))
    if jump_at:
        idx = sorted(set(idx) | set(pd.date_range(jump_at, periods=20, freq="1min", tz=mm.EASTERN)))
    df = pd.DataFrame(index=pd.DatetimeIndex(idx))
    df["c"] = 20 * (1 + rng.normal(0, 0.001, len(df)))
    t = df.index.time
    df["v"] = np.where((t >= mm.OPEN) & (t < mm.CLOSE), 1000.0, 50.0)
    if jump_at:
        j = pd.Timestamp(jump_at, tz=mm.EASTERN)
        df.loc[df.index >= j, "c"] = to
        df.loc[j, "v"] = vol
    return df


def test_first_move_premarket_on_day0():
    t, note = mm.first_move(bars("2023-11-02 06:00"), SESSIONS, DAY0)
    assert t == pd.Timestamp("2023-11-02 06:00", tz=mm.EASTERN) and "+25.0%" in note


def test_first_move_after_hours_uses_same_day_close():
    t, _ = mm.first_move(bars("2023-11-01 16:05"), SESSIONS, DAY0)
    assert t == pd.Timestamp("2023-11-01 16:05", tz=mm.EASTERN)


def test_first_move_sessions_before_day0():
    t, _ = mm.first_move(bars("2023-10-31 11:00"), SESSIONS, DAY0)
    assert t.date() == date(2023, 10, 31)


def test_spike_that_reverts_or_thin_volume_is_ignored():
    df = bars("2023-11-02 08:00")
    df.loc[df.index > pd.Timestamp("2023-11-02 08:00", tz=mm.EASTERN), "c"] = 20.0  # one-minute spike
    assert mm.first_move(df, SESSIONS, DAY0)[0] is None
    t, note = mm.first_move(bars("2023-11-02 08:00", vol=100), SESSIONS, DAY0)
    assert t is None and "no move met the rules" in note


def test_ticker_from_text():
    text = ("Six Flags Entertainment Corporation (NYSE: SIX) and Cedar Fair, L.P. (NYSE: FUN) today announced...")
    assert mm.ticker_from_text(text, "Six Flags Entertainment Corp/OLD")[0] == "SIX"
    assert mm.ticker_from_text(text, "CEDAR FAIR L P")[0] == "FUN"
    assert mm.ticker_from_text("Fitbit, Inc. (NYSE: FIT) today announced", "FITBIT, INC.")[0] == "FIT"
    assert mm.ticker_from_text("Acme (Nasdaq: ACME) and Beta (NASDAQ:BETA)", "Gamma Corp")[0] is None
    assert mm.ticker_from_text("no exchange listed", "Acme")[0] is None


def test_market_command_end_to_end(tmp_path):
    df = bars("2023-11-01 16:05")

    def handler(request):
        p = request.url.params
        assert p["timeframe"] == "1Min" and p["symbols"] == "SIX"
        recs = [{"t": t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), "c": c, "v": v}
                for t, c, v in zip(df.index, df.c, df.v)]
        return httpx.Response(200, json={"bars": {"SIX": recs}, "next_page_token": None})
    api = Alpaca("k", "s", transport=httpx.MockTransport(handler), max_per_minute=100000)
    csv = tmp_path / "d.csv"
    pd.DataFrame({"event_id": ["e1"], "cik": ["1"], "company": ["Six Flags"], "accepted_et": ["2023-11-02 06:13:06"],
                  "day0": ["2023-11-02"], "filing_index": ["x"], "press_release_et": [""],
                  "ticker": ["SIX"]}).to_csv(csv, index=False)
    day0check.market(str(csv), api=api, sec=object(), cal=CAL)
    out = pd.read_csv(csv, dtype=str, encoding="utf-8-sig").fillna("")
    assert out.market_move_et.tolist() == ["2023-11-01 16:05"]
    scored = day0check.score(str(csv), "market_move_et")
    assert scored.session_diff.tolist() == [0]  # after the Nov 1 close: trades first on Nov 2, day 0
