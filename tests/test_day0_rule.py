from datetime import date

import numpy as np
import pandas as pd

from insider_screen import day0_rule as r

SESS = pd.bdate_range("2022-01-03", periods=300).date
S = SESS[-1]


def frame(jumps: dict[int, float] | None = None, vol: dict[int, float] | None = None, seed: int = 1):
    """SPY and a stock with 1% daily noise; jumps {sessions before S: return} with 10x volume by default."""
    rng = np.random.default_rng(seed)
    n = len(SESS)
    spy_ret = rng.normal(0, 0.008, n)
    ret = spy_ret + rng.normal(0, 0.01, n)
    volume = np.full(n, 1e6)
    for k, j in (jumps or {}).items():
        ret[n - 1 - k] = j + spy_ret[n - 1 - k]
        volume[n - 1 - k] = 1e7
    for k, v in (vol or {}).items():
        volume[n - 1 - k] = v
    return pd.DataFrame({"close": 20 * np.cumprod(1 + ret), "spy_close": 400 * np.cumprod(1 + spy_ret),
                         "volume": volume}, index=SESS)


def test_move_one_session_before_8k():
    got = r.choose_day0(frame({1: 0.30}), S)
    assert (got.day0, got.basis, got.shift) == (SESS[-2], "announcement_move", 1)


def test_move_on_8k_session_confirms():
    got = r.choose_day0(frame({0: 0.25}), S)
    assert (got.day0, got.basis) == (S, "8k_confirmed")


def test_no_move_keeps_8k_day0():
    got = r.choose_day0(frame(), S)
    assert (got.day0, got.basis) == (S, "8k_no_move")


def test_move_outside_lookback_ignored():
    got = r.choose_day0(frame({3: 0.30}), S)
    assert got.basis == "8k_no_move" and got.day0 == S


def test_drift_and_thin_volume_dont_move_day0():
    # pre-announcement drift: +3% a day for two days, normal volume, then the 8-K-day jump
    got = r.choose_day0(frame({2: 0.03, 1: 0.03, 0: 0.25}, vol={2: 1e6, 1: 1e6}), S)
    assert (got.day0, got.basis) == (S, "8k_confirmed")
    # a big move on ordinary volume
    assert r.choose_day0(frame({1: 0.30}, vol={1: 1.2e6}), S).basis == "8k_no_move"


def test_reversed_spike_ignored():
    got = r.choose_day0(frame({1: 0.30, 0: -0.25}, vol={0: 1e6}), S)
    assert got.basis == "8k_no_move"


def test_short_history():
    assert r.choose_day0(frame().iloc[-40:], S).basis == "8k_no_data"


def test_daily_command(tmp_path):
    import exchange_calendars as xc
    import httpx

    from insider_screen import day0check
    from insider_screen.prices import Alpaca
    cal = xc.get_calendar("XNYS", start="2021-01-01")
    sess = [d.date() for d in cal.sessions_window(pd.Timestamp("2023-11-02"), -300)]
    rng = np.random.default_rng(2)
    n = len(sess)
    spy = 400 * np.cumprod(1 + rng.normal(0, 0.008, n))
    ret = rng.normal(0, 0.01, n)
    ret[-2] = 0.30  # Nov 1: the deal news, a session before the 8-K's day 0
    stock = 20 * np.cumprod(1 + ret)
    vol = np.full(n, 1e6)
    vol[-2] = 1e7

    def bar(d, c, v):
        return {"t": f"{d}T04:00:00Z", "o": c, "h": c, "l": c, "c": c, "v": v, "n": 1, "vw": c}

    def handler(request):
        assert request.url.params["adjustment"] == "all"
        return httpx.Response(200, json={"bars": {
            "SIX": [bar(d, c, v) for d, c, v in zip(sess, stock, vol)],
            "SPY": [bar(d, c, 1e8) for d, c in zip(sess, spy)]}, "next_page_token": None})
    api = Alpaca("k", "s", transport=httpx.MockTransport(handler), max_per_minute=100000)
    csv = tmp_path / "d.csv"
    pd.DataFrame({"company": ["Six Flags", "No Ticker Co"], "day0": ["2023-11-02", "2023-11-02"],
                  "ticker": ["SIX", ""], "market_move_et": ["2023-11-01 07:36", ""]}).to_csv(csv, index=False)
    out = day0check.daily(str(csv), api=api, cal=cal)
    assert out.day0_daily.tolist() == ["2023-11-01", ""]
    assert out.day0_daily_basis.tolist() == ["announcement_move", ""]
