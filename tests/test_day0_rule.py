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


def at(day, hhmm):
    return pd.Timestamp(f"{day} {hhmm}")


def test_late_8k_shifts_to_acceptance_day():
    # news moved the stock on S-1; the 8-K was accepted after that day's close
    got = r.choose_day0(frame({1: 0.30}), S, at(SESS[-2], "17:18"))
    assert (got.day0, got.basis, got.shift) == (SESS[-2], "late_8k_shift", 1)


def test_move_before_preopen_8k_is_flagged_not_used():
    # 8-K accepted before the open on S: a big move on S-1 isn't explained by a late filing (leak or tip)
    got = r.choose_day0(frame({1: 0.30}), S, at(S, "07:00"))
    assert (got.day0, got.basis, got.prior_moves) == (S, "8k_no_move", [SESS[-2]])


def test_move_two_sessions_back_never_moves_day0():
    got = r.choose_day0(frame({2: 0.30}), S, at(SESS[-2], "17:00"))
    assert (got.day0, got.prior_moves) == (S, [SESS[-3]])


def test_no_acceptance_time_no_shift():
    assert r.choose_day0(frame({1: 0.30}), S).day0 == S


def test_move_on_8k_session_confirms():
    got = r.choose_day0(frame({0: 0.25}), S, at(S, "07:00"))
    assert (got.day0, got.basis) == (S, "8k_confirmed")


def test_no_move_keeps_8k_day0():
    assert r.choose_day0(frame(), S, at(SESS[-2], "17:00")).basis == "8k_no_move"


def test_pre_announcement_trading_stays_in_the_window():
    # five days of insider-style buying (+2%/day on 3x volume), then the deal; 8-K after the close on the
    # announcement day. Day 0 moves to the announcement session; all five days stay inside days -20..-1.
    jumps = {k: 0.02 for k in range(2, 7)} | {1: 0.30}
    vol = {k: 3e6 for k in range(2, 7)}
    got = r.choose_day0(frame(jumps, vol), S, at(SESS[-2], "16:45"))
    assert got.day0 == SESS[-2]
    window = set(SESS[-2 - 20:-2])
    assert {SESS[-1 - k] for k in range(2, 7)} <= window
    # same buying with the 8-K before the open: day 0 stays on the 8-K session
    got = r.choose_day0(frame(jumps | {1: 0.0, 0: 0.30}, vol), S, at(S, "07:00"))
    assert (got.day0, got.basis) == (S, "8k_confirmed")


def test_day0_only_ever_moves_to_the_acceptance_day():
    rng = np.random.default_rng(7)
    for seed in range(60):
        k = int(rng.integers(0, 4))
        jumps = {k: float(rng.choice([0.3, -0.3, 0.06, 0.02]))}
        acc = at(SESS[-1 - int(rng.integers(0, 3))], rng.choice(["07:00", "12:00", "16:30", "19:00"]))
        got = r.choose_day0(frame(jumps, seed=seed), S, acc)
        assert got.shift in (0, 1)
        assert got.day0 == S or (got.day0 == acc.date() and acc.hour >= 16 and got.day0 == SESS[-2])


def test_drift_and_thin_volume_dont_qualify():
    got = r.choose_day0(frame({2: 0.03, 1: 0.03, 0: 0.25}, vol={2: 1e6, 1: 1e6}), S, at(SESS[-2], "17:00"))
    assert (got.day0, got.basis, got.prior_moves) == (S, "8k_confirmed", [])
    assert r.choose_day0(frame({1: 0.30}, vol={1: 1.2e6}), S, at(SESS[-2], "17:00")).basis == "8k_no_move"


def test_reversed_spike_ignored():
    got = r.choose_day0(frame({1: 0.30, 0: -0.25}, vol={0: 1e6}), S, at(SESS[-2], "17:00"))
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
    pd.DataFrame({"company": ["Six Flags", "Pre-open Co", "No Ticker Co"],
                  "day0": ["2023-11-02"] * 3, "accepted_et": ["2023-11-01 17:18:27", "2023-11-02 07:00:00", ""],
                  "ticker": ["SIX", "SIX", ""], "market_move_et": ["2023-11-01 07:36", "", ""]}).to_csv(csv, index=False)
    out = day0check.daily(str(csv), api=api, cal=cal)
    assert out.day0_daily.tolist() == ["2023-11-01", "2023-11-02", ""]
    assert out.day0_daily_basis.tolist() == ["late_8k_shift", "8k_no_move", ""]
    assert out.prior_moves.tolist() == ["", "2023-11-01", ""]


def test_shock_days_in_baseline_dont_raise_the_bar():
    # Paragon 28: six -30% days in the baseline year inflated a plain SD threshold to 15.6%,
    # so the +14% deal day after a late 8-K didn't count
    f = frame({1: 0.14})
    rets = f.close.pct_change().to_numpy(copy=True)
    for k in (60, 90, 120, 150, 180, 210):
        rets[len(f) - 1 - k] = -0.30
    f["close"] = 20 * np.cumprod(np.nan_to_num(1 + rets, nan=1.0))
    got = r.choose_day0(f, S, at(SESS[-2], "17:02"))
    assert (got.day0, got.basis) == (SESS[-2], "late_8k_shift")
    plain = 3 * (f.close.pct_change() - f.spy_close.pct_change()).iloc[-251:-30].std()
    assert plain > 0.14  # the old threshold would have missed it


def test_small_premium_on_heavy_volume_paragon28():
    """Paragon 28's real closes and volumes, Jan 21-30 2025, after a volatile baseline (3.7% daily SD):
    +8.7% abnormal on 24x volume the day after an after-hours deal release; 8-K after the close that day."""
    rng = np.random.default_rng(3)
    n = len(SESS)
    spy = 590 * np.cumprod(1 + rng.normal(0, 0.008, n))
    stock = 11 * np.cumprod(1 + rng.normal(0, 0.037, n) + (spy / np.roll(spy, 1) - 1) * 0)
    vol = np.full(n, 870_000.0)
    tail = [(11.35, 530730, 591.40), (11.69, 719200, 594.73), (11.69, 297607, 597.97), (11.62, 344903, 596.23),
            (11.67, 360104, 587.79), (12.00, 873893, 592.85), (12.99, 20937104, 590.19), (13.03, 3764805, 593.36)]
    for k, (c, v, sp) in enumerate(tail):
        j = n - len(tail) + k
        stock[j], vol[j], spy[j] = c, v, sp
    f = pd.DataFrame({"close": stock, "spy_close": spy, "volume": vol}, index=SESS)
    got = r.choose_day0(f, S, at(SESS[-2], "17:02"))
    assert (got.day0, got.basis) == (SESS[-2], "late_8k_shift")
    assert float(got.note.split("threshold ")[1].rstrip("%")) > 8.7  # the 3-SD bar alone would have missed it


def test_heavy_volume_path_keeps_its_limits():
    # below the 5% floor, even on 20x volume
    assert r.choose_day0(frame({1: 0.04}, vol={1: 2e7}), S, at(SESS[-2], "17:00")).basis == "8k_no_move"
    # volatile stock (3-SD bar ~9%): +7% on 5x volume clears neither path
    rng = np.random.default_rng(5)
    f = frame({1: 0.07}, vol={1: 5e6})
    noisy = f.close.pct_change().to_numpy(copy=True)
    noisy[1:-2] += rng.normal(0, 0.03, len(noisy) - 3)
    noisy[-2] = 0.07 + f.spy_close.pct_change().iloc[-2]
    f["close"] = 20 * np.cumprod(np.nan_to_num(1 + noisy, nan=1.0))
    assert r.choose_day0(f, S, at(SESS[-2], "17:00")).basis == "8k_no_move"


def test_apply_writes_target_day0_and_spot_check(tmp_path):
    import duckdb
    ref, edgar, prices, finra = (str(tmp_path / f) for f in
                                 ("reference.duckdb", "edgar.duckdb", "prices.duckdb", "finra.duckdb"))
    con = duckdb.connect(ref)
    con.execute("CREATE SCHEMA ref")
    con.execute("""CREATE TABLE ref.ticker_cik AS SELECT * FROM (VALUES ('ACME', 1::BIGINT, 'Acme', 'CS',
                   NULL::DATE, NULL::DATE)) t(ticker, cik, name, type, valid_from, valid_to)""")
    con.close()
    con = duckdb.connect(edgar)
    con.execute("CREATE SCHEMA events; CREATE SCHEMA raw")
    con.execute("CREATE TABLE raw.edgar_companies AS SELECT 1 AS cik, 'Acme Corp' AS name UNION ALL SELECT 2, 'Private Co'")
    con.execute(f"""CREATE TABLE events.announcements AS SELECT * FROM (VALUES
        ('late', 1, 'acquisition_target', TIMESTAMP '{S}', TIMESTAMP '{SESS[-2]} 16:30', '0001-22-000001'),
        ('none', 2, 'acquisition_target', TIMESTAMP '{S}', TIMESTAMP '{S} 08:00', '0001-22-000002'))
        t(event_id, cik, event_type, day0, accepted_et, accession)""")
    con.close()
    f = frame({1: 0.30})
    con = duckdb.connect(prices)
    con.execute("CREATE SCHEMA raw")
    bars = pd.concat([pd.DataFrame({"symbol": "ACME", "date": f.index, "close": f.close, "volume": f.volume}),
                      pd.DataFrame({"symbol": "SPY", "date": f.index, "close": f.spy_close, "volume": 1e8})])
    bars["adjustment"] = "all"
    con.register("bars", bars)
    con.execute("CREATE TABLE raw.alpaca_bars_daily AS SELECT symbol, CAST(date AS DATE) AS date, close, volume, adjustment FROM bars")
    con.close()
    out = r.apply(ref, edgar, prices, finra, sample_out=str(tmp_path / "s.csv"), dev_csv=str(tmp_path / "none.csv"))
    got = out.set_index("event_id")
    assert got.loc["late", "basis"] == "late_8k_shift" and got.loc["late", "day0"] == SESS[-2]
    assert got.loc["none", "basis"] == "no_ticker"
    sample = pd.read_csv(tmp_path / "s.csv")
    assert list(sample.event_id) == ["late"] and "0001-22-000001-index.htm" in sample.filing_index[0]
    con = duckdb.connect(edgar, read_only=True)
    assert con.execute("SELECT count(*) FROM events.target_day0").fetchone()[0] == 2
