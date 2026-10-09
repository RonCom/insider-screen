import math

import numpy as np
import pandas as pd
import pytest

from insider_screen import features as ft

SESS = list(pd.bdate_range("2020-01-01", periods=420).date)
DAY0 = SESS[380]


def data(seed=3):
    rng = np.random.default_rng(seed)
    n = len(SESS)
    m = rng.normal(0, 0.01, n)
    r = 0.0002 + 1.2 * m + rng.normal(0, 0.004, n)
    stock = pd.DataFrame({"close": 50 * np.cumprod(1 + r), "volume": np.full(n, 1e6)},
                         index=SESS)
    stock["close_raw"] = stock.close
    spy = pd.Series(300 * np.cumprod(1 + m), index=SESS)
    short = pd.DataFrame({"short_volume": rng.integers(3e5, 5e5, n), "total_volume": np.full(n, 1e6)}, index=SESS)
    trades = pd.DataFrame({"short_trades": rng.integers(800, 1200, n), "small_trades": rng.integers(400, 600, n)},
                          index=SESS)
    return stock, spy, short, trades


def same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        (isinstance(a[k], float) and math.isnan(a[k]) and math.isnan(b[k])) or a[k] == b[k] for k in a)


@pytest.mark.parametrize("window", list(ft.WINDOWS))
def test_no_feature_uses_day0_or_later(window):
    """Changing every row dated on or after day 0 must leave every feature unchanged."""
    stock, spy, short, trades = data()
    before = ft.event_features(DAY0, SESS, stock, spy, short, trades, window)
    late = [d >= DAY0 for d in SESS]
    s2, spy2, sh2, tr2 = stock.copy(), spy.copy(), short.copy(), trades.copy()
    s2.loc[late, :] *= 7
    spy2[late] *= 0.3
    sh2.loc[late, :] = 0
    tr2.loc[late, :] = 99999
    after = ft.event_features(DAY0, SESS + [pd.Timestamp("2030-01-02").date()], s2, spy2, sh2, tr2, window)
    assert same(before, after)
    assert {"abn_volume", "car", "short_share_abn", "abn_short_trades"} <= before.keys()


def test_features_pick_up_window_activity():
    stock, spy, short, trades = data()
    i0 = SESS.index(DAY0)
    win = SESS[i0 - 20:i0]
    stock.loc[win, "volume"] *= 3                            # triple volume
    stock.loc[SESS[i0 - 5:i0], "volume"] *= 2                # most of it late
    short.loc[win, "short_volume"] = 800_000                 # short share 0.8 vs 0.4
    k = stock.index.get_loc(win[0])
    stock.iloc[k:, stock.columns.get_loc("close")] *= 1.10   # +10% jump on day -20
    f = ft.event_features(DAY0, SESS, stock, spy, short, trades)
    assert f["abn_volume"] == pytest.approx(np.log(3 * 1.25), abs=0.08)
    assert f["last5_share"] == pytest.approx(10 / 25, abs=0.03)
    assert f["car"] == pytest.approx(0.10, abs=0.08) and f["scar"] > 1
    assert f["short_share_abn"] == pytest.approx(0.4, abs=0.02) and f["short_share_z"] > 10
    assert f["n_window"] == 20 and f["n_baseline"] == 220


def test_short_history_leaves_features_empty():
    stock, spy, short, trades = data()
    f = ft.event_features(SESS[80], SESS, stock, spy, short, trades)
    assert "abn_volume" not in f and f["n_baseline"] < ft.MIN_BASELINE


def test_build_targets_end_to_end(tmp_path):
    import duckdb
    edgar, ref, prices, finra, out = (str(tmp_path / f) for f in
                                      ("edgar.duckdb", "reference.duckdb", "prices.duckdb", "finra.duckdb", "event_features.duckdb"))
    cal_sessions = [s.date() for s in ft.xc.get_calendar("XNYS", start="2015-01-01").sessions_in_range("2019-01-02", "2021-06-30")]
    day0 = cal_sessions[400]
    con = duckdb.connect(edgar)
    con.execute("CREATE SCHEMA events; CREATE SCHEMA raw")
    con.execute(f"""CREATE TABLE events.target_day0 AS SELECT 'acq-1' AS event_id, 'ACME' AS ticker, 'map' AS ticker_source,
                    DATE '{day0}' AS day0, DATE '{day0}' AS day0_8k, '8k_confirmed' AS basis""")
    con.execute(f"""CREATE TABLE events.announcements AS SELECT * FROM (VALUES
        ('acq-1', 7, 'acquisition_target', TIMESTAMP '{day0}', 'index_header'),
        ('ear-1', 7, 'earnings', TIMESTAMP '{cal_sessions[300]}', 'earliest_candidate'))
        t(event_id, cik, event_type, day0, day0_basis)""")
    con.execute("CREATE TABLE raw.edgar_companies AS SELECT 7 AS cik, 'Acme' AS name, '2834' AS sic")
    con.close()
    con = duckdb.connect(ref)
    con.execute("CREATE SCHEMA ref")
    con.execute("""CREATE TABLE ref.ticker_cik AS SELECT 'ACME' AS ticker, 7::BIGINT AS cik, 'Acme' AS name,
                   'CS' AS type, NULL::DATE AS valid_from, NULL::DATE AS valid_to""")
    con.close()
    rng = np.random.default_rng(1)
    n = len(cal_sessions)
    rows = []
    for sym, base in (("ACME", 20.0), ("SPY", 300.0)):
        close = base * np.cumprod(1 + rng.normal(0, 0.01, n))
        for adj in ("raw", "all"):
            rows.append(pd.DataFrame({"symbol": sym, "date": cal_sessions, "adjustment": adj, "close": close,
                                      "volume": 1e6}))
    con = duckdb.connect(prices)
    con.execute("CREATE SCHEMA raw")
    con.register("b", pd.concat(rows))
    con.execute("CREATE TABLE raw.alpaca_bars_daily AS SELECT symbol, CAST(date AS DATE) AS date, adjustment, close, volume FROM b")
    con.close()
    con = duckdb.connect(finra)
    con.execute("CREATE SCHEMA raw")
    con.register("s", pd.DataFrame({"symbol": "ACME", "date": cal_sessions, "short_volume": 400_000,
                                    "total_volume": 1_000_000}))
    con.execute("CREATE TABLE raw.finra_short_daily AS SELECT symbol, CAST(date AS DATE) AS date, short_volume, total_volume FROM s")
    con.close()
    df = ft.build_targets(edgar, ref, prices, finra, out)
    pre = df[df.window == "pre"].iloc[0]
    assert pre.in_universe and pre.abn_volume == pytest.approx(0) and pre.short_share_abn == pytest.approx(0)
    assert set(df.window) == {"pre", "placebo"}
    assert not math.isnan(pre.day0_ar)
    ear = ft.build_earnings(edgar, ref, prices, finra, out)
    assert list(ear.event_id.unique()) == ["ear-1"] and ear[ear.window == "pre"].iloc[0].n_window == 20
