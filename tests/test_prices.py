import duckdb
import httpx

from insider_screen import prices


def bar(t, c):
    return {"t": f"{t}T05:00:00Z", "o": c, "h": c, "l": c, "c": c, "v": 100, "n": 5, "vw": c}


def make_api(handler):
    return prices.Alpaca("k", "s", transport=httpx.MockTransport(handler), max_per_minute=100000)


def test_fetch_bars_follows_pages():
    def handler(request):
        assert request.headers["APCA-API-KEY-ID"] == "k"
        if request.url.params.get("page_token") == "p2":
            return httpx.Response(200, json={"bars": {"BBB": [bar("2018-01-03", 2.0)]}, "next_page_token": None})
        return httpx.Response(200, json={"bars": {"AAA": [bar("2018-01-02", 1.0)]}, "next_page_token": "p2"})
    df = prices.fetch_bars(make_api(handler), ["AAA", "BBB"], "2018-01-01", "2018-01-05", "all")
    assert sorted(df.symbol) == ["AAA", "BBB"] and str(df.date.iloc[0]) == "2018-01-02"


def test_bad_symbol_is_isolated():
    def handler(request):
        syms = request.url.params["symbols"].split(",")
        if "BAD" in syms:
            return httpx.Response(400, json={"message": "invalid symbol: BAD"})
        return httpx.Response(200, json={"bars": {s: [bar("2018-01-02", 1.0)] for s in syms},
                                         "next_page_token": None})
    df, errors = prices.fetch_batch(make_api(handler), ["AAA", "BAD", "CCC", "DDD"], "2018-01-01", "2018-01-05", "raw")
    assert sorted(df.symbol) == ["AAA", "CCC", "DDD"]
    assert [s for s, _ in errors] == ["BAD"]


def test_load_resumes_and_adds_market(tmp_path):
    calls = []

    def handler(request):
        syms = request.url.params["symbols"].split(",")
        calls.append((request.url.params["adjustment"], syms))
        return httpx.Response(200, json={"bars": {s: [bar("2018-01-02", 1.0)] for s in syms if s != "GONE"},
                                         "next_page_token": None})
    db = str(tmp_path / "p.duckdb")
    prices.load(make_api(handler), db, "2018-01-01", "2018-01-05", ["AAA", "brk/b", "GONE", "ABR-D"])
    assert {tuple(s) for _, s in calls} == {("AAA", "BRK.B", "GONE", "SPY")}
    assert sorted(a for a, _ in calls) == ["all", "raw"]
    con = duckdb.connect(db)
    assert con.execute("SELECT count(*) FROM raw.alpaca_bars_daily").fetchone()[0] == 6
    assert con.execute("SELECT n_bars FROM raw.alpaca_symbols_done WHERE symbol = 'GONE' LIMIT 1").fetchone()[0] == 0
    con.close()
    calls.clear()
    prices.load(make_api(handler), db, "2018-01-01", "2018-01-05", ["AAA", "brk/b", "GONE"])
    assert calls == []


def test_rejected_credentials_exit_with_hint():
    import pytest
    api = make_api(lambda request: httpx.Response(401, json={"message": "unauthorized."}))
    with pytest.raises(SystemExit, match="Key ID starts 'k'"):
        prices.fetch_bars(api, ["AAA"], "2018-01-01", "2018-01-05", "raw")
