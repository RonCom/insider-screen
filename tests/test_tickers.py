from datetime import date

import duckdb
import httpx
import pandas as pd
import pytest

from insider_screen import tickers as tk


def rec(ticker, cik, name, active, delisted=None, type_="CS", figi=None):
    return {"ticker": ticker, "cik": cik, "name": name, "type": type_, "active": active,
            "primary_exchange": "XNYS", "composite_figi": figi, "share_class_figi": None,
            "delisted_utc": delisted, "last_updated_utc": "2026-01-01T00:00:00Z"}


ACTIVE = [rec("META", "0001326801", "Meta Platforms", True), rec("SIX", "0001999001", "Six Flags (new)", True)]
INACTIVE = [rec("FB", "0001326801", "Meta Platforms", False, "2022-06-09T00:00:00Z"),
            rec("SIX", "0000701374", "Six Flags Entertainment Corp", False, "2024-07-01T00:00:00Z"),
            rec("CELG", "0000816284", "Celgene", False, "2019-11-21T00:00:00Z"),
            rec("CELG.WS", None, "Celgene rights", False, "2019-11-21T00:00:00Z", type_="RIGHT")]


def api_with(pages_seen, fail_host=None):
    def handler(request):
        pages_seen.append(str(request.url))
        assert request.headers["Authorization"] == "Bearer k"
        if fail_host and request.url.host == fail_host:
            raise httpx.ConnectError("no such host")
        if request.url.params.get("cursor") == "p2":
            return httpx.Response(200, json={"status": "OK", "results": INACTIVE[2:], "next_url": None})
        if request.url.params.get("active") == "true":
            return httpx.Response(200, json={"status": "OK", "results": ACTIVE})
        return httpx.Response(200, json={"status": "OK", "results": INACTIVE[:2],
                                         "next_url": f"https://{request.url.host}/v3/reference/tickers?cursor=p2"})
    return tk.Massive("k", transport=httpx.MockTransport(handler), per_minute=100000,
                      base_urls=["https://api.massive.com", "https://api.polygon.io"])


def test_download_pages_and_resume(tmp_path):
    db = str(tmp_path / "reference.duckdb")
    seen = []
    tk.download(api_with(seen), db)
    con = duckdb.connect(db)
    assert con.execute("SELECT count(*) FROM raw.massive_tickers").fetchone()[0] == 6
    assert con.execute("SELECT count(*) FROM raw.massive_progress WHERE done").fetchone()[0] == 2
    con.close()
    seen.clear()
    tk.download(api_with(seen), db)  # both lists done: no requests
    assert seen == []


def test_falls_back_to_polygon_host(tmp_path):
    seen = []
    tk.download(api_with(seen, fail_host="api.massive.com"), str(tmp_path / "reference.duckdb"))
    assert any("api.polygon.io" in u for u in seen)


def test_build_dated_ranges(tmp_path):
    db = str(tmp_path / "reference.duckdb")
    tk.download(api_with([]), db)
    out = tk.build(db).set_index(["ticker", "cik"])
    # reused ticker: old Six Flags until its delisting, new holder from the next day
    old = out.loc[("SIX", 701374)]
    new = out.loc[("SIX", 1999001)]
    assert (old.valid_from, old.valid_to) == (None, date(2024, 7, 1))
    assert (new.valid_from, new.valid_to) == (date(2024, 7, 2), None)
    # ticker change, same CIK: FB until 2022-06-09, META open-ended
    assert out.loc[("FB", 1326801)].valid_to == date(2022, 6, 9)
    assert out.loc[("META", 1326801)].valid_to is None
    # rights and other non-stock types are left out
    assert "CELG.WS" not in out.index.get_level_values(0)


def test_coverage_joins_on_day0(tmp_path):
    db, edgar = str(tmp_path / "reference.duckdb"), str(tmp_path / "edgar.duckdb")
    tk.download(api_with([]), db)
    tk.build(db)
    con = duckdb.connect(edgar)
    con.execute("CREATE SCHEMA events")
    con.execute("""CREATE TABLE events.announcements AS SELECT * FROM (VALUES
        ('e1', 701374, 'acquisition_target', DATE '2023-11-02'),
        ('e2', 1326801, 'earnings', DATE '2021-07-28'),
        ('e3', 1326801, 'earnings', DATE '2023-07-26'),
        ('e4', 999, 'earnings', DATE '2023-07-26')) t(event_id, cik, event_type, day0)""")
    con.close()
    con = duckdb.connect(db)
    con.execute(f"ATTACH '{edgar}' AS edgar (READ_ONLY)")
    got = dict(con.execute(tk.TICKER_ON_SQL.format(events="edgar.events.announcements", map="ref.ticker_cik")
                           .replace("SELECT e.*, t.ticker", "SELECT e.event_id, t.ticker")).fetchall())
    con.close()
    assert got == {"e1": "SIX", "e2": "FB", "e3": "META", "e4": None}
    df = tk.coverage(db, edgar)
    assert df.with_ticker.sum() == 3


def test_rejected_key_message():
    api = tk.Massive("bad", transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"status": "ERROR"})),
                     per_minute=100000)
    with pytest.raises(SystemExit, match="MASSIVE_API_KEY"):
        api.get("/v3/reference/tickers")


def test_all_null_dates_still_typed(tmp_path):
    """No reused tickers and no delistings: the date columns must still be DATE, or the day-0 join fails."""
    db = str(tmp_path / "reference.duckdb")
    con = duckdb.connect(db)
    tk._setup(con)
    con.execute("INSERT INTO raw.massive_tickers VALUES ('AAA', 'A', '0000000001', 'CS', TRUE, 'XNYS', NULL, NULL, NULL, NULL, now())")
    con.close()
    tk.build(db)
    con = duckdb.connect(db)
    types = dict(con.execute("SELECT column_name, data_type FROM information_schema.columns "
                             "WHERE table_name = 'ticker_cik'").fetchall())
    assert types["valid_from"] == "DATE" and types["valid_to"] == "DATE"
    con.execute("CREATE TABLE ev AS SELECT 'e1' AS event_id, 1::BIGINT AS cik, TIMESTAMP '2023-01-05' AS day0")
    assert con.execute(tk.TICKER_ON_SQL.format(events="ev", map="ref.ticker_cik")).fetchall()[0][-1] == "AAA"
