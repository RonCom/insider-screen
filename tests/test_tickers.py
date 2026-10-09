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
    got = dict(con.execute(tk.ticker_on_sql(con, "edgar.events.announcements")
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
    assert con.execute(tk.ticker_on_sql(con, "ev")).fetchall()[0][-1] == "AAA"


class FakeSec:
    """Filing index and press release per accession; the release names the given ticker."""
    def __init__(self, releases):
        self.releases, self.urls = releases, []

    def get(self, url, use_cache=True, store=True):
        self.urls.append(url)
        for acc, text in self.releases.items():
            folder = acc.replace("-", "")
            if url.endswith(f"{acc}-index.htm"):
                return 200, (f'<table class="tableFile"><tr><th>h</th></tr><tr><td>1</td><td>x</td>'
                             f'<td><a href="/Archives/edgar/data/1/{folder}/ex991.htm">d</a></td><td>EX-99.1</td><td>1</td></tr>'
                             f'</table>').encode()
            if url.endswith(f"{folder}/ex991.htm"):
                return 200, f"<p>{text}</p>".encode()
        return 404, b""


def _edgar(tmp_path):
    edgar = str(tmp_path / "edgar.duckdb")
    con = duckdb.connect(edgar)
    con.execute("CREATE SCHEMA events; CREATE SCHEMA raw")
    con.execute("""CREATE TABLE raw.edgar_companies AS SELECT * FROM (VALUES
        (701374, 'SIX FLAGS ENTERTAINMENT CORP'), (1326801, 'Meta Platforms, Inc.'), (555, 'Gone Corp')) t(cik, name)""")
    con.execute("""CREATE TABLE events.announcements AS SELECT *, day0 + INTERVAL 7 HOUR AS accepted_et FROM (VALUES
        ('e1', 701374, 'acquisition_target', TIMESTAMP '2023-11-02', '0001-23-000001'),
        ('e2', 1326801, 'earnings', TIMESTAMP '2021-07-28', '0001-21-000002'),
        ('e3', 555, 'earnings', TIMESTAMP '2018-05-01', '0001-18-000003'),
        ('e4', 555, 'other_material_candidate', TIMESTAMP '2018-09-01', '0001-18-000004')
        ) t(event_id, cik, event_type, day0, accession)""")
    con.close()
    return edgar


def test_check_classifies(tmp_path):
    db = str(tmp_path / "reference.duckdb")
    tk.download(api_with([]), db)
    tk.build(db)
    edgar = _edgar(tmp_path)
    sec = FakeSec({"0001-23-000001": "Six Flags Entertainment Corporation (NYSE: SIX) today announced",
                   "0001-21-000002": "Meta Platforms, Inc. (Nasdaq: MSFT) reported"})
    df = tk.check(db, edgar, n=6, out=str(tmp_path / "c.csv"), sec=sec)
    got = dict(zip(df.event_id, df.outcome))
    assert got == {"e1": "agree", "e2": "release_ticker_not_in_map"}


def test_fill_supplements_missing_company_years(tmp_path):
    db = str(tmp_path / "reference.duckdb")
    tk.download(api_with([]), db)
    tk.build(db)
    edgar = _edgar(tmp_path)
    sec = FakeSec({"0001-18-000004": "Gone Corp (NYSE: GONE) today announced",
                   "0001-18-000003": "Gone Corp (NYSE: GONE) reported results"})
    tk.fill(db, edgar, sec=sec)
    con = duckdb.connect(db)
    assert con.execute("SELECT cik, year, ticker, conflict FROM ref.ticker_supplement").fetchall() == [(555, 2018, "GONE", False)]
    con.execute(f"ATTACH '{edgar}' AS edgar (READ_ONLY)")
    rows = con.execute(f"SELECT event_id, ticker, ticker_source FROM ({tk.event_tickers_sql(con)}) ORDER BY 1").fetchall()
    con.close()
    assert rows == [("e1", "SIX", "map"), ("e2", "FB", "map"), ("e3", "GONE", "press_release"),
                    ("e4", "GONE", "press_release")]
    n = len(sec.urls)
    tk.fill(db, edgar, sec=sec)  # done company-years aren't looked up again
    assert len(sec.urls) == n


def test_fill_rejects_ticker_held_by_another_company(tmp_path):
    db = str(tmp_path / "reference.duckdb")
    tk.download(api_with([]), db)
    tk.build(db)
    edgar = _edgar(tmp_path)
    tk.fill(db, edgar, sec=FakeSec({"0001-18-000004": "Gone Corp (NYSE: CELG) today announced"}))
    con = duckdb.connect(db)
    row = con.execute("SELECT ticker, conflict, how FROM ref.ticker_supplement").fetchone()
    assert row[:2] == ("CELG", True) and "Celgene" in row[2]


def test_finra_volume_picks_the_traded_ticker(tmp_path):
    """Several tickers valid for one company on day 0: the one that traded most that month wins."""
    db, edgar, finra = (str(tmp_path / f) for f in ("reference.duckdb", "edgar.duckdb", "finra.duckdb"))
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA ref")
    con.execute("""CREATE TABLE ref.ticker_cik AS SELECT * FROM (VALUES
        ('VRNT', 1166388, 'CS', NULL::DATE), ('VRNTV', 1166388, 'CS', DATE '2016-06-01'),
        ('FB', 1326801, 'CS', DATE '2022-06-09'), ('META', 1326801, 'CS', NULL::DATE),
        ('SOI', 1697500, 'CS', DATE '2024-07-31'), ('SEI', 1697500, 'CS', NULL::DATE)
        ) t(ticker, cik, type, valid_to)""")
    con.execute("ALTER TABLE ref.ticker_cik ADD COLUMN valid_from DATE")
    con.close()
    con = duckdb.connect(edgar)
    con.execute("CREATE SCHEMA events")
    con.execute("""CREATE TABLE events.announcements AS SELECT * FROM (VALUES
        ('verint', 1166388, TIMESTAMP '2016-03-29'), ('meta21', 1326801, TIMESTAMP '2021-07-28'),
        ('meta23', 1326801, TIMESTAMP '2023-07-26'), ('solaris', 1697500, TIMESTAMP '2024-07-10'))
        t(event_id, cik, day0)""")
    con.close()
    con = duckdb.connect(finra)
    con.execute("CREATE SCHEMA raw")
    con.execute("""CREATE TABLE raw.finra_short_daily AS SELECT * FROM (VALUES
        (DATE '2016-03-15', 'VRNT', 900000), (DATE '2016-03-15', 'VRNTV', 2000),
        (DATE '2021-07-15', 'FB', 5000000), (DATE '2023-07-14', 'META', 6000000),
        (DATE '2024-07-09', 'SOI', 300000)) t(date, symbol, total_volume)""")
    con.close()
    con = duckdb.connect(db, read_only=True)
    assert tk.attach(con, edgar, finra)
    sql = tk.ticker_on_sql(con, "edgar.events.announcements").replace("SELECT e.*, t.ticker", "SELECT e.event_id, t.ticker")
    got = dict(con.execute(sql).fetchall())
    con.close()
    assert got == {"verint": "VRNT", "meta21": "FB", "meta23": "META", "solaris": "SOI"}


def test_units_are_not_errors():
    ranges = pd.DataFrame({"ticker": ["PHYT"], "cik": [1]})
    assert tk.classify("PHYT", "PHYT.U", 1, ranges) == "units_or_warrants"
    assert tk.classify("ASAX", "ASAXU", 1, ranges) == "units_or_warrants"
    assert tk.classify("ASAX", "ASAXX", 1, ranges) == "release_ticker_not_in_map"


def test_attach_without_finra_warns(tmp_path, capsys):
    con = duckdb.connect()
    edgar = str(tmp_path / "e.duckdb")
    duckdb.connect(edgar).close()
    assert tk.attach(con, edgar, str(tmp_path / "missing.duckdb")) is False
    assert "no FINRA volume" in capsys.readouterr().out


def test_fill_retry_and_8k_body(tmp_path):
    """A company-year with no EX-99 is retried with --retry and found in the 8-K's own text."""
    db = str(tmp_path / "reference.duckdb")
    tk.download(api_with([]), db)
    tk.build(db)
    edgar = _edgar(tmp_path)
    tk.fill(db, edgar, sec=FakeSec({}))  # nothing found anywhere
    con = duckdb.connect(db)
    assert con.execute("SELECT count(*) FROM ref.ticker_supplement WHERE ticker IS NULL").fetchone()[0] == 1
    con.close()

    class BodySec(FakeSec):
        def get(self, url, use_cache=True, store=True):
            self.urls.append(url)
            if url.endswith("0001-18-000004-index.htm"):
                return 200, (b'<table class="tableFile"><tr><th>h</th></tr><tr><td>1</td><td>x</td>'
                             b'<td><a href="/Archives/edgar/data/555/000118000004/d8k.htm">d</a></td><td>8-K</td><td>1</td></tr></table>')
            if url.endswith("000118000004/d8k.htm"):
                return 200, b"<p>Gone Corp (NYSE: GONE) announced today</p>"
            return 404, b""
    tk.fill(db, edgar, sec=BodySec({}), retry=True)
    con = duckdb.connect(db)
    assert con.execute("SELECT ticker FROM ref.ticker_supplement WHERE cik = 555").fetchall() == [("GONE",)]


def test_ticker_wordings():
    from insider_screen.market_move import ticker_from_text
    assert ticker_from_text("Acme Inc. (NASDAQ Capital Market: ACME) today", "ACME INC")[0] == "ACME"
    assert ticker_from_text("Acme Inc. (NYSE American: AMX) today", "ACME INC")[0] == "AMX"
    assert ticker_from_text("Acme Inc. (Nasdaq Global Select Market: ACMG) today", "ACME INC")[0] == "ACMG"
    assert ticker_from_text("Acme Inc. (NasdaqGS: ACMS) today", "ACME INC")[0] == "ACMS"
    # lowercase words after an exchange name aren't tickers
    assert ticker_from_text("Acme Inc. listed on the NYSE: the company said", "ACME INC")[0] is None
    # the reason lists the tickers found
    assert "['BETA']" in ticker_from_text("Beta Corp (NYSE: BETA) will buy it", "ACME INC")[1]


def test_untyped_delisted_rows_kept_unless_not_stock(tmp_path):
    db = str(tmp_path / "reference.duckdb")
    con = duckdb.connect(db)
    tk._setup(con)
    rows = [
        ("OLDC", "Old Company Inc", "0000000101", None, "2017-05-01T00:00:00Z"),    # untyped stock: kept
        ("OLDCW", "Old Company Inc", "0000000101", None, "2017-05-01T00:00:00Z"),   # base + W: dropped
        ("OLDC.U", "Old Company Inc", "0000000101", None, "2016-01-01T00:00:00Z"),  # base + .U: dropped
        ("ZZW", "Zeta Acquisition Corp Warrants", "0000000102", None, "2019-01-01T00:00:00Z"),  # name: dropped
        ("PFX", "Pfx Corp 6.5% Notes due 2025", "0000000103", None, "2019-01-01T00:00:00Z"),     # name: dropped
        ("NOCIK", "No Cik Inc", None, None, "2018-01-01T00:00:00Z"),                # no CIK: dropped
        ("TYPED", "Typed Inc", "0000000104", "CS", "2018-01-01T00:00:00Z"),          # typed stock: kept
    ]
    for t, name, cik, typ, delisted in rows:
        con.execute("INSERT INTO raw.massive_tickers VALUES (?, ?, ?, ?, FALSE, 'XNYS', NULL, NULL, ?, NULL, now())",
                    [t, name, cik, typ, delisted])
    con.close()
    out = tk.build(db)
    assert sorted(out.ticker) == ["OLDC", "TYPED"]
    assert dict(zip(out.ticker, out.type)) == {"OLDC": "untyped", "TYPED": "CS"}


def test_release_ticker_gets_a_date(monkeypatch):
    """DuckDB returns list elements as datetime; the filing-date comparisons need a date."""
    from datetime import date, datetime

    from insider_screen import market_move
    seen = []
    monkeypatch.setattr(market_move, "release_ticker", lambda sec, url, filed, company: seen.append(filed) or (None, "x"))
    tk._release_ticker(None, 1, "0001-18-000004", "Gone Corp", datetime(2018, 5, 1))
    tk._release_ticker(None, 1, "0001-18-000004", "Gone Corp", pd.NaT)
    assert seen == [date(2018, 5, 1), None] and type(seen[0]) is date
