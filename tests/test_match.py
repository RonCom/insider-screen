import exchange_calendars as xc
import pandas as pd

from insider_screen.match import match, normalize


def test_normalize():
    assert normalize("APPLE INC /CA/") == "apple"
    assert normalize("Kindred Biosciences, Inc.") == "kindred biosciences"
    assert normalize("Johnson & Johnson") == "johnson and johnson"
    assert normalize("The Goodness Growth Holdings, Inc.") == "goodness growth"


CAL = xc.get_calendar("XNYS", start="2015-01-01")
COMPANIES = pd.DataFrame({
    "cik": [1, 2, 3],
    "name": ["KINDRED BIOSCIENCES, INC.", "NEOPHOTONICS CORP", "UNRELATED CO"],
    "former_names": ["", "", ""],
})
EVENTS = pd.DataFrame({
    "event_id": ["acq-1", "ear-1", "acq-2"],
    "cik": [1, 1, 2],
    "event_type": ["acquisition_target", "earnings", "acquisition_target"],
    "day0": pd.to_datetime(["2021-06-16", "2021-06-17", "2021-11-04"]),
})


def traded(name, ann, etype="acquisition_target", last_trade=None):
    return pd.DataFrame({"lr_no": [1], "issuer_name": [name], "announcement_date": [pd.Timestamp(ann) if ann else pd.NaT],
                         "last_trade_date": [pd.Timestamp(last_trade) if last_trade else pd.NaT],
                         "event_type": [etype]})


def test_exact_and_type_preference():
    out = match(traded("Kindred Biosciences, Inc.", "2021-06-16"), COMPANIES, EVENTS, CAL).iloc[0]
    assert (out.cik, out.name_method, out.event_id, out.session_gap) == (1, "exact", "acq-1", 0)


def test_fuzzy_name_and_gap():
    out = match(traded("NeoPhotonics Corporation Inc", "2021-11-03"), COMPANIES, EVENTS, CAL).iloc[0]
    assert out.event_id == "acq-2" and out.session_gap == 1


def test_reasons():
    assert match(traded("Zzyzx Widgets", "2021-06-16"), COMPANIES, EVENTS, CAL).iloc[0].reason == "no_company"
    assert match(traded("Kindred Biosciences", None), COMPANIES, EVENTS, CAL).iloc[0].reason == "no_dates"
    assert match(traded("Kindred Biosciences", "2021-03-01"), COMPANIES, EVENTS, CAL).iloc[0].reason == "no_event_in_window"


def test_run_reads_events_from_edgar_file(tmp_path):
    import duckdb

    from insider_screen import match as m
    from insider_screen.extract import TABLE, model_key

    rel, edg = str(tmp_path / "releases.duckdb"), str(tmp_path / "edgar.duckdb")
    con = duckdb.connect(edg)
    con.execute("CREATE SCHEMA raw; CREATE SCHEMA events")
    con.register("c", COMPANIES)
    con.execute("CREATE TABLE raw.edgar_companies AS SELECT * FROM c")
    con.register("e", EVENTS)
    con.execute("CREATE TABLE events.announcements AS SELECT * FROM e")
    con.close()
    con = duckdb.connect(rel)
    con.execute("CREATE SCHEMA extracted")
    con.execute(
        f"""CREATE TABLE {TABLE} AS SELECT 1 AS lr_no, 'Kindred Biosciences' AS issuer_name,
           DATE '2021-06-16' AS announcement_date, NULL::DATE AS last_trade_date, 'acquisition_target' AS event_type,
           TRUE AS is_insider_trading_case, ? AS model""", [model_key("m")])
    con.close()
    m.run(rel, "m", edg)
    con = duckdb.connect(rel, read_only=True)
    assert con.execute("SELECT event_id FROM labels.release_event_matches").fetchone() == ("acq-1",)
    assert con.execute("SELECT count(*), sum(is_charged::INT) FROM labels.charged_events").fetchone() == (3, 1)


def test_trade_date_fallback():
    out = match(traded("Kindred Biosciences", None, last_trade="2021-06-10"), COMPANIES, EVENTS, CAL).iloc[0]
    assert (out.event_id, out.date_source) == ("acq-1", "last_trade_date")  # acquisition preferred over earnings day after
    out = match(traded("Kindred Biosciences", None, last_trade="2021-04-01"), COMPANIES, EVENTS, CAL).iloc[0]
    assert out.reason == "no_event_after_trades"  # 76 days before the event: outside the 30-day window
