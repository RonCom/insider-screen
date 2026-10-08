import duckdb

from insider_screen import db


def test_route():
    assert db.route("raw", "sec_litigation_releases") == db.RELEASES
    assert db.route("extracted", "traded_events") == db.RELEASES
    assert db.route("raw", "edgar_filings") == db.EDGAR
    assert db.route("events", "announcements") == db.EDGAR
    assert db.route("raw", "finra_short_daily") == db.FINRA
    assert db.route("raw", "alpaca_bars_daily") == db.PRICES
    assert db.route("raw", "other") is None


def test_split(tmp_path):
    old = str(tmp_path / "insider.duckdb")
    con = duckdb.connect(old)
    for s in ("raw", "extracted", "events", "labels"):
        con.execute(f"CREATE SCHEMA {s}")
    con.execute("CREATE TABLE raw.sec_litigation_releases AS SELECT 1 AS lr_no")
    con.execute("CREATE TABLE extracted.traded_events_v2 AS SELECT 1 AS lr_no, 'X' AS issuer_name")
    con.execute("CREATE VIEW extracted.traded_events AS SELECT * FROM extracted.traded_events_v2")
    con.execute("CREATE TABLE events.announcements AS SELECT 'e1' AS event_id")
    con.execute("CREATE TABLE labels.release_event_matches AS SELECT 'e1' AS event_id, 1 AS lr_no")
    con.execute("""CREATE VIEW labels.charged_events AS SELECT a.*, m.lr_no IS NOT NULL AS is_charged
                   FROM events.announcements a LEFT JOIN labels.release_event_matches m USING (event_id)""")
    con.execute("CREATE TABLE raw.finra_short_daily AS SELECT 'AAA' AS symbol")
    con.execute("CREATE TABLE raw.stray AS SELECT 1 AS x")
    con.close()
    files = {t: str(tmp_path / t.split("/")[-1]) for t in (db.RELEASES, db.EDGAR, db.FINRA, db.PRICES)}

    log = {obj: action for obj, _, action in db.split(old, files)}
    assert log["extracted.traded_events"] == "view recreated"
    assert log["labels.charged_events"] == "view copied as table"
    assert log["raw.stray"].startswith("left")

    lab = duckdb.connect(files[db.RELEASES], read_only=True)
    assert lab.execute("SELECT issuer_name FROM extracted.traded_events").fetchone() == ("X",)
    assert lab.execute("SELECT is_charged FROM labels.charged_events").fetchone() == (True,)
    lab.close()
    ed = duckdb.connect(files[db.EDGAR], read_only=True)
    assert ed.execute("SELECT count(*) FROM events.announcements").fetchone() == (1,)
    ed.close()
    assert {a for _, _, a in db.split(old, files)} == {"already there", "left in old file (no route)"}
