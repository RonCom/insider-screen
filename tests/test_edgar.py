import json
import zipfile

import duckdb
import exchange_calendars as xc
import pandas as pd

from insider_screen import edgar


def _filings(rows):
    keys = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items"]
    return {k: [r[i] for r in rows] for i, k in enumerate(keys)}


TARGET = {  # target company: credit agreement long before, deal 8-K, 14D-9C same day, earnings
    "cik": 1001, "name": "Target Co", "entityType": "operating", "sic": "2834",
    "tickers": [], "exchanges": [], "formerNames": [{"name": "Target Holdings"}],
    "filings": {"recent": _filings([
        ("0001-23-000001", "2023-01-10", "2023-01-10T08:00:00.000Z", "8-K", "1.01,2.03,9.01"),
        ("0001-23-000002", "2023-06-05", "2023-06-05T07:15:00.000Z", "8-K", "1.01,8.01,9.01"),
        ("0001-23-000003", "2023-06-05", "2023-06-05T07:20:00.000Z", "SC14D9C", ""),
        ("0001-23-000004", "2023-06-20", "2023-06-20T16:30:00.000Z", "SC 14D9", ""),
        ("0001-23-000005", "2023-05-02", "2023-05-02T16:05:00.000Z", "8-K", "2.02,9.01"),
        ("0001-23-000006", "2023-03-01", "2023-03-01T10:00:00.000Z", "10-Q", ""),
    ])},
}
OTHER = {  # older filings in a continuation file; a fund that should be excluded
    "cik": 1002, "name": "Fund Trust", "entityType": "other", "sic": "6726",
    "tickers": ["FND"], "exchanges": ["NYSE"], "formerNames": [],
    "filings": {"recent": _filings([
        ("0002-23-000001", "2023-03-01", "2023-03-01T10:00:00.000Z", "10-Q", ""),
        ("0002-23-000002", "2023-07-03", "2023-07-03T17:00:00.000Z", "8-K", "2.02"),
    ])},
}
CONT = _filings([("0001-17-000001", "2017-02-01", "2017-02-04T09:00:00.000Z", "8-K", "8.01")])


def _zip(tmp_path):
    p = tmp_path / "submissions.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("CIK0000001001.json", json.dumps(TARGET))
        zf.writestr("CIK0000001002.json", json.dumps(OTHER))
        zf.writestr("CIK0000001001-submissions-001.json", json.dumps(CONT))
    return p


def test_day0_rules():
    cal = xc.get_calendar("XNYS", start="2015-01-01")
    ts = pd.Series(pd.to_datetime([
        "2023-06-05 07:15",  # pre-open Monday -> same day
        "2023-06-05 16:05",  # after close -> Tuesday
        "2023-06-03 09:00",  # Saturday -> Monday
        "2023-07-03 17:00",  # after close before July 4 holiday -> July 5
        "2023-06-05 15:59",  # during session -> same day
    ]))
    got = edgar.day0(ts, cal).dt.strftime("%Y-%m-%d").tolist()
    assert got == ["2023-06-05", "2023-06-06", "2023-06-05", "2023-07-05", "2023-06-05"]


def test_load_and_events(tmp_path):
    db = str(tmp_path / "t.duckdb")
    edgar.load(str(_zip(tmp_path)), db)
    ev = edgar.build_events(db, start="2016-01-01", end="2025-12-31")
    con = duckdb.connect(db)
    assert con.execute("SELECT former_names FROM raw.edgar_companies WHERE cik = 1001").fetchone()[0] == "Target Holdings"
    assert con.execute("SELECT count(*) FROM raw.edgar_filings WHERE cik = 1001").fetchone()[0] == 7
    got = {(r.event_type, r.accession): r for r in ev.itertuples()}
    tgt = got[("acquisition_target", "0001-23-000002")]
    assert tgt.evidence == "SC14D9C 2023-06-05"
    assert str(tgt.day0.date()) == "2023-06-05"
    assert ("acquisition_target", "0001-23-000001") not in got  # credit agreement, 146 days earlier
    assert str(got[("earnings", "0001-23-000005")].day0.date()) == "2023-05-03"
    assert ("other_material_candidate", "0001-17-000001") in got  # from the continuation file
    assert not (ev.cik == 1002).any()  # fund excluded


def test_src_tag_and_tz_sample(tmp_path, monkeypatch):
    db = str(tmp_path / "t.duckdb")
    edgar.load(str(_zip(tmp_path)), db)
    con = duckdb.connect(db)
    assert set(con.execute("SELECT DISTINCT src FROM raw.edgar_filings").fetchall()) == {("recent",), ("file",)}
    stamps = {a: (t, s) for a, t, s in con.execute("SELECT accession, accepted_json, src FROM raw.edgar_filings").fetchall()}
    con.close()

    # headers say Eastern time: recent JSON runs 4 hours ahead (UTC in June), continuation files match
    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def get(self, url):
            acc = url.rsplit("/", 1)[1].replace("-index-headers.html", "")
            ts, src = stamps[acc]
            et = ts - pd.Timedelta(hours=4) if src == "recent" else ts
            return 200, f"<ACCEPTANCE-DATETIME>{et:%Y%m%d%H%M%S}".encode()

    import insider_screen.http
    monkeypatch.setattr(insider_screen.http, "PoliteClient", FakeClient)
    monkeypatch.setattr(edgar, "ERAS", [("2016-01-01", "2025-12-31")])
    edgar.tz_sample(db, per_cell=10)
    con = duckdb.connect(db)
    assert edgar.tz_rules(con) == {"recent": "UTC", "file": "ET"}
    con.close()

    ev = edgar.build_events(db, start="2016-01-01", end="2025-12-31")
    tgt = ev[ev.accession == "0001-23-000002"].iloc[0]  # JSON 07:15 read as UTC -> 03:15 ET, still pre-open
    assert str(tgt.accepted_et) == "2023-06-05 03:15:00" and str(tgt.day0.date()) == "2023-06-05"
    earn = ev[ev.accession == "0001-23-000005"].iloc[0]  # JSON 16:05 UTC -> 12:05 ET, during the session
    assert str(earn.day0.date()) == "2023-05-02"


def test_mixed_offsets_stop_events(tmp_path):
    import pytest
    db = str(tmp_path / "m.duckdb")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA raw")
    con.execute("CREATE TABLE raw.edgar_tz_check AS SELECT * FROM (VALUES ('recent', 0.0), ('recent', 4.0)) t(src, offset_hours)")
    with pytest.raises(SystemExit):
        edgar.tz_rules(con)
