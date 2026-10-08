import json
from datetime import date

import duckdb
import httpx

from insider_screen import extract
from insider_screen.extract import ReleaseExtraction, call_ollama, check_date, clean_events, is_named

TEXT = """Litigation Release No. 25999 / March 3, 2024. The complaint alleges Doe bought call options
in Target Co. ahead of the June 5, 2023 announcement that Buyer Inc. would acquire Target Co.
He also bought Target Co. stock. In 2011 he traded ahead of news about a pharmaceutical company."""


def ev(**kw):
    base = {"issuer_name": "Target Co.", "ticker": None, "announcement_date": "2023-06-05",
            "announcement_evidence": "ahead of the June 5, 2023 announcement that Buyer Inc. would acquire Target Co.",
            "event_type": "acquisition_target", "instruments": "options", "direction": "long"}
    base.update(kw)
    return base


def ext(events, kind="new_charges"):
    return ReleaseExtraction.model_validate({"release_kind": kind, "is_insider_trading_case": True, "events": events})


def test_named_issuers():
    assert is_named("Target Co.") and is_named("SPSS Inc.") and is_named("at&t Inc.")
    assert not is_named("pharmaceutical company")
    assert not is_named("at least 15 stocks")
    assert not is_named("null") and not is_named(None)


def test_date_checks():
    e = ext([ev()]).events[0]
    assert check_date(e, TEXT) == (date(2023, 6, 5), "verified")
    e = ext([ev(announcement_date="2011-01-01", announcement_evidence="In 2011 he traded")]).events[0]
    assert check_date(e, TEXT) == (None, "quote_lacks_full_date")
    e = ext([ev(announcement_evidence="ahead of the June 5, 2023 merger announcement")]).events[0]
    assert check_date(e, TEXT) == (None, "quote_not_in_release")


def test_clean_merges_and_drops():
    rows = clean_events(ext([
        ev(),
        ev(instruments="stock", event_type="other", announcement_date=None, announcement_evidence=None),
        ev(issuer_name="pharmaceutical company"),
        ev(issuer_name="Target Co"),  # same issuer, punctuation differs
    ]), TEXT)
    assert len(rows) == 1
    r = rows[0]
    assert (r["instruments"], r["direction"], r["event_type"], r["date_check"]) == ("both", "long", "acquisition_target", "verified")


def _transport(replies):
    it = iter(replies)
    def handler(request):
        body = json.loads(request.content)
        assert body["format"]["title"] == "ReleaseExtraction"
        return httpx.Response(200, json={"message": {"content": next(it)}})
    return httpx.MockTransport(handler)


def test_call_ollama_retries_bad_json():
    good = json.dumps({"release_kind": "new_charges", "is_insider_trading_case": True, "events": [ev()]})
    client = httpx.Client(transport=_transport(["{not json", good]))
    assert call_ollama("text", "m", client).events[0].issuer_name == "Target Co."


def test_run_writes_clean_rows(tmp_path, monkeypatch):
    db = str(tmp_path / "t.duckdb")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA raw")
    con.execute("CREATE TABLE raw.sec_litigation_releases AS SELECT 1 AS lr_no, ? AS text, TRUE AS is_insider_candidate, 'u' AS url", [TEXT])
    con.close()
    monkeypatch.setattr(extract, "call_ollama", lambda text, model, client: ext([ev(), ev()]))
    extract.run(db, "m", None)
    extract.run(db, "m", None)  # second run skips done releases
    con = duckdb.connect(db)
    rows = con.execute("SELECT model, issuer_name, announcement_date::VARCHAR, date_check FROM extracted.traded_events").fetchall()
    assert rows == [("m#v3", "Target Co.", "2023-06-05", "verified")]


def test_trade_date_verified_and_latest_kept():
    text = TEXT + " Doe last bought Target Co. stock on June 2, 2023, three days before the news."
    rows = clean_events(ext([
        ev(last_trade_date="2023-06-02", trade_evidence="Doe last bought Target Co. stock on June 2, 2023"),
        ev(last_trade_date="2023-05-01", trade_evidence="bought on May 1, 2023"),  # quote not in release
    ]), text)
    assert (rows[0]["last_trade_date"], rows[0]["trade_check"]) == (date(2023, 6, 2), "verified")


def test_failed_release_is_retried(tmp_path, monkeypatch):
    db = str(tmp_path / "t.duckdb")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA raw")
    con.execute("CREATE TABLE raw.sec_litigation_releases AS SELECT 1 AS lr_no, ? AS text, TRUE AS is_insider_candidate, 'u' AS url", [TEXT])
    con.close()

    def boom(text, model, client):
        raise httpx.ReadTimeout("timed out")
    monkeypatch.setattr(extract, "call_ollama", boom)
    extract.run(db, "m", None)
    monkeypatch.setattr(extract, "call_ollama", lambda text, model, client: ext([ev()]))
    extract.run(db, "m", None)
    con = duckdb.connect(db)
    assert con.execute("SELECT ok FROM extracted.release_extractions").fetchall() == [(True,)]
