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
    assert not is_named("Unknown")
    assert not is_named("Post's employer (pharmaceutical company)")
    assert not is_named("Company A")
    assert is_named("Johnson & Johnson") and is_named("McDonald's Corporation")


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
        ev(instruments="stock", event_type="unknown", announcement_date=None, announcement_evidence=None),
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
    assert rows == [(extract.model_key("m"), "Target Co.", "2023-06-05", "verified")]


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


def test_run_works_on_table_without_primary_key(tmp_path, monkeypatch):
    """`db split` copies release_extractions with CREATE TABLE AS, which drops the primary key."""
    db = str(tmp_path / "t.duckdb")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA raw; CREATE SCHEMA extracted")
    con.execute("CREATE TABLE raw.sec_litigation_releases AS SELECT 1 AS lr_no, ? AS text, TRUE AS is_insider_candidate, 'u' AS url", [TEXT])
    con.execute("""CREATE TABLE extracted.release_extractions AS
                   SELECT 1 AS lr_no, ? AS model, FALSE AS ok, 'timed out' AS error, NULL::JSON AS payload""",
                [extract.model_key("m")])
    con.close()
    monkeypatch.setattr(extract, "call_ollama", lambda text, model, client: ext([ev()]))
    extract.run(db, "m", None)
    con = duckdb.connect(db)
    assert con.execute("SELECT lr_no, ok FROM extracted.release_extractions").fetchall() == [(1, True)]


def test_alias_parenthetical_kept_as_name():
    from insider_screen.extract import clean_issuer
    assert clean_issuer('Potash Corporation of Saskatchewan ("Potash")') == "Potash Corporation of Saskatchewan"
    assert clean_issuer("Acme Corp. (the \u201cCompany\u201d)") == "Acme Corp."
    assert not is_named(clean_issuer("Post's employer (pharmaceutical company)"))


def test_reclean_rebuilds_from_stored_output(tmp_path, monkeypatch):
    db = str(tmp_path / "t.duckdb")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA raw")
    con.execute("CREATE TABLE raw.sec_litigation_releases AS SELECT 1 AS lr_no, ? AS text, TRUE AS is_insider_candidate, 'u' AS url", [TEXT])
    con.close()
    monkeypatch.setattr(extract, "call_ollama", lambda text, model, client: ext([ev(issuer_name='Target Co. ("Target")')]))
    extract.run(db, "m", None)
    con = duckdb.connect(db)
    con.execute(f"DELETE FROM {extract.TABLE}")
    con.close()
    extract.reclean(db, "m")
    con = duckdb.connect(db)
    assert con.execute(f"SELECT issuer_name FROM {extract.TABLE}").fetchall() == [("Target Co.",)]


def test_two_announcements_for_one_issuer_stay_separate():
    text = TEXT + " On January 2, 2023 Target Co. announced a license agreement with Gamma Inc."
    rows = clean_events(ext([
        ev(),
        ev(event_type="other", announcement_date="2023-01-02",
           announcement_evidence="On January 2, 2023 Target Co. announced a license agreement"),
    ]), text)
    assert sorted((r["event_type"], str(r["announcement_date"])) for r in rows) == [
        ("acquisition_target", "2023-06-05"), ("other", "2023-01-02")]
    # same type, different verified dates: two earnings releases
    rows = clean_events(ext([
        ev(event_type="earnings"),
        ev(event_type="earnings", announcement_date="2023-01-02",
           announcement_evidence="On January 2, 2023 Target Co. announced a license agreement"),
    ]), text)
    assert len(rows) == 2


def test_acquirer_dropped_when_target_present():
    rows = clean_events(ext([ev(), ev(issuer_name="Buyer Inc.", event_type="acquirer")]), TEXT)
    assert [r["issuer_name"] for r in rows] == ["Target Co."]
    rows = clean_events(ext([ev(issuer_name="Buyer Inc.", event_type="acquirer")]), TEXT)
    assert [r["issuer_name"] for r in rows] == ["Buyer Inc."]


def test_instruments_checked_against_release():
    from insider_screen.extract import check_instruments
    securities = "Penna traded in the securities of each of the three companies."
    assert check_instruments("stock", securities) == ("unknown", "stock_not_in_release")
    assert check_instruments("unknown", securities) == ("unknown", "ok")
    employee = "Ying exercised all of his vested Equifax stock options and then sold the shares."
    assert check_instruments("options", employee) == ("stock", "employee_options")
    assert check_instruments("stock", employee) == ("stock", "ok")
    calls = "He bought 200 call options and later sold his shares."
    assert check_instruments("both", calls) == ("both", "ok")
    assert check_instruments("options", "Doe bought QLogic call options.") == ("options", "ok")
    assert check_instruments("both", "Doe bought QLogic call options.") == ("options", "both_not_in_release")


def test_sell_direction_accepted():
    e = ext([ev(direction="sell", instruments="stock")]).events[0]
    assert e.direction == "sell"


def test_trade_dates_split_only_when_far_apart():
    text = TEXT + " Doe bought on May 20, 2023 and again on June 1, 2023. Earlier he bought on January 3, 2023."
    def tev(d, q):
        return ev(announcement_date=None, announcement_evidence=None, last_trade_date=d, trade_evidence=q)
    rows = clean_events(ext([tev("2023-05-20", "Doe bought on May 20, 2023"),
                             tev("2023-06-01", "again on June 1, 2023")]), text)
    assert [str(r["last_trade_date"]) for r in rows] == ["2023-06-01"]
    rows = clean_events(ext([tev("2023-06-01", "again on June 1, 2023"),
                             tev("2023-01-03", "Earlier he bought on January 3, 2023")]), text)
    assert len(rows) == 2


def test_compare_against_reviewed_handcheck(tmp_path):
    import duckdb
    import pandas as pd

    from insider_screen import extract as ex
    db = str(tmp_path / "releases.duckdb")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA extracted")
    con.execute(f"""CREATE TABLE {ex.TABLE} AS SELECT * FROM (VALUES
        (1, '{ex.model_key("small")}', 'Acme Corp.', DATE '2020-01-02'),
        (2, '{ex.model_key("small")}', 'Beta Inc', DATE '2020-05-05'),
        (3, '{ex.model_key("small")}', 'Lumentum', DATE '2021-01-19'),
        (5, '{ex.model_key("small")}', 'Delta Co', NULL)) t(lr_no, model, issuer_name, announcement_date)""")
    con.close()
    csv = tmp_path / "h.csv"
    pd.DataFrame({"lr_no": ["1", "2", "3", "4", "5"],
                  "issuer_name": ["Acme Corporation", "Beta, Inc.", "Coherent", "Old", "Delta Company"],
                  "announcement_date": ["2020-01-02", "2020-05-06", "2021-01-19", "", ""],
                  "ok_issuer_name": ["Y", "Y", "Y", "N", "Y"],
                  "ok_announcement_date": ["Y", "Y", "Y", "Y", "Y"]}).to_csv(csv, index=False)
    ir, dr = ex.compare(db, "small", str(csv))
    # Delta: no date in the release and none extracted counts as a match
    assert (ir, dr) == (0.75, 0.5)
