import json

import duckdb
import httpx

from insider_screen import extract
from insider_screen.extract import ReleaseExtraction, call_ollama

GOOD = {
    "release_kind": "new_charges",
    "is_insider_trading_case": True,
    "events": [{
        "issuer_name": "Target Co.", "ticker": None, "announcement_date": "2023-06-05",
        "event_type": "acquisition_target", "instruments": "options", "direction": "long",
        "first_trade_date": None, "last_trade_date": "2023-06-02",
    }],
}


def _transport(replies):
    it = iter(replies)
    def handler(request):
        body = json.loads(request.content)
        assert body["format"]["title"] == "ReleaseExtraction"
        return httpx.Response(200, json={"message": {"content": next(it)}})
    return httpx.MockTransport(handler)


def test_call_ollama_retries_bad_json():
    client = httpx.Client(transport=_transport(["{not json", json.dumps(GOOD)]))
    out = call_ollama("text", "m", client)
    assert out.events[0].issuer_name == "Target Co."


def test_traded_events_view(tmp_path, monkeypatch):
    db = str(tmp_path / "t.duckdb")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA raw")
    con.execute("CREATE TABLE raw.sec_litigation_releases AS SELECT 1 AS lr_no, 'x' AS text, TRUE AS is_insider_candidate, 'u' AS url")
    con.close()
    monkeypatch.setattr(extract, "call_ollama", lambda text, model, client: ReleaseExtraction.model_validate(GOOD))
    extract.run(db, "m", None)
    con = duckdb.connect(db)
    row = con.execute("SELECT issuer_name, announcement_date::VARCHAR, instruments FROM extracted.traded_events").fetchone()
    assert row == ("Target Co.", "2023-06-05", "options")
    extract.run(db, "m", None)  # second run skips done rows
    assert con.execute("SELECT count(*) FROM extracted.release_extractions").fetchone()[0] == 1
