"""Extract traded events from insider-trading releases with a local model (Ollama).

Ollama's structured output takes a JSON schema in `format`; the reply is validated
with pydantic, and a failed validation is retried once before being logged.

Usage:
    uv run python -m insider_screen.extract run --db data/insider.duckdb --model qwen2.5:14b
    uv run python -m insider_screen.extract sample --db data/insider.duckdb --out data/handcheck.csv
    uv run python -m insider_screen.extract score --csv data/handcheck.csv
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from typing import Literal

import duckdb
import httpx
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
GATE = 0.90  # spec: >= 90% accuracy on issuer and announcement date
NUM_CTX = 8192  # 12,000 chars of release (~3k tokens) + prompt + schema can pass Ollama's 4,096 default,
# and Ollama drops the start of an overlong prompt (the instructions) without an error


class TradedEvent(BaseModel):
    issuer_name: str = Field(description="Company whose securities were traded")
    ticker: str | None = Field(None, description="Ticker only if stated in the release")
    announcement_date: date | None = Field(None, description="Date the news traded on became public")
    event_type: Literal[
        "acquisition_target", "acquirer", "earnings", "clinical_or_regulatory", "financing", "other", "unknown"
    ]
    instruments: Literal["stock", "options", "both", "other", "unknown"]
    direction: Literal["long", "short", "both", "unknown"]
    first_trade_date: date | None = None
    last_trade_date: date | None = None


class ReleaseExtraction(BaseModel):
    release_kind: Literal["new_charges", "final_judgment", "settlement", "other"]
    is_insider_trading_case: bool
    events: list[TradedEvent]


SYSTEM = """You extract facts from SEC litigation releases. Use only facts stated in the text.
Return null for any field the release doesn't state. Do not infer tickers.
announcement_date is the date the information the defendants traded on became public,
not the filing or judgment date. One event per issuer traded.
Direction is "long" for purchases of stock or call options, "short" for short sales or put purchases.
If the release isn't about trading on material nonpublic information, set is_insider_trading_case to false
and return an empty events list."""


def call_ollama(text: str, model: str, client: httpx.Client) -> ReleaseExtraction:
    payload = {
        "model": model,
        "stream": False,
        "format": ReleaseExtraction.model_json_schema(),
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text[:12000]},
        ],
    }
    last_err: Exception | None = None
    for _ in range(2):
        resp = client.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        try:
            return ReleaseExtraction.model_validate_json(content)
        except ValidationError as err:
            last_err = err
    raise last_err  # type: ignore[misc]


def require_model(client: httpx.Client, model: str) -> None:
    """Exit with the installed list if Ollama doesn't have `model`; a missing model 404s on every call."""
    resp = client.get(f"{OLLAMA_URL}/api/tags", timeout=30)
    resp.raise_for_status()
    installed = sorted(m["name"] for m in resp.json().get("models", []))
    want = model if ":" in model else f"{model}:latest"
    if want not in installed:
        raise SystemExit(f"model {model!r} isn't installed in Ollama. Installed: {', '.join(installed) or 'none'}. "
                         f"Pull it with: ollama pull {model}")


def run(db: str, model: str, limit: int | None) -> None:
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA IF NOT EXISTS extracted")
    con.execute(
        """CREATE TABLE IF NOT EXISTS extracted.release_extractions (
            lr_no INTEGER, model VARCHAR, ok BOOLEAN, error VARCHAR, payload JSON,
            PRIMARY KEY (lr_no, model))"""
    )
    todo = con.execute(
        """SELECT r.lr_no, r.text FROM raw.sec_litigation_releases r
           WHERE r.is_insider_candidate
             AND r.lr_no NOT IN (SELECT lr_no FROM extracted.release_extractions WHERE model = ? AND ok)
           ORDER BY r.lr_no""",
        [model],
    ).fetchall()
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} releases to extract with {model}")
    failed: list[tuple[int, str]] = []
    with httpx.Client() as client:
        if todo:
            require_model(client, model)
        for i, (lr_no, text) in enumerate(todo, 1):
            try:
                ext = call_ollama(text, model, client)
                row = (lr_no, model, True, None, ext.model_dump_json())
            except (ValidationError, httpx.HTTPError, KeyError) as err:
                row = (lr_no, model, False, str(err)[:500], None)
                failed.append((lr_no, row[3]))
            con.execute("INSERT OR REPLACE INTO extracted.release_extractions VALUES (?, ?, ?, ?, ?)", row)
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")
    print(f"{len(todo) - len(failed)} extracted, {len(failed)} failed (retried on the next run)")
    for lr_no, err in failed[:5]:
        print(f"  LR {lr_no}: {err}")
    con.execute(
        """CREATE OR REPLACE VIEW extracted.traded_events AS
           SELECT x.lr_no, x.model,
                  json_extract_string(x.payload, '$.release_kind') AS release_kind,
                  CAST(json_extract(x.payload, '$.is_insider_trading_case') AS BOOLEAN) AS is_insider_trading_case,
                  e.unnest ->> 'issuer_name' AS issuer_name,
                  e.unnest ->> 'ticker' AS ticker,
                  TRY_CAST(e.unnest ->> 'announcement_date' AS DATE) AS announcement_date,
                  e.unnest ->> 'event_type' AS event_type,
                  e.unnest ->> 'instruments' AS instruments,
                  e.unnest ->> 'direction' AS direction,
                  TRY_CAST(e.unnest ->> 'first_trade_date' AS DATE) AS first_trade_date,
                  TRY_CAST(e.unnest ->> 'last_trade_date' AS DATE) AS last_trade_date
           FROM extracted.release_extractions x,
                unnest(CAST(json_extract(x.payload, '$.events') AS JSON[])) AS e
           WHERE x.ok"""
    )
    con.close()


HAND_FIELDS = ["issuer_name", "announcement_date", "event_type", "instruments", "direction"]


def sample(db: str, model: str, out: str, n: int = 100, seed: int = 42) -> None:
    """Write n random releases with the model's output and blank Y/N columns to fill by hand."""
    con = duckdb.connect(db, read_only=True)
    df = con.execute(
        """SELECT r.lr_no, r.url, t.issuer_name, t.announcement_date, t.event_type, t.instruments, t.direction
           FROM extracted.traded_events t JOIN raw.sec_litigation_releases r USING (lr_no)
           WHERE t.model = ?""",
        [model],
    ).df()
    con.close()
    picked = df.drop_duplicates("lr_no").sample(n=min(n, df.lr_no.nunique()), random_state=seed)
    for f in HAND_FIELDS:
        picked[f"ok_{f}"] = ""
    picked["missed_events"] = ""
    picked["notes"] = ""
    picked.to_csv(out, index=False)
    print(f"Wrote {len(picked)} rows to {out}. Fill ok_* with Y or N, and missed_events with a count.")


def score(csv: str) -> dict[str, float]:
    df = pd.read_csv(csv, dtype=str).fillna("")
    res = {}
    for f in HAND_FIELDS:
        col = df[f"ok_{f}"].str.upper().str.strip()
        filled = col.isin(["Y", "N"])
        res[f] = round((col[filled] == "Y").mean(), 3) if filled.any() else float("nan")
    for f, v in res.items():
        flag = "" if f not in ("issuer_name", "announcement_date") else ("PASS" if v >= GATE else "FAIL")
        print(f"{f:20s} {v:.3f} {flag}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--db", default="data/insider.duckdb")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--limit", type=int)
    s = sub.add_parser("sample")
    s.add_argument("--db", default="data/insider.duckdb")
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--out", default="data/handcheck.csv")
    c = sub.add_parser("score")
    c.add_argument("--csv", default="data/handcheck.csv")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.db, a.model, a.limit)
    elif a.cmd == "sample":
        sample(a.db, a.model, a.out)
    else:
        score(a.csv)


if __name__ == "__main__":
    main()
