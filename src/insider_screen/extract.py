"""Extract traded events from insider-trading releases with a local model (Ollama).

Ollama's structured output takes a JSON schema in `format`; the reply is validated with pydantic,
then checked against the release text before it's stored:
- an event needs a named issuer (at least one capitalized word; "a pharmaceutical company" is dropped);
- announcement_date is kept only if the model's quoted evidence appears in the release and states
  that full date (month, day and year); otherwise it's set to null and date_check records why;
- events for the same issuer within a release are merged (instruments and directions combined).

Runs are keyed by model and PROMPT_VERSION, so a prompt change re-extracts without deleting old rows.

Usage:
    uv run python -m insider_screen.extract run --model gemma4:26b --limit 20
    uv run python -m insider_screen.extract sample --model gemma4:26b --out data/handcheck.csv
    uv run python -m insider_screen.extract score --csv data/handcheck.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date
from typing import Literal

import duckdb
import httpx
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:26b")
PROMPT_VERSION = "v2"
GATE = 0.90  # spec: >= 90% accuracy on issuer and announcement date

EventType = Literal["acquisition_target", "acquirer", "earnings", "clinical_or_regulatory", "financing",
                    "other", "unknown"]


class TradedEvent(BaseModel):
    issuer_name: str | None = Field(None, description="Named company whose securities were traded; null if unnamed")
    ticker: str | None = Field(None, description="Ticker only if stated in the release")
    announcement_date: date | None = Field(None, description="Date the news traded on became public")
    announcement_evidence: str | None = Field(
        None, description="Exact sentence fragment from the release, at most 40 words, that states the announcement date")
    event_type: EventType
    instruments: Literal["stock", "options", "both", "other", "unknown"]
    direction: Literal["long", "short", "both", "unknown"]


class ReleaseExtraction(BaseModel):
    release_kind: Literal["new_charges", "final_judgment", "settlement", "criminal_outcome", "other"]
    is_insider_trading_case: bool
    events: list[TradedEvent]


SYSTEM = """You extract facts from SEC litigation releases. Use only what the release states.

is_insider_trading_case: true if the release concerns trading, or tipping others to trade, on material
nonpublic information. False for cases about disclosure (Regulation FD), compliance policies, subpoenas,
or fraud without such trading.

release_kind:
- new_charges: the SEC announces it filed a complaint or brought charges.
- final_judgment: a court entered a judgment, injunction or penalty order.
- settlement: a defendant agreed or consented to settle, with no judgment described.
- criminal_outcome: the release reports a guilty plea, conviction or sentence in a parallel criminal case.
- other: anything else.

events: one per issuer whose securities were traded. Skip issuers the release doesn't name
(e.g. "a pharmaceutical company"). Use the company's name as written.

event_type, from the news traded on:
- acquisition_target: the issuer agreed to be acquired, received a tender offer, or announced a merger in which it is bought.
- acquirer: the issuer announced it would buy another company.
- earnings: quarterly or annual results, or guidance.
- clinical_or_regulatory: drug trial results, FDA or other regulatory decisions.
- financing: a securities offering or financing.
- other: any other named news. unknown: the release doesn't say.

announcement_date: the date the news became public, only if the release states the full date
(month, day and year). If it gives only a month or year, or no date, return null.
announcement_evidence: copy the words from the release that state that date, unchanged.

instruments: stock, options, or both. direction: long for buying stock or call options; short for
short sales or buying puts; both if the release describes each.

Example. Release text: "...Doe bought call options in Acme Corp. ahead of the March 4, 2019 announcement
that Acme would be acquired by Beta Inc..." Event: issuer_name "Acme Corp.", announcement_date 2019-03-04,
announcement_evidence "ahead of the March 4, 2019 announcement that Acme would be acquired by Beta Inc.",
event_type acquisition_target, instruments options, direction long."""


def model_key(model: str) -> str:
    return f"{model}#{PROMPT_VERSION}"


def call_ollama(text: str, model: str, client: httpx.Client) -> ReleaseExtraction:
    payload = {
        "model": model,
        "stream": False,
        "format": ReleaseExtraction.model_json_schema(),
        "options": {"temperature": 0, "num_ctx": 8192},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text[:16000]},
        ],
    }
    last_err: Exception | None = None
    for _ in range(2):
        resp = client.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        try:
            return ReleaseExtraction.model_validate_json(content)
        except ValidationError as err:
            last_err = err
    raise last_err  # type: ignore[misc]


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def date_patterns(d: date) -> list[str]:
    m = MONTHS[d.month - 1]
    return [
        rf"{m}\.?\s+{d.day},?\s+{d.year}",
        rf"{m[:3]}\.?\s+{d.day},?\s+{d.year}",
        rf"\b{d.month}/{d.day}/{d.year}\b",
        rf"\b{d.month:02d}/{d.day:02d}/{d.year}\b",
    ]


def check_date(ev: TradedEvent, text: str) -> tuple[date | None, str]:
    """Return (date or None, reason)."""
    if ev.announcement_date is None:
        return None, "none_given"
    quote = ev.announcement_evidence or ""
    if not quote or _norm(quote) not in _norm(text):
        return None, "quote_not_in_release"
    if not any(re.search(p, quote, re.IGNORECASE) for p in date_patterns(ev.announcement_date)):
        return None, "quote_lacks_full_date"
    return ev.announcement_date, "verified"


NAMED_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.\-]*")
UNNAMED_RE = re.compile(r"^(?:a|an|the|at least|several|various|certain|unnamed|unknown|null|none)\b", re.I)


def is_named(issuer: str | None) -> bool:
    if not issuer or issuer.strip().lower() in {"null", "none", "n/a"}:
        return False
    return bool(NAMED_RE.search(issuer)) and not (UNNAMED_RE.match(issuer) and issuer[0].islower())


def _combine(values: list[str], both: str = "both") -> str:
    known = {v for v in values if v not in ("unknown",)}
    if not known:
        return "unknown"
    if len(known) == 1:
        return known.pop()
    if known <= {"stock", "options", "both"} or known <= {"long", "short", "both"}:
        return both
    return "other"


def clean_events(ext: ReleaseExtraction, text: str) -> list[dict]:
    rows: dict[str, dict] = {}
    for ev in ext.events:
        if not is_named(ev.issuer_name):
            continue
        d, why = check_date(ev, text)
        key = _norm(re.sub(r"[^\w\s]", "", ev.issuer_name))
        row = rows.get(key)
        if row is None:
            rows[key] = {
                "issuer_name": ev.issuer_name.strip(), "ticker": ev.ticker,
                "announcement_date": d, "date_check": why, "announcement_evidence": ev.announcement_evidence,
                "event_types": [ev.event_type], "instruments": [ev.instruments], "directions": [ev.direction],
            }
        else:
            row["event_types"].append(ev.event_type)
            row["instruments"].append(ev.instruments)
            row["directions"].append(ev.direction)
            if row["announcement_date"] is None and d is not None:
                row.update(announcement_date=d, date_check=why, announcement_evidence=ev.announcement_evidence)
    out = []
    for r in rows.values():
        types = [t for t in r.pop("event_types") if t not in ("other", "unknown")]
        r["event_type"] = types[0] if types else "other"
        r["instruments"] = _combine(r.pop("instruments"))
        r["direction"] = _combine(r.pop("directions"))
        out.append(r)
    return out


def run(db: str, model: str, limit: int | None) -> None:
    key = model_key(model)
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA IF NOT EXISTS extracted")
    con.execute(
        """CREATE TABLE IF NOT EXISTS extracted.release_extractions (
            lr_no INTEGER, model VARCHAR, ok BOOLEAN, error VARCHAR, payload JSON,
            PRIMARY KEY (lr_no, model))"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS extracted.traded_events_v2 (
            lr_no INTEGER, model VARCHAR, release_kind VARCHAR, is_insider_trading_case BOOLEAN,
            issuer_name VARCHAR, ticker VARCHAR, announcement_date DATE, date_check VARCHAR,
            announcement_evidence VARCHAR, event_type VARCHAR, instruments VARCHAR, direction VARCHAR)"""
    )
    todo = con.execute(
        """SELECT r.lr_no, r.text FROM raw.sec_litigation_releases r
           WHERE r.is_insider_candidate
             AND r.lr_no NOT IN (SELECT lr_no FROM extracted.release_extractions WHERE model = ?)
           ORDER BY r.lr_no""",
        [key],
    ).fetchall()
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} releases to extract with {key}")
    with httpx.Client() as client:
        for i, (lr_no, text) in enumerate(todo, 1):
            try:
                ext = call_ollama(text, model, client)
            except (ValidationError, httpx.HTTPError, KeyError) as err:
                con.execute("INSERT OR REPLACE INTO extracted.release_extractions VALUES (?, ?, ?, ?, ?)",
                            (lr_no, key, False, str(err)[:500], None))
                continue
            con.execute("INSERT OR REPLACE INTO extracted.release_extractions VALUES (?, ?, ?, ?, ?)",
                        (lr_no, key, True, None, ext.model_dump_json()))
            con.execute("DELETE FROM extracted.traded_events_v2 WHERE lr_no = ? AND model = ?", [lr_no, key])
            for ev in clean_events(ext, text):
                con.execute(
                    "INSERT INTO extracted.traded_events_v2 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [lr_no, key, ext.release_kind, ext.is_insider_trading_case, ev["issuer_name"], ev["ticker"],
                     ev["announcement_date"], ev["date_check"], ev["announcement_evidence"], ev["event_type"],
                     ev["instruments"], ev["direction"]],
                )
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")
    con.execute("CREATE OR REPLACE VIEW extracted.traded_events AS SELECT * FROM extracted.traded_events_v2")
    summary = con.execute(
        """SELECT count(DISTINCT lr_no) AS releases, count(*) AS events,
                  sum((date_check = 'verified')::INT) AS dated,
                  sum((date_check = 'quote_lacks_full_date')::INT) AS partial_date,
                  sum((date_check = 'quote_not_in_release')::INT) AS quote_mismatch,
                  sum((event_type = 'acquisition_target')::INT) AS acquisition_targets
           FROM extracted.traded_events_v2 WHERE model = ?""",
        [key],
    ).df()
    con.close()
    print(summary.to_string(index=False))


HAND_FIELDS = ["issuer_name", "announcement_date", "event_type", "instruments", "direction"]


def sample(db: str, model: str, out: str, n: int = 100, seed: int = 42) -> None:
    """Write n random releases with the model's output and blank Y/N columns to fill by hand."""
    con = duckdb.connect(db, read_only=True)
    df = con.execute(
        """SELECT r.lr_no, r.url, t.issuer_name, t.announcement_date, t.date_check, t.event_type,
                  t.instruments, t.direction
           FROM extracted.traded_events_v2 t JOIN raw.sec_litigation_releases r USING (lr_no)
           WHERE t.model = ?""",
        [model_key(model)],
    ).df()
    con.close()
    ids = pd.Series(df.lr_no.unique()).sample(n=min(n, df.lr_no.nunique()), random_state=seed)
    picked = df[df.lr_no.isin(ids)].copy()
    for f in HAND_FIELDS:
        picked[f"ok_{f}"] = ""
    picked["missed_events"] = ""
    picked["notes"] = ""
    picked.to_csv(out, index=False)
    print(f"Wrote {len(picked)} rows from {len(ids)} releases to {out}. Fill ok_* with Y or N.")


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
