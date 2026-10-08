"""Extract traded events from insider-trading releases with a local model (Ollama).

Ollama's structured output takes a JSON schema in `format`; the reply is validated with pydantic,
then checked against the release text before it's stored:
- an event needs a named issuer (at least one capitalized word; "a pharmaceutical company" is dropped);
- announcement_date is kept only if the model's quoted evidence appears in the release and states
  that full date (month, day and year); otherwise it's set to null and date_check records why;
- events for the same issuer within a release are merged (instruments and directions combined);
- last_trade_date gets the same quote check; the matcher uses it when the announcement date is missing.

Runs are keyed by model and PROMPT_VERSION, so a prompt change re-extracts without deleting old rows.

Usage:
    uv run python -m insider_screen.extract run --model qwen3:8b --limit 20
    uv run python -m insider_screen.extract sample --model qwen3:8b --out data/handcheck.csv

Settings (shell or .env): OLLAMA_MODEL, OLLAMA_NUM_CTX (default 8192; 5120 fits qwen3:8b on an 8 GB GPU),
OLLAMA_THINK (false turns off qwen3's reasoning; leave unset for models without it).
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

from insider_screen.db import RELEASES

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
PROMPT_VERSION = "v4"
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
THINK = {"true": True, "false": False}.get(os.environ.get("OLLAMA_THINK", "").lower())
TABLE = f"extracted.traded_events_{PROMPT_VERSION}"
GATE = 0.90  # spec: >= 90% accuracy on issuer and announcement date

EventType = Literal["acquisition_target", "acquirer", "earnings", "clinical_or_regulatory", "financing",
                    "other", "unknown"]


class TradedEvent(BaseModel):
    issuer_name: str | None = Field(None, description="Named company whose securities were traded; null if unnamed")
    ticker: str | None = Field(None, description="Ticker only if stated in the release")
    announcement_date: date | None = Field(None, description="Date the news traded on became public")
    announcement_evidence: str | None = Field(
        None, description="Exact sentence fragment from the release, at most 40 words, that states the announcement date")
    last_trade_date: date | None = Field(None, description="Last date the defendants traded before the announcement")
    trade_evidence: str | None = Field(
        None, description="Exact sentence fragment from the release, at most 40 words, that states that trade date")
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

events: one per issuer whose securities were traded. The issuer is the company whose stock or options
the defendants bought or sold. It is not the defendant's employer, and not the source of the information,
unless that company's own securities were traded. Example: an analyst at a bank who used card data to
trade retailers' stock before their earnings: the issuers are the retailers, not the bank.
Skip issuers the release doesn't name (e.g. "a pharmaceutical company"). Use the company's name as written.

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

last_trade_date: the last date the release says the defendants traded that issuer before the news,
only if it states the full date. trade_evidence: copy the words that state it, unchanged.

instruments: stock, options, or both. direction: the position taken before the news, ignoring the sale
or cover that closed it afterward. long for buying stock or call options; short for short sales or buying
puts; both only if the defendants took long and short positions in that issuer before the news.

Example. Release text: "...Doe bought call options in Acme Corp. ahead of the March 4, 2019 announcement
that Acme would be acquired by Beta Inc..." Event: issuer_name "Acme Corp.", announcement_date 2019-03-04,
announcement_evidence "ahead of the March 4, 2019 announcement that Acme would be acquired by Beta Inc.",
last_trade_date null (no trade date stated), event_type acquisition_target, instruments options, direction long."""


def model_key(model: str) -> str:
    return f"{model}#{PROMPT_VERSION}"


def call_ollama(text: str, model: str, client: httpx.Client) -> ReleaseExtraction:
    payload = {
        "model": model,
        "stream": False,
        "format": ReleaseExtraction.model_json_schema(),
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
        **({"think": THINK} if THINK is not None else {}),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text[: (NUM_CTX - 1500) * 4]},
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


def verify_date(d: date | None, quote: str | None, text: str) -> tuple[date | None, str]:
    """Keep d only if the quote appears in the release and states d in full. Returns (date or None, reason)."""
    if d is None:
        return None, "none_given"
    if not quote or _norm(quote) not in _norm(text):
        return None, "quote_not_in_release"
    if not any(re.search(p, quote, re.IGNORECASE) for p in date_patterns(d)):
        return None, "quote_lacks_full_date"
    return d, "verified"


def check_date(ev: TradedEvent, text: str) -> tuple[date | None, str]:
    return verify_date(ev.announcement_date, ev.announcement_evidence, text)


NAMED_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.\-]*")
DESCRIPTOR_RE = re.compile(
    r"(?i:\b(?:unknown|unnamed|undisclosed|employer)\b)|\b(?i:company) [A-Z]\b|'s\s+[a-z]|\(")
UNNAMED_RE = re.compile(r"^(?:a|an|the|at least|several|various|certain|unnamed|unknown|null|none)\b", re.I)


def is_named(issuer: str | None) -> bool:
    if not issuer or issuer.strip().lower() in {"null", "none", "n/a"}:
        return False
    if DESCRIPTOR_RE.search(issuer):  # "Unknown", "Post's employer (pharmaceutical company)", "Company A"
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
        td, twhy = verify_date(ev.last_trade_date, ev.trade_evidence, text)
        key = _norm(re.sub(r"[^\w\s]", "", ev.issuer_name))
        row = rows.get(key)
        if row is None:
            rows[key] = {
                "issuer_name": ev.issuer_name.strip(), "ticker": ev.ticker,
                "announcement_date": d, "date_check": why, "announcement_evidence": ev.announcement_evidence,
                "last_trade_date": td, "trade_check": twhy, "trade_evidence": ev.trade_evidence,
                "event_types": [ev.event_type], "instruments": [ev.instruments], "directions": [ev.direction],
            }
        else:
            row["event_types"].append(ev.event_type)
            row["instruments"].append(ev.instruments)
            row["directions"].append(ev.direction)
            if row["announcement_date"] is None and d is not None:
                row.update(announcement_date=d, date_check=why, announcement_evidence=ev.announcement_evidence)
            if td is not None and (row["last_trade_date"] is None or td > row["last_trade_date"]):
                row.update(last_trade_date=td, trade_check=twhy, trade_evidence=ev.trade_evidence)
    out = []
    for r in rows.values():
        types = [t for t in r.pop("event_types") if t not in ("other", "unknown")]
        r["event_type"] = types[0] if types else "other"
        r["instruments"] = _combine(r.pop("instruments"))
        r["direction"] = _combine(r.pop("directions"))
        out.append(r)
    return out


def _record(con, row: tuple) -> None:
    """Replace the row for (lr_no, model). Delete-then-insert works whether or not the table kept its
    primary key; `db split` copies tables with CREATE TABLE AS, which drops constraints."""
    con.execute("DELETE FROM extracted.release_extractions WHERE lr_no = ? AND model = ?", [row[0], row[1]])
    con.execute("INSERT INTO extracted.release_extractions VALUES (?, ?, ?, ?, ?)", row)


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
        f"""CREATE TABLE IF NOT EXISTS {TABLE} (
            lr_no INTEGER, model VARCHAR, release_kind VARCHAR, is_insider_trading_case BOOLEAN,
            issuer_name VARCHAR, ticker VARCHAR, announcement_date DATE, date_check VARCHAR,
            announcement_evidence VARCHAR, last_trade_date DATE, trade_check VARCHAR, trade_evidence VARCHAR,
            event_type VARCHAR, instruments VARCHAR, direction VARCHAR)"""
    )
    todo = con.execute(
        """SELECT r.lr_no, r.text FROM raw.sec_litigation_releases r
           WHERE r.is_insider_candidate
             AND r.lr_no NOT IN (SELECT lr_no FROM extracted.release_extractions WHERE model = ? AND ok)
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
                _record(con, (lr_no, key, False, str(err)[:500], None))
                continue
            _record(con, (lr_no, key, True, None, ext.model_dump_json()))
            con.execute(f"DELETE FROM {TABLE} WHERE lr_no = ? AND model = ?", [lr_no, key])
            for ev in clean_events(ext, text):
                con.execute(
                    f"INSERT INTO {TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [lr_no, key, ext.release_kind, ext.is_insider_trading_case, ev["issuer_name"], ev["ticker"],
                     ev["announcement_date"], ev["date_check"], ev["announcement_evidence"],
                     ev["last_trade_date"], ev["trade_check"], ev["trade_evidence"], ev["event_type"],
                     ev["instruments"], ev["direction"]],
                )
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")
    con.execute(f"CREATE OR REPLACE VIEW extracted.traded_events AS SELECT * FROM {TABLE}")
    summary = con.execute(
        f"""SELECT count(DISTINCT lr_no) AS releases, count(*) AS events,
                  sum((date_check = 'verified')::INT) AS dated,
                  sum((date_check <> 'verified' AND trade_check = 'verified')::INT) AS trade_date_only,
                  sum((date_check IN ('quote_lacks_full_date', 'quote_not_in_release')
                       OR trade_check IN ('quote_lacks_full_date', 'quote_not_in_release'))::INT) AS rejected_dates,
                  sum((event_type = 'acquisition_target')::INT) AS acquisition_targets
           FROM {TABLE} WHERE model = ?""",
        [key],
    ).df()
    failed = con.execute(
        "SELECT count(*) FROM extracted.release_extractions WHERE model = ? AND NOT ok", [key]).fetchone()[0]
    no_events = con.execute(
        f"""SELECT count(*) FROM extracted.release_extractions x WHERE x.model = ? AND x.ok
            AND x.lr_no NOT IN (SELECT lr_no FROM {TABLE} WHERE model = ?)""", [key, key]).fetchone()[0]
    con.close()
    print(summary.to_string(index=False))
    print(f"{failed} releases failed extraction; {no_events} returned no named events")


HAND_FIELDS = ["issuer_name", "announcement_date", "event_type", "instruments", "direction"]


def sample(db: str, model: str, out: str, n: int = 100, seed: int = 42) -> None:
    """Write n random releases with the model's output and blank Y/N columns to fill by hand."""
    con = duckdb.connect(db, read_only=True)
    df = con.execute(
        f"""SELECT r.lr_no, r.url, t.issuer_name, t.announcement_date, t.date_check, t.last_trade_date,
                  t.event_type, t.instruments, t.direction
           FROM {TABLE} t JOIN raw.sec_litigation_releases r USING (lr_no)
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
    r.add_argument("--db", default=RELEASES)
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--limit", type=int)
    s = sub.add_parser("sample")
    s.add_argument("--db", default=RELEASES)
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
