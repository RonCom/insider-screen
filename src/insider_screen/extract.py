"""Extract traded events from insider-trading releases with a local model (Ollama).

Ollama's structured output takes a JSON schema in `format`; the reply is validated with pydantic,
then checked against the release text before it's stored:
- an event needs a named issuer (at least one capitalized word; "a pharmaceutical company" is dropped);
- announcement_date is kept only if the model's quoted evidence appears in the release and states
  that full date (month, day and year); otherwise it's set to null and date_check records why;
- events for the same issuer within a release are merged unless they conflict (different event types,
  different verified announcement dates, or verified last-trade dates more than 30 days apart), so two
  announcements for one issuer stay two rows;
- instruments are checked against the release: stock needs a mention of stock or shares, options a mention
  of options, calls or puts; employee options that were exercised and sold count as stock;
- an acquirer event is dropped when the same release has a target event (the target is what was traded);
- last_trade_date gets the same quote check; the matcher uses it when the announcement date is missing.

Runs are keyed by model and PROMPT_VERSION, so a prompt change re-extracts without deleting old rows.

Usage:
    uv run python -m insider_screen.extract run --model qwen3:8b --limit 20
    uv run python -m insider_screen.extract sample --model qwen3:8b --out data/handcheck.csv

Settings (shell or .env): OLLAMA_MODEL, OLLAMA_NUM_CTX (default 8192; 5120 fits qwen3:8b on an 8 GB GPU),
OLLAMA_THINK (false turns off qwen3's reasoning; leave unset for models without it).
    uv run python -m insider_screen.extract score --csv data/handcheck.csv
    uv run python -m insider_screen.extract reclean --model gemma4:e4b   # re-apply checks to stored output
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import date
from typing import Literal

import duckdb
import httpx
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from insider_screen.db import RELEASES

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
PROMPT_VERSION = "v5"
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
    direction: Literal["long", "short", "sell", "both", "unknown"]


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

events: one per announcement the defendants traded ahead of. If they traded one issuer ahead of two
different announcements (two earnings releases, a license deal and later an acquisition), return two events.
List every named issuer that was traded, including ones mentioned in a single sentence.
The issuer is the company whose stock or options the defendants bought or sold. It is not the defendant's
employer, and not the source of the information, unless that company's own securities were traded.
Example: an analyst at a bank who used card data to trade retailers' stock before their earnings: the issuers
are the retailers, not the bank. In an acquisition the traded issuer is almost always the company being
bought; return the buyer only if the release says the defendants traded the buyer's own securities.
Example: "trading ahead of Lumentum's acquisition of Coherent" -> issuer Coherent, not Lumentum.
Skip issuers the release doesn't name (e.g. "a pharmaceutical company"). Use the company's name as written.

event_type, from the news traded on:
- acquisition_target: the issuer agreed to be acquired, received a tender offer, or announced a merger in which it is bought.
- acquirer: the issuer announced it would buy another company.
- earnings: quarterly or annual results, or guidance.
- clinical_or_regulatory: drug trial results, FDA or other regulatory decisions.
- financing: a securities offering or financing.
- other: any other named news, including license, collaboration, supply or sales agreements between
  companies (these are not acquisitions, even between drug companies). unknown: the release doesn't say.
A SPAC that agrees to merge with a private company is the acquirer.

announcement_date: the date the news became public, only if the release states the full date
(month, day and year). If it gives only a month or year, or no date, return null.
announcement_evidence: copy the words from the release that state that date, unchanged.

last_trade_date: the last date the release says the defendants traded that issuer before the news,
only if it states the full date. trade_evidence: copy the words that state it, unchanged.

instruments: what the release says was traded. stock if it says stock, shares or ADRs; options if it says
options, calls or puts; both if it says both. If it says only "securities" or "traded", return unknown;
don't guess. Employee stock options that were exercised and the shares sold are stock, not options.

direction: the position taken before the news, ignoring the sale or cover that closed it afterward.
- long: bought stock or call options.
- short: short sales or bought put options. Only these.
- sell: sold shares the defendant already owned (including shares from exercised employee options),
  usually to avoid a loss before bad news. This is not short.
- both: long and short (or sell) positions in that issuer before the same news.
- unknown: the release doesn't say which way they traded.

Example. Release text: "...Doe bought call options in Acme Corp. ahead of the March 4, 2019 announcement
that Acme would be acquired by Beta Inc..." Event: issuer_name "Acme Corp.", announcement_date 2019-03-04,
announcement_evidence "ahead of the March 4, 2019 announcement that Acme would be acquired by Beta Inc.",
last_trade_date null (no trade date stated), event_type acquisition_target, instruments options, direction long."""


def model_key(model: str) -> str:
    """Rows are stored per model, prompt version and reasoning setting, so a run with OLLAMA_THINK=false
    never mixes with (or skips because of) a run with reasoning on."""
    return f"{model}#{PROMPT_VERSION}{'-nothink' if THINK is False else ''}"


WAIT_FOR_OLLAMA = 600  # seconds to keep retrying when Ollama isn't answering (starting up, loading the model)


def _post_waiting(client: httpx.Client, payload: dict) -> httpx.Response:
    """POST to Ollama; if it refuses the connection, wait and retry for up to WAIT_FOR_OLLAMA seconds."""
    waited, step = 0, 10
    while True:
        try:
            return client.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=900)
        except httpx.ConnectError:
            if waited >= WAIT_FOR_OLLAMA:
                raise
            print(f"    Ollama not answering at {OLLAMA_URL}; retrying in {step} s "
                  f"(is `ollama serve` or the Ollama app running?)", flush=True)
            time.sleep(step)
            waited += step


NOTES: list[str] = []  # how the last call_ollama got its answer, when it needed a retry


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
    NOTES.clear()
    last_err: Exception | None = None
    for _ in range(2):
        resp = _post_waiting(client, payload)
        resp.raise_for_status()
        body = resp.json()
        content = body["message"]["content"]
        if body.get("done_reason") == "length":
            # the context filled before the answer finished. If reasoning took the space (it can loop for
            # 30,000 characters without answering), retry once with reasoning off; otherwise retry once with
            # twice the context. Only these releases are affected; the retry is noted with the extraction.
            thought = len(body["message"].get("thinking") or "")
            last_err = ValueError(f"output cut off at the token limit ({len(content)} characters of answer, "
                                  f"{thought} of reasoning, context {payload['options']['num_ctx']})")
            if thought > max(2000, 4 * len(content)):
                payload = {**payload, "think": False}
                NOTES.append("retried without reasoning: reasoning filled the context")
            else:
                payload = {**payload, "options": {**payload["options"], "num_ctx": payload["options"]["num_ctx"] * 2}}
                NOTES.append("retried with twice the context")
            continue
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


ALIAS_RE = re.compile(r"""\s*\(\s*(?:the\s+)?["\u201c\u2018'][^()]{1,60}["\u201d\u2019']\s*\)""", re.I)


def clean_issuer(issuer: str | None) -> str | None:
    """Drop a defined-term alias: 'Potash Corporation of Saskatchewan ("Potash")' -> the name alone."""
    if issuer is None:
        return None
    return re.sub(r"\s+", " ", ALIAS_RE.sub("", issuer)).strip()


def is_named(issuer: str | None) -> bool:
    if not issuer or issuer.strip().lower() in {"null", "none", "n/a"}:
        return False
    if DESCRIPTOR_RE.search(issuer):  # "Unknown", "Post's employer (pharmaceutical company)", "Company A"
        return False
    return bool(NAMED_RE.search(issuer)) and not (UNNAMED_RE.match(issuer) and issuer[0].islower())


def _combine(values: list[str]) -> str:
    known = {v for v in values if v not in ("unknown",)}
    if not known:
        return "unknown"
    if len(known) == 1:
        return known.pop()
    if known <= {"stock", "options", "both"} or known <= {"long", "short", "sell", "both"}:
        return "both"
    return "other"


STOCK_RE = re.compile(
    r"\bstocks?\b(?!\s*(?:-\s*)?(?:broker|price|market|exchange|options?|trading plan|purchase plan))"
    r"|\bshares?\b(?!\s+(?:price|rose|fell|increased|decreased|dropped|jumped|climbed|declined))"
    r"|\bADRs?\b|\bADSs?\b|American Deposit[ao]ry", re.I)
EMPLOYEE_OPTION_RE = re.compile(
    r"\b(?:vested|employee|exercis\w*)\b[^.]{0,60}?\b(?:stock )?options?\b", re.I)
OPTION_RE = re.compile(r"\boptions?\b|\b(?:call|put)s\b", re.I)


def check_instruments(value: str, text: str) -> tuple[str, str]:
    """Hold the model's instrument to what the release says. Returns (instruments, reason).
    Employee options that were exercised count as stock, so they're removed before looking for options."""
    stock = bool(STOCK_RE.search(text))
    employee = bool(EMPLOYEE_OPTION_RE.search(text))
    options = bool(OPTION_RE.search(EMPLOYEE_OPTION_RE.sub(" ", text)))
    if value == "options" and not options:
        return ("stock", "employee_options") if employee else ("unknown", "options_not_in_release")
    if value == "stock" and not stock:
        return "unknown", "stock_not_in_release"
    if value == "both" and not (stock and options):
        return ("stock" if stock else "options" if options else "unknown"), "both_not_in_release"
    return value, "ok"


def _dates_conflict(a: date | None, b: date | None, days: int = 0) -> bool:
    return a is not None and b is not None and abs((a - b).days) > days


TRADE_GAP_DAYS = 30  # purchases weeks apart can precede one deal; further apart, separate news


def _types_conflict(a: str, b: str) -> bool:
    return a != b and "unknown" not in (a, b)


def clean_events(ext: ReleaseExtraction, text: str) -> list[dict]:
    rows: list[dict] = []
    for ev in ext.events:
        ev = ev.model_copy(update={"issuer_name": clean_issuer(ev.issuer_name)})
        if not is_named(ev.issuer_name):
            continue
        d, why = check_date(ev, text)
        td, twhy = verify_date(ev.last_trade_date, ev.trade_evidence, text)
        key = _norm(re.sub(r"[^\w\s]", "", ev.issuer_name))
        row = next((r for r in rows if r["key"] == key
                    and not _types_conflict(r["event_type"], ev.event_type)
                    and not _dates_conflict(r["announcement_date"], d)
                    and not _dates_conflict(r["last_trade_date"], td, TRADE_GAP_DAYS)), None)
        if row is None:
            rows.append({
                "key": key, "issuer_name": ev.issuer_name.strip(), "ticker": ev.ticker,
                "announcement_date": d, "date_check": why, "announcement_evidence": ev.announcement_evidence,
                "last_trade_date": td, "trade_check": twhy, "trade_evidence": ev.trade_evidence,
                "event_type": ev.event_type, "instruments": [ev.instruments], "directions": [ev.direction],
            })
            continue
        row["instruments"].append(ev.instruments)
        row["directions"].append(ev.direction)
        if row["event_type"] == "unknown":
            row["event_type"] = ev.event_type
        if row["announcement_date"] is None and d is not None:
            row.update(announcement_date=d, date_check=why, announcement_evidence=ev.announcement_evidence)
        if td is not None and (row["last_trade_date"] is None or td > row["last_trade_date"]):
            row.update(last_trade_date=td, trade_check=twhy, trade_evidence=ev.trade_evidence)
    if any(r["event_type"] == "acquisition_target" for r in rows):
        rows = [r for r in rows if r["event_type"] != "acquirer"]
    for r in rows:
        r.pop("key")
        if r["event_type"] == "unknown":
            r["event_type"] = "other"
        r["instruments"], r["instruments_check"] = check_instruments(_combine(r.pop("instruments")), text)
        r["direction"] = _combine(r.pop("directions"))
    return rows


def _record(con, row: tuple) -> None:
    """Replace the row for (lr_no, model). Delete-then-insert works whether or not the table kept its
    primary key; `db split` copies tables with CREATE TABLE AS, which drops constraints."""
    con.execute("DELETE FROM extracted.release_extractions WHERE lr_no = ? AND model = ?", [row[0], row[1]])
    con.execute("INSERT INTO extracted.release_extractions VALUES (?, ?, ?, ?, ?)", row)


def _insert_events(con, lr_no: int, key: str, ext: ReleaseExtraction, text: str) -> int:
    events = clean_events(ext, text)
    for ev in events:
        con.execute(
            f"INSERT INTO {TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [lr_no, key, ext.release_kind, ext.is_insider_trading_case, ev["issuer_name"], ev["ticker"],
             ev["announcement_date"], ev["date_check"], ev["announcement_evidence"],
             ev["last_trade_date"], ev["trade_check"], ev["trade_evidence"], ev["event_type"],
             ev["instruments"], ev["instruments_check"], ev["direction"]],
        )
    return len(events)


def run(db: str, model: str, limit: int | None, lrs_from: str | None = None) -> None:
    """lrs_from: a hand-check CSV; only its releases are extracted (to check a new model against it)."""
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
            event_type VARCHAR, instruments VARCHAR, instruments_check VARCHAR, direction VARCHAR)"""
    )
    todo = con.execute(
        """SELECT r.lr_no, r.text FROM raw.sec_litigation_releases r
           WHERE r.is_insider_candidate
             AND r.lr_no NOT IN (SELECT lr_no FROM extracted.release_extractions WHERE model = ? AND ok)
           ORDER BY r.lr_no""",
        [key],
    ).fetchall()
    if lrs_from:
        keep = set(pd.read_csv(lrs_from, dtype=str).lr_no.astype(int))
        todo = [t for t in todo if t[0] in keep]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} releases to extract with {key}", flush=True)
    start, failed_n = time.monotonic(), 0
    with httpx.Client() as client:
        for i, (lr_no, text) in enumerate(todo, 1):
            t0 = time.monotonic()
            try:
                ext = call_ollama(text, model, client)
                _record(con, (lr_no, key, True, "; ".join(NOTES) or None, ext.model_dump_json()))
                con.execute(f"DELETE FROM {TABLE} WHERE lr_no = ? AND model = ?", [lr_no, key])
                n_events = _insert_events(con, lr_no, key, ext, text)
                status = f"{n_events} event{'s' if n_events != 1 else ''}" + (f" ({'; '.join(NOTES)})" if NOTES else "")
            except (ValidationError, ValueError, httpx.HTTPError, KeyError) as err:
                _record(con, (lr_no, key, False, str(err)[:500], None))
                failed_n += 1
                first = str(err).strip().splitlines()[0][:120] if str(err).strip() else ""
                status = f"FAILED ({type(err).__name__}: {first})"
            took = time.monotonic() - t0
            elapsed = time.monotonic() - start
            left = (len(todo) - i) * elapsed / i
            print(f"  {i}/{len(todo)} LR {lr_no}: {status}, {took:.0f} s | average {elapsed / i:.0f} s, "
                  f"{elapsed / 3600:.1f} h so far, about {left / 3600:.1f} h left "
                  f"(done around {time.strftime('%a %H:%M', time.localtime(time.time() + left))})"
                  f"{f' | {failed_n} failed' if failed_n else ''}", flush=True)
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


def reclean(db: str, model: str) -> None:
    """Rebuild the events table for one model from the stored model output, without calling the model.
    Use after changing clean_events (filters, merging, date checks)."""
    key = model_key(model)
    con = duckdb.connect(db)
    rows = con.execute(
        """SELECT x.lr_no, x.payload, r.text FROM extracted.release_extractions x
           JOIN raw.sec_litigation_releases r USING (lr_no) WHERE x.model = ? AND x.ok""", [key]).fetchall()
    con.execute(f"DELETE FROM {TABLE} WHERE model = ?", [key])
    n = 0
    for lr_no, payload, text in rows:
        ext = ReleaseExtraction.model_validate_json(payload)
        n += _insert_events(con, lr_no, key, ext, text)
    con.close()
    print(f"Rebuilt {n} events from {len(rows)} stored extractions for {key}")


HAND_FIELDS = ["issuer_name", "announcement_date", "event_type", "instruments", "direction"]


def sample(db: str, model: str, out: str, n: int = 100, seed: int = 42) -> None:
    """Write n random releases with the model's output and blank Y/N columns to fill by hand."""
    con = duckdb.connect(db, read_only=True)
    df = con.execute(
        f"""SELECT r.lr_no, r.url, t.issuer_name, t.announcement_date, t.date_check, t.last_trade_date,
                  t.event_type, t.instruments, t.instruments_check, t.direction
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


def compare(db: str, model: str, csv: str) -> tuple[float, float]:
    """Check a model against a reviewed hand-check of another model. The known answers are the rows marked
    Y for both issuer and announcement date; the model passes a known answer when one of its events in the
    same release names that issuer (normalized names, fuzzy ratio >= 85) and, for date, gives the same date.
    Rows marked N have no recorded answer, so this is a check on the answers known, not a full hand-check."""
    from rapidfuzz import fuzz

    from insider_screen.match import normalize
    hc = pd.read_csv(csv, dtype=str).fillna("")
    known = hc[(hc.ok_issuer_name.str.upper() == "Y") & (hc.ok_announcement_date.str.upper() == "Y")]
    con = duckdb.connect(db, read_only=True)
    got = con.execute(f"SELECT lr_no, issuer_name, CAST(announcement_date AS VARCHAR) AS d FROM {TABLE} "
                      "WHERE model = ?", [model_key(model)]).df()
    con.close()
    got["lr_no"] = got.lr_no.astype(str)
    done = set(got.lr_no)
    missing = sorted(set(known.lr_no) - done)
    if missing:
        print(f"{len(missing)} hand-checked releases have no events from {model_key(model)} "
              f"(not extracted yet, failed, or no events found): {missing[:10]}")
    issuer_hits = date_hits = 0
    misses = []
    for r in known.itertuples(index=False):
        cand = got[got.lr_no == r.lr_no]
        names = [(fuzz.token_set_ratio(normalize(r.issuer_name), normalize(n or "")), d) for n, d in zip(cand.issuer_name, cand.d)]
        best = max(names, default=(0, None))
        if best[0] >= 85:
            issuer_hits += 1
            if ("" if pd.isna(best[1]) else best[1]) == (r.announcement_date or ""):  # both missing counts
                date_hits += 1
            else:
                misses.append((r.lr_no, r.issuer_name, f"date {'none' if pd.isna(best[1]) else best[1]} vs "
                                                        f"{r.announcement_date or 'none'}"))
        else:
            misses.append((r.lr_no, r.issuer_name, "issuer not found"))
    n = len(known)
    ir, dr = issuer_hits / n if n else float("nan"), date_hits / n if n else float("nan")
    print(f"{n} known answers. issuer_name {ir:.3f} {'PASS' if ir >= GATE else 'FAIL'}; "
          f"announcement_date {dr:.3f} {'PASS' if dr >= GATE else 'FAIL'}")
    for m in misses[:40]:
        print("  ", *m)
    return ir, dr


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
    r.add_argument("--lrs-from", help="only the releases in this hand-check CSV")
    cp = sub.add_parser("compare", help="check a model against the reviewed hand-check of another model")
    cp.add_argument("--db", default=RELEASES)
    cp.add_argument("--model", default=DEFAULT_MODEL)
    cp.add_argument("--csv", default="data/handcheck.csv")
    s = sub.add_parser("sample")
    s.add_argument("--db", default=RELEASES)
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--out", default="data/handcheck.csv")
    rc = sub.add_parser("reclean")
    rc.add_argument("--db", default=RELEASES)
    rc.add_argument("--model", default=DEFAULT_MODEL)
    c = sub.add_parser("score")
    c.add_argument("--csv", default="data/handcheck.csv")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.db, a.model, a.limit, a.lrs_from)
    elif a.cmd == "compare":
        compare(a.db, a.model, a.csv)
    elif a.cmd == "reclean":
        reclean(a.db, a.model)
    elif a.cmd == "sample":
        sample(a.db, a.model, a.out)
    else:
        score(a.csv)


if __name__ == "__main__":
    main()
