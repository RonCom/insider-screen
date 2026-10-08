# insider-screen

Screens material announcements by US-listed companies for abnormal trading beforehand, across options, exchange and off-exchange venues, and checks the screen against events the SEC later charged as insider trading. The plan and every expectation are in [docs/spec.md](docs/spec.md), written before any data was pulled.

## Setup

```powershell
uv sync
copy .env.example .env   # then fill in .env: API keys and your SEC User-Agent (name and email)
uv run pytest
```

`.env` is listed in `.gitignore`, so keys in it stay on your machine. The package loads it on import; a variable already set in the shell takes precedence.

## Database files

Each loader writes its own DuckDB file, so long runs (an extraction and a price load, say) can go at the same time. DuckDB lets one process write a file at a time.

| File | Written by |
| --- | --- |
| `data/releases.duckdb` | `sec_releases`, `extract`, `match` |
| `data/edgar.duckdb` | `edgar` |
| `data/finra.duckdb` | `shortsale daily`, `shortsale monthly` |
| `data/prices.duckdb` | `prices` |

`match` reads `edgar.duckdb` and `prices` reads `finra.duckdb`, both read-only; each fails while the other file's loader is running. Data from before this split, in `data/insider.duckdb`, moves over with:

```powershell
uv run python -m insider_screen.db split
```

## Labels: SEC litigation releases

```powershell
# 1. Download and parse releases from 2016 on (roughly 3,000-4,000 pages at 5 requests/second: 10-15 minutes the first run, cached after)
uv run python -m insider_screen.sec_releases --since 2016-01-01

# Release dates are read from the header line; the run lists any missing or out of sequence with
# neighbouring release numbers. After a parser change, re-read dates from the stored text (no download):
uv run python -m insider_screen.sec_releases --fix-dates

# 2. Extract traded events from insider-trading candidates with a local model (Ollama running).
#    On an 8 GB GPU: set OLLAMA_FLASH_ATTENTION=1 and OLLAMA_KV_CACHE_TYPE=q8_0 for Ollama itself (setx, then
#    restart Ollama), and OLLAMA_NUM_CTX=5120 and OLLAMA_THINK=false in .env for qwen3:8b.
uv run python -m insider_screen.extract run --model qwen3:8b --limit 20   # check a few first
uv run python -m insider_screen.extract run --model qwen3:8b

# 3. Hand-check 100 releases: fill the ok_* columns with Y or N, then score (gate: 0.90 on issuer and date)
uv run python -m insider_screen.extract sample
uv run python -m insider_screen.extract score
```

Tables land in `data/releases.duckdb`: `raw.sec_litigation_releases`, `extracted.release_extractions`, `extracted.traded_events_v3`, and the view `extracted.traded_events` over it.

## Events: EDGAR 8-K filings

```powershell
# 1. Download the nightly bulk file of the Submissions API (several GB)
uv run python -m insider_screen.edgar download

# 2. Load filer details and the filing types the event rules use (raw.edgar_companies, raw.edgar_filings)
uv run python -m insider_screen.edgar load

# 3. Check the timestamps: hour histogram of earnings 8-Ks, then a sample compared with each filing's
#    index header (Eastern time). The bulk-file times carry 0, 1 or 2 times the UTC offset.
uv run python -m insider_screen.edgar check-tz
uv run python -m insider_screen.edgar tz-sample

# 4. Build events.announcements, then read exact acceptance times for acquisition targets from their
#    index headers (~2,700 requests, ~10 minutes) and rebuild. Other events use the earliest consistent day 0.
uv run python -m insider_screen.edgar events
uv run python -m insider_screen.edgar exact-times
uv run python -m insider_screen.edgar events
```

## Labels matched to events

```powershell
uv run python -m insider_screen.match --model qwen3:8b
```

Writes `labels.release_event_matches` (one row per extracted event, with a reason when unmatched) and `labels.charged_events` (every event with an `is_charged` flag). Reads events from `data/edgar.duckdb`.

## Off-exchange short sales: FINRA Reg SHO files

```powershell
# 1. Check the file paths respond for sample dates before the full pull
uv run python -m insider_screen.shortsale probe

# 2. Daily short and total off-exchange volume per symbol (raw.finra_short_daily); about 2,750 sessions at 2 requests/second
uv run python -m insider_screen.shortsale daily --start 2015-01-01 --end 2025-12-31

# 3. Trade-level files aggregated to daily counts (raw.finra_short_trades_daily); streamed, aggregated and deleted.
#    Nasdaq TRF months are split into parts; August 2026 was four files of about 1 GB each. Plan for a long run.
uv run python -m insider_screen.shortsale monthly --start 2015-01 --end 2025-12
```

## Stock prices: Alpaca daily bars

```powershell
# Free Alpaca account (paper trading is enough); put its keys in .env as ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY

# 1. Test run on a few symbols, including delisted ones
uv run python -m insider_screen.prices --symbols CELG,TWTR,ATVI --db data/test.duckdb

# 2. Every symbol in raw.finra_short_daily plus SPY, raw and adjusted (raw.alpaca_bars_daily in data/prices.duckdb); resumable
uv run python -m insider_screen.prices --start 2016-01-01 --end 2025-12-31
```

## Hand checks

```powershell
# Extraction (spec gate: 0.90 on issuer and announcement date)
uv run python -m insider_screen.extract sample     # data/handcheck.csv: fill ok_* with Y or N
uv run python -m insider_screen.extract score

# Day 0 for acquisition targets (spec: switch to press-release time if more than 5 of 50 differ)
uv run python -m insider_screen.day0check sample   # data/day0_check.csv: fill press_release_et
uv run python -m insider_screen.day0check score
```
