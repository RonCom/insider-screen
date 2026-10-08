# insider-screen

Screens material announcements by US-listed companies for abnormal trading beforehand, across options, exchange and off-exchange venues, and checks the screen against events the SEC later charged as insider trading. The plan and every expectation are in [docs/spec.md](docs/spec.md), written before any data was pulled.

## Setup

```powershell
uv sync
$env:SEC_USER_AGENT = "Chris L cflave@gmail.com"   # SEC requires a name and email
uv run pytest
```

## Labels: SEC litigation releases

```powershell
# 1. Download and parse releases from 2016 on (roughly 3,000-4,000 pages at 5 requests/second: 10-15 minutes the first run, cached after)
uv run python -m insider_screen.sec_releases --since 2016-01-01

# 2. Extract traded events from insider-trading candidates with a local model (Ollama running)
uv run python -m insider_screen.extract run --model gemma4:26b --limit 20   # check a few first
uv run python -m insider_screen.extract run --model gemma4:26b

# 3. Hand-check 100 releases: fill the ok_* columns with Y or N, then score (gate: 0.90 on issuer and date)
uv run python -m insider_screen.extract sample
uv run python -m insider_screen.extract score
```

Tables land in `data/insider.duckdb`: `raw.sec_litigation_releases`, `extracted.release_extractions`, and the view `extracted.traded_events`.

## Events: EDGAR 8-K filings

```powershell
# 1. Download the nightly bulk file of the Submissions API (several GB)
uv run python -m insider_screen.edgar download

# 2. Load filer details and the filing types the event rules use (raw.edgar_companies, raw.edgar_filings)
uv run python -m insider_screen.edgar load

# 3. Check the timestamp reading: earnings 8-Ks should cluster at 16-17h and 6-9h
uv run python -m insider_screen.edgar check-tz

# 4. Build events.announcements: earnings, acquisition targets, other material candidates; day 0 per NYSE calendar
uv run python -m insider_screen.edgar events
```

## Labels matched to events

```powershell
uv run python -m insider_screen.match --model gemma4:26b
```

Writes `labels.release_event_matches` (one row per extracted event, with a reason when unmatched) and the view `labels.charged_events`.

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
