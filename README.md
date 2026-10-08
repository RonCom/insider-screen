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
uv run python -m insider_screen.extract run --model qwen2.5:14b --limit 20   # check a few first
uv run python -m insider_screen.extract run --model qwen2.5:14b

# 3. Hand-check 100 releases: fill the ok_* columns with Y or N, then score (gate: 0.90 on issuer and date)
uv run python -m insider_screen.extract sample
uv run python -m insider_screen.extract score
```

Tables land in `data/insider.duckdb`: `raw.sec_litigation_releases`, `extracted.release_extractions`, and the view `extracted.traded_events`.
