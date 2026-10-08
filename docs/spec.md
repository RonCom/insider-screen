# Pre-announcement trading screen: specification

Written before any data is pulled or code runs. Results go in `docs/results.md`, scored against each expectation below. Changes after data is seen are logged at the bottom with a date and reason.

## Question

Across material announcements by US-listed companies from 2016 to 2025, which events show abnormal trading in the 20 trading days before the announcement, across options, exchange and off-exchange venues? Do events the SEC later charged as insider trading rank at the top when scored without access to the charges?

## Data

| Source | Use | Cost | Pre-check before building |
| --- | --- | --- | --- |
| SEC EDGAR 8-K filings | Event list and announcement timestamps | Free; declared User-Agent, 10 requests/second limit | Acceptance timestamps present for 2016–2025 |
| SEC litigation releases | Insider-trading labels | Free, public domain | Count of insider-trading releases 2016–2026 |
| FINRA Monthly Short Sale Transaction Files | Off-exchange short sales: trade time, price, size | Free | Coverage of symbols that later delisted |
| Massive (formerly Polygon) stocks | Daily and minute bars, total volume, delisted tickers | $29–79/mo | Delisted tickers return full history on trial |
| Massive options | Daily option volume and prices by contract | Separate subscription | Expired contracts available back to 2016 |

If the Massive pre-checks fail, the options features are dropped and H2 is reported as not tested.

## Events

- **Universe:** US common stocks with an 8-K in the event types below. ADRs, funds and stocks under $1 or under $50M market cap at day −30 are excluded.
- **Event types:** each 8-K is tagged with one of three types:
  - M&A target: Item 1.01 with a merger agreement exhibit, where the filer is the target
  - Earnings: Item 2.02
  - Other material: Item 8.01 or 7.01 with a same-day absolute abnormal return above 10%
- **Event day 0:** the 8-K acceptance time. A filing accepted after 16:00 ET sets day 0 to the next trading day. For 50 sampled M&A events, the 8-K time is compared against the press-release time; if more than 5 of the 50 differ by a trading day or more, day 0 switches to the press-release time.

## Labels

1. **Extraction:** pull SEC litigation releases from 2016 to 2026-09-30 tagged or describing insider trading. A local model (Ollama) extracts these fields into a fixed schema:
   - issuer name, and ticker if stated
   - announcement date
   - instrument (stock, options, or both)
   - direction (long or short)
   - trade dates if stated
2. **Hand-check:** check 100 releases by hand. The gate is ≥90% field accuracy on issuer and announcement date; below that, fix the prompt or schema and re-check before going further.
3. **Matching:** match releases to events by issuer CIK and announcement date within ±3 trading days. Unmatched releases are listed with a reason.
4. **Positive definition:** an event is positive if at least one matched release charges trading before it. All other events are unlabeled, not negative, because most insider trading is never charged.

## Features

The pre-event window is days −20 to −1. The baseline window is days −250 to −31.

- **Stock:** abnormal volume against the baseline; cumulative abnormal return from a market model fitted on the baseline; and the share of volume in the last 5 days of the window.
- **Off-exchange (FINRA):** short-sale share of off-exchange volume against the baseline, and median short-sale trade size against the baseline.
- **Options:** total volume against the baseline; out-of-the-money call volume share (put share for negative events); share of volume in contracts expiring within 30 days of day 0; and the change in at-the-money implied volatility.
- **Peer benchmarking:** every feature is expressed against events of the same type, market-cap quintile and sector.
- **Shrinkage:** an empirical-Bayes adjustment pulls estimates for stocks with fewer than 20 baseline days of options trading toward their peer group.

No feature uses data timestamped at or after day 0. A unit test enforces this on every build.

## Scores

1. **Composite:** mean of robust z-scores across features.
2. **Isolation Forest:** fitted on all events in the development period.
3. **Supervised model:** gradient-boosted classifier on development-period labels (positive vs. unlabeled), reported with that caveat.

## Splits

- **Development:** events from 2016 to 2020. All tuning happens here.
- **Test:** events from 2021 to 2025, scored once after the spec and code are frozen.
- **Label lag:** SEC charges trail the trades by years, so test-period events are under-labeled. Results report positives per event year alongside every metric.

## Expectations

| ID | Test | Expectation | If it fails |
| --- | --- | --- | --- |
| H1 | Top-5% lift of charged events, M&A targets, test period, best score | ≥ 3× | Report as failed; check whether lift appears within the top 1% |
| H2 | Options features added to stock-only features, M&A targets, test AUC | Gain ≥ 0.03 | Report; check whether options-only cases (from labels) are ranked higher |
| H3 | Off-exchange short-sale features added, negative-return earnings events | Gain < 0.02 (expected to add little) | If gain ≥ 0.02, report as a finding and re-test on a 2024–2025 holdout |
| H4 | Shrinkage vs. unshrunk z, share of top-5% alerts from the smallest market-cap quintile | Falls by ≥ 25% | Report; keep both versions in the dashboard |
| H5 | Placebo window (days −70 to −51), top-5% lift of charged events | ≤ 1.5× | Lift above 1.5× means the score picks up stock traits, not pre-event trading; fix before reporting H1 |
| H6 | Alert budget of 50 events per month, share of test-period charged events captured | ≥ 25% | Report the budget needed to reach 25% |

## Limits stated in advance

- SEC cases include cases found through surveillance screens of this kind, which inflates measured lift.
- Charged events skew toward large, liquid names and M&A, where trading leaves records. Lift on other event types will rest on few positives.
- FINRA short-sale files cover reported off-exchange short sales only and aren't combined with exchange data.
- Unlabeled events include uncharged insider trading, so precision is understated.

## Build

uv, DuckDB, dbt with a Snowflake target, MLflow for runs, Streamlit alert queue with per-event drill-down, and GitHub Actions running dbt tests and unit tests on every push.

## Milestones

1. Labels extracted, hand-checked and matched to events.
2. Event table and FINRA short-sale ingestion.
3. Stock and off-exchange features; scores; development results; placebo test (H5).
4. Options features (H2).
5. Frozen test-period run; `results.md`; write-up and dashboard.

## Change log

| Date | Change | Reason |
| --- | --- | --- |
