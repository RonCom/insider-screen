# Pre-announcement trading screen: specification

Written before any data is pulled or code runs. Results go in `docs/results.md`, scored against each expectation below. Changes after data is seen are logged at the bottom with a date and reason.

## Question

Across material announcements by US-listed companies from 2017 to 2025, which events show abnormal trading in the 20 trading days before the announcement, across options, exchange and off-exchange venues? Do events the SEC later charged as insider trading rank at the top when scored without access to the charges?

## Data

| Source | Use | Cost | Pre-check before building |
| --- | --- | --- | --- |
| SEC EDGAR 8-K filings | Event list and announcement timestamps | Free; declared User-Agent, 10 requests/second limit | Acceptance timestamps present for 2016–2025 |
| SEC litigation releases | Insider-trading labels | Free, public domain | Count of insider-trading releases 2016–2026 |
| FINRA Daily Short Sale Volume Files | Off-exchange short and total volume per symbol per day, 2015–2025 | Free | Files available back to 2015 at the CDN path |
| FINRA Monthly Short Sale Transaction Files | Off-exchange short trades with size, 2015–2025 | Free | Coverage of symbols that later delisted |
| Alpaca Market Data, free plan (SIP feed) | Daily bars, split- and dividend-adjusted and raw, total volume, delisted tickers, 2016–2025 | Free; 200 requests/minute | Delisted tickers return history: passed 2026-10-08 (CELG, TWTR, ATVI); history starts January 2016 |
| Massive reference data, free Basic key | Dated ticker-to-CIK map, delisted tickers included | Free; 5 requests/minute | Delisted tickers carry a CIK |
| ~~Massive stocks~~ | Replaced by Alpaca (see change log) | | |
| ~~Massive options~~ | Dropped (see change log) | | |

No free source has expired option contracts back to 2017, so the options features are dropped and H2 and H4 are reported as not tested.

## Events

- **Universe:** US common stocks with an 8-K in the event types below. ADRs, funds and stocks under $1 or under $50M market cap at day −30 are excluded.
- **Event types:** each 8-K is tagged with one of three types:
  - M&A target: Item 1.01 with a merger agreement exhibit, where the filer is the target
  - Earnings: Item 2.02
  - Other material: Item 8.01 or 7.01 with a same-day absolute abnormal return above 10%
- **Event day 0:** the 8-K acceptance time. A filing accepted after 16:00 ET sets day 0 to the next trading day. Acquisition targets take the acceptance time from the filing's index header. Other events take the earliest day 0 consistent with EDGAR's hours and the filing date, because the bulk-file time can carry 0, 1 or 2 times the UTC offset; the pre-event window then never contains the announcement, and an ambiguous event can lose its last pre-event day. For 50 sampled M&A events, the 8-K time is compared against the time the news reached the market (first abnormal minute-bar move; see change log); if more than 5 of the 50 differ by a trading day or more, day 0 switches to the press-release time.

## Labels

1. **Extraction:** pull SEC litigation releases from 2016 to 2026-09-30 tagged or describing insider trading. A local model (Ollama) extracts these fields into a fixed schema:
   - issuer name, and ticker if stated
   - announcement date
   - instrument (stock, options, or both)
   - direction (long, short, or sell: shares already held sold before bad news)
   - trade dates if stated
2. **Hand-check:** check 100 releases by hand. The gate is ≥90% field accuracy on issuer and announcement date; below that, fix the prompt or schema and re-check before going further.
3. **Matching:** match releases to events by issuer CIK and announcement date within ±3 trading days. Unmatched releases are listed with a reason.
4. **Positive definition:** an event is positive if at least one matched release charges trading before it. All other events are unlabeled, not negative, because most insider trading is never charged.

## Features

The pre-event window is days −20 to −1. The baseline window is days −250 to −31.

- **Stock:** abnormal volume against the baseline; cumulative abnormal return from a market model fitted on the baseline; and the share of volume in the last 5 days of the window.
- **Off-exchange (FINRA):** short-sale share of off-exchange volume against the baseline, from the daily files. Short-sale trade counts and the share of trades of 100 shares or fewer, from the monthly transaction files.
- **Options (dropped, see change log):** total volume against the baseline; out-of-the-money call volume share (put share for negative events); share of volume in contracts expiring within 30 days of day 0; and the change in at-the-money implied volatility.
- **Peer benchmarking:** every feature is expressed against events of the same type, market-cap quintile and sector.
- **Shrinkage (dropped with the options features):** an empirical-Bayes adjustment pulls estimates for stocks with fewer than 20 baseline days of options trading toward their peer group.

No feature uses data timestamped at or after day 0. A unit test enforces this on every build.

## Scores

1. **Composite:** mean of robust z-scores across features.
2. **Isolation Forest:** fitted on all events in the development period.
3. **Supervised model:** gradient-boosted classifier on development-period labels (positive vs. unlabeled), reported with that caveat.

## Splits

- **Development:** events from 2017 to 2020. All tuning happens here.
- **Test:** events from 2021 to 2025, scored once after the spec and code are frozen.
- **Label lag:** SEC charges trail the trades by years, so test-period events are under-labeled. Results report positives per event year alongside every metric.

## Expectations

| ID | Test | Expectation | If it fails |
| --- | --- | --- | --- |
| H1 | Top-5% lift of charged events, M&A targets, test period, best score | ≥ 3× | Report as failed; check whether lift appears within the top 1% |
| H2 | Options features added to stock-only features, M&A targets, test AUC | Gain ≥ 0.03 | Not tested: options features dropped (see change log) |
| H3 | Off-exchange short-sale features added, negative-return earnings events | Gain < 0.02 (expected to add little) | If gain ≥ 0.02, report as a finding and re-test on a 2024–2025 holdout |
| H4 | Shrinkage vs. unshrunk z, share of top-5% alerts from the smallest market-cap quintile | Falls by ≥ 25% | Not tested: shrinkage was defined on options trading days (see change log) |
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
4. ~~Options features (H2).~~ Dropped (see change log).
5. Frozen test-period run; `results.md`; write-up and dashboard.

## Change log

| Date | Change | Reason |
| --- | --- | --- |
| 2026-10-08 | Off-exchange short share from the daily files; trade-size features development-only | FINRA's TRF pages list monthly transaction files only through 2021, and its ADF files are empty after January 2015. Seen before any data pull. |
| 2026-10-08 | Previous change reversed: trade-size features return for all years | FINRA's data catalog lists the monthly files for 2009 through 2026 at a new location (cdn.finra.org); the old TRF pages were out of date. Found the same day, before any data pull. |
| 2026-10-08 | Alpaca's free plan replaces Massive for daily stock bars; Massive's free reference data supplies the ticker-to-CIK map | Free source that passed the delisted-ticker pre-check (CELG, TWTR, ATVI returned full bars). Massive stocks would cost $79–199 for the history needed. Before any price data pull. |
| 2026-10-08 | Development period starts in 2017 (was 2016); question range is 2017–2025 | Alpaca's history starts in January 2016 (a request for January 2015 returned no bars), so 2016 events have no full −250-day baseline. Test period unchanged. Before any price data pull. |
| 2026-10-08 | Options features and shrinkage dropped; H2 and H4 reported as not tested | No free source of expired option contracts back to 2017 (Twelve Data, Yahoo Finance, Interactive Brokers and Alpaca checked); the spec's fallback for a failed options pre-check. Before any price data pull. |
| 2026-10-08 | Day 0 from index headers for acquisition targets; earliest consistent day 0 for other events | A check of 90 bulk-file times against filing index headers found offsets of 0, 4–5 and 8–10 hours from Eastern time in both parts of the bulk file. Before any feature was computed. |
| 2026-10-08 | Direction adds `sell` (shares already held, sold before bad news); `short` is short sales and puts only | The first hand-check pass found 11 of 100 releases where defendants sold held shares to avoid a loss, which neither long nor short describes. A sale shows in trading data as selling, not short selling. Before any event was matched. |
| 2026-10-08 | Hand-check done as a model-assisted first pass, then every row reviewed by hand | The release pages couldn't be fetched in the environment used; a second model filled the first pass from the stored release texts. The person running the study reviews all rows against the releases before scoring. |
| 2026-10-09 | Day-0 check compares the 8-K time with the first abnormal market move (minute bars), not the newswire time | Newswire times couldn't be collected by script: GlobeNewswire's search stalls, PR Newswire's returned no results, Business Wire has no usable search, and DuckDuckGo and Bing refuse scripted queries. The move rule (at least 5% or 3 daily standard deviations from the prior close, 5x median minute volume, held 15 minutes, searched from two sessions before day 0) was set before any event was checked. A move before day 0 can come from a leak or press report; it counts, since that is when the news was public. Same pass rule: more than 5 of 50 a session or more apart switches day 0. |
| 2026-10-09 | Day-0 check failed; day 0 for acquisition targets now set by a daily-bar rule | 8 of 16 events with a detected market move differed from the 8-K day 0 by a session (6 clear cases: 8-K filed after the close on the announcement day, or the day after an after-hours release), above the 5-of-50 limit. Rule: from the 8-K session, look back 2 sessions for the earliest session with abnormal return (stock minus SPY) of at least 5% or 3 baseline SDs, on 3x median volume, holding through the 8-K session; otherwise keep the 8-K day 0. The lookback and the announcement-size bar keep pre-announcement drift, which the screen measures, from moving day 0. The 8-K day 0 is kept beside the new one. Set before any feature was computed. |
| 2026-10-09 | Day-0 rule narrowed: day 0 moves only when the 8-K was accepted after the close, only to that day's session, and only if that session had the announcement-sized move | The earlier rule (earliest qualifying move in 2 sessions) could put day 0 on a leak or tipped trading, which would move pre-announcement trading out of the window the screen measures. All 6 failures in the day-0 check were this one pattern. Other announcement-sized moves before day 0 are flagged (prior_moves) and stay in the window. Same day, before any feature was computed. |
| 2026-10-09 | Day-0 rule's threshold uses a robust SD (1.4826 x median absolute deviation of baseline abnormal returns) in place of the plain SD | On the sample, Paragon 28's baseline year held shock days that raised a plain-SD threshold to 15.6%, so its deal-day jump after a late 8-K didn't count. A robust SD isn't inflated by a few shock days. A lower bar can only add the late-8-K shift to the acceptance day, or add flags to prior_moves. Seen on the 50-event sample, before any feature was computed. |
| 2026-10-09 | Day-0 rule: a session also qualifies with abnormal return of at least 5% on at least 10x median volume | Paragon 28 (+8.7% abnormal on 24x volume the day after an after-hours deal release, with an 11% 3-SD bar) was still missed: a small premium on a volatile stock. This is the second change fitted to the 50-event sample; the day-0 check's verdict doesn't depend on either change (6 late 8-Ks under every version). The shifts the rule makes across all events will be spot-checked on a fresh sample before features are computed. |
| 2026-10-09 | Ticker supplement: press releases are matched to former EDGAR names as well as the current one, and company-years still without a ticker are matched by name to map rows that have no CIK or a CIK EDGAR doesn't contradict (one ticker only, valid during the year's events, not held by a differently named company); the press-release lookup tries up to four 8-Ks, earnings releases first | A sample of 15 targets without a ticker held 5 listed companies: two renamed after their deals (Westar Energy is now Evergy Kansas Central; Questar is Dominion Questar), and three delisted. Checking them: Versar's VSR sits in the map under another CIK; Lime Energy and Questar's STR aren't in Massive's data at all; CEB's target event already had its ticker. The other 10 have no listed stock (non-traded REITs, subsidiaries, OTC). Done before any feature was computed. |
| 2026-10-09 | Ticker name match: a ticker is refused when another company's events carry it through the CIK join between the company-year's first and last event | A check of 20 name matches found 5 wrong: subsidiaries given the parent's ticker (HD Supply, Inc. got HDS; Regency Centers LP got REG; Duke Energy Carolinas, whose former name is Duke Energy Corp, got DUK), a different firm with the same short name (PHI Group got PHI Inc's PHII), and a company matched before its listing (Coeptis Therapeutics in 2021). In each, the ticker's own company has events carrying it in the same window. Name matches are recomputed under the rule; the check is repeated before any feature is computed. |
| 2026-10-09 | Ticker name match tightened again: the ticker must have FINRA volume during the company-year's events (from the month before the first); no other company's events may carry it that calendar year; a map row under another CIK needs a name of at least 5 characters | A second check of 25 found 6 wrong: tickers the company only got later (FS KKR Capital Corp. II in 2017, FS Specialty Lending Fund in 2021, EMI Holding as Emmaus in 2017, Amplify Energy in 2016), a short name shared by two firms (Metalert, formerly GTX Corp, got GTx Inc's GTXI), and an operating partnership given its parent's ticker on days the parent had no event (Life Storage LP). Each new rule removes at least one; the year-wide rule also drops correct matches (HarborOne's second step, AspenTech's reorganization). Checked again on a fresh sample before any feature is computed. |
| 2026-10-09 | Spot check of day-0 shifts, rule set before any shift is seen: 20 late_8k_shift events drawn from all targets, excluding the 50-event development sample. Each is right if the deal press release (EX-99 dateline, or the newswire page) is dated the acceptance day or earlier. Pass: at least 18 of 20. On a fail, the shift is dropped and every target keeps its 8-K day 0, with the announcement-day moves flagged | The rule was fitted on the development sample (two changes); a fresh sample measures how often the shift is right outside it. A release dated the next day would mean the acceptance-day move came before any public news, which the screen must keep in the window. |
| 2026-10-09 | Day-0 shift spot check passed: 19 of 20 shifted targets had the deal release dated the acceptance day (Clover Leaf Capital's release date couldn't be found and is counted as a miss). The late-8-K shift stays | Rule and pass line were set before the sample was drawn (entry above). The sample also showed events typed as targets that aren't takeovers of the filer (Ribbon was the acquirer, Seres sold an asset, Clover Leaf is a SPAC) and one ticker that is a note, not stock (DHCNI for Diversified Healthcare Trust); both are audited next. |
| 2026-10-09 | Target set: an acquisition-target event stays only if the filer also filed SC 14D9 or SC 13E3, or was delisted or deregistered (25-NSE, 15-12B) within 730 days after day 0; SPACs (SIC 6770) are dropped. Exchange test symbols (NTEST.B, ZVZZT and the like) are removed from the ticker map | The audit of 2,671 target events found 161 SPACs and 502 with no sign of a takeover. 15 of those, at random: acquirers paying in stock (Gen Digital, Tamboran), reverse mergers into a listed shell (KalVista into Carbylan, Larimar into Zafgen, Peraso into MoSys), asset sales (FG Nexus, Atreca, Determine), non-traded or private filers, a SPAC no longer coded 6770 (HNR Acquisition), one broken deal (Avenue Therapeutics), and ANSYS, whose deal took 18 months to close (the window was 540 days, now 730). Broken deals are lost from the target set; the screen's charged cases are checked against this set before features. The ticker check turned up Cash America mapped to the test symbol NTEST.B. |
| 2026-10-09 | Ticker map drops rows named as notes, debentures, preferred, warrants or when-issued lines whatever their type; press-release tickers for companies with a normalized name under 5 characters must have the name within 40 characters before the ticker | Massive types Diversified Healthcare Trust's senior notes (DHCNI, DHCNL) as CS and has no DHC row, so DHC's events were priced on a note. PHH Corp got RLGY: its release named Realogy a sentence after "PHH", and a 3-letter name always matches somewhere in 150 characters. Press-release tickers found for short names are redone under the new window. |
| 2026-10-09 | Target set kept as audited (no rule added for broken deals) | 49 of 51 charged target events stay in the set. The two dropped are Lattice Semiconductor (Canyon Bridge deal of 2016-11-03, blocked in 2017) and Skyline's 2018-01-05 combination with Champion (a reverse merger: the listed company survived). A rule for terminated deals (Item 1.02 after the merger 8-K) would also bring back terminated stock deals on the acquirer side; for 2 events it isn't worth it. The validation counts 49 charged targets. |
| 2026-10-09 | Features: windows counted in NYSE sessions from the final day 0; the placebo window (-70 to -51) gets its own baseline, -300 to -81, keeping the 220-session length and 10-session gap; size is the baseline median dollar volume, and the $50M market-cap floor isn't applied | The spec's baseline (-250 to -31) contains the placebo window, so a placebo measured against it compares the window with itself in part. No free source gives historical shares outstanding, so market cap can't be computed; dollar volume stands in for the market-cap quintiles in peer benchmarking. The $1 price floor (raw close at day -30) and the ADR exclusion are applied. Set before any feature value was looked at. |
