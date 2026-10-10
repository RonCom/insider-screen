# Results

Scored against the expectations in [spec.md](spec.md). The test period (events from 2021 to 2025) was scored once, on code tagged `freeze-1` (commit a829140); the full output is `reports/test_scores.md`. Every rule changed after data was seen is in the spec's change log, dated, with the reason.

## Expectations

| ID | Test | Expectation | Result | Verdict |
| --- | --- | --- | --- | --- |
| H1 | Top-5% lift of charged events, M&A targets, test period, best score | ≥ 3× | 2.82× (composite: 3 of 21 charged targets among the top 37 of 731) | Failed |
| H2 | Options features added, M&A targets | Gain ≥ 0.03 AUC | No free source of expired option contracts | Not tested |
| H3 | Off-exchange short share added, negative-return earnings events | Gain < 0.02 AUC | +0.097 (0.852 to 0.949), on 3 charged events | Failed (no evidence either way) |
| H4 | Shrinkage vs. unshrunk z, small-cap share of top-5% alerts | Falls by ≥ 25% | Shrinkage was defined on options trading days | Not tested |
| H5 | Placebo window (−70 to −51), top-5% lift | ≤ 1.5× | 0× (0 of 37), composite | Met |
| H6 | 50 alerts a month, share of charged events caught | ≥ 25% | 28% (7 of 25), targets and earnings | Met |

## Acquisition targets (H1, H5)

| Score | Top-5% lift | Charged in top 5% | Top-1% | AUC | Placebo top-5% | Placebo AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Composite | 2.82× | 3 / 37 | 0 / 8 | 0.517 | 0 / 37 | 0.386 |
| Isolation Forest | 1.88× | 2 / 37 | 0 / 8 | 0.480 | 0 / 37 | 0.460 |
| Supervised (fitted on 2017–2020) | 0× | 0 / 37 | 0 / 8 | 0.541 | 0 / 37 | 0.481 |

The test period holds 731 targets with features, and 21 of them are charged. At random, the top 37 would hold about 1.1 charged events. The composite's 3 would occur by chance about 9% of the time (Poisson), and its AUC of 0.517 means it doesn't rank charged targets above the rest as a group. The 3 hits are consistent with the composite's top ranks being deals with visible run-ups. A charged target can sit among them when its deal also leaked or drew rumors.

The placebo window catches no charged event in the top 5% under any score, so H5 is met. The composite's placebo AUC of 0.386 is below 0.5: charged targets were, if anything, quieter than other targets 51 to 70 sessions before their announcements.

Development (2017–2020) showed the same pattern: none of 9 charged targets in the composite's top 5%, composite AUC 0.66.

## Why H1 failed

This was recorded in the change log before the test run (2026-10-09). The trades the SEC charges are too small for daily data to see:

- Of the 48 charged targets, 27 were traded in stock only, 10 in options only, 4 in both, and 7 aren't stated (extraction v5).
- Where a release states a share count, the largest charged trade is a median 0.07% of the stock's volume over the 20-day window, and none reaches 5%. The largest, TravelCenters, is 2.3%.
- Stated profits run from $31,000 to $5.2 million.

A trade that size doesn't move daily volume, returns or the off-exchange short share. The composite's top ranks hold deals with visible run-ups from press leaks and rumors, which the SEC rarely charges, because the information was already public. The screen measures abnormal trading before announcements. In this data, charged insider trading isn't where that trading comes from.

Options are where a small trade stands out against normal volume, and options data was the one source dropped (no free source of expired contracts). H2 would have tested that channel.

## Earnings, negative day-0 return (H3)

There are 40,682 negative-return earnings events in the test period, and 3 of them are charged. Adding the short share raises the composite's AUC from 0.852 to 0.949. The spec says a gain of 0.02 or more is reported as a finding and re-tested on 2024–2025. With 3 charged events, one event moving a few thousand places changes the AUC by tenths, so the gain is noise. A 2024–2025 re-test would rest on one or two events and can't settle it. H3 is reported as failed against its threshold, without a claim that off-exchange data helps.

## Alert budget (H6)

Taking the top 50 events each month by composite, across targets and earnings, catches 7 of 25 charged events (28%), against the 25% expected. About 1,300 events a month are scored, so 50 alerts flag roughly 4% of events. The report doesn't split the 7 between targets and earnings, though 21 of the 25 charged events are targets. The two kinds compete for the same 50 alerts on their composites, each a robust z within its own peer groups. That puts them on similar scales, but how they compare across kinds was never checked.

## Labels per year

| Year | Targets | Charged targets | Earnings events | Charged earnings |
| --- | --- | --- | --- | --- |
| 2021 | 164 | 7 | 16,239 | 1 |
| 2022 | 132 | 5 | 16,481 | 0 |
| 2023 | 133 | 6 | 15,736 | 2 |
| 2024 | 137 | 3 | 15,316 | 1 |
| 2025 | 165 | 0 | 15,020 | 0 |

SEC charges trail the trades by years, so 2024 and 2025 are under-labeled: 3 and 0 charged targets, against 5 to 7 a year before. A score that finds 2025's insider trading has nothing yet to be credited with.

## Limits

- **Small counts:** 21 charged targets in the test period and 9 in development. One event more or less in the top 5% moves the lift by about 1×.
- **Lost charged targets:** 48 charged targets passed the target audit, but only 30 were scored (9 + 21). The rest had no ticker, no price bars, a price under $1, were ADRs, or fell in 2016, before the development period.
- **Broken deals left out:** the target audit keeps only deals confirmed by a tender offer, a going-private filing, or a delisting. Deals that broke are out of the target set; two charged events were lost that way (Lattice Semiconductor, Skyline–Champion).
- **Labels from one local model:** extraction ran on gemma4:26b without reasoning, with reasoning on for releases where that pass named no company. Its hand-check score, on the answers known from the reviewed check, was 90.4% for issuer and 89.6% for announcement date. That's one answer under the date gate, and was accepted by the study owner.
- **No market cap:** no free source of historical shares outstanding, so size peers use dollar volume, and the $50 million market-cap floor wasn't applied.
- **Not covered:** other material events (8-K Items 7.01 and 8.01) weren't scored. The monthly short-sale trade counts were dropped as too large to download for two features.
- **Stated in advance:** SEC cases include some found by screens like this one, and unlabeled events include uncharged insider trading.
