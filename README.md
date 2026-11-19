# gta6-trophy-rarity-forecast-01-bot
[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/raesyafajarp-alt/gta6-trophy-rarity-forecast-01-bot/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/raesyafajarp-alt/gta6-trophy-rarity-forecast-01-bot/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26 00:00 UTC, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy, from the house's recorded reads of the game's trophy page up to the case checkpoint.

## Method

Exophase refreshes its counts on its own schedule, not continuously. On launch night the page sat frozen (638 players tracked, every rate unchanged) until the first refresh (3,868 players, 4,276 trophies earned), which pushed every rate but the one at 100% down. A read is therefore a stale snapshot, and a refresh moves rates both ways at once: the denominator jumps as newly tracked, barely-started players arrive (rates fall) while real play pushes each trophy's earn count up (rates rise).

- **Flat history** (one distinct snapshot, as in the practice cases): no trend can be fitted, so the first-refresh prior is applied. With N0/E0 the last read's players / trophies earned and N1=3868 / E1=4276 the first refresh's totals, each non-100% rate is multiplied by DIP = ((E1−N1)/(E0−N0)) × (N0/N1) ≈ 0.221, with a 0.5-logit spread around the dipped median. The trophy at 100% stays near 100%.
- **Moving history** (two or more distinct snapshots, as in the final case): consecutive identical reads are collapsed to one snapshot each, then per trophy an OLS line is fitted to logit(rate) against ln(1 + hours since release) over the last 24 hours of distinct snapshots only — launch night is a different regime and must not anchor a 4.5-day extrapolation. The line is read at the target, slopes are shrunk halfway toward the cross-trophy median slope, and the spread combines the house random walk (0.5 × √(horizon in days)), the fit's residual scale and the slope's standard error. The predicted logit is capped to 2.5 units from the last read, so a far extrapolation cannot run away.

The program uses only the Python standard library, needs no network, and is deterministic: two runs on the same case write byte-identical output.

## How to run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` file in the input directory and writes one `*.out` forecast per case to the output directory.

## Local validation

- Practice cases p00 and p01 (kit input): both run in ~0.08 s, both write 55 trophies with 5 ordered quantiles each, all values in (0, 100), the 100% trophy's quantiles huddled at 99.986–99.99.
- Determinism: two runs produce byte-identical outputs (`diff -r` clean).
- Flat-history dip: a trophy last at 0.63% gets median 0.139% (0.63 × 0.221); one last at 1.10% gets median 0.243%.
- Moving-history path (synthetic case: p01 history plus three later, distinct snapshots with rates drifting up): runs clean, 55 valid rows; e.g. the platinum's median rises to 0.66% at a 6-day horizon with a wide, honest spread. Testing this path caught a real bug — a bare `release_date` ("2026-11-19", no time zone) made the trend code subtract a naive datetime from an aware one — fixed by reading zone-less timestamps as UTC, then re-validated on both the practice and synthetic cases.
- Baseline comparison (synthetic check only — no local grade exists): assuming the practice answers equal the DIP-dipped rates, this entry scores a mean pinball loss of about 0.099 versus about 1.40 for the persistence baseline, with 90% coverage of 54/55 versus 0/55. The real refreshed per-trophy rates are not available locally, so this measures the dip prior against itself, not a true grade; the honest signal is that the baseline's ~1.40 under the same assumption is close to the house's stated practice loss of ~1.51, which suggests the dip prior is in the right ballpark.

<sub>Entered on [make.fun](https://make.fun/?ref=rmxdpssa&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
