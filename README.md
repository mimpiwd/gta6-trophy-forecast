# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mimpiwd/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mimpiwd/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI trophy's earn rate on Exophase at 2026-11-26T00:00:00Z — a week after launch — as five quantiles per trophy, from the house's recording of the trophy page up to the case's checkpoint.

## How it works

Exophase does not update its counts continuously. It refreshes a game's counts on its own schedule, hours apart: between refreshes the page is frozen, and at a refresh the tracked player count jumps and every rate moves at once. Launch night is one flat page (638 players, every rate unchanged), then the first refresh jumps to 3,868 players and every rate but the one at 100% falls, because the newly tracked players hold fewer trophies.

So the program works with the refreshes, not the reads. It collapses the history to the distinct pages (one per refresh), then separates the two things that move a rate:

```
rate_i(t) = 100 * earners_i(t) / players(t)
```

- **players(t)**, the tracked count, is one clean series, the same for every trophy. It is fit on the log scale against log(1 + hours since the release day began) and read at the target.
- **earners_i(t)**, the count holding trophy i, is fit the same way, per trophy, on log(earners + 0.5). Its slope is shrunk toward the players slope — earners growing exactly with players means a flat rate, which is the honest null when a trophy's own history is thin — with the shrinkage fading as the trophy accumulates refresh points.

The median rate is the ratio of the two forecasts. The spread is the two fits' residual scatter plus a random-walk term that widens with the square root of the horizon, floored and capped. When the history is a single frozen page (no refresh seen yet), a dilution prior applies instead of a flat persistence: the players are assumed to keep growing and the rates to fall toward the rate new players earn them at.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is Python 3.13, standard library only, no network. It reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Validation

- **Practice (launch night, 90-min horizon):** the model expects a refresh — its median falls for 54 of 55 trophies and its spread is wide enough to cover one. Against reconstructed refresh answers it beats the persistence baseline's pinball loss in every scenario (e.g. 0.22 vs 1.10 under a strong refresh drop).
- **Synthetic final-like case (12 refresh points over 2.5 days, 4.5-day horizon):** the model captures the early-up / late-down shape — the earliest trophy's median rises while late trophies fall — and scores a pinball loss of 0.26 vs 0.33 for persistence.

---

<sub>Entered on [make.fun](https://make.fun/?ref=6hry4cng&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
