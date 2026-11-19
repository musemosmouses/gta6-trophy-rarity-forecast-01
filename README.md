# GTA VI trophy rarity forecast — make.fun entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/musemosmouses/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/musemosmouses/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the percent
of the players Exophase tracks who will have earned it at 2026-11-26T00:00:00Z,
as five quantiles per trophy, scored by pinball loss on the logit scale.

- Bounty: https://make.fun/bounty/gta6-trophy-rarity-forecast-01
- Referral: https://make.fun/r/w3zshnah
- Entry: `main.ts` (Node 22, standard library only, no network at runtime)

## The model

A trophy's Exophase earn rate is **earners / players_tracked**, and both move:
players_tracked jumps when Exophase refreshes the page, while earners grow as
players work through the story. So the program models the two counts
separately instead of extrapolating the rate itself.

**Fitted path (the final case).** `log(players_tracked)` and `log(earners)`
per trophy are each fit by least squares against `u = ln(1 + hours since the
2026-11-19T00:00:00Z release)` over the last 24 hours of reads, read at the
target, and divided back into a rate. The fitted move is damped toward the
last rate (regimes change over a multi-day horizon), and the spread combines
the fits' residuals, a random-walk term in `sqrt(days)`, and one refresh
worth of jump surprise.

**Flat-history path (the practice cases).** When the whole history is one
frozen page, no trend exists to fit, so a documented refresh prior takes
over. The key arithmetic: the first refresh took players 638 → 3,868 and
trophies earned 943 → 4,276, but the 100% trophy alone accounts for all 3,230
new players — so only **103 new earners** are left for the other 54 trophies.
Every rate therefore falls to roughly a fifth of its frozen value, with the
103 earners shared out by difficulty (platinum / 100% / long grinds get
~none; story chapters a few; easy one-offs the most). The forecast is the
two-component mixture (no refresh / refresh) and its quantiles are inverted
from that mixture, so the band honestly covers both outcomes.

**Difficulty grounding (public trophy guides).** The story takes 60–80
hours, the platinum about 200; hardest are A Long Road (100% completion),
The Shootist (gold in all 42 shooting ranges), Cunning Stunts (250 stunts)
and You WAiNKer! (100 locations). No Rockstar logo, artwork or font is used.

**Honest expectations.** Two days of launch data cannot pin a rate a week
out. The quantiles are deliberately wide, the medians drift down on a frozen
page and are damped on a fitted trend, and the program never invents
precision the data does not support.

## Practice results (answers not in the kit — scored against estimated answers)

The kit holds no practice answers, so the practice was scored against three
estimated answer scenarios built from the documented refresh totals
(players 638 → 3,868, trophies earned 943 → 4,276):

| scenario | ours (p00 / p01) | house baseline |
| --- | --- | --- |
| S4 total-consistent (103 new earners shared by difficulty) | **0.214 / 0.137** | 1.422 |
| S2 zero new earners (every rate × 0.165) | 0.297 / 0.186 | 1.688 |
| S1 proportional earners (every rate × 0.748) | 0.503 / 0.807 | 0.198 |

S4 is the scenario the documented numbers support: under it the house
baseline scores 1.42, close to the rules' stated ~1.51, while S1's 0.198
contradicts the stated figure — and S1 would need 14 new platinum earners
in the first five hours of a ~200-hour platinum, which is not credible.
Under the realistic scenarios the entry beats the baseline by roughly 7–10×
on the practice, with 90% coverage 1.00 (baseline: 0.00). All 54
non-100% medians sit below their last rate, as dilution requires.

## Verification

- `main.ts` on the practice input, twice to two output dirs: byte-identical
  (`diff -r` clean). Runtime ~0.3 s per run (cap 8 s).
- Outputs valid: 55 rows, each id once, five finite numbers 0–100,
  non-decreasing, ≤ 65,536 bytes (p00: 2,777 bytes).
- Grader simulation (read-only repo copy, user `nobody`, no network,
  1 CPU, 10 s cap): exit 0 on the practice case and on a synthetic
  non-flat final-like case, outputs identical to a normal run.
- Synthetic final-like case (2-day history, N 638 → ~35k, story rates
  rising, grinds flat): story medians rise above last rates, grind medians
  stay near 0.01%, bands wide — the fitted path behaves as designed.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` in the input directory and writes one `*.out` per case.
Deterministic: no clock, no randomness, no network. Never removes the
output directory (`mkdirSync(out, { recursive: true })` only).

## License

MIT — see `LICENSE` (`SPDX-License-Identifier: MIT`).
