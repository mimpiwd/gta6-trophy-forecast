"""GTA VI trophy rarity forecast — refresh-aware earners-over-players model.

Forecasts every Grand Theft Auto VI trophy's earn rate on Exophase at the
case's target, as five quantiles per trophy, from the house's recording of
the trophy page up to the case's checkpoint.

Why this shape
--------------
Exophase does not update its counts continuously. It refreshes a game's
counts on its own schedule, hours apart: between refreshes the page is
frozen (same players tracked, same rates), and at a refresh the tracked
player count jumps and every rate moves at once. The practice cases show
it plainly — launch night is one flat page (638 players, every rate
unchanged), then the first refresh jumps to 3,868 players and every rate
but the one at 100% falls, because the newly tracked players hold fewer
trophies. A forecast that treats the flat stretches as information, or
that holds every rate where it stands, misses the whole motion.

So this program works with the refreshes, not the reads. It collapses
the history to the distinct pages (one per refresh), then separates the
two things that move a rate:

    rate_i(t) = 100 * earners_i(t) / players(t)

* players(t), the tracked count, is one clean series, the same for every
  trophy. It is fit on the log scale against log(1 + hours since the
  release day began) and read at the target.
* earners_i(t), the count holding trophy i, is fit the same way, per
  trophy, on log(earners + 0.5). Its slope is shrunk toward the players
  slope — earners growing exactly with players means a flat rate, which
  is the honest null when a trophy's own history is thin — with the
  shrinkage fading as the trophy accumulates refresh points.

The median rate is the ratio of the two forecasts. The spread is the
two fits' residual scatter plus a random-walk term that widens with the
square root of the horizon, floored and capped so no trophy is forecast
absurdly tight or absurdly wide. The 100% trophy (everyone holds it) is
kept at 100%: its earners grow with the players, so the ratio holds.

What the model assumes, honestly
---------------------------------
* The tracked count keeps growing roughly as a power of time since
  launch, and each trophy's earners do too — the log-log line is the
  shape, and the shrinkage keeps a thin history from bending it wild.
* A trophy's past refreshes say how fast its earners grow relative to
  the players. Early, easy trophies grow faster than the players (their
  rates rise); late, hard ones grow slower (their rates fall).
* The refresh pattern itself — a jump every few hours — continues, so
  the spread must cover several more jumps before the target.
* When the history is a single frozen page (no refresh seen yet), the
  trend says nothing, but a refresh is still coming: the practice's
  first refresh multiplied the tracked players by six and cut every
  rate but the one at 100%. So a flat history gets a dilution prior —
  the players are assumed to keep growing and the rates to fall toward
  the rate new players earn them at — instead of a flat persistence.

Run: `python3 -P -s /bot/main.py` (Python 3.13, stdlib only, no network),
reads every <id>.in in /input (or $MAKEFUN_INPUT_DIR), writes <id>.out
to /output (or $MAKEFUN_OUTPUT_DIR): {"trophies": [{"id": n, "q": [q05,
q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import
# model` finds a module shipped beside this file (none is needed here, but
# the line keeps the entrypoint importable the same way as the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The release day began here; all log-time is measured from it.
LAUNCH_EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc).timestamp()
# How strongly a trophy's slope is pulled toward the players slope when
# its own history is thin (n refresh points: weight n / (n + SHRINK_K)).
SHRINK_K = 4.0
# Random-walk logit volatility per square-root day, added to the spread.
RW_SIGMA = 0.35
# The spread (logit SD) is never tighter than this, never wider than this.
SPREAD_FLOOR = 0.35
SPREAD_CAP = 1.6
# The dilution prior, used only when the history is a single frozen page
# (too few refresh points to fit a trend): a refresh is still coming.
# The practice's first refresh multiplied the tracked players by six and
# cut every rate but the one at 100%, because the newly tracked players
# had earned almost nothing yet. So the prior assumes at least one
# refresh in the horizon, each multiplying the players by about four,
# and the new players to earn each trophy at DILUTION_ALPHA of its
# current rate.
DILUTION_REFRESH_HOURS = 4.0
DILUTION_GROWTH_PER_REFRESH = 4.0
DILUTION_ALPHA = 0.25
# The spread floor is wider when the history is a single frozen page.
SPREAD_FLOOR_FLAT = 0.7


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100 - p))


def inv_logit(l: float) -> float:
    return 100 / (1 + math.exp(-l))


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def log_hours(at_epoch: float) -> float:
    """log(1 + hours since the release day began)."""
    h = max(0.0, (at_epoch - LAUNCH_EPOCH) / 3600.0)
    return math.log1p(h)


def refresh_points(history: list) -> list:
    """The distinct pages: consecutive reads with the same players count
    and the same rates are one frozen page, so keep only its first read."""
    pts = []
    prev = None
    for h in history:
        key = (h["players_tracked"], tuple(h["rates"]))
        if key != prev:
            pts.append(h)
            prev = key
    return pts


def fit_line(xs: list, ys: list):
    """Least-squares y = a + b*x. Returns (a, b, resid_sd, n)."""
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0, 0)
    if n == 1:
        return (ys[0], 0.0, 0.0, 1)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return (my, 0.0, 0.0, n)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    else:
        var = 0.0
    return (a, b, math.sqrt(max(0.0, var)), n)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    pts = refresh_points(case["history"])
    last = case["history"][-1]
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = log_hours(parse_time(case["target"]))

    # --- players: one log-linear fit over the refresh points. ---
    xs_p = [log_hours(parse_time(h["at"])) for h in pts]
    ys_p = [math.log(max(1, h["players_tracked"])) for h in pts]
    a_p, b_p, sd_p, n_p = fit_line(xs_p, ys_p)
    x_bar_p = sum(xs_p) / len(xs_p)
    y_bar_p = sum(ys_p) / len(ys_p)
    log_p_target = y_bar_p + b_p * (x_target - x_bar_p)
    p_target = math.exp(log_p_target)
    p_last = last["players_tracked"]
    # The tracked count only grows; never forecast fewer players.
    p_target = max(p_target, float(p_last))
    # And never an absurd jump: cap the growth at 200x the last count.
    p_target = min(p_target, float(p_last) * 200.0)

    rows = []
    # Whether the history is a single frozen page: too few refresh points
    # to fit a trend, so the dilution prior applies (see the module docs).
    flat = n_p <= 2
    if flat:
        n_refresh = (case["horizon_minutes"] / 60.0) / DILUTION_REFRESH_HOURS
        # A refresh is expected in the horizon: at least one, since the
        # page only moves at a refresh and the horizon spans one.
        n_refresh = max(1.0, n_refresh)
        g_prior = DILUTION_GROWTH_PER_REFRESH ** n_refresh
        dilution = (1.0 + (g_prior - 1.0) * DILUTION_ALPHA) / g_prior
    else:
        dilution = 1.0
    for i, trophy in enumerate(trophies):
        last_rate = last["rates"][i]
        if last_rate is None:
            last_rate = 0.0
        # The 100% trophy: everyone holds it, so it stays at 100%.
        if last_rate >= 100.0:
            rows.append({"id": trophy["id"], "q": [round4(v) for v in (99.5, 99.8, 100, 100, 100)]})
            continue
        # --- earners: per-trophy log-linear fit, slope shrunk toward b_p. ---
        xs_e = []
        ys_e = []
        for h in pts:
            r = h["rates"][i]
            if r is None:
                continue
            e = r * h["players_tracked"] / 100.0
            xs_e.append(log_hours(parse_time(h["at"])))
            ys_e.append(math.log(e + 0.5))
        if not xs_e:
            # Never seen: hold the last rate with the baseline spread.
            lg = logit(last_rate)
            spread = RW_SIGMA * math.sqrt(horizon_days)
            spread = min(SPREAD_CAP, max(SPREAD_FLOOR, spread))
            rows.append({"id": trophy["id"], "q": [round4(inv_logit(lg + z * spread)) for z in Z]})
            continue
        a_e, b_e, sd_e, n_e = fit_line(xs_e, ys_e)
        x_bar_e = sum(xs_e) / len(xs_e)
        y_bar_e = sum(ys_e) / len(ys_e)
        # Shrink the slope toward the players slope (a flat rate); the
        # pull fades as the trophy accumulates refresh points.
        w = n_e / (n_e + SHRINK_K)
        b_shrunk = b_p + (b_e - b_p) * w
        log_e_target = y_bar_e + b_shrunk * (x_target - x_bar_e)
        e_target = math.exp(log_e_target) - 0.5
        e_target = max(0.0, e_target)
        median = 100.0 * e_target / p_target
        # A flat history gets the dilution prior: the trend cannot see
        # the refresh coming, so pull the median down toward the rate
        # new players earn the trophy at.
        median = median * dilution
        median = min(100.0, max(0.01, median))
        # --- spread: the two fits' scatter plus a random-walk term. ---
        var = sd_e * sd_e + sd_p * sd_p + (RW_SIGMA ** 2) * horizon_days
        spread = math.sqrt(max(0.0, var))
        floor = SPREAD_FLOOR_FLAT if flat else SPREAD_FLOOR
        spread = min(SPREAD_CAP, max(floor, spread))
        lg = logit(median)
        rows.append({"id": trophy["id"], "q": [round4(inv_logit(lg + z * spread)) for z in Z]})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
