"""Forecast GTA VI trophy earn rates on Exophase.

Reads one case per *.in file in the input directory (or $MAKEFUN_INPUT_DIR)
and writes one *.out forecast per case to the output directory (or
$MAKEFUN_OUTPUT_DIR). Each forecast gives five quantiles (5/25/50/75/95%)
of every trophy's earn rate, in percent, at the case's target instant.

Method, in short:

* Exophase does not refresh its counts continuously. On launch night the
  page sat frozen (638 players tracked, every rate unchanged) until the
  first refresh (3,868 players, 4,276 trophies earned), which pushed every
  rate but the one at 100% down. A read is therefore a stale snapshot, and
  a refresh moves rates both ways at once: the denominator jumps as newly
  tracked, barely-started players arrive (rates fall) while real play
  pushes each trophy's earn count up (rates rise).

* Flat history (only one distinct snapshot in the case): no trend can be
  fitted, so the first-refresh prior is applied. With N0/E0 the last
  read's players / trophies earned and N1=3868 / E1=4276 the first
  refresh's totals, each non-100% rate is multiplied by
  DIP = ((E1-N1)/(E0-N0)) * (N0/N1) (about 0.221), and a 0.5-logit
  spread is put around the dipped median. The trophy at 100% stays
  near 100%.

* Moving history (two or more distinct snapshots): consecutive identical
  reads are collapsed to one snapshot each, then per trophy an OLS line
  is fitted to logit(rate) against ln(1 + hours since release) over the
  last 24 hours of distinct snapshots only (launch night is a different
  regime and must not anchor a 4.5-day extrapolation). The line is read
  at the target, slopes are shrunk halfway toward the cross-trophy
  median slope, and the spread combines the house random walk
  (0.5 * sqrt(horizon in days)), the fit's residual scale and the
  slope's standard error. The predicted logit is also capped to 2.5
  units from the last read, so a far extrapolation cannot run away.

Only the Python standard library is used. The program is deterministic:
the same input always produces byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

# Standard-normal quantiles for the five forecast levels.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# The first refresh Exophase made of the game's counts, on launch night:
# 638 players / 943 trophies earned frozen, then 3,868 players / 4,276
# trophies earned. Used only by the flat-history (first-refresh) prior.
FIRST_REFRESH_PLAYERS = 3868
FIRST_REFRESH_EARNED = 4276

# A rate at or above this is treated as "everyone has it" and kept near
# 100% instead of being dipped or trended.
NEAR_FULL = 99.5

# A fitted line must not be allowed to run away from the last observed
# rate when the target sits far outside the data's range: cap the
# predicted logit's move from the last read.
EXTRAP_CAP = 2.5

# Fallback release instant if the case does not name one.
FALLBACK_RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)


def parse_time(text):
    """Parse an ISO 8601 timestamp (with a trailing Z) into an aware datetime.

    A bare date such as a release date carries no zone; read it as UTC.
    """
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def clamp_pct(rate):
    """Clamp a percent into the range the logit can take (0.01 .. 99.99)."""
    if rate < 0.01:
        return 0.01
    if rate > 99.99:
        return 99.99
    return rate


def logit_pct(rate):
    """logit of a percent, clamped so 0% and 100% stay finite."""
    p = clamp_pct(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(value):
    """Inverse logit, returned as a percent."""
    if value >= 0.0:
        z = math.exp(-value)
        return 100.0 / (1.0 + z)
    z = math.exp(value)
    return 100.0 * z / (1.0 + z)


def median(values):
    """Median of a non-empty list of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def distinct_snapshots(history):
    """Collapse consecutive identical reads to one snapshot each.

    Two reads are the same snapshot when they carry the same players
    tracked, the same trophies earned and the same rates. The result is
    a list of (time, players, earned, rates) tuples, oldest first.
    """
    snaps = []
    last_key = None
    for read in history:
        key = (read.get("players_tracked"),
               read.get("trophies_earned"),
               tuple(read.get("rates") or ()))
        if key == last_key:
            continue
        last_key = key
        snaps.append((parse_time(read["at"]),
                      read.get("players_tracked"),
                      read.get("trophies_earned"),
                      read.get("rates") or ()))
    return snaps


def quantiles_from_logit(centre, sigma):
    """Five quantiles around a logit centre, as percents, non-decreasing."""
    out = []
    prev = None
    for z in Z:
        q = inv_logit_pct(centre + z * sigma)
        q = round(q, 4)
        if q < 0.01:
            q = 0.01
        if q > 99.99:
            q = 99.99
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def near_full_row():
    """Quantiles for a trophy everyone holds: all huddled just under 100%."""
    centre = logit_pct(99.99)
    return quantiles_from_logit(centre, 0.2)


def flat_history_forecast(case, snaps, horizon_days):
    """First-refresh prior: one distinct snapshot, no trend to fit."""
    last = snaps[-1]
    n0 = last[1]
    e0 = last[2]
    rates = last[3]
    dip = 1.0
    if n0 and e0 is not None and n0 > 0 and (e0 - n0) > 0:
        dip = ((FIRST_REFRESH_EARNED - FIRST_REFRESH_PLAYERS)
               / (e0 - n0)) * (n0 / FIRST_REFRESH_PLAYERS)
    rows = []
    for trophy in case["trophies"]:
        idx = trophy["id"] - 1
        rate = rates[idx] if idx < len(rates) and rates[idx] is not None else None
        if rate is None:
            rate = 1.0
        if rate >= NEAR_FULL:
            rows.append({"id": trophy["id"], "q": near_full_row()})
            continue
        centre = logit_pct(rate * dip)
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_logit(centre, 0.5)})
    return rows


def ols_line(xs, ys):
    """Least-squares line through (x, y) points.

    Returns (intercept, slope, residual_sd, slope_se, x_mean, sxx).
    With fewer than 3 points the residual scale and slope standard
    error are undefined, so conservative floors are returned instead.
    """
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0
    intercept = y_mean - slope * x_mean
    if n > 2 and sxx > 0:
        resid_sq = sum((y - (intercept + slope * x)) ** 2
                       for x, y in zip(xs, ys))
        resid_sd = math.sqrt(resid_sq / (n - 2))
        slope_se = resid_sd / math.sqrt(sxx)
    else:
        # Two points fix a line exactly; allow honest uncertainty.
        resid_sd = 0.2
        slope_se = 0.5
    return intercept, slope, resid_sd, slope_se, x_mean, sxx


def moving_history_forecast(case, snaps, checkpoint, target, horizon_days,
                           release):
    """Trend forecast: OLS on logit(rate) vs ln(1 + hours since release).

    Only distinct snapshots from the last 24 hours before the checkpoint
    are used, so launch night cannot anchor the extrapolation. Slopes
    are shrunk halfway toward the cross-trophy median slope.
    """
    window_start = checkpoint - timedelta(hours=24)
    recent = [s for s in snaps if s[0] >= window_start]
    if len(recent) < 2:
        recent = snaps
    if len(recent) < 2:
        return None

    horizon_rw = 0.5 * math.sqrt(horizon_days)
    x_target = math.log1p((target - release).total_seconds() / 3600.0)

    # First pass: raw slope per trophy, to find the median slope.
    raw = {}
    for trophy in case["trophies"]:
        idx = trophy["id"] - 1
        xs, ys = [], []
        for when, _players, _earned, rates in recent:
            if idx >= len(rates) or rates[idx] is None:
                continue
            xs.append(math.log1p((when - release).total_seconds() / 3600.0))
            ys.append(logit_pct(rates[idx]))
        if len(xs) < 2 or max(xs) == min(xs):
            raw[trophy["id"]] = None
            continue
        raw[trophy["id"]] = ols_line(xs, ys)
    slopes = [fit[1] for fit in raw.values() if fit is not None]
    median_slope = median(slopes) if slopes else 0.0

    rows = []
    for trophy in case["trophies"]:
        idx = trophy["id"] - 1
        last_rate = None
        for when, _p, _e, rates in reversed(recent):
            if idx < len(rates) and rates[idx] is not None:
                last_rate = rates[idx]
                break
        if last_rate is None:
            last_rate = 1.0
        if last_rate >= NEAR_FULL:
            rows.append({"id": trophy["id"], "q": near_full_row()})
            continue
        fit = raw[trophy["id"]]
        if fit is None:
            # No usable movement for this trophy: stay where it is,
            # with a random-walk spread over the horizon.
            centre = logit_pct(last_rate)
            rows.append({"id": trophy["id"],
                         "q": quantiles_from_logit(centre, horizon_rw)})
            continue
        intercept, slope, resid_sd, slope_se, x_mean, _sxx = fit
        # Shrink the slope halfway toward the cross-trophy median,
        # keeping the line anchored at the data's centroid.
        shrunk = 0.5 * slope + 0.5 * median_slope
        anchored = (intercept + slope * x_mean) - shrunk * x_mean
        centre = anchored + shrunk * x_target
        # Guard against a runaway extrapolation: never let the forecast
        # sit further than EXTRAP_CAP logit units from the last read.
        last_centre = logit_pct(last_rate)
        centre = min(max(centre, last_centre - EXTRAP_CAP),
                     last_centre + EXTRAP_CAP)
        dx = x_target - x_mean
        sigma = math.sqrt(horizon_rw ** 2 + resid_sd ** 2
                          + (slope_se * dx) ** 2)
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_logit(centre, sigma)})
    return rows


def forecast_case(case):
    """Forecast one case, returning the output object."""
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    release_text = (case.get("game") or {}).get("release_date")
    if release_text:
        try:
            release = parse_time(release_text)
        except ValueError:
            release = FALLBACK_RELEASE
    else:
        release = FALLBACK_RELEASE

    history = case.get("history") or []
    snaps = distinct_snapshots(history)

    if not snaps:
        # No reads at all: admit ignorance with a wide, low forecast.
        rows = [{"id": t["id"],
                 "q": quantiles_from_logit(logit_pct(1.0), 2.0)}
                for t in case["trophies"]]
        return {"trophies": rows}

    if len(snaps) == 1:
        rows = flat_history_forecast(case, snaps, horizon_days)
    else:
        rows = moving_history_forecast(case, snaps, checkpoint, target,
                                       horizon_days, release)
        if rows is None:
            rows = flat_history_forecast(case, snaps, horizon_days)
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        stem = name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
