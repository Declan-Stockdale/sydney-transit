"""
Slowest stretches between stations: how fast trains actually run between each
pair of neighbouring stations, and how much time is lost against that
stretch's own good runs.

For every train and every pair of consecutive stations (A then B) where both
were matched to GPS:
    running time = first GPS fix near B - last GPS fix near A (so it excludes the stops)
    speed        = straight-line distance A to B / running time
Per stretch (both directions kept apart, all lines combined since they share the track):
    best run     = 10th percentile running time (what the track allows on a good run)
    time lost    = average of (running time - best run): slow running from congestion,
                   signals, speed restrictions or waiting for a platform
    train-minutes lost per day = time lost x trains / days collected

Straight-line distance understates curving track, so speeds are a lower bound;
compare stretches with each other rather than with line speeds.

Output: app/data/slow_segments.csv
    mode, from_station, station, lines, trains, dist_km, median_run_s, best_run_s,
    p90_run_s, median_kmh, lost_s, lost_min_per_day, peak_lost_s, from_lat, from_lon, lat, lon

Usage:
    python -m analysis.reliability
    python -m analysis.slow_segments
"""

import numpy as np
import pandas as pd

from analysis.stop_events import rail_events, station_coords
from tfnsw import PROJECT_DIR

OUT = PROJECT_DIR / "app" / "data"
MIN_TRAINS = 20
MAX_RUN_S = 60 * 60
PEAK_HOURS = [6, 7, 8, 9, 16, 17, 18]


def runs(rail: pd.DataFrame) -> pd.DataFrame:
    """Station-to-station running times for consecutive GPS-matched stops of the same trip."""
    d = rail.sort_values(["service_date", "trip_id", "sched_time"])
    g = d.groupby(["service_date", "trip_id"], sort=False)
    gps = d["delay_source"] == "gps"
    d = d.assign(prev_dep=g["dep_actual"].shift().where(g["delay_source"].shift() == "gps"),
                 arr=d["arr_actual"].where(gps))
    r = d[d["prev_station"].notna()].dropna(subset=["prev_dep", "arr"]).copy()
    r["run_s"] = r["arr"] - r["prev_dep"]
    return r[(r["run_s"] > 0) & (r["run_s"] < MAX_RUN_S)]


def summarise(r: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    days = max(r["date"].nunique(), 1)
    key = ["mode", "prev_station", "station"]
    best = r.groupby(key)["run_s"].transform(lambda s: s.quantile(0.1))
    r = r.assign(lost_s=(r["run_s"] - best).clip(lower=0), peak=r["hour"].isin(PEAK_HOURS))
    s = r.groupby(key).agg(
        lines=("line", lambda x: " ".join(sorted(set(x)))), trains=("run_s", "size"),
        median_run_s=("run_s", "median"), best_run_s=("run_s", lambda x: x.quantile(0.1)),
        p90_run_s=("run_s", lambda x: x.quantile(0.9)), lost_s=("lost_s", "mean"), lost_total=("lost_s", "sum")).reset_index()
    peak = r[r["peak"]].groupby(key)["lost_s"].mean().rename("peak_lost_s")
    s = s.merge(peak.reset_index(), on=key, how="left")
    s["lost_min_per_day"] = s.pop("lost_total") / 60 / days
    c = coords.set_index("station")
    a, b = c.reindex(s["prev_station"]), c.reindex(s["station"])
    s["dist_km"] = np.hypot(a["x"].to_numpy() - b["x"].to_numpy(), a["y"].to_numpy() - b["y"].to_numpy()) / 1000
    s["median_kmh"] = s["dist_km"] / (s["median_run_s"] / 3600)
    s["from_lat"], s["from_lon"], s["lat"], s["lon"] = a["lat"].values, a["lon"].values, b["lat"].values, b["lon"].values
    s = s[(s["trains"] >= MIN_TRAINS) & s["dist_km"].notna()].rename(columns={"prev_station": "from_station"})
    return s.round({"median_run_s": 0, "best_run_s": 0, "p90_run_s": 0, "lost_s": 1, "peak_lost_s": 1,
                    "lost_min_per_day": 1, "dist_km": 2, "median_kmh": 1})


def main() -> None:
    s = summarise(runs(rail_events()), station_coords())
    s.to_csv(OUT / "slow_segments.csv", index=False)
    print(f"slow_segments.csv: {len(s):,} stretches")
    cols = ["lines", "from_station", "station", "trains", "dist_km", "median_kmh", "lost_s", "lost_min_per_day"]
    print("Slowest (over 1 km):")
    print(s[s["dist_km"] >= 1].nsmallest(10, "median_kmh")[cols])
    print("Most train-minutes lost a day:")
    print(s.nlargest(10, "lost_min_per_day")[cols])


if __name__ == "__main__":
    main()
