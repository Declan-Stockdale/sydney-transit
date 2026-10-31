"""
Bunching and gaps: are frequent services evenly spaced, or do they arrive in
clumps followed by long waits? Trains, Metro and buses.

For each stop and direction (trains: station + line + next station; buses: the
kerbside stop + route), consecutive services in timetable order:
    scheduled gap h_s = difference of scheduled times
    actual gap    h_a = difference of actual times (GPS for trains; the operator's own
                        prediction for buses, see analysis/stop_events.py)
    bunched  h_a < half of h_s       (two services close together)
    big gap  h_a > 1.5 x h_s         (a long wait, usually just before a bunch)
Only "turn up and go" services are judged: scheduled gaps of 15 minutes or less.
Gaps of over an hour, or across a pause in collection, are dropped.

Buses are only measured at stops within 250 m of a rail station (where bus
timings are collected; analysis/stop_events.py).

Output: app/data/headways.csv
    mode, line, hour, gaps, pct_bunched, pct_big_gap, regularity (std of h_a / h_s; 0 = perfectly even)
and app/data/headway_stops.csv: the same per mode, line and station.

Usage:
    python -m analysis.reliability
    python -m analysis.stop_events
    python -m analysis.headways
"""

import numpy as np
import pandas as pd

from analysis.stop_events import BUS_EVENTS, bus_events, rail_events
from tfnsw import PROJECT_DIR

OUT = PROJECT_DIR / "app" / "data"
MAX_SCHED_GAP_S = 15 * 60
MIN_SCHED_GAP_S = 60
MAX_ACTUAL_GAP_S = 60 * 60
MIN_GAPS = 20


def gaps(ev: pd.DataFrame, group: list[str], sched: str, actual: str) -> pd.DataFrame:
    """Scheduled and actual gaps between consecutive services at each group (stop + direction), in timetable order."""
    ev = ev.dropna(subset=[sched, actual]).sort_values([*group, sched])
    g = ev.groupby(group, sort=False)
    ev = ev.assign(h_s=g[sched].diff(), h_a=g[actual].diff())
    ev = ev[ev["h_s"].between(MIN_SCHED_GAP_S, MAX_SCHED_GAP_S) & ev["h_a"].notna() & (ev["h_a"].abs() < MAX_ACTUAL_GAP_S)]
    ev["h_a"] = ev["h_a"].clip(lower=0)  # overtaking: the second service arrived first
    ev["ratio"] = ev["h_a"] / ev["h_s"]
    ev["bunched"] = ev["ratio"] < 0.5
    ev["big_gap"] = ev["ratio"] > 1.5
    return ev


def summarise(h: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    s = h.groupby(by).agg(gaps=("ratio", "size"), pct_bunched=("bunched", "mean"), pct_big_gap=("big_gap", "mean"),
                          regularity=("ratio", "std"), sched_gap_min=("h_s", "median")).reset_index()
    s[["pct_bunched", "pct_big_gap"]] *= 100
    s["sched_gap_min"] /= 60
    return s[s["gaps"] >= MIN_GAPS].round({"pct_bunched": 1, "pct_big_gap": 1, "regularity": 2, "sched_gap_min": 1})


def all_gaps() -> pd.DataFrame:
    rail = rail_events()
    rail = rail[rail["next_station"].notna()]  # departures
    parts = [gaps(rail, ["mode", "station", "line_dir"], "sched_time", "dep_actual")[["mode", "line", "station", "hour", "h_s", "h_a", "ratio", "bunched", "big_gap"]]]
    if BUS_EVENTS.exists():
        bus = bus_events().assign(mode="Bus")
        bus = bus[~bus["last_stop"]]
        parts.append(gaps(bus, ["mode", "stop_id", "line"], "sched_arr", "arr_actual")[["mode", "line", "station", "hour", "h_s", "h_a", "ratio", "bunched", "big_gap"]])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    h = all_gaps()
    by_hour = summarise(h, ["mode", "line", "hour"])
    by_stop = summarise(h, ["mode", "line", "station"])
    by_hour.to_csv(OUT / "headways.csv", index=False)
    by_stop.to_csv(OUT / "headway_stops.csv", index=False)
    print(f"headways.csv: {len(by_hour):,} rows; headway_stops.csv: {len(by_stop):,} rows")
    print(summarise(h, ["mode"]))
    print(summarise(h, ["mode", "line"]).query("gaps >= 200").sort_values("pct_bunched", ascending=False).head(12))


if __name__ == "__main__":
    main()
