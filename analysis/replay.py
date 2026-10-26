"""
Replay data for the animated train map: where every train was, and how late it
was running, for the period collected.

Built from the vehicle positions feeds already collected (Sydney Trains, Metro
and NSW TrainLink trains; coaches dropped), joined to the measured delays from
reliability.py:
    - one GPS fix per train per minute (fixes arrive every ~30 s)
    - lateness at each fix = the train's most recent measured delay at or before
      that moment (0 until its first measured stop)
    - fixes with stale GPS clocks (more than 10 minutes away from when they were
      collected) and fixes outside NSW are dropped
    - gaps when collection paused (PC asleep or off) are kept as gaps: a train is
      only drawn between fixes less than 5 minutes apart

Output: app/data/replay.json
    {"start": <epoch s>, "end": <epoch s>, "trips": [{"line", "mode", "p": [[lon, lat, t, late_min], ...]}]}
    where t is seconds since "start".

Usage:
    python -m analysis.reliability
    python -m analysis.replay                # everything collected
    python -m analysis.replay --hours 24     # only the latest 24 hours
"""

import argparse
import json

import numpy as np
import pandas as pd

from analysis.reliability import MARTS_DIR, vehicle_points
from tfnsw import PROJECT_DIR

OUT = PROJECT_DIR / "app" / "data" / "replay.json"
FEEDS = {"sydneytrains": "Trains", "metro": "Metro", "nswtrains": "TrainLink"}
OUT_OF_SERVICE = "Out of service"  # route RTTA (Out Of Service / Non Revenue) and NonTimetabled.* trips


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the train replay dataset for the dashboard.")
    parser.add_argument("--hours", type=float, default=None, help="Only the latest N hours (default: all)")
    args = parser.parse_args()

    frames = []
    for feed, mode in FEEDS.items():
        vp = vehicle_points(feed, None)
        if vp.empty:
            continue
        if feed == "nswtrains":
            vp = vp[vp["route_id"].str.startswith("4T.T.")]  # trains only, no coaches
        frames.append(vp.assign(mode=mode))
    vp = pd.concat(frames, ignore_index=True)
    vp = vp[vp["lon"].between(149.5, 153.0) & vp["lat"].between(-35.5, -31.5)]
    # Drop stale GPS clocks: keep fixes within the bulk of the collected period
    lo, hi = vp["vp_ts"].quantile([0.005, 1.0])
    vp = vp[vp["vp_ts"].between(lo, hi)]
    if args.hours:
        vp = vp[vp["vp_ts"] >= hi - args.hours * 3600]

    # Lateness at each fix: most recent measured delay for that trip
    d = pd.read_parquet(MARTS_DIR / "stop_delays.parquet")
    d = d[~d["cancelled"] & d["delay_s"].notna()]
    lines = d.drop_duplicates("trip_id").set_index("trip_id")["route_short_name"]
    dd = d[["trip_id", "obs_time", "delay_s"]].dropna().astype({"obs_time": "int64"}).sort_values("obs_time")
    vp["vp_ts"] = vp["vp_ts"].astype("int64")
    vp = pd.merge_asof(vp.sort_values("vp_ts"), dd.rename(columns={"obs_time": "vp_ts"}), on="vp_ts", by="trip_id", direction="backward")
    vp["late"] = (vp["delay_s"].fillna(0) / 60).clip(-5, 60).round(1)

    # One fix per train per minute
    vp["minute"] = vp["vp_ts"] // 60
    vp = vp.sort_values("vp_ts").drop_duplicates(["trip_id", "minute"], keep="last")
    start, end = int(vp["vp_ts"].min()), int(vp["vp_ts"].max())

    trips = []
    for trip_id, g in vp.groupby("trip_id", sort=False):
        if len(g) < 2:
            continue
        line = lines.get(trip_id)
        if not isinstance(line, str) or not line:
            line = g["route_id"].iloc[0].split("_")[0].replace("4T.T.", "")
        if line in ("RTTA", "") or trip_id.startswith("NonTimetabled"):
            line = OUT_OF_SERVICE  # empty runs and shunting: drawn grey, lateness not meaningful
        pts = np.column_stack([g["lon"].round(5), g["lat"].round(5), g["vp_ts"] - start, g["late"]])
        trips.append({"line": str(line), "mode": g["mode"].iloc[0], "p": pts.tolist()})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"start": start, "end": end, "trips": trips}, separators=(",", ":")))
    print(f"{len(trips):,} trips, {len(vp):,} fixes, {OUT.stat().st_size / 1e6:.1f} MB, "
          f"{pd.Timestamp(start, unit='s', tz='Australia/Sydney'):%d %b %H:%M} to {pd.Timestamp(end, unit='s', tz='Australia/Sydney'):%d %b %H:%M}")


if __name__ == "__main__":
    main()
