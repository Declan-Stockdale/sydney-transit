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
    - bad GPS fixes are dropped (see clean_track and off_network): short bursts
      where a train "jumps" faster than any train can go and then jumps back, and
      Sydney Trains and Metro fixes more than 2 km from any of their lines
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

from analysis.reliability import MARTS_DIR, TO_METRES, vehicle_points
from tfnsw import PROJECT_DIR

OUT = PROJECT_DIR / "app" / "data" / "replay.json"
FEEDS = {"sydneytrains": "Trains", "metro": "Metro", "nswtrains": "TrainLink"}
OUT_OF_SERVICE = "Out of service"  # route RTTA (Out Of Service / Non Revenue) and NonTimetabled.* trips
SEGMENTS = PROJECT_DIR / "app" / "data" / "segments.csv"

# Bad GPS. No train here exceeds 160 km/h (the XPT's top speed; suburban trains ~130), so a move faster
# than that is a glitch, often one fix landing on a parallel line before snapping back. Moves under 1 km
# are never flagged, because GPS clocks jitter by a few seconds and would inflate speeds over short gaps.
MAX_KMH = 160
MIN_JUMP_M = 1000
MAX_GLITCH_FIXES = 4  # a run of this many fixes (~2 min) or fewer between impossible jumps is a glitch
MAX_OFF_NETWORK_M = 2000  # segments are straight station-to-station lines, so curved track can sit ~1 km off


def impossible_moves(x, y, t) -> np.ndarray:
    """True for each step between consecutive fixes (metres, seconds) that no train could make."""
    dist = np.hypot(np.diff(x), np.diff(y))
    kmh = dist / np.maximum(np.diff(t), 1) * 3.6
    return (dist > MIN_JUMP_M) & (kmh > MAX_KMH)


def clean_track(x, y, t) -> np.ndarray:
    """Keep-mask for one train's time-sorted fixes. Cuts the track at every impossible move and drops
    the short runs between cuts (a glitch away and back). If every run is short, keeps the longest."""
    x, y, t = (np.asarray(a, dtype=float) for a in (x, y, t))
    if len(t) < 2:
        return np.ones(len(t), dtype=bool)
    run = np.concatenate([[0], np.cumsum(impossible_moves(x, y, t))])
    sizes = np.bincount(run)
    good = sizes > MAX_GLITCH_FIXES
    if not good.any():
        good = np.arange(len(sizes)) == sizes.argmax()
    return good[run]


def off_network(x, y, seg: pd.DataFrame, chunk=5000) -> np.ndarray:
    """True for points more than MAX_OFF_NETWORK_M from every segment (ax, ay, bx, by, all in metres)."""
    ax, ay, bx, by = (seg[c].to_numpy(float)[None, :] for c in ("ax", "ay", "bx", "by"))
    vx, vy = bx - ax, by - ay
    len2 = np.maximum(vx**2 + vy**2, 1e-9)
    out = np.empty(len(x), dtype=bool)
    for i in range(0, len(x), chunk):
        px, py = np.asarray(x[i:i + chunk], float)[:, None], np.asarray(y[i:i + chunk], float)[:, None]
        f = np.clip(((px - ax) * vx + (py - ay) * vy) / len2, 0, 1)
        out[i:i + chunk] = np.hypot(px - ax - f * vx, py - ay - f * vy).min(axis=1) > MAX_OFF_NETWORK_M
    return out


def drop_bad_gps(vp: pd.DataFrame) -> pd.DataFrame:
    seg = pd.read_csv(SEGMENTS)
    seg["ax"], seg["ay"] = TO_METRES.transform(seg["a_lon"].to_numpy(), seg["a_lat"].to_numpy())
    seg["bx"], seg["by"] = TO_METRES.transform(seg["b_lon"].to_numpy(), seg["b_lat"].to_numpy())
    # Only Sydney Trains and Metro: segments.csv doesn't cover TrainLink's regional lines (Bathurst, Dubbo, ...)
    off = off_network(vp["x"].to_numpy(), vp["y"].to_numpy(), seg) & vp["mode"].isin(["Trains", "Metro"]).to_numpy()
    vp = vp[~off].sort_values(["trip_id", "vp_ts"])
    keep = np.concatenate([clean_track(g["x"], g["y"], g["vp_ts"]) for _, g in vp.groupby("trip_id", sort=True)])
    print(f"Bad GPS dropped: {off.sum():,} fixes off the rail network, {(~keep).sum():,} in impossible jumps "
          f"(of {len(off):,})")
    return vp[keep]


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
    vp = drop_bad_gps(vp)

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
