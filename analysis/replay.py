"""
Replay data for the animated map: where every train, bus, ferry and light rail
vehicle was, and how late it was running, plus the per-minute table behind the
"Delays by line and hour" dashboard page.

Built from the vehicle positions feeds already collected:
    - one GPS fix per vehicle per minute (fixes arrive every 30 to 60 s)
    - lateness at each fix:
        trains and Metro: the train's most recent measured delay at a station, from
            reliability.py (0 until its first measured stop)
        buses, ferries, light rail: the delay the trip updates feed was reporting for
            the vehicle's next stop at that moment (the operator's own estimate, so less
            reliable than the GPS-checked train delays); unknown if the feed gave none
    - fixes with stale GPS clocks and fixes outside NSW are dropped
    - bad GPS fixes are dropped (see clean_track, frozen_fixes and off_network): short
      bursts where a vehicle "jumps" faster than its mode can go (SPEED_LIMITS) and then
      jumps back; frozen positions that end in a catch-up jump (some bus trackers);
      Sydney Trains and Metro fixes more than 2 km from any of their lines; and light
      rail fixes more than 250 m from its track
    - gaps when collection paused (PC asleep or off) are kept as gaps: a vehicle is
      only drawn between fixes less than 5 minutes apart

Outputs:
    data/marts/vehicle_minutes.parquet   every vehicle-minute, all history (mode, line, time, lateness)
    app/static/replay/<mode>.json        the latest --hours for the map, one file per mode, served by
                                         Streamlit's static file serving and loaded only when ticked:
        {"mode", "start": <epoch s>, "lines": [names], "trips": [[line_index, out_of_service, p], ...]}
        p is a flat list of integers, 4 per fix: lon x 1e4, lat x 1e4, seconds since "start", minutes
        late x 10 (UNKNOWN_LATE if unknown). After the first fix, lon, lat and seconds are differences
        from the previous fix, which keeps ~1,700 buses to a few MB a day.

Usage:
    python -m analysis.reliability
    python -m analysis.replay                # map shows the latest 24 hours
    python -m analysis.replay --hours 48
"""

import argparse
import json

import numpy as np
import pandas as pd
from google.transit import gtfs_realtime_pb2

from analysis.reliability import MARTS_DIR, SYDNEY, TO_METRES, iter_snapshots, vehicle_points
from tfnsw import DATA_DIR, PROJECT_DIR

OUT_DIR = PROJECT_DIR / "app" / "static" / "replay"
MINUTES = MARTS_DIR / "vehicle_minutes.parquet"
STATIC_DIR = DATA_DIR / "static"
RAIL = {"sydneytrains": "Trains", "metro": "Metro", "nswtrains": "TrainLink"}
OTHER = {"buses": "Bus", "sydneyferries": "Ferry", "lightrail_cbdandsoutheast": "Light rail",
         "lightrail_innerwest": "Light rail", "lightrail_parramatta": "Light rail", "lightrail_newcastle": "Light rail"}
OUT_OF_SERVICE = "Out of service"  # route RTTA (Out Of Service / Non Revenue) and NonTimetabled.* trips
SEGMENTS = PROJECT_DIR / "app" / "data" / "segments.csv"
UNKNOWN_LATE = 9999
MAX_DELAY_AGE_S = 600  # a bus/ferry/light rail delay older than 10 minutes is too stale to colour by

# Bad GPS. A move faster than the mode's top speed is a glitch: often one fix landing on a parallel line or
# street before snapping back. Short moves are never flagged, because GPS clocks jitter by a few seconds and
# would inflate speeds over short gaps. Measured 27 Oct 2026: buses' 99.9th percentile ~95 km/h (motorway
# routes), light rail max ~61 km/h, ferries max ~50 km/h.
#   mode: (top speed km/h, shortest move that can be flagged in m)
SPEED_LIMITS = {
    "Trains": (160, 1000),  # the XPT's top speed; suburban trains ~130
    "Metro": (160, 1000),
    "TrainLink": (160, 1000),
    "Bus": (110, 500),  # 100 km/h motorway limit plus GPS slack
    "Light rail": (80, 300),
    "Ferry": (60, 300),
}
MAX_KMH, MIN_JUMP_M = SPEED_LIMITS["Trains"]
MAX_GLITCH_FIXES = 4  # a run of this many fixes (~2 to 4 min) or fewer between impossible jumps is a glitch
MAX_OFF_NETWORK_M = 2000  # rail segments are straight station-to-station lines, so curved track can sit ~1 km off
MAX_OFF_TRACK_LR_M = 250  # light rail is checked against its real track shapes (shapes.txt), so much tighter
LIGHT_RAIL_FEEDS = [f for f, m in OTHER.items() if m == "Light rail"]


def impossible_moves(x, y, t, max_kmh=MAX_KMH, min_jump_m=MIN_JUMP_M) -> np.ndarray:
    """True for each step between consecutive fixes (metres, seconds) that the vehicle couldn't make."""
    dist = np.hypot(np.diff(x), np.diff(y))
    kmh = dist / np.maximum(np.diff(t), 1) * 3.6
    return (dist > min_jump_m) & (kmh > max_kmh)


def frozen_fixes(x, y, t, max_kmh=MAX_KMH, min_jump_m=MIN_JUMP_M) -> np.ndarray:
    """True for repeats of a frozen position that end in a catch-up jump. Some bus trackers stop updating
    and resend the same coordinates for minutes, then jump to where the bus really is; the jump looks
    impossible from the last repeat but not from the first. A bus really standing still (at a terminus)
    keeps its repeats, because it moves off slowly afterwards."""
    n = len(t)
    frozen = np.zeros(n, dtype=bool)
    same = (np.diff(x) == 0) & (np.diff(y) == 0)
    i = 0
    while i < n - 1:
        if not same[i]:
            i += 1
            continue
        j = i
        while j < n - 1 and same[j]:
            j += 1  # fixes i..j share one position
        if j < n - 1 and impossible_moves(x[j:j + 2], y[j:j + 2], t[j:j + 2], max_kmh, min_jump_m)[0]:
            frozen[i + 1:j + 1] = True
        i = j
    return frozen


def clean_track(x, y, t, max_kmh=MAX_KMH, min_jump_m=MIN_JUMP_M) -> np.ndarray:
    """Keep-mask for one vehicle's time-sorted fixes. Drops frozen repeats before a catch-up jump, then cuts
    the track at every remaining impossible move and drops the short runs between cuts (a glitch away and
    back). If every run is short, keeps the longest."""
    x, y, t = (np.asarray(a, dtype=float) for a in (x, y, t))
    keep = np.ones(len(t), dtype=bool)
    if len(t) < 2:
        return keep
    keep &= ~frozen_fixes(x, y, t, max_kmh, min_jump_m)
    idx = np.flatnonzero(keep)
    run = np.concatenate([[0], np.cumsum(impossible_moves(x[idx], y[idx], t[idx], max_kmh, min_jump_m))])
    sizes = np.bincount(run)
    good = sizes > MAX_GLITCH_FIXES
    if not good.any():
        good = np.arange(len(sizes)) == sizes.argmax()
    keep[idx] = good[run]
    return keep


def off_network(x, y, seg: pd.DataFrame, max_m=MAX_OFF_NETWORK_M, chunk=5000) -> np.ndarray:
    """True for points more than max_m from every segment (ax, ay, bx, by, all in metres)."""
    ax, ay, bx, by = (seg[c].to_numpy(float)[None, :] for c in ("ax", "ay", "bx", "by"))
    vx, vy = bx - ax, by - ay
    len2 = np.maximum(vx**2 + vy**2, 1e-9)
    out = np.empty(len(x), dtype=bool)
    for i in range(0, len(x), chunk):
        px, py = np.asarray(x[i:i + chunk], float)[:, None], np.asarray(y[i:i + chunk], float)[:, None]
        f = np.clip(((px - ax) * vx + (py - ay) * vy) / len2, 0, 1)
        out[i:i + chunk] = np.hypot(px - ax - f * vx, py - ay - f * vy).min(axis=1) > max_m
    return out


def light_rail_track() -> pd.DataFrame:
    """Light rail track as short straight segments (metres), from the shapes in each line's timetable."""
    frames = []
    for feed in LIGHT_RAIL_FEEDS:
        f = STATIC_DIR / feed / "shapes.txt"
        if not f.exists():
            continue
        sh = pd.read_csv(f, dtype={"shape_id": str}, encoding="utf-8-sig").sort_values(["shape_id", "shape_pt_sequence"])
        sh["x"], sh["y"] = TO_METRES.transform(sh["shape_pt_lon"].to_numpy(), sh["shape_pt_lat"].to_numpy())
        nxt = sh.groupby("shape_id")[["x", "y"]].shift(-1)
        frames.append(pd.DataFrame({"ax": sh["x"], "ay": sh["y"], "bx": nxt["x"], "by": nxt["y"]}).dropna())
    seg = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ax", "ay", "bx", "by"])
    return seg.round(0).drop_duplicates()


def drop_bad_gps(vp: pd.DataFrame) -> pd.DataFrame:
    # Off the network: Sydney Trains and Metro against the station-to-station segments; light rail against its
    # track shapes. Not TrainLink (segments don't cover regional lines), buses (no road network here) or ferries.
    rail = vp["mode"].isin(["Trains", "Metro"]).to_numpy()
    seg = pd.read_csv(SEGMENTS)
    seg = seg[seg["mode"].isin(["Trains", "Metro"])]
    seg["ax"], seg["ay"] = TO_METRES.transform(seg["a_lon"].to_numpy(), seg["a_lat"].to_numpy())
    seg["bx"], seg["by"] = TO_METRES.transform(seg["b_lon"].to_numpy(), seg["b_lat"].to_numpy())
    off = np.zeros(len(vp), dtype=bool)
    off[rail] = off_network(vp["x"].to_numpy()[rail], vp["y"].to_numpy()[rail], seg)
    lr = (vp["mode"] == "Light rail").to_numpy()
    track = light_rail_track()
    if lr.any() and len(track):
        off[lr] = off_network(vp["x"].to_numpy()[lr], vp["y"].to_numpy()[lr], track, MAX_OFF_TRACK_LR_M)
    by_mode = vp[off].groupby("mode").size().to_dict()
    vp = vp[~off].sort_values(["key", "vp_ts"])
    keep = np.concatenate([clean_track(g["x"], g["y"], g["vp_ts"], *SPEED_LIMITS[g["mode"].iloc[0]])
                           for _, g in vp.groupby("key", sort=True)])
    jumps = vp[~keep].groupby("mode").size().to_dict()
    print(f"Bad GPS dropped (of {len(off):,} fixes): off the network/track {by_mode}, in jumps or frozen {jumps}")
    return vp[keep]


def route_names(feed: str) -> pd.Series:
    """route_id -> line name from the feed's latest timetable: the short name (T1, 333, F1, L2), except
    regional TrainLink trains (route type 106), whose short names are bare numbers, so they get the
    region instead (e.g. "North Coast NSW")."""
    f = STATIC_DIR / feed / "routes.txt"
    if not f.exists():
        return pd.Series(dtype=str)
    r = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
    name = r["route_short_name"].where(r["route_type"] != "106", r["route_long_name"])
    return pd.Series(name.values, index=r["route_id"])


def drop_intercity_duplicates(vp: pd.DataFrame) -> pd.DataFrame:
    """Intercity trains (Blue Mountains, Central Coast, South Coast, ...) can appear in both the Sydney
    Trains and TrainLink feeds. Drop a TrainLink fix when a Sydney Trains train on the same line was
    within 1 km in the same minute (reliability.py dedupes the delays the same way, by station)."""
    tl = vp["mode"] == "TrainLink"
    st = vp.loc[vp["mode"] == "Trains", ["line", "minute", "x", "y"]]
    cand = vp[tl].reset_index().merge(st, on=["line", "minute"], suffixes=("", "_st"))
    near = cand.loc[(cand["x"] - cand["x_st"]).pow(2) + (cand["y"] - cand["y_st"]).pow(2) < 1000**2, "index"].unique()
    print(f"Intercity duplicates dropped: {len(near):,} TrainLink fixes")
    return vp.drop(index=near)


def running_delays(feed: str) -> pd.DataFrame:
    """Per trip and snapshot, the delay the trip updates feed reported for the next stop that has one."""
    rows = []
    for raw in iter_snapshots(feed, None):
        msg = gtfs_realtime_pb2.FeedMessage()
        msg.ParseFromString(raw)
        ts = msg.header.timestamp
        for e in msg.entity:
            tu = e.trip_update
            for s in tu.stop_time_update:
                ev = s.arrival if s.arrival.HasField("delay") else s.departure if s.departure.HasField("delay") else None
                if ev is not None:
                    rows.append((tu.trip.trip_id, ts, ev.delay))
                    break
    return pd.DataFrame(rows, columns=["trip_id", "obs_time", "delay_s"])


def load_feed(feed: str, mode: str) -> pd.DataFrame:
    vp = vehicle_points(feed, None)
    if vp.empty:
        return vp
    if feed == "nswtrains":
        vp = vp[vp["route_id"].str.startswith("4T.T.")]  # trains only, no coaches
    if feed == "metro":
        vp = vp[~vp["route_id"].str.startswith("1001_")]  # CBD light rail vehicles also listed in the Metro feed
    if feed not in RAIL:
        vp = vp[vp["trip_id"] != ""]  # buses parked or between runs
    vp = vp[vp["lon"].between(149.5, 153.6) & vp["lat"].between(-35.5, -31.5)]
    # Drop stale GPS clocks: keep fixes within the bulk of this feed's collected period
    vp = vp[vp["vp_ts"] >= vp["vp_ts"].quantile(0.005)].copy()
    vp["mode"], vp["feed"], vp["key"] = mode, feed, feed + ":" + vp["trip_id"]
    vp["vp_ts"] = vp["vp_ts"].astype("int64")
    if feed not in RAIL:
        vp["line"] = vp["route_id"].map(route_names(feed)).fillna(vp["route_id"].str.split("_").str[-1])
    return vp


def add_lateness(vp: pd.DataFrame) -> pd.DataFrame:
    is_rail = vp["feed"].isin(list(RAIL))
    out = []
    if is_rail.any():
        d = pd.read_parquet(MARTS_DIR / "stop_delays.parquet")
        d = d[~d["cancelled"] & d["delay_s"].notna()]
        lines = d.drop_duplicates("trip_id").set_index("trip_id")["route_short_name"]
        dd = d[["trip_id", "obs_time", "delay_s"]].dropna().astype({"obs_time": "int64"}).sort_values("obs_time")
        rail = pd.merge_asof(vp[is_rail].sort_values("vp_ts"), dd.rename(columns={"obs_time": "vp_ts"}),
                             on="vp_ts", by="trip_id", direction="backward")
        rail["late"] = rail["delay_s"].fillna(0) / 60
        names = pd.concat([route_names(f) for f in RAIL])
        line = rail["trip_id"].map(lines)
        line = line.where(line.notna() & (line != ""), rail["route_id"].map(names))
        fallback = rail["route_id"].str.split("_").str[0].str.replace("4T.T.", "", regex=False)
        rail["line"] = line.where(line.notna() & (line != ""), fallback)
        oos = rail["line"].isin(["RTTA", ""]) | rail["trip_id"].str.startswith("NonTimetabled")
        rail.loc[oos, "line"] = OUT_OF_SERVICE  # empty runs and shunting: drawn grey, lateness not meaningful
        out.append(rail)
    for feed, g in vp[~is_rail].groupby("feed"):
        dd = running_delays(feed).astype({"obs_time": "int64"}).sort_values("obs_time")
        g = pd.merge_asof(g.sort_values("vp_ts"), dd.rename(columns={"obs_time": "vp_ts"}), on="vp_ts", by="trip_id",
                          direction="backward", tolerance=MAX_DELAY_AGE_S)
        g["late"] = g["delay_s"] / 60  # NaN = unknown
        out.append(g)
    vp = pd.concat(out, ignore_index=True)
    vp["late"] = vp["late"].clip(-5, 60).round(1)
    return vp


def encode(g: pd.DataFrame, start: int) -> list[int]:
    """One vehicle's fixes as the flat delta-encoded integer list described in the module docstring."""
    lon = np.round(g["lon"].to_numpy() * 1e4).astype(np.int64)
    lat = np.round(g["lat"].to_numpy() * 1e4).astype(np.int64)
    t = g["vp_ts"].to_numpy() - start
    late = np.where(g["late"].isna(), UNKNOWN_LATE, np.round(g["late"].fillna(0).to_numpy() * 10)).astype(np.int64)
    delta = lambda a: np.concatenate([a[:1], np.diff(a)])
    return np.column_stack([delta(lon), delta(lat), delta(t), late]).ravel().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the replay map data and vehicle-minute table.")
    parser.add_argument("--hours", type=float, default=24, help="Hours shown on the map, latest first (default 24)")
    args = parser.parse_args()

    frames = [load_feed(feed, mode) for feed, mode in {**RAIL, **OTHER}.items()]
    vp = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    vp = drop_bad_gps(vp)
    vp = add_lateness(vp)

    # One fix per vehicle per minute
    vp["minute"] = vp["vp_ts"] // 60
    vp = vp.sort_values("vp_ts").drop_duplicates(["key", "minute"], keep="last")
    vp = drop_intercity_duplicates(vp)

    # Every vehicle-minute, all history, for the delays-by-hour page (out-of-service runs excluded)
    vm = vp.loc[vp["line"] != OUT_OF_SERVICE, ["mode", "line", "key", "vp_ts", "late"]]
    vm.to_parquet(MINUTES, index=False)
    print(f"{MINUTES.name}: {len(vm):,} vehicle-minutes")

    # Map files: the latest --hours only
    end = int(vp["vp_ts"].max())
    vp = vp[vp["vp_ts"] >= end - args.hours * 3600]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mode, m in vp.groupby("mode"):
        start = int(m["vp_ts"].min())
        lines = sorted(m["line"].astype(str).unique())
        index = {name: i for i, name in enumerate(lines)}
        trips = [[index[str(g["line"].iloc[0])], int(g["line"].iloc[0] == OUT_OF_SERVICE), encode(g, start)]
                 for _, g in m.groupby("key", sort=False) if len(g) >= 2]
        f = OUT_DIR / f"{mode.lower().replace(' ', '_')}.json"
        f.write_text(json.dumps({"mode": mode, "start": start, "lines": lines, "trips": trips}, separators=(",", ":")))
        print(f"{f.name}: {len(trips):,} trips, {len(m):,} fixes, {f.stat().st_size / 1e6:.1f} MB, "
              f"{pd.Timestamp(start, unit='s', tz=SYDNEY):%d %b %H:%M} to "
              f"{pd.Timestamp(int(m['vp_ts'].max()), unit='s', tz=SYDNEY):%d %b %H:%M}")


if __name__ == "__main__":
    main()
