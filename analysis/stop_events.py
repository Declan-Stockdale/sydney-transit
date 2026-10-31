"""
Arrival and departure events at stations, the shared input for the operations
analyses (connections, delay origins, slow stretches, headways, dwell, crowding).

Rail (trains, Metro, TrainLink): read from data/marts/stop_delays.parquet
(analysis/reliability.py), one row per train per station with:
    arr_actual  first GPS fix within 300 m of the station (the departure time where no fix matched)
    dep_actual  last GPS fix within 300 m (reliability's measured departure), or the last prediction
    prev_station, next_station  the stations before and after on that trip
    line_dir    line + next station, e.g. "T1>Seven Hills": one direction and stopping pattern

Buses: only stops within HUB_RADIUS_M of a rail station (bus stops that feed the
rail network). Built from the bus trip updates feed the same way reliability.py
treats trains: the observed time at a stop is the LAST prediction seen before the
stop drops out of the trip, trusted only if the next snapshot came within 3
minutes. These are the operator's own estimates, not GPS-checked like the trains.

Output: data/marts/bus_stop_events.parquet
    trip_id, route_id, line, stop_id, stop_sequence, first_stop, last_stop, station, walk_m,
    sched_arr, arr_actual, sched_dep, dep_actual   (epoch seconds, UTC)

Usage:
    python -m analysis.reliability
    python -m analysis.stop_events
"""

import argparse

import numpy as np
import pandas as pd
from google.transit import gtfs_realtime_pb2

from analysis.reliability import MARTS_DIR, SYDNEY, TO_METRES, iter_snapshots
from analysis.replay import route_names
from tfnsw import DATA_DIR

STOP_DELAYS = MARTS_DIR / "stop_delays.parquet"
BUS_EVENTS = MARTS_DIR / "bus_stop_events.parquet"
STATIC_DIR = DATA_DIR / "static"
HUB_RADIUS_M = 250  # a bus stop this close to a station (straight line) counts as serving it
MAX_SNAPSHOT_GAP_S = 180
MODES = {"sydneytrains": "Trains", "metro": "Metro", "nswtrains": "TrainLink"}


def add_neighbours(d: pd.DataFrame) -> pd.DataFrame:
    """Previous and next station of each row's trip (rows ordered by scheduled time) and a line_dir key."""
    d = d.sort_values(["service_date", "trip_id", "sched_time"])
    g = d.groupby(["service_date", "trip_id"], sort=False)["station"]
    d["prev_station"] = g.shift(1)
    d["next_station"] = g.shift(-1)
    d["line_dir"] = d["line"] + ">" + d["next_station"].fillna("(terminates)")
    return d


def rail_events(path=STOP_DELAYS) -> pd.DataFrame:
    """Served train-station events with actual arrival and departure times, neighbours and Sydney hour."""
    d = pd.read_parquet(path)
    d = d[~d["cancelled"] & d["station"].notna() & d["sched_time"].notna() & d["delay_s"].notna()].copy()
    d["mode"] = d["feed"].map(MODES)
    d["line"] = d["route_short_name"].fillna("").astype(str)
    d = d[d["line"].ne("") & d["line"].ne("RTTA")]
    d["dep_actual"] = d["obs_time"]
    gps_first = d["gps_first_time"] if "gps_first_time" in d else pd.Series(np.nan, index=d.index)
    d["arr_actual"] = gps_first.where(d["delay_source"] == "gps").fillna(d["obs_time"])
    d = add_neighbours(d)
    d["hour"] = d["sched_local"].dt.hour
    d["date"] = d["sched_local"].dt.strftime("%Y-%m-%d")
    return d.reset_index(drop=True)


def station_coords(feeds=("sydneytrains", "metro", "nswtrains")) -> pd.DataFrame:
    """Rail stations from the latest timetables: station (name as in stop_delays), stop_id, lat, lon, x, y (metres)."""
    stations = []
    for feed in feeds:
        f = STATIC_DIR / feed / "stops.txt"
        if f.exists():
            s = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
            stations.append(s[s["location_type"] == "1"][["stop_id", "stop_name", "stop_lat", "stop_lon"]])
    st = pd.concat(stations)
    st["station"] = st["stop_name"].str.replace(r"\s+Station$", "", regex=True)
    st = st.drop_duplicates("station").rename(columns={"stop_lat": "lat", "stop_lon": "lon"}).astype({"lat": float, "lon": float})
    st["x"], st["y"] = TO_METRES.transform(st["lon"].to_numpy(), st["lat"].to_numpy())
    return st[["station", "stop_id", "lat", "lon", "x", "y"]].reset_index(drop=True)


def hub_stops(max_m: float = HUB_RADIUS_M) -> pd.DataFrame:
    """Bus stops within max_m of a Sydney Trains or Metro station: stop_id, station, walk_m (straight line)."""
    st = station_coords(("sydneytrains", "metro"))
    sx, sy = st["x"].to_numpy(), st["y"].to_numpy()
    bus = pd.read_csv(STATIC_DIR / "buses" / "stops.txt", dtype=str, encoding="utf-8-sig")
    bus = bus[bus["location_type"].fillna("").isin(["", "0"])]
    bx, by = TO_METRES.transform(bus["stop_lon"].astype(float).to_numpy(), bus["stop_lat"].astype(float).to_numpy())
    rows = []
    for i in range(0, len(bus), 2000):  # chunked: ~37k bus stops x ~330 stations
        d = np.hypot(bx[i:i + 2000, None] - sx[None, :], by[i:i + 2000, None] - sy[None, :])
        j = d.argmin(axis=1)
        near = d[np.arange(len(j)), j] <= max_m
        rows.append(pd.DataFrame({"stop_id": bus["stop_id"].to_numpy()[i:i + 2000][near],
                                  "station": st["station"].to_numpy()[j[near]],
                                  "walk_m": d[np.arange(len(j)), j][near].round(0)}))
    return pd.concat(rows, ignore_index=True)


def bus_hub_events(hubs: pd.DataFrame, dates=None, feed: str = "buses") -> pd.DataFrame:
    """Last prediction per (start date, trip, hub stop) before the stop dropped out of the feed."""
    want = set(hubs["stop_id"])
    cols = ["snapshot_ts", "start_date", "trip_id", "route_id", "stop_id", "stop_sequence", "arr_time", "arr_delay",
            "dep_time", "dep_delay"]
    kept, batch, snaps, last_seq = None, [], [], {}
    for n, raw in enumerate(iter_snapshots(feed, dates), 1):
        msg = gtfs_realtime_pb2.FeedMessage()
        msg.ParseFromString(raw)
        ts = msg.header.timestamp
        snaps.append(ts)
        for e in msg.entity:
            tu = e.trip_update
            if tu.stop_time_update:  # the trip's final stop, so a bus ending at a station isn't offered as a departure
                key = (tu.trip.start_date, tu.trip.trip_id)
                last_seq[key] = max(last_seq.get(key, 0), tu.stop_time_update[-1].stop_sequence)
            for s in tu.stop_time_update:
                if s.stop_id in want:
                    a, d = s.arrival, s.departure
                    batch.append((ts, tu.trip.start_date, tu.trip.trip_id, tu.trip.route_id, s.stop_id, s.stop_sequence,
                                  a.time or None, a.delay if a.HasField("delay") else None,
                                  d.time or None, d.delay if d.HasField("delay") else None))
        if n % 300 == 0:  # fold into the running "last seen" table to keep memory flat
            kept = _fold(kept, pd.DataFrame(batch, columns=cols))
            batch = []
            print(f"  {n:,} bus snapshots read")
    kept = _fold(kept, pd.DataFrame(batch, columns=cols))
    if kept is None or kept.empty:
        return pd.DataFrame()
    snaps = np.unique(np.array(snaps))
    nxt = pd.Series(np.append(snaps[1:], np.iinfo(np.int64).max), index=snaps)
    gap = kept["snapshot_ts"].map(nxt) - kept["snapshot_ts"]
    kept = kept[(kept["snapshot_ts"] < snaps.max()) & (gap <= MAX_SNAPSHOT_GAP_S)].copy()
    kept["last_sequence"] = [last_seq.get(k) for k in zip(kept["start_date"], kept["trip_id"])]
    return kept


def _fold(kept, new):
    df = new if kept is None else pd.concat([kept, new], ignore_index=True)
    return df.sort_values("snapshot_ts").drop_duplicates(["start_date", "trip_id", "stop_id"], keep="last")


def build_bus_events(dates=None) -> pd.DataFrame:
    hubs = hub_stops()
    ev = bus_hub_events(hubs, dates)
    if ev.empty:
        return ev
    ev = ev.merge(hubs, on="stop_id")
    ev["arr_actual"] = ev["arr_time"].fillna(ev["dep_time"])
    ev["dep_actual"] = ev["dep_time"].fillna(ev["arr_time"])
    ev["sched_arr"] = ev["arr_actual"] - ev["arr_delay"].fillna(ev["dep_delay"]).fillna(0)
    ev["sched_dep"] = ev["dep_actual"] - ev["dep_delay"].fillna(ev["arr_delay"]).fillna(0)
    names = route_names("buses")
    ev["line"] = ev["route_id"].map(names).fillna(ev["route_id"].str.split("_").str[-1])
    ev["first_stop"] = ev["stop_sequence"] <= 1
    ev["last_stop"] = ev["stop_sequence"] >= ev["last_sequence"]
    return ev[["trip_id", "route_id", "line", "stop_id", "stop_sequence", "first_stop", "last_stop", "station", "walk_m",
               "sched_arr", "arr_actual", "sched_dep", "dep_actual"]].dropna(subset=["arr_actual"])


def bus_events(path=BUS_EVENTS) -> pd.DataFrame:
    ev = pd.read_parquet(path)
    t = pd.to_datetime(ev["sched_arr"], unit="s", utc=True).dt.tz_convert(SYDNEY)
    ev["hour"], ev["date"] = t.dt.hour, t.dt.strftime("%Y-%m-%d")
    return ev


def main() -> None:
    parser = argparse.ArgumentParser(description="Bus arrivals and departures at stops next to rail stations.")
    parser.add_argument("--dates", nargs="+", default=None, help="UTC day folders, YYYY-MM-DD")
    args = parser.parse_args()
    ev = build_bus_events(args.dates)
    ev.to_parquet(BUS_EVENTS, index=False)
    print(f"{BUS_EVENTS.name}: {len(ev):,} bus stop events at {ev['stop_id'].nunique():,} stops next to "
          f"{ev['station'].nunique()} stations, {ev['line'].nunique()} routes")


if __name__ == "__main__":
    main()
