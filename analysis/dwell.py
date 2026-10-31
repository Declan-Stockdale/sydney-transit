"""
Dwell times: how long trains spend at each station's platform, by hour.
Long dwells in the peak are a sign of crowded platforms and doors.

The two position feeds report differently, so each gets its own measure of
"platform time" (the dwell plus the last few metres of braking and the first
few of pulling away):
    Metro         GPS every ~30 s with a status: time from the first fix STOPPED_AT
                  a station to the first fix after it leaves (accurate to ~30 s)
    Sydney Trains positions snap to signalling locations, and a platform is its
                  own location ("Ourimbah 2 Loc" = platform 2): time from entering
                  the platform location to entering the next one
Each trip's first and last station are left out (trains wait there to turn
around), as are platform times over 15 minutes (held trains, data gaps).

Because the two measures differ, compare stations within a mode, and compare
each station's peak with its own quieter hours ("extra in peak").

Output: app/data/dwell.csv
    mode, station, hour, stops, median_s, p90_s
and app/data/dwell_stations.csv: mode, station, stops, median_s, offpeak_median_s,
    peak_median_s, extra_peak_s, lat, lon

Usage:
    python -m analysis.reliability
    python -m analysis.dwell
"""

import numpy as np
import pandas as pd
from google.transit import gtfs_realtime_pb2

from analysis.reliability import TO_METRES, iter_vp_snapshots
from analysis.stop_events import rail_events, station_coords
from tfnsw import PROJECT_DIR

OUT = PROJECT_DIR / "app" / "data"
PLATFORM_LOC = r"\.[A-Za-z' ]+ \d+ Loc$"  # e.g. "CentralCoast.Ourimbah 2 Loc"; signals look like "Hornsby.N22.97 Loc"
STOPPED_AT = gtfs_realtime_pb2.VehiclePosition.VehicleStopStatus.Value("STOPPED_AT")
MAX_PLATFORM_S = 15 * 60
MAX_STATION_M = 400  # a platform location (or a Metro stop) must be this close to the station it's matched to
PEAK_HOURS = [7, 8, 17, 18]
OFFPEAK_HOURS = [10, 11, 12, 13, 14]
MIN_STOPS = 10


def fixes(feed: str, dates=None) -> pd.DataFrame:
    rows = []
    for raw in iter_vp_snapshots(feed, dates):
        msg = gtfs_realtime_pb2.FeedMessage()
        msg.ParseFromString(raw)
        for e in msg.entity:
            v = e.vehicle
            if e.HasField("vehicle") and v.timestamp and v.trip.trip_id:
                rows.append((v.trip.trip_id, v.timestamp, v.stop_id, v.current_status, v.position.longitude, v.position.latitude))
    f = pd.DataFrame(rows, columns=["trip_id", "ts", "loc", "status", "lon", "lat"]).drop_duplicates(["trip_id", "ts"])
    return f.sort_values(["trip_id", "ts"]).reset_index(drop=True)


def platform_visits(f: pd.DataFrame, at_platform: pd.Series) -> pd.DataFrame:
    """Runs of consecutive fixes of a trip at the same platform; platform time = first fix after the run - first fix in it."""
    f = f.assign(at=at_platform.to_numpy(), idx=np.arange(len(f)))
    new_run = (f["trip_id"] != f["trip_id"].shift()) | (f["loc"] != f["loc"].shift()) | (f["at"] != f["at"].shift())
    f["run"] = new_run.cumsum()
    nxt_ts = f.groupby("trip_id")["ts"].shift(-1)
    runs = f[f["at"]].groupby("run").agg(trip_id=("trip_id", "first"), loc=("loc", "first"), start=("ts", "first"),
                                         last_idx=("idx", "last"), lon=("lon", "first"), lat=("lat", "first"))
    runs["end"] = nxt_ts.reindex(runs["last_idx"]).to_numpy()
    runs["platform_s"] = runs["end"] - runs["start"]
    return runs.dropna(subset=["platform_s"]).drop(columns="last_idx")


def match_station(v: pd.DataFrame, coords: pd.DataFrame, chunk: int = 5000) -> pd.DataFrame:
    """Nearest station to each visit, kept if within MAX_STATION_M."""
    x, y = TO_METRES.transform(v["lon"].to_numpy(), v["lat"].to_numpy())
    cx, cy = coords["x"].to_numpy()[None, :], coords["y"].to_numpy()[None, :]
    j, dist = np.empty(len(v), dtype=int), np.empty(len(v))
    for i in range(0, len(v), chunk):
        d = np.hypot(x[i:i + chunk, None] - cx, y[i:i + chunk, None] - cy)
        j[i:i + chunk] = d.argmin(axis=1)
        dist[i:i + chunk] = d.min(axis=1)
    v = v.assign(station=coords["station"].to_numpy()[j], dist_m=dist)
    return v[v["dist_m"] <= MAX_STATION_M]


def visits(dates=None) -> pd.DataFrame:
    coords = station_coords()
    out = []
    st = fixes("sydneytrains", dates)
    if not st.empty:
        v = platform_visits(st, st["loc"].str.contains(PLATFORM_LOC, regex=True))
        out.append(match_station(v, coords).assign(mode="Trains"))
    mt = fixes("metro", dates)
    if not mt.empty:
        v = platform_visits(mt, mt["status"] == STOPPED_AT)
        out.append(match_station(v, coords).assign(mode="Metro"))
    v = pd.concat(out, ignore_index=True)
    return v[(v["platform_s"] > 0) & (v["platform_s"] <= MAX_PLATFORM_S)]


def with_trip_context(v: pd.DataFrame, rail: pd.DataFrame) -> pd.DataFrame:
    """Keep intermediate stops only, adding line and the scheduled hour from the matching train-station row."""
    r = rail[rail["prev_station"].notna() & rail["next_station"].notna()][["trip_id", "station", "line", "sched_time", "hour", "date", "delay_s"]]
    m = v.merge(r, on=["trip_id", "station"])
    m = m[(m["start"] - m["sched_time"]).abs() < 3600]  # the right pass for trips that visit a station twice
    return m.drop_duplicates(["trip_id", "station", "start"])


def summarise(m: pd.DataFrame, coords: pd.DataFrame):
    q90 = lambda s: s.quantile(0.9)
    by_hour = m.groupby(["mode", "station", "hour"]).agg(stops=("platform_s", "size"), median_s=("platform_s", "median"),
                                                         p90_s=("platform_s", q90)).reset_index()
    by_hour = by_hour[by_hour["stops"] >= MIN_STOPS]
    st = m.groupby(["mode", "station"]).agg(stops=("platform_s", "size"), median_s=("platform_s", "median")).reset_index()
    for name, hours in [("offpeak_median_s", OFFPEAK_HOURS), ("peak_median_s", PEAK_HOURS)]:
        st = st.merge(m[m["hour"].isin(hours)].groupby(["mode", "station"])["platform_s"].median().rename(name).reset_index(),
                      on=["mode", "station"], how="left")
    st["extra_peak_s"] = st["peak_median_s"] - st["offpeak_median_s"]
    st = st[st["stops"] >= MIN_STOPS].merge(coords[["station", "lat", "lon"]], on="station", how="left")
    return by_hour.round(0), st.round({"median_s": 0, "offpeak_median_s": 0, "peak_median_s": 0, "extra_peak_s": 0})


def main() -> None:
    coords = station_coords()
    m = with_trip_context(visits(), rail_events())
    by_hour, st = summarise(m, coords)
    by_hour.to_csv(OUT / "dwell.csv", index=False)
    st.to_csv(OUT / "dwell_stations.csv", index=False)
    print(f"dwell.csv: {len(by_hour):,} rows from {len(m):,} platform stops")
    print(m.groupby("mode")["platform_s"].describe().round(0))
    print(st[st["stops"] >= 100].nlargest(12, "extra_peak_s"))


if __name__ == "__main__":
    main()
