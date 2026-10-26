"""
Reliability tracker, first version: turn saved realtime snapshots into one
"observed" delay per train per stop.

How the feed behaves (checked 2026-10-26): each snapshot lists, for every
running trip, predictions for its remaining stops. A stop drops out of the trip
once the train has served it. So the observed arrival at a stop is taken as
the LAST prediction seen before the stop drops out (see DECISIONS.md: realtime
GTFS gives predictions, not confirmed arrivals).

Steps:
    1. Read every trip-update snapshot for the chosen feeds and dates, both
       day folders of .pb.gz files and compacted <date>.tar.xz archives.
    2. Keep, per (service date, trip, stop), the row from the last snapshot
       that contains it. Rows still present in the final snapshot are
       dropped, because that stop hasn't been served yet.
    3. Delay = the feed's own delay field where given; otherwise predicted
       time minus scheduled time from the static timetable version for that
       service date (data/raw/static/<feed>/<date>.zip).
    4. Cancelled trips are kept as rows with cancelled = True and no delay.

Output: data/marts/stop_delays.parquet, one row per train per stop, with the
station name, line, scheduled and observed times (UTC), delay_s, cancelled.

Usage:
    python -m analysis.reliability                       # all collected days
    python -m analysis.reliability --dates 2026-10-26
"""

import argparse
import gzip
import tarfile
import zipfile
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from google.transit import gtfs_realtime_pb2

from tfnsw import DATA_DIR

RAW_RT = DATA_DIR / "raw" / "realtime" / "trip_updates"
RAW_STATIC = DATA_DIR / "raw" / "static"
MARTS_DIR = DATA_DIR / "marts"
SYDNEY = ZoneInfo("Australia/Sydney")
FEEDS = ["sydneytrains", "metro"]
TRIP_SR = gtfs_realtime_pb2.TripDescriptor.ScheduleRelationship


def iter_snapshots(feed: str, dates: list[str] | None):
    """Yield raw snapshot bytes in time order, from day folders and .tar.xz archives."""
    base = RAW_RT / feed
    for day in sorted(p for p in base.iterdir()):
        name = day.name.removesuffix(".tar.xz")
        if dates and name not in dates:
            continue
        if day.is_dir():
            for f in sorted(day.glob("*.pb.gz")):
                yield gzip.decompress(f.read_bytes())
        elif day.name.endswith(".tar.xz"):
            with tarfile.open(day) as tf:
                for member in sorted(tf.getmembers(), key=lambda m: m.name):
                    yield tf.extractfile(member).read()


def flatten(raw: bytes) -> list[dict]:
    msg = gtfs_realtime_pb2.FeedMessage()
    msg.ParseFromString(raw)
    ts = msg.header.timestamp
    # Sydney Trains leaves start_date blank. Its service day runs to about 3am,
    # so the service date is the Sydney date three hours before the snapshot.
    fallback_date = (datetime.fromtimestamp(ts, SYDNEY) - timedelta(hours=3)).strftime("%Y%m%d")
    rows = []
    for e in msg.entity:
        if not e.HasField("trip_update"):
            continue
        tu = e.trip_update
        base = {"snapshot_ts": ts, "trip_id": tu.trip.trip_id, "start_date": tu.trip.start_date or fallback_date,
                "route_id": tu.trip.route_id, "cancelled": tu.trip.schedule_relationship == TRIP_SR.Value("CANCELED")}
        if base["cancelled"] or not tu.stop_time_update:
            rows.append({**base, "stop_id": None})
            continue
        for s in tu.stop_time_update:
            ev = s.arrival if s.HasField("arrival") else s.departure
            rows.append({
                **base,
                "stop_id": s.stop_id,
                "feed_delay_s": ev.delay if ev.HasField("delay") else None,
                "pred_time": ev.time if ev.HasField("time") else None,
            })
    return rows


def last_seen(feed: str, dates) -> pd.DataFrame:
    frames, last_ts = [], 0
    for raw in iter_snapshots(feed, dates):
        frames.append(pd.DataFrame(flatten(raw)))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    last_ts = df["snapshot_ts"].max()
    df = df.sort_values("snapshot_ts").drop_duplicates(["start_date", "trip_id", "stop_id"], keep="last")
    df = df[(df["snapshot_ts"] < last_ts) | df["cancelled"]]  # still in the last snapshot = not served yet
    df["feed"] = feed
    return df


def static_zip(feed: str, service_date: str):
    """The timetable version in force on a service date: the newest zip dated on or before it."""
    d = f"{service_date[:4]}-{service_date[4:6]}-{service_date[6:]}"
    zips = sorted(p for p in (RAW_STATIC / feed).glob("*.zip") if p.name[:10] <= d)
    return zips[-1] if zips else None


def schedule(feed: str, service_date: str) -> pd.DataFrame:
    z = static_zip(feed, service_date)
    if z is None:
        return pd.DataFrame()
    with zipfile.ZipFile(z) as zf:
        st = pd.read_csv(zf.open("stop_times.txt"), dtype=str, usecols=["trip_id", "stop_id", "arrival_time", "departure_time"])
        stops = pd.read_csv(zf.open("stops.txt"), dtype=str, usecols=["stop_id", "stop_name", "parent_station"])
        trips = pd.read_csv(zf.open("trips.txt"), dtype=str, usecols=["trip_id", "route_id"])
        routes = pd.read_csv(zf.open("routes.txt"), dtype=str, usecols=["route_id", "route_short_name"])
    midnight = datetime.strptime(service_date, "%Y%m%d").replace(tzinfo=SYDNEY)
    t = st["arrival_time"].fillna(st["departure_time"]).str.split(":", expand=True).astype(int)
    secs = t[0] * 3600 + t[1] * 60 + t[2]
    st["sched_time"] = secs.map(lambda s: int((midnight + timedelta(seconds=int(s))).timestamp()))
    parent = stops.set_index("stop_id")["parent_station"].fillna("")
    names = stops.set_index("stop_id")["stop_name"]
    station_id = st["stop_id"].map(lambda s: parent.get(s) or s)
    st["station"] = station_id.map(names).str.replace(r"\s+Station$", "", regex=True)
    st = st.merge(trips.merge(routes, on="route_id"), on="trip_id", how="left")
    return st[["trip_id", "stop_id", "sched_time", "station", "route_short_name"]]


def build(dates=None) -> pd.DataFrame:
    out = []
    for feed in FEEDS:
        obs = last_seen(feed, dates)
        if obs.empty:
            continue
        for sd, part in obs.groupby("start_date"):
            sched = schedule(feed, sd)
            if sched.empty:
                continue
            m = part.merge(sched, on=["trip_id", "stop_id"], how="left")
            m["delay_s"] = m["feed_delay_s"].where(m["feed_delay_s"].notna(), m["pred_time"] - m["sched_time"])
            m["obs_time"] = m["pred_time"].where(m["pred_time"].notna(), m["sched_time"] + m["feed_delay_s"])
            m["service_date"] = sd
            out.append(m)
    df = pd.concat(out, ignore_index=True)
    df["sched_local"] = pd.to_datetime(df["sched_time"], unit="s", utc=True).dt.tz_convert(SYDNEY)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Observed delay per train per stop from realtime snapshots.")
    parser.add_argument("--dates", nargs="+", default=None, help="UTC day folders, YYYY-MM-DD")
    args = parser.parse_args()
    df = build(args.dates)
    MARTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MARTS_DIR / "stop_delays.parquet", index=False)

    served = df[~df["cancelled"] & df["delay_s"].notna() & df["station"].notna()]
    print(f"{len(served):,} train-stop observations, {served['trip_id'].nunique():,} trips, "
          f"{df.loc[df['cancelled'], 'trip_id'].nunique()} cancelled; "
          f"{served['sched_local'].min():%H:%M} to {served['sched_local'].max():%H:%M %d %b} Sydney time")
    print(f"Unmatched to timetable: {df['station'].isna().mean():.1%}")
    print(f"On time (<5 min late): {(served['delay_s'] < 300).mean():.1%}; median delay {served['delay_s'].median():.0f} s")
    print(served.groupby("route_short_name")["delay_s"].agg(["count", "median", lambda s: (s < 300).mean()]).round(2))


if __name__ == "__main__":
    main()
