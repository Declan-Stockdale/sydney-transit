"""
Week 1 exploration: pull TfNSW realtime trip updates and the static timetable,
decode the protocol buffers, and run first sanity checks.

Usage:
    export TFNSW_API_KEY="your-key-here"        # Windows PowerShell: $env:TFNSW_API_KEY="your-key-here"
    python explore_feeds.py                      # Sydney Trains (default)
    python explore_feeds.py --feed metro         # Metro
    python explore_feeds.py --skip-static        # realtime only (static zip is large, no need to re-download often)

Outputs (under ./data):
    raw/realtime/<feed>/<timestamp>.pb    raw snapshot, keep a few as test fixtures
    flat/<feed>_trip_updates_<timestamp>.csv   one row per trip x stop
    static/<feed>/                         extracted GTFS text files
"""

import argparse
import io
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from google.transit import gtfs_realtime_pb2

load_dotenv()

BASE_URL = "https://api.transport.nsw.gov.au"

# Endpoint paths. Sydney Trains paths are confirmed on the TfNSW forum.
# Check the Metro paths in the Open Data Hub API explorer before relying on them.
REALTIME_PATHS = {
    "sydneytrains": "/v2/gtfs/realtime/sydneytrains",
    "metro": "/v2/gtfs/realtime/metro",
}
STATIC_PATHS = {
    "sydneytrains": "/v1/gtfs/schedule/sydneytrains",
    "metro": "/v1/gtfs/schedule/metro",
}

DATA_DIR = Path("data")


def get_api_key() -> str:
    key = os.environ.get("TFNSW_API_KEY")
    if not key:
        sys.exit("Set the TFNSW_API_KEY environment variable first (never hard-code it).")
    return key


def tfnsw_get(path: str, api_key: str, accept: str, timeout: int) -> bytes:
    """GET a TfNSW endpoint. The key must go in a header, with a space after 'apikey'."""
    resp = requests.get(
        BASE_URL + path,
        headers={"Authorization": f"apikey {api_key}", "Accept": accept},
        timeout=timeout,
    )
    if resp.status_code != 200:
        sys.exit(f"HTTP {resp.status_code} from {path}: {resp.text[:300]}")
    return resp.content


def enum_name(enum_type, value: int) -> str:
    """Name for an enum value, tolerating values the standard bindings don't know
    (the Sydney Trains feed still uses REPLACEMENT, which was removed from the spec)."""
    try:
        return enum_type.Name(value)
    except ValueError:
        return f"UNKNOWN_{value}"


def flatten_trip_updates(feed: gtfs_realtime_pb2.FeedMessage) -> pd.DataFrame:
    """One row per trip x stop_time_update. Trips with no stop updates (e.g. cancellations)
    still get one row so they aren't lost."""
    snapshot_ts = feed.header.timestamp
    trip_sr = gtfs_realtime_pb2.TripDescriptor.ScheduleRelationship
    stop_sr = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship
    rows = []

    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue
        tu = entity.trip_update
        trip = tu.trip
        base = {
            "snapshot_ts": snapshot_ts,
            "entity_id": entity.id,
            "trip_id": trip.trip_id,
            "route_id": trip.route_id,
            "start_date": trip.start_date,
            "start_time": trip.start_time,
            "trip_schedule_relationship": enum_name(trip_sr, trip.schedule_relationship),
            "vehicle_id": tu.vehicle.id if tu.HasField("vehicle") else None,
        }

        if not tu.stop_time_update:
            rows.append(base)
            continue

        for stu in tu.stop_time_update:
            row = dict(base)
            row.update(
                {
                    "stop_sequence": stu.stop_sequence if stu.HasField("stop_sequence") else None,
                    "stop_id": stu.stop_id,
                    "stop_schedule_relationship": enum_name(stop_sr, stu.schedule_relationship),
                    "arrival_delay_s": stu.arrival.delay if stu.HasField("arrival") and stu.arrival.HasField("delay") else None,
                    "arrival_time": stu.arrival.time if stu.HasField("arrival") and stu.arrival.HasField("time") else None,
                    "departure_delay_s": stu.departure.delay if stu.HasField("departure") and stu.departure.HasField("delay") else None,
                    "departure_time": stu.departure.time if stu.HasField("departure") and stu.departure.HasField("time") else None,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def explore_realtime(feed_name: str, api_key: str) -> pd.DataFrame:
    print(f"\n=== Realtime trip updates: {feed_name} ===")
    raw = tfnsw_get(REALTIME_PATHS[feed_name], api_key, "application/x-google-protobuf", timeout=30)

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = DATA_DIR / "raw" / "realtime" / feed_name / f"{now}.pb"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw)

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(raw)

    header_time = datetime.fromtimestamp(feed.header.timestamp, tz=timezone.utc)
    print(f"Saved raw snapshot: {raw_path} ({len(raw) / 1024:.1f} KB)")
    print(f"Feed timestamp (UTC): {header_time}  |  GTFS-RT version: {feed.header.gtfs_realtime_version}")
    print(f"Entities: {len(feed.entity)}")

    # Rough storage estimate if polled every 60 seconds
    per_day_mb = len(raw) * 1440 / 1024 / 1024
    print(f"Storage estimate at 60 s polling: ~{per_day_mb:.0f} MB/day uncompressed")

    df = flatten_trip_updates(feed)
    if df.empty:
        print("No trip updates in this snapshot.")
        return df

    print(f"Flattened rows (trip x stop): {len(df)}  |  distinct trips: {df['trip_id'].nunique()}")
    print("\nTrip schedule relationships:")
    print(df.drop_duplicates("trip_id")["trip_schedule_relationship"].value_counts().to_string())
    if "stop_schedule_relationship" in df:
        print("\nStop schedule relationships:")
        print(df["stop_schedule_relationship"].value_counts(dropna=False).to_string())
    if "arrival_delay_s" in df:
        delays = df["arrival_delay_s"].dropna()
        if not delays.empty:
            print("\nArrival delay (seconds), across all predicted stops:")
            print(delays.describe().round(1).to_string())

    # Print one full trip update so you can see the raw structure
    for entity in feed.entity:
        if entity.HasField("trip_update"):
            print("\nExample entity (truncated):")
            print(str(entity)[:1500])
            break

    flat_path = DATA_DIR / "flat" / f"{feed_name}_trip_updates_{now}.csv"
    flat_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(flat_path, index=False)
    print(f"\nSaved flattened CSV: {flat_path}")
    return df


def explore_static(feed_name: str, api_key: str) -> Path:
    print(f"\n=== Static timetable: {feed_name} ===")
    # Static zips are large; TfNSW recommends a timeout over 60 seconds.
    raw = tfnsw_get(STATIC_PATHS[feed_name], api_key, "application/zip", timeout=180)

    out_dir = DATA_DIR / "static" / feed_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        zf.extractall(out_dir)
        names = zf.namelist()

    print(f"Downloaded {len(raw) / 1024 / 1024:.1f} MB, extracted to {out_dir}")
    for name in sorted(names):
        path = out_dir / name
        if name.endswith(".txt"):
            with open(path, encoding="utf-8-sig") as f:
                n_rows = sum(1 for _ in f) - 1
            print(f"  {name:<22} {n_rows:>10,} rows")
    return out_dir


def check_trip_match(rt: pd.DataFrame, static_dir: Path) -> None:
    """How many realtime trip_ids exist in today's static trips.txt?
    A low match rate means the timetable version is out of sync (trip keys change daily)."""
    trips_path = static_dir / "trips.txt"
    if rt.empty or not trips_path.exists():
        return
    static_trips = set(pd.read_csv(trips_path, usecols=["trip_id"], dtype=str)["trip_id"])
    rt_trips = set(rt["trip_id"].dropna().astype(str))
    matched = rt_trips & static_trips
    rate = len(matched) / len(rt_trips) if rt_trips else 0
    print(f"\n=== Join check ===\nRealtime trips found in static trips.txt: {len(matched)}/{len(rt_trips)} ({rate:.0%})")
    unmatched = sorted(rt_trips - static_trips)[:5]
    if unmatched:
        print(f"Examples not matched (worth investigating): {unmatched}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore TfNSW GTFS and GTFS-realtime feeds.")
    parser.add_argument("--feed", choices=list(REALTIME_PATHS), default="sydneytrains")
    parser.add_argument("--skip-static", action="store_true", help="Don't download the static timetable")
    args = parser.parse_args()

    api_key = get_api_key()
    rt = explore_realtime(args.feed, api_key)

    static_dir = DATA_DIR / "static" / args.feed
    if not args.skip_static:
        static_dir = explore_static(args.feed, api_key)
    check_trip_match(rt, static_dir)


if __name__ == "__main__":
    main()
