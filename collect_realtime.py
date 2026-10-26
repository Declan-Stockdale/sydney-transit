"""
Collect TfNSW realtime feeds (trip updates, alerts, vehicle positions) for
Sydney Trains and Metro, and save each new snapshot untouched.

This replaces collect_snapshots.py. It is meant to run all the time: Windows Task
Scheduler starts it at logon (see scheduling/register_tasks.ps1). In week 2 the
same script moves to a cloud VM.

What it does, every --interval seconds:
    1. Requests each of the 6 endpoints in turn (3 feed types x 2 networks).
    2. Reads the feed's header timestamp. If it matches the last snapshot saved
       for that endpoint, the feed hasn't changed, so the snapshot is skipped.
    3. Otherwise saves the raw protobuf, gzip-compressed (about 5x smaller), to
           data/raw/realtime/<kind>/<feed>/<YYYY-MM-DD>/<HHMMSS>Z.pb.gz
       Date and time come from the feed's header timestamp, in UTC.
    4. Once an hour, writes how often each feed actually changed to
       data/logs/feed_intervals.csv. Use this to choose the polling interval
       (record the result in DECISIONS.md).

Each night, compact_raw.py bundles finished days into one .tar.xz per feed,
which is a further ~5 to 8x smaller.

Errors never stop the collector: a failed request is retried a few times, then
logged and skipped until the next cycle. Logs go to data/logs/collect_realtime.log.

Usage:
    python collect_realtime.py                     # run until stopped (Ctrl+C)
    python collect_realtime.py --minutes 5         # short test run
    python collect_realtime.py --interval 15       # poll more often
    python collect_realtime.py --kinds trip_updates alerts
"""

import argparse
import csv
import gzip
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone

import requests
from google.protobuf.message import DecodeError
from google.transit import gtfs_realtime_pb2

from tfnsw import DATA_DIR, LOG_DIR, REALTIME_PATHS, get_api_key, setup_logging, tfnsw_get

RAW_DIR = DATA_DIR / "raw" / "realtime"
INTERVALS_CSV = LOG_DIR / "feed_intervals.csv"
SUMMARY_EVERY_S = 3600

log = setup_logging("collect_realtime")


def snapshot_path(kind: str, feed: str, header_ts: int):
    t = datetime.fromtimestamp(header_ts, tz=timezone.utc)
    return RAW_DIR / kind / feed / t.strftime("%Y-%m-%d") / f"{t.strftime('%H%M%S')}Z.pb.gz"


def poll_once(session, api_key, kind, feed, last_ts, stats) -> None:
    """Fetch one endpoint and save it if it has changed since the last save."""
    key = (kind, feed)
    try:
        raw = tfnsw_get(session, REALTIME_PATHS[kind][feed], api_key)
    except requests.RequestException as exc:
        stats[key]["errors"] += 1
        log.warning("%s/%s request failed: %s", kind, feed, exc)
        return

    msg = gtfs_realtime_pb2.FeedMessage()
    try:
        msg.ParseFromString(raw)
    except DecodeError:
        stats[key]["errors"] += 1
        log.warning("%s/%s returned %d bytes that are not valid protobuf", kind, feed, len(raw))
        return

    # Fall back to the time we fetched it if the feed has no header timestamp
    header_ts = msg.header.timestamp or int(time.time())
    if header_ts == last_ts.get(key):
        stats[key]["skipped"] += 1
        return

    path = snapshot_path(kind, feed, header_ts)
    if not path.exists():  # also guards against a duplicate after a restart
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(raw, compresslevel=9))
        tmp.replace(path)  # rename at the end, so a crash never leaves half a file
        stats[key]["bytes"] += len(raw)

    if key in last_ts:
        stats[key]["intervals"].append(header_ts - last_ts[key])
    last_ts[key] = header_ts
    stats[key]["saved"] += 1


def write_summary(stats, polls: int) -> None:
    """Log and append to feed_intervals.csv how often each feed changed this period."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new_file = not INTERVALS_CSV.exists()
    with open(INTERVALS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                ["period_end_utc", "kind", "feed", "polls", "saved", "skipped", "errors",
                 "median_interval_s", "min_interval_s", "max_interval_s", "raw_mb"]
            )
        for (kind, feed), s in sorted(stats.items()):
            iv = s["intervals"]
            median = statistics.median(iv) if iv else None
            writer.writerow(
                [now, kind, feed, polls, s["saved"], s["skipped"], s["errors"], median,
                 min(iv) if iv else None, max(iv) if iv else None, round(s["bytes"] / 1e6, 1)]
            )
            log.info(
                "%-17s %-12s saved=%d skipped=%d errors=%d median update interval=%ss",
                kind, feed, s["saved"], s["skipped"], s["errors"], median,
            )


def new_stats():
    return defaultdict(lambda: {"saved": 0, "skipped": 0, "errors": 0, "bytes": 0, "intervals": []})


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect TfNSW realtime snapshots.")
    parser.add_argument("--interval", type=float, default=30, help="Seconds between polls (default 30)")
    parser.add_argument("--minutes", type=float, default=None, help="Stop after this many minutes (default: run forever)")
    parser.add_argument("--kinds", nargs="+", choices=list(REALTIME_PATHS), default=list(REALTIME_PATHS))
    parser.add_argument("--feeds", nargs="+", choices=["sydneytrains", "metro"], default=["sydneytrains", "metro"])
    args = parser.parse_args()

    api_key = get_api_key()
    session = requests.Session()
    last_ts: dict = {}
    stats = new_stats()
    polls = 0
    started = time.monotonic()
    next_summary = started + SUMMARY_EVERY_S
    log.info("Started: kinds=%s feeds=%s interval=%ss", args.kinds, args.feeds, args.interval)

    try:
        while args.minutes is None or time.monotonic() - started < args.minutes * 60:
            cycle_start = time.monotonic()
            for kind in args.kinds:
                for feed in args.feeds:
                    poll_once(session, api_key, kind, feed, last_ts, stats)
            polls += 1

            if time.monotonic() >= next_summary:
                write_summary(stats, polls)
                stats, polls = new_stats(), 0
                next_summary += SUMMARY_EVERY_S

            time.sleep(max(0, args.interval - (time.monotonic() - cycle_start)))
    except KeyboardInterrupt:
        log.info("Stopped by user")
    finally:
        if polls:
            write_summary(stats, polls)


if __name__ == "__main__":
    main()
