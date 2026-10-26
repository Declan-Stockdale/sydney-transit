"""
Collect TfNSW realtime feeds (trip updates, alerts, vehicle positions) for
Sydney Trains, Metro, NSW TrainLink, buses, Sydney Ferries and light rail, and
save each new snapshot untouched.

This replaces collect_snapshots.py. It is meant to run all the time: Windows Task
Scheduler starts it at logon (see scheduling/register_tasks.ps1). In week 2 the
same script moves to a cloud VM.

What it does, every --interval seconds (trains) or every 60 s (buses, ferries,
light rail; see tfnsw.POLL_EVERY_S):
    1. Requests each of the 24 endpoints in turn (see tfnsw.REALTIME_PATHS),
       with a short pause between calls to respect the API's rate limit.
    2. Reads the feed's header timestamp. If it matches the last snapshot saved
       for that endpoint, the feed hasn't changed, so the snapshot is skipped.
    3. Otherwise saves the raw protobuf, gzip-compressed (about 5x smaller), to
           data/raw/realtime/<kind>/<feed>/<YYYY-MM-DD>/<HHMMSS>Z.pb.gz
       Date and time come from the feed's header timestamp, in UTC.
    4. Once an hour, writes how often each feed actually changed to
       data/logs/feed_intervals.csv. Use this to choose the polling interval
       (record the result in DECISIONS.md).

It also launches the other frequent collectors as separate processes on their
own clocks (see SIDE_JOBS): rain radar every 30 min, Park&Ride car parks every
15 min, weather every 6 h. Use --no-side-jobs to switch that off.

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
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import requests
from google.protobuf.message import DecodeError
from google.transit import gtfs_realtime_pb2

from tfnsw import (DATA_DIR, DEFAULT_POLL_EVERY_S, LOG_DIR, POLL_EVERY_S, PROJECT_DIR, REALTIME_FEEDS, REALTIME_PATHS,
                   get_api_key, setup_logging, tfnsw_get)

RAW_DIR = DATA_DIR / "raw" / "realtime"
# Pause between API calls: the API allows about 5 calls a second, and back-to-back
# calls to small feeds tripped HTTP 429 (rate limited) when tested on 27 Oct 2026
CALL_GAP_S = 0.3
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


# Other collectors launched from this always-running loop, as separate processes so a
# failure in one never stops realtime collection. Windows Task Scheduler's repeating
# triggers proved unreliable for these (on 26 Oct 2026 radar and car parks stopped after
# 5:30pm with the PC awake), and radar frames are only kept for 2 hours upstream.
SIDE_JOBS = {"pull_radar.py": 30 * 60, "pull_carpark.py": 15 * 60, "pull_weather.py": 6 * 3600}


def run_side_jobs(last_run: dict, running: dict) -> None:
    now = time.monotonic()
    for script, every in SIDE_JOBS.items():
        proc = running.get(script)
        if proc is not None and proc.poll() is None:
            continue  # still running from last time
        if proc is not None and proc.returncode not in (0, None):
            log.warning("%s exited with code %s", script, proc.returncode)
        running.pop(script, None)
        if now - last_run.get(script, -1e9) >= every:
            running[script] = subprocess.Popen([sys.executable, str(PROJECT_DIR / script)], cwd=PROJECT_DIR)
            last_run[script] = now
            log.info("Started %s", script)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect TfNSW realtime snapshots.")
    parser.add_argument("--no-side-jobs", action="store_true", help="Don't also run radar, car park and weather collection")
    parser.add_argument("--interval", type=float, default=30, help="Seconds between polls (default 30)")
    parser.add_argument("--minutes", type=float, default=None, help="Stop after this many minutes (default: run forever)")
    parser.add_argument("--kinds", nargs="+", choices=list(REALTIME_PATHS), default=list(REALTIME_PATHS))
    parser.add_argument("--feeds", nargs="+", choices=REALTIME_FEEDS, default=REALTIME_FEEDS)
    args = parser.parse_args()

    api_key = get_api_key()
    session = requests.Session()
    last_ts: dict = {}
    stats = new_stats()
    polls = 0
    started = time.monotonic()
    next_summary = started + SUMMARY_EVERY_S
    log.info("Started: kinds=%s feeds=%s interval=%ss", args.kinds, args.feeds, args.interval)
    side_last, side_running = {}, {}
    last_poll: dict = {}

    try:
        while args.minutes is None or time.monotonic() - started < args.minutes * 60:
            cycle_start = time.monotonic()
            if not args.no_side_jobs:
                run_side_jobs(side_last, side_running)
            for kind in args.kinds:
                for feed in args.feeds:
                    if feed not in REALTIME_PATHS[kind]:
                        continue
                    # Trains every cycle; other modes only when their own interval is due
                    every = POLL_EVERY_S.get(feed, DEFAULT_POLL_EVERY_S)
                    if cycle_start - last_poll.get((kind, feed), -1e9) < every - 1:
                        continue
                    last_poll[(kind, feed)] = cycle_start
                    poll_once(session, api_key, kind, feed, last_ts, stats)
                    time.sleep(CALL_GAP_S)
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
