"""
Download the static GTFS timetable for Sydney Trains and Metro, and keep one
dated copy of each version.

Why keep every version: Sydney Trains trip_ids change daily (the timetable is
refreshed around 2am Sydney time), so each realtime record has to be joined to
the timetable that was current on its service date. A single "latest" copy
would make older realtime data impossible to join.

Run daily by Task Scheduler at 3:30am Sydney time, after TfNSW's refresh
(see scheduling/register_tasks.ps1).

For each feed it:
    1. Downloads the zip (tens of MB, so the timeout is long).
    2. Hashes it (SHA-256). If the hash matches the last saved version, nothing
       changed and nothing new is stored.
    3. Otherwise saves the zip untouched (it is already compressed) as
           data/raw/static/<feed>/<YYYY-MM-DD>.zip   (date = Sydney date)
       and adds a row to data/raw/static/<feed>/manifest.csv.
    4. Extracts the newest version to data/static/<feed>/ for quick exploration
       in notebooks and explore_feeds.py.

Usage:
    python pull_static.py                  # both feeds
    python pull_static.py --feeds metro
"""

import argparse
import csv
import hashlib
import shutil
import zipfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

from tfnsw import DATA_DIR, STATIC_PATHS, get_api_key, setup_logging, tfnsw_get

RAW_STATIC_DIR = DATA_DIR / "raw" / "static"
EXTRACTED_DIR = DATA_DIR / "static"
SYDNEY = ZoneInfo("Australia/Sydney")

log = setup_logging("pull_static")


def last_saved_hash(manifest) -> str | None:
    if not manifest.exists():
        return None
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1]["sha256"] if rows else None


def pull_feed(session, api_key: str, feed: str) -> None:
    raw = tfnsw_get(session, STATIC_PATHS[feed], api_key, accept="application/zip", timeout=300)
    digest = hashlib.sha256(raw).hexdigest()
    feed_dir = RAW_STATIC_DIR / feed
    manifest = feed_dir / "manifest.csv"

    if digest == last_saved_hash(manifest):
        log.info("%s: unchanged since last download (%.1f MB), nothing saved", feed, len(raw) / 1e6)
        return

    now_utc = datetime.now(timezone.utc)
    sydney_date = now_utc.astimezone(SYDNEY).strftime("%Y-%m-%d")
    zip_path = feed_dir / f"{sydney_date}.zip"
    if zip_path.exists():  # a second, different version on the same day
        zip_path = feed_dir / f"{sydney_date}_{now_utc.strftime('%H%M%S')}Z.zip"

    feed_dir.mkdir(parents=True, exist_ok=True)
    zip_path.write_bytes(raw)

    new_manifest = not manifest.exists()
    with open(manifest, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_manifest:
            writer.writerow(["file", "downloaded_utc", "sha256", "bytes"])
        writer.writerow([zip_path.name, now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), digest, len(raw)])
    log.info("%s: saved new version %s (%.1f MB)", feed, zip_path.name, len(raw) / 1e6)

    # Refresh the extracted copy. Extract to a temp folder first so a failure
    # never leaves a half-written timetable behind.
    out_dir = EXTRACTED_DIR / feed
    tmp_dir = EXTRACTED_DIR / f"{feed}.tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    log.info("%s: extracted to %s", feed, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and version the static GTFS timetables.")
    parser.add_argument("--feeds", nargs="+", choices=list(STATIC_PATHS), default=list(STATIC_PATHS))
    args = parser.parse_args()

    api_key = get_api_key()
    session = requests.Session()
    failed = []
    for feed in args.feeds:
        try:
            pull_feed(session, api_key, feed)
        except Exception:
            log.exception("%s: static download failed", feed)
            failed.append(feed)
    if failed:
        raise SystemExit(f"Failed: {', '.join(failed)}")  # non-zero exit shows in Task Scheduler


if __name__ == "__main__":
    main()
