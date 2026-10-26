"""
Bundle each finished day of realtime snapshots into one compressed archive.

The collector saves one gzip file per snapshot (about 5x smaller than raw).
Consecutive snapshots are nearly identical, and gzip can't see across files, so
bundling a whole day into one .tar.xz shrinks it a further ~5 to 8x (measured
on 2026-10-26: about 40x smaller than raw overall for trip updates).

For every day folder older than today (UTC), this:
    1. Writes data/raw/realtime/<kind>/<feed>/<YYYY-MM-DD>.tar.xz holding the
       original, uncompressed .pb files, named <HHMMSS>Z.pb.
    2. Re-opens the archive and checks every snapshot is in it, byte for byte.
    3. Only then deletes the day folder.

Nothing is lost: the .pb bytes inside the archive are exactly what the API sent.
To read one day back in Python:

    import tarfile
    with tarfile.open("data/raw/realtime/trip_updates/sydneytrains/2026-10-26.tar.xz") as tf:
        for member in tf:
            raw = tf.extractfile(member).read()   # parse with gtfs_realtime_pb2.FeedMessage

Run daily by Task Scheduler, straight after pull_static.py.

Usage:
    python compact_raw.py              # compact all finished days
    python compact_raw.py --dry-run    # list what would be compacted
"""

import argparse
import gzip
import io
import lzma
import shutil
import tarfile
from datetime import datetime, timezone

from tfnsw import DATA_DIR, setup_logging

RAW_DIR = DATA_DIR / "raw" / "realtime"
XZ_PRESET = 6  # good ratio with modest memory; preset 9 needs ~700 MB of RAM

log = setup_logging("compact_raw")


def compact_day(day_dir) -> None:
    snapshots = sorted(day_dir.glob("*.pb.gz"))
    archive = day_dir.with_suffix(".tar.xz")
    tmp = day_dir.with_suffix(".tar.xz.tmp")

    raw_bytes = 0
    with tarfile.open(tmp, mode="w:xz", preset=XZ_PRESET) as tf:
        for snap in snapshots:
            data = gzip.decompress(snap.read_bytes())
            raw_bytes += len(data)
            info = tarfile.TarInfo(snap.name.removesuffix(".gz"))
            info.size = len(data)
            info.mtime = int(snap.stat().st_mtime)
            tf.addfile(info, io.BytesIO(data))

    # Verify before deleting anything
    with tarfile.open(tmp, mode="r:xz") as tf:
        for snap in snapshots:
            member = tf.extractfile(snap.name.removesuffix(".gz"))
            if member is None or member.read() != gzip.decompress(snap.read_bytes()):
                raise RuntimeError(f"Verification failed for {snap}")

    tmp.replace(archive)
    shutil.rmtree(day_dir)
    size = archive.stat().st_size
    log.info(
        "%s: %d snapshots, %.0f MB raw -> %.1f MB (%.0fx smaller)",
        archive.relative_to(DATA_DIR), len(snapshots), raw_bytes / 1e6, size / 1e6,
        raw_bytes / size if size else 0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact finished days of realtime snapshots.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    day_dirs = sorted(
        d for d in RAW_DIR.glob("*/*/*")
        if d.is_dir() and d.name < today and len(d.name) == 10
    )
    if not day_dirs:
        log.info("Nothing to compact")
    failed = 0
    for day_dir in day_dirs:
        if args.dry_run:
            log.info("Would compact %s", day_dir.relative_to(DATA_DIR))
            continue
        try:
            compact_day(day_dir)
        except (OSError, RuntimeError, lzma.LZMAError, tarfile.TarError):
            log.exception("Failed to compact %s (left as is)", day_dir)
            day_dir.with_suffix(".tar.xz.tmp").unlink(missing_ok=True)
            failed += 1
    if failed:
        raise SystemExit(f"{failed} day(s) failed to compact")


if __name__ == "__main__":
    main()
