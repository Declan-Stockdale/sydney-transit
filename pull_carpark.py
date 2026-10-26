"""
Collect live occupancy of TfNSW Park&Ride car parks (about 40 commuter car parks
at stations such as Tallawong, Kellyville, Ashfield and Kogarah).

Why: the catchment analysis shows hub stations drawing riders from far beyond
walking distance. Car park fill times (e.g. full by 7:15am) show how much of
that is park-and-ride and when parking runs out.

The API only gives the current state, so Task Scheduler runs this every 15
minutes. Each run saves one gzipped JSON file with every facility's response:
    data/raw/carpark/<YYYY-MM-DD>/<HHMMSS>Z.json.gz   (UTC)

Facilities the API marks "historical only" are skipped.

Usage:
    python pull_carpark.py
"""

import gzip
import json
import time
from datetime import datetime, timezone

import requests

from tfnsw import DATA_DIR, get_api_key, setup_logging, tfnsw_get

OUT_DIR = DATA_DIR / "raw" / "carpark"
LIST_PATH = "/v1/carpark"

log = setup_logging("pull_carpark")


def main() -> None:
    api_key = get_api_key()
    session = requests.Session()
    facilities = json.loads(tfnsw_get(session, LIST_PATH, api_key, accept="application/json"))
    active = {fid: name for fid, name in facilities.items() if "historical" not in name.lower()}

    now = datetime.now(timezone.utc)
    results, failed = [], 0
    for fid in sorted(active, key=int):
        try:
            raw = tfnsw_get(session, f"{LIST_PATH}?facility={fid}", api_key, accept="application/json", retries=1)
            results.append(json.loads(raw))
        except (requests.RequestException, ValueError) as exc:
            failed += 1
            log.warning("Facility %s (%s) failed: %s", fid, active[fid], exc)
        time.sleep(0.25)  # stay well under the API's 5 requests per second

    path = OUT_DIR / now.strftime("%Y-%m-%d") / f"{now.strftime('%H%M%S')}Z.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fetched_utc": now.isoformat(), "facilities": results}
    path.write_bytes(gzip.compress(json.dumps(payload).encode(), compresslevel=9))
    log.info("Car parks: saved %d of %d facilities", len(results), len(active))
    if failed:
        raise SystemExit(f"{failed} facilities failed")


if __name__ == "__main__":
    main()
