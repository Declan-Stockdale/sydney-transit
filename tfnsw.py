"""
Shared helpers for talking to the TfNSW Open Data Hub API.

Every collector script imports from here, so endpoint paths, authentication and
retry behaviour live in one place.

Key points:
    - The API key is read from the TFNSW_API_KEY environment variable, which
      python-dotenv loads from the .env file in the project folder. It is never
      printed or logged.
    - The key goes in a header, "Authorization: apikey <key>" (the space matters).
    - HTTPS certificates are checked against the Windows certificate store (via
      the truststore package) rather than Python's bundled list. Without this,
      some Windows machines fail with CERTIFICATE_VERIFY_FAILED.
"""

import logging
import os
import sys
import time
from pathlib import Path

import requests
import truststore
from dotenv import load_dotenv

truststore.inject_into_ssl()

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
LOG_DIR = DATA_DIR / "logs"

# Load .env by absolute path so scripts work no matter which folder they are
# started from (Task Scheduler does not start in the project folder by default).
load_dotenv(PROJECT_DIR / ".env")

BASE_URL = "https://api.transport.nsw.gov.au"

# Realtime endpoints, confirmed working on 2026-10-26 (trains) and 2026-10-27 (buses,
# ferries, light rail). Alerts only exist under v2 (the v1 alerts paths return 401).
# Buses, ferries and light rail only have v1 trip updates and positions, except the
# Inner West light rail, which is v2 only. Alerts for these modes are one feed per mode,
# not per line. Manly Fast Ferry (/v1/.../ferries/MFF) is left out: its feed stopped
# updating in June 2026.
_LIGHT_RAIL = ["cbdandsoutheast", "parramatta", "newcastle"]
REALTIME_PATHS = {
    "trip_updates": {
        "sydneytrains": "/v2/gtfs/realtime/sydneytrains",
        "metro": "/v2/gtfs/realtime/metro",
        # NSW TrainLink: the only realtime source for the Hunter line and regional
        # trains (and coaches, which analysis filters out). Only v1 exists.
        "nswtrains": "/v1/gtfs/realtime/nswtrains",
        "buses": "/v1/gtfs/realtime/buses",
        "sydneyferries": "/v1/gtfs/realtime/ferries/sydneyferries",
        **{f"lightrail_{line}": f"/v1/gtfs/realtime/lightrail/{line}" for line in _LIGHT_RAIL},
        "lightrail_innerwest": "/v2/gtfs/realtime/lightrail/innerwest",
    },
    "alerts": {
        "sydneytrains": "/v2/gtfs/alerts/sydneytrains",
        "metro": "/v2/gtfs/alerts/metro",
        "nswtrains": "/v2/gtfs/alerts/nswtrains",
        "buses": "/v2/gtfs/alerts/buses",
        "ferries": "/v2/gtfs/alerts/ferries",
        "lightrail": "/v2/gtfs/alerts/lightrail",
    },
    "vehicle_positions": {
        "sydneytrains": "/v2/gtfs/vehiclepos/sydneytrains",
        "metro": "/v2/gtfs/vehiclepos/metro",
        "nswtrains": "/v1/gtfs/vehiclepos/nswtrains",
        "buses": "/v1/gtfs/vehiclepos/buses",
        "sydneyferries": "/v1/gtfs/vehiclepos/ferries/sydneyferries",
        **{f"lightrail_{line}": f"/v1/gtfs/vehiclepos/lightrail/{line}" for line in _LIGHT_RAIL},
        "lightrail_innerwest": "/v2/gtfs/vehiclepos/lightrail/innerwest",
    },
}
# Every feed name, in polling order (not every feed exists for every kind)
REALTIME_FEEDS = list(dict.fromkeys(f for paths in REALTIME_PATHS.values() for f in paths))
# Trains are polled every collector cycle (--interval, default 30 s). The other modes every
# 60 s, to stay inside the API's daily request limit (9 train endpoints every 30 s plus
# 15 others every 60 s = 47,520 requests a day).
POLL_EVERY_S = {"sydneytrains": 0, "metro": 0, "nswtrains": 0}
DEFAULT_POLL_EVERY_S = 60

# Metro must use v2: the v1 Metro timetable stopped updating in Sept 2024 (its
# calendar ends 2024-12-26) and its trip_ids don't match the realtime feed.
# Sydney Trains has no v2 timetable (404).
STATIC_PATHS = {
    "sydneytrains": "/v1/gtfs/schedule/sydneytrains",
    "metro": "/v2/gtfs/schedule/metro",
    # All NSW bus operators (timetable only; buses are used for access analysis,
    # not reliability). Large, so it isn't part of the daily pull.
    "buses": "/v1/gtfs/schedule/buses",
    # NSW TrainLink (intercity and regional). Only v1 exists.
    "nswtrains": "/v1/gtfs/schedule/nswtrains",
    # Ferries and light rail: small (0.1 to 1.2 MB each), only v1 exists
    "sydneyferries": "/v1/gtfs/schedule/ferries/sydneyferries",
    **{f"lightrail_{line}": f"/v1/gtfs/schedule/lightrail/{line}" for line in _LIGHT_RAIL + ["innerwest"]},
}
# Every feed with realtime collection is pulled daily, because realtime trip IDs only join
# to the timetable of the same day. Buses are ~108 MB a day (kept only when changed).
DAILY_STATIC_FEEDS = [f for f in STATIC_PATHS]


def get_api_key() -> str:
    key = os.environ.get("TFNSW_API_KEY")
    if not key:
        sys.exit("TFNSW_API_KEY is not set. Add it to the .env file in the project folder.")
    return key


def tfnsw_get(
    session: requests.Session,
    path: str,
    api_key: str,
    accept: str = "application/x-google-protobuf",
    timeout: int = 30,
    retries: int = 3,
) -> bytes:
    """GET a TfNSW endpoint and return the raw response body.

    Retries network errors, rate limits (429) and server errors (5xx) with a
    growing wait (5 s, 10 s, 20 s). Other HTTP errors, such as 401 for a bad key,
    are not retried. Raises requests.RequestException if every attempt fails.
    """
    headers = {"Authorization": f"apikey {api_key}", "Accept": accept}
    for attempt in range(retries + 1):
        try:
            resp = session.get(BASE_URL + path, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp.content
            retryable = resp.status_code == 429 or resp.status_code >= 500
            error = requests.HTTPError(f"HTTP {resp.status_code} from {path}: {resp.text[:200]}")
            if not retryable:
                raise error
        except requests.HTTPError:
            raise
        except requests.RequestException as exc:
            error = exc
        if attempt < retries:
            time.sleep(5 * 2**attempt)
    raise error


def setup_logging(name: str) -> logging.Logger:
    """Log to data/logs/<name>.log (one file per day, 30 days kept) and to the console."""
    from logging.handlers import TimedRotatingFileHandler

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = TimedRotatingFileHandler(
        LOG_DIR / f"{name}.log", when="midnight", backupCount=30, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    # pythonw.exe (used by Task Scheduler, no console window) has no stdout
    if sys.stdout is not None:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)
    return logger
