"""
Collect weather data to compare against rail delays: BOM station observations
and Open-Meteo gridded weather at every rail station.

Two sources, saved untouched (gzipped JSON) under data/raw/weather/:

1. BOM observations  ->  bom_obs/<wmo id>/<YYYY-MM-DD>/<HHMMSS>Z.json.gz
   Half-hourly readings from 29 automatic weather stations around Sydney, the
   Blue Mountains, Central Coast and Illawarra: rain since 9am, temperature,
   wind speed and gusts. The BOM only keeps the last ~72 hours, so this must run
   at least every couple of days; Task Scheduler runs it every 6 hours.
   Note "rain_trace" is cumulative since 9am local time: difference consecutive
   readings to get rain per half hour.

2. Open-Meteo hourly weather  ->  open_meteo/<recent|archive>/<stamp>.json.gz
   Gridded hourly precipitation, rain, temperature, wind gusts and weather code,
   at the location of every rail station (Sydney Trains + Metro). Free, no key.
   Grid cells are roughly 10 to 15 km across around Sydney, so nearby stations
   share a cell. The point list is saved beside it as points.csv.
   - Scheduled runs use the forecast API with the last 3 days of observed-model
     data (overlapping, deduplicate downstream).
   - --backfill START END uses the historical archive (Open-Meteo's default
     model mix, ~10 km grid) to fill gaps or go back in time.

For high-resolution rain (about 0.5 km, every 5 minutes) see pull_radar.py.

Data sources: Bureau of Meteorology (attribute the Bureau; terms at
bom.gov.au/other/copyright.shtml) and Open-Meteo (open-meteo.com, CC BY 4.0).

Usage:
    python pull_weather.py                                   # BOM + Open-Meteo recent
    python pull_weather.py --only bom
    python pull_weather.py --backfill 2026-09-01 2026-10-25  # Open-Meteo archive
"""

import argparse
import gzip
import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from tfnsw import DATA_DIR, setup_logging  # tfnsw also sets up Windows HTTPS certificates

WEATHER_DIR = DATA_DIR / "raw" / "weather"
STATIC_DIR = DATA_DIR / "static"
USER_AGENT = "sydney-transit portfolio project (personal, non-commercial)"

# BOM automatic weather stations (WMO ids), confirmed 2026-10-26 in product IDN60901
BOM_STATIONS = {
    94743: "Mount Boyce", 94744: "Katoomba", 94749: "Bellambi", 94752: "Badgerys Creek",
    94755: "Camden", 94757: "Campbelltown", 94759: "Terrey Hills", 94760: "Horsley Park",
    94763: "Penrith", 94764: "Parramatta", 94765: "Bankstown", 94766: "Canterbury",
    94767: "Sydney Airport", 94768: "Observatory Hill", 94769: "Fort Denison",
    94774: "Newcastle Nobbys", 94776: "Williamtown", 94780: "Little Bay", 95752: "Wattamolla",
    95753: "Richmond", 95756: "Kurnell", 95757: "Lucas Heights", 95761: "Holsworthy",
    95765: "Sydney Olympic Park", 95766: "Sydney Harbour", 95767: "Cooranbong",
    95768: "North Head", 95770: "Norah Head", 95774: "Mangrove Mountain",
}
BOM_URL = "https://www.bom.gov.au/fwo/IDN60901/IDN60901.{wmo}.json"

OPEN_METEO_RECENT = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "precipitation,rain,temperature_2m,wind_gusts_10m,weather_code"
BATCH = 100  # locations per Open-Meteo request

log = setup_logging("pull_weather")


def utc_stamp() -> datetime:
    return datetime.now(timezone.utc)


def save_json_gz(path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(payload, compresslevel=9))
    tmp.replace(path)


def pull_bom(session: requests.Session) -> int:
    now = utc_stamp()
    failed = 0
    for wmo, name in BOM_STATIONS.items():
        try:
            resp = session.get(BOM_URL.format(wmo=wmo), timeout=30)
            resp.raise_for_status()
            json.loads(resp.content)  # make sure it is real JSON, not an error page
            path = WEATHER_DIR / "bom_obs" / str(wmo) / now.strftime("%Y-%m-%d") / f"{now.strftime('%H%M%S')}Z.json.gz"
            save_json_gz(path, resp.content)
        except (requests.RequestException, ValueError) as exc:
            failed += 1
            log.warning("BOM %s (%s) failed: %s", wmo, name, exc)
        time.sleep(0.5)  # be polite to the BOM website
    log.info("BOM: saved %d of %d stations", len(BOM_STATIONS) - failed, len(BOM_STATIONS))
    return failed


def station_points() -> pd.DataFrame:
    """One point per rail station (parent stops) in the Sydney Trains and Metro timetables."""
    frames = []
    for feed in ("sydneytrains", "metro"):
        stops = pd.read_csv(STATIC_DIR / feed / "stops.txt", dtype=str, encoding="utf-8-sig")
        stations = stops[stops["location_type"] == "1"][["stop_id", "stop_name", "stop_lat", "stop_lon"]]
        frames.append(stations.assign(feed=feed))
    pts = pd.concat(frames)
    pts[["stop_lat", "stop_lon"]] = pts[["stop_lat", "stop_lon"]].astype(float)
    # Greater Sydney and the intercity lines; drops regional NSW stations
    in_area = pts["stop_lat"].between(-34.9, -32.7) & pts["stop_lon"].between(149.9, 152.0)
    return pts[in_area].reset_index(drop=True)


def pull_open_meteo(session: requests.Session, start: str | None = None, end: str | None = None) -> int:
    """Recent (last 3 days) when start/end are None, else the archive for that date range."""
    pts = station_points()
    now = utc_stamp()
    kind = "recent" if start is None else "archive"
    out_dir = WEATHER_DIR / "open_meteo" / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    pts.to_csv(out_dir.parent / "points.csv", index=False)

    failed = 0
    for i in range(0, len(pts), BATCH):
        chunk = pts.iloc[i : i + BATCH]
        params = {
            "latitude": ",".join(f"{v:.5f}" for v in chunk["stop_lat"]),
            "longitude": ",".join(f"{v:.5f}" for v in chunk["stop_lon"]),
            "hourly": HOURLY_VARS,
            "timezone": "UTC",
        }
        if kind == "recent":
            url = OPEN_METEO_RECENT
            params.update(past_days=3, forecast_days=1)
            name = f"{now.strftime('%Y-%m-%dT%H%M%SZ')}_part{i // BATCH}.json.gz"
        else:
            url = OPEN_METEO_ARCHIVE
            # Default model mix. Don't use era5_land: on 2026-10-26 it returned
            # only nulls for precipitation around Sydney.
            params.update(start_date=start, end_date=end)
            name = f"{start}_{end}_part{i // BATCH}.json.gz"
        try:
            resp = session.get(url, params=params, timeout=120)
            resp.raise_for_status()
            body = resp.json()
            # Record which station ids are in this chunk, in request order
            wrapped = {"stop_ids": chunk["stop_id"].tolist(), "response": body}
            save_json_gz(out_dir / name, json.dumps(wrapped).encode())
        except (requests.RequestException, ValueError) as exc:
            failed += 1
            log.warning("Open-Meteo %s part %d failed: %s", kind, i // BATCH, exc)
        time.sleep(1)
    log.info("Open-Meteo %s: %d points in %d requests, %d failed", kind, len(pts), -(-len(pts) // BATCH), failed)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect weather observations and gridded weather.")
    parser.add_argument("--only", choices=["bom", "open_meteo"], default=None)
    parser.add_argument("--backfill", nargs=2, metavar=("START", "END"), help="Open-Meteo archive, dates as YYYY-MM-DD")
    args = parser.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    if args.backfill:
        failed = pull_open_meteo(session, *args.backfill)
    else:
        failed = 0
        if args.only in (None, "bom"):
            failed += pull_bom(session)
        if args.only in (None, "open_meteo"):
            failed += pull_open_meteo(session)
    if failed:
        raise SystemExit(f"{failed} request(s) failed")  # non-zero exit shows in Task Scheduler


if __name__ == "__main__":
    main()
