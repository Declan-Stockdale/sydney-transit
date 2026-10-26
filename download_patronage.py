"""
Download TfNSW patronage (demand) datasets from the Open Data portal.

These are periodic files, not an API. TfNSW refreshes the monthly ones around
mid-month, so rerunning this monthly picks up new data (files are re-downloaded
and overwritten). No API key needed.

Saved to data/raw/patronage/:
    station_entry_exit_monthly.csv
        Entries and exits per train, Metro and light rail station per month,
        from November 2024 (Opal batch data). The main demand measure per station.
    station_entry_exit_typical_day_2016_2023.xlsx
        "Typical day" entries/exits per station by time of day, 2016 to 2023.
        No longer updated, but the only source with peak vs off-peak by station.
    trips_train_monthly.csv / trips_metro_monthly.csv
        Estimated trips per month by line and card type.
    peak_train_loads_sep2019.xlsx
        Load estimates (passengers per train vs seats) on peak services, the
        last survey before COVID. Old, but shows where crowding was.

Source: Transport for NSW Open Data (opendata.transport.nsw.gov.au), CC BY 4.0.

Usage:
    python download_patronage.py
"""

import requests

from tfnsw import DATA_DIR, setup_logging  # tfnsw also sets up Windows HTTPS certificates

OUT_DIR = DATA_DIR / "raw" / "patronage"
BASE = "https://opendata.transport.nsw.gov.au/data/dataset"
FILES = {
    "station_entry_exit_monthly.csv":
        f"{BASE}/3977df59-a1fa-422e-91ff-cfaeac355cc9/resource/f8bb2918-0540-4bb3-9ccf-f7aef04d4249/download/entry_exit.csv",
    "station_entry_exit_typical_day_2016_2023.xlsx":
        f"{BASE}/3977df59-a1fa-422e-91ff-cfaeac355cc9/resource/491763fc-82ed-4e44-9fca-8f3285db526e/download/train-station-entries-and-exits-version-2.2.xlsx",
    "trips_train_monthly.csv":
        f"{BASE}/a1ebe80d-442e-43cc-9ea9-e075bba96eb5/resource/7d49ca0b-1fb6-4524-81b0-68c8775b4d82/download/train.csv",
    "trips_metro_monthly.csv":
        f"{BASE}/a1ebe80d-442e-43cc-9ea9-e075bba96eb5/resource/0ae59bfc-e14d-4d5b-bb27-156a67b9759a/download/metro.csv",
    "peak_train_loads_sep2019.xlsx":
        f"{BASE}/69b0f082-84cb-41a0-8b53-bfd81b763d01/resource/3b18f6b2-89aa-41ce-b0a3-a4349447871d/download/peak-trains-load-report-02-06-sep-2019-final.xlsx",
}

log = setup_logging("download_patronage")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "sydney-transit portfolio project"
    for name, url in FILES.items():
        dest = OUT_DIR / name
        resp = session.get(url, timeout=300)
        resp.raise_for_status()
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.replace(dest)
        log.info("Saved %s (%.1f MB)", name, len(resp.content) / 1e6)


if __name__ == "__main__":
    main()
