"""
Scheduled rail service per station for one service date, from the static GTFS.

Counts how many trains (and Metro services) depart each station on a chosen
date, overall and in each time band. This is the "supply" side of the service
equity analysis; service_equity.py joins it to where people live.

Rules:
    - A departure is a scheduled stop where passengers can board
      (pickup_type is not 1). Empty and terminating moves are excluded.
    - Platforms are rolled up to their parent station, so both directions and
      all platforms count together.
    - Stations shared by Sydney Trains and Metro (e.g. Chatswood, Central) are
      merged by name, so an interchange counts its combined service.
    - Included operators: Sydney Trains, Sydney Metro and all NSW TrainLink rail
      services: intercity lines (Blue Mountains, Central Coast & Newcastle,
      Hunter, South Coast, Southern Highlands) and regional trains (XPT,
      Xplorer). The Sydney Trains timetable feed already carries every TrainLink
      rail trip, so the separate NSW TrainLink timetable feed is not added on
      top: it repeats a subset of the same trains under different trip IDs (it
      would double count them), and adds only road coaches, which are out of scope.
    - GTFS times can pass 24:00 (after-midnight trips belong to the previous
      service date); they are kept as seconds since the start of the service day.

Time bands use Opal's weekday peak definitions:
    am_peak 06:30-10:00, interpeak 10:00-15:00, pm_peak 15:00-19:00,
    evening 19:00 to end of service, early before 06:30.

Usage:
    python -m analysis.station_service                 # next Tuesday
    python -m analysis.station_service --date 2026-10-27
"""

import argparse
from datetime import date, timedelta

import pandas as pd

from tfnsw import DATA_DIR

STATIC_DIR = DATA_DIR / "static"
MARTS_DIR = DATA_DIR / "marts"

INTERCITY_LINES = {"BMT", "CCN", "SCO", "SHL", "HUN"}
REGIONAL_LINES = {"NRC", "NRW", "STH", "WST"}  # NSW TrainLink XPT and Xplorer
RAIL_AGENCIES = {"SydneyTrains", "NSWTrains"}
BANDS = [  # name, start hour, end hour (end exclusive)
    ("early", 0, 6.5),
    ("am_peak", 6.5, 10),
    ("interpeak", 10, 15),
    ("pm_peak", 15, 19),
    ("evening", 19, 30),
]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def read(feed: str, name: str, **kwargs) -> pd.DataFrame:
    path = STATIC_DIR / feed / f"{name}.txt"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", **kwargs)


def active_service_ids(feed: str, service_date: date) -> set[str]:
    """Services running on the date, from calendar.txt plus calendar_dates.txt exceptions."""
    ymd = service_date.strftime("%Y%m%d")
    cal = read(feed, "calendar")
    running = cal[
        (cal["start_date"] <= ymd)
        & (cal["end_date"] >= ymd)
        & (cal[WEEKDAYS[service_date.weekday()]] == "1")
    ]
    ids = set(running["service_id"])
    exceptions = read(feed, "calendar_dates")
    if not exceptions.empty:
        today = exceptions[exceptions["date"] == ymd]
        ids |= set(today.loc[today["exception_type"] == "1", "service_id"])
        ids -= set(today.loc[today["exception_type"] == "2", "service_id"])
    return ids


def gtfs_seconds(times: pd.Series) -> pd.Series:
    """'25:10:00' -> 90600. Hours can exceed 24 for after-midnight trips."""
    parts = times.str.split(":", expand=True).astype(int)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def rail_routes(routes: pd.DataFrame, feed: str) -> pd.DataFrame:
    """Rail routes to analyse from a feed: in the Sydney Trains feed, Sydney Trains and NSW TrainLink."""
    if feed == "sydneytrains" and "agency_id" in routes:
        return routes[routes["agency_id"].isin(RAIL_AGENCIES)]
    return routes


def feed_departures(feed: str, service_date: date) -> pd.DataFrame:
    """One row per boardable scheduled departure on the date, with its station and line."""
    routes = rail_routes(read(feed, "routes", usecols=["route_id", "agency_id", "route_short_name"]), feed)

    trips = read(feed, "trips", usecols=["route_id", "service_id", "trip_id"])
    trips = trips[trips["service_id"].isin(active_service_ids(feed, service_date))]
    trips = trips.merge(routes, on="route_id")

    st = read(feed, "stop_times", usecols=["trip_id", "departure_time", "stop_id", "pickup_type"])
    st = st[st["trip_id"].isin(set(trips["trip_id"])) & (st["pickup_type"].fillna("0") != "1")]
    st = st.merge(trips[["trip_id", "route_short_name"]], on="trip_id")

    stops = read(feed, "stops", usecols=["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"])
    stops["parent_station"] = stops["parent_station"].fillna("")
    platform_to_station = dict(
        zip(stops["stop_id"], stops["parent_station"].where(stops["parent_station"] != "", stops["stop_id"]))
    )
    st["station_id"] = st["stop_id"].map(platform_to_station)
    stations = stops[stops["stop_id"].isin(set(st["station_id"]))]
    st = st.merge(
        stations[["stop_id", "stop_name", "stop_lat", "stop_lon"]].rename(columns={"stop_id": "station_id"}),
        on="station_id",
    )
    st["dep_s"] = gtfs_seconds(st["departure_time"])
    st["feed"] = feed
    return st[["feed", "trip_id", "route_short_name", "station_id", "stop_name", "stop_lat", "stop_lon", "dep_s"]]


def station_service(service_date: date) -> pd.DataFrame:
    """Departures per station (merged across feeds by name), total and per time band."""
    deps = pd.concat([feed_departures(f, service_date) for f in ("sydneytrains", "metro")])
    deps["station"] = deps["stop_name"].str.replace(r"\s+Station$", "", regex=True).str.strip()
    hours = deps["dep_s"] / 3600
    for name, start, end in BANDS:
        deps[name] = ((hours >= start) & (hours < end)).astype(int)

    band_names = [b[0] for b in BANDS]
    out = deps.groupby("station").agg(
        lat=("stop_lat", lambda s: s.astype(float).mean()),
        lon=("stop_lon", lambda s: s.astype(float).mean()),
        lines=("route_short_name", lambda s: " ".join(sorted(set(s)))),
        networks=("feed", lambda s: " ".join(sorted(set(s)))),
        daily_departures=("trip_id", "size"),
        first_departure_s=("dep_s", "min"),
        last_departure_s=("dep_s", "max"),
        **{b: (b, "sum") for b in band_names},
    ).reset_index()

    for name, start, end in BANDS:
        if name in ("am_peak", "interpeak", "pm_peak"):
            out[f"{name}_tph"] = (out[name] / (end - start)).round(1)  # trains per hour
    out["service_date"] = service_date.isoformat()
    return out.sort_values("daily_departures", ascending=False)


def next_tuesday(today: date | None = None) -> date:
    today = today or date.today()
    return today + timedelta(days=(1 - today.weekday()) % 7 or 7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduled departures per station for one date.")
    parser.add_argument("--date", type=date.fromisoformat, default=None, help="Service date (default: next Tuesday)")
    args = parser.parse_args()
    service_date = args.date or next_tuesday()

    out = station_service(service_date)
    MARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = MARTS_DIR / "station_service.csv"
    out.to_csv(path, index=False)
    print(f"{len(out)} stations, {out['daily_departures'].sum():,} departures on {service_date} -> {path}")
    print(out.head(10)[["station", "lines", "daily_departures", "am_peak_tph"]].to_string(index=False))


if __name__ == "__main__":
    main()
