"""
Commuter car parks: what time each one fills on a weekday, and how much of the
morning's train service is left for people who can't park.

From the Park&Ride snapshots (pull_carpark.py, every 15 minutes):
    fill time   the first time a car park reached FULL_SHARE of its spaces that morning,
                interpolated between the two readings either side (it can't be more exact
                than the 15 minute readings it comes from). When collection was paused (the PC
                asleep overnight), a car park already full at the first reading is only known
                to be full BY then: fill_seen = False, and pct_am_after_full is then a minimum
    peak trains after full  share of the station's scheduled departures between 6 and 10am
                (a typical weekday in the latest timetable, all lines, both directions) that leave
                after the car park is full: the service a driver arriving later can't use

Car parks are matched to their station by the stop number in the API ("tsn"), the
station's ID in the timetable. Car parks next to bus stops only (the B-Line car
parks) have no train service to compare.

Output: app/data/carpark_fill.csv
    carpark, station, date, spaces, max_pct, fill_time (H:MM Sydney, "by H:MM" if not seen, blank if it
    didn't fill), fill_hour (decimal), fill_seen, am_departures, pct_am_after_full
and app/data/carpark_curves.csv: carpark, date, time_local (decimal hour), pct_full

Usage:
    python -m analysis.carpark_fill
"""

import gzip
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from analysis.reliability import SYDNEY
from analysis.station_service import active_service_ids, feed_departures
from analysis.stop_events import station_coords
from tfnsw import DATA_DIR, PROJECT_DIR

RAW = DATA_DIR / "raw" / "carpark"
OUT = PROJECT_DIR / "app" / "data"
FULL_SHARE = 0.95
AM_START, AM_END = 6, 10
MAX_READING_GAP_H = 0.5  # readings come every 15 min; a longer gap (PC asleep) hides when it filled


def readings() -> pd.DataFrame:
    rows = []
    for f in sorted(RAW.glob("*/*.json.gz")):
        try:
            payload = json.loads(gzip.decompress(f.read_bytes()))
        except (OSError, EOFError, ValueError):
            continue
        t = pd.Timestamp(payload["fetched_utc"])
        for fac in payload["facilities"]:
            occ = pd.to_numeric((fac.get("occupancy") or {}).get("total"), errors="coerce")
            spots = pd.to_numeric(fac.get("spots"), errors="coerce")
            rows.append((t, fac["facility_name"].replace("Park&Ride - ", ""), str(fac.get("tsn")), spots, occ))
    df = pd.DataFrame(rows, columns=["t", "carpark", "tsn", "spaces", "occupied"]).dropna()
    df = df[df["spaces"] > 0]
    local = df["t"].dt.tz_convert(SYDNEY)
    df["date"] = local.dt.strftime("%Y-%m-%d")
    df["hour"] = local.dt.hour + local.dt.minute / 60
    df["weekday"] = local.dt.day_of_week < 5
    df["pct_full"] = (100 * df["occupied"] / df["spaces"]).clip(upper=100)
    return df.sort_values(["carpark", "t"]).reset_index(drop=True)


def fill_time(hours: np.ndarray, pct: np.ndarray, threshold: float = 100 * FULL_SHARE) -> tuple[float, bool]:
    """(decimal hour the series first reaches threshold, whether the filling was seen).

    Interpolated between the readings either side when they are at most MAX_READING_GAP_H apart. If it was
    already full at the first reading, or after a gap, it is only known to be full BY that reading: returns
    (that hour, False). (NaN, True) if it never filled."""
    hit = np.flatnonzero(pct >= threshold)
    if not len(hit):
        return np.nan, True
    i = hit[0]
    if i == 0 or hours[i] - hours[i - 1] > MAX_READING_GAP_H:
        return float(hours[i]), False
    if pct[i] == pct[i - 1]:
        return float(hours[i]), True
    return float(hours[i - 1] + (threshold - pct[i - 1]) / (pct[i] - pct[i - 1]) * (hours[i] - hours[i - 1])), True


def daily(df: pd.DataFrame) -> pd.DataFrame:
    morning = df[df["weekday"] & (df["hour"] >= 3) & (df["hour"] <= 14)]  # the car park empties overnight
    out = []
    for (cp, date), g in morning.groupby(["carpark", "date"]):
        hour, seen = fill_time(g["hour"].to_numpy(), g["pct_full"].to_numpy())
        out.append({"carpark": cp, "tsn": g["tsn"].iloc[0], "date": date, "spaces": int(g["spaces"].max()),
                    "max_pct": round(g["pct_full"].max(), 1), "readings": len(g), "fill_hour": hour, "fill_seen": seen})
    d = pd.DataFrame(out)
    return d[d["readings"] >= 12]  # at least 3 hours of the morning collected


def weekday_in_timetable(feed: str):
    """A Tuesday to Thursday the feed's latest timetable (data/static) covers: Sydney Trains files only cover a few days."""
    today = date.today()
    for offset in range(-7, 15):
        day = today + timedelta(days=offset)
        if day.weekday() in (1, 2, 3) and active_service_ids(feed, day):
            return day
    return None


def am_departures() -> pd.DataFrame:
    """Scheduled departures between AM_START and AM_END on a typical weekday, trains and Metro: station_id, dec_hour."""
    frames = []
    for feed in ("sydneytrains", "metro"):
        day = weekday_in_timetable(feed)
        if day is not None:
            frames.append(feed_departures(feed, day))
    deps = pd.concat(frames)
    deps = deps.assign(dec_hour=deps["dep_s"] / 3600)
    return deps[(deps["dec_hour"] >= AM_START) & (deps["dec_hour"] < AM_END)][["station_id", "dec_hour"]]


def after_full(d: pd.DataFrame, deps: pd.DataFrame) -> pd.DataFrame:
    """AM peak departures at the car park's station, and the share leaving after the car park filled."""
    by_station = {k: v.to_numpy() for k, v in deps.groupby("station_id")["dec_hour"]}
    n, pct = [], []
    for r in d.itertuples():
        t = by_station.get(r.tsn, np.array([]))
        after = (t >= r.fill_hour).sum() if not np.isnan(r.fill_hour) else 0
        n.append(len(t))
        pct.append(100 * after / len(t) if len(t) else np.nan)
    return d.assign(am_departures=n, pct_am_after_full=pct)


def as_clock(h: float) -> str:
    if np.isnan(h):
        return ""
    hh, mm = divmod(int(round(h * 60)), 60)
    return f"{hh}:{mm:02d}"


def main() -> None:
    df = readings()
    d = daily(df)
    stations = station_coords().drop_duplicates("stop_id").set_index("stop_id")["station"]
    d["station"] = d["tsn"].map(stations)
    d = after_full(d, am_departures())
    d["fill_time"] = [as_clock(h) if seen else f"by {as_clock(h)}" for h, seen in zip(d["fill_hour"], d["fill_seen"])]
    d = d.round({"fill_hour": 2, "pct_am_after_full": 0})
    cols = ["carpark", "station", "date", "spaces", "max_pct", "fill_time", "fill_hour", "fill_seen", "am_departures",
            "pct_am_after_full"]
    d[cols].to_csv(OUT / "carpark_fill.csv", index=False)
    curves = df[df["weekday"]][["carpark", "date", "hour", "pct_full"]].rename(columns={"hour": "time_local"})
    curves.round(2).to_csv(OUT / "carpark_curves.csv", index=False)
    print(f"carpark_fill.csv: {len(d)} car park-days, {d['carpark'].nunique()} car parks; "
          f"{d['station'].isna().sum()} rows with no rail station (bus car parks); filling seen on "
          f"{(d['fill_seen'] & d['fill_hour'].notna()).sum()} of {d['fill_hour'].notna().sum()} days that filled")
    s = d.groupby(["carpark", "station"], dropna=False).agg(days=("date", "size"), median_fill=("fill_hour", "median"),
                                                           pct_after=("pct_am_after_full", "median")).reset_index()
    s["median_fill"] = s["median_fill"].map(as_clock)
    print(s.sort_values("pct_after", ascending=False).head(20))


if __name__ == "__main__":
    main()
