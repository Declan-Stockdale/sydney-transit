"""
Demand vs supply per station: how many people board in the morning peak for
every scheduled train?

Demand
    Station entries from the monthly Opal file, last 12 months, turned into an
    average day. The AM peak share (06:00 to 10:00) comes from each station's
    2023 "typical day" profile, the latest available; stations without one
    (e.g. new Metro stations) use the network median share and are flagged:
    the median badly overstates AM boardings at CBD destination stations such
    as Gadigal and Victoria Cross, so leave flagged rows out of rankings.
    Terminus stations (Bondi Junction, Tallawong, Leppington) score high partly
    because every departure goes the busy way; at through stations half of the
    departures run counter-peak.
    Note the average day includes weekends, so weekday peaks are higher than
    shown. The ranking between stations is what matters.

Supply
    Scheduled departures from each station between 06:00 and 10:00 on the
    chosen service date (both directions), from station_service.py's logic.

The same is done for the PM peak (15:00 to 19:00): pm_peak_entries_est,
pm_departures, pm_entries_per_departure. In the evening, the busiest boarding
stations are the CBD and job centres rather than the suburbs.

Output: data/marts/station_demand_supply.csv, one row per station, with
    am_peak_entries_est, am_departures, entries_per_departure (AM) and the PM
    equivalents.

Usage:
    python -m analysis.demand_supply --date 2026-10-27
"""

import argparse
from datetime import date

import numpy as np
import pandas as pd

from analysis.station_service import feed_departures, next_tuesday
from tfnsw import DATA_DIR

PATRONAGE_DIR = DATA_DIR / "raw" / "patronage"
MARTS_DIR = DATA_DIR / "marts"
AM_START, AM_END = 6, 10  # hours, matching the typical-day file's 06:00 to 10:00 band
PM_START, PM_END = 15, 19  # the typical-day file's 15:00 to 19:00 band
# Patronage files name a few stations differently from the timetable
ALIASES = {"Domestic": "Domestic Airport", "International": "International Airport",
           "Bomaderry": "Bomaderry (Nowra)"}


def clean_station(names: pd.Series) -> pd.Series:
    s = names.str.replace(r"\s+Station$", "", regex=True).str.strip()
    return s.replace(ALIASES)


def monthly_entries(months: int = 12) -> pd.DataFrame:
    """Average daily entries and exits per station over the last `months` months."""
    df = pd.read_csv(PATRONAGE_DIR / "station_entry_exit_monthly.csv")
    df = df[df["Station_Type"] != "Light rail"]
    # Counts under 50 are suppressed for privacy; use the midpoint
    df["trips"] = pd.to_numeric(df["Trip"].replace("Less than 50", "25"), errors="coerce")
    df["month"] = pd.to_datetime(df["MonthYear"])
    recent = df[df["month"] > df["month"].max() - pd.DateOffset(months=months)]
    days = recent.drop_duplicates("month")["month"].dt.days_in_month.sum()

    out = (
        recent.assign(station=clean_station(recent["Station"]))
        .pivot_table(index="station", columns="Entry_Exit", values="trips", aggfunc="sum")
        .div(days)
        .rename(columns={"Entry": "avg_daily_entries", "Exit": "avg_daily_exits"})
        .reset_index()
    )
    out.attrs["period"] = f"{recent['month'].min():%b %Y} to {recent['month'].max():%b %Y}"
    return out


def peak_share(column: str = "Entries 06:00 to 10:00") -> pd.Series:
    """Share of each station's entries in a time band, from the 2023 typical day."""
    td = pd.read_excel(PATRONAGE_DIR / "station_entry_exit_typical_day_2016_2023.xlsx", sheet_name="train_stn_entry_exits")
    td = td[td["YEAR"].astype(str) == "2023"]
    band = pd.to_numeric(td[column], errors="coerce")
    day = pd.to_numeric(td["Entries 24 hours"], errors="coerce")
    share = (band / day).where(day > 0)
    return pd.Series(share.to_numpy(), index=clean_station(td["STATION"])).dropna()


def am_peak_share() -> pd.Series:
    return peak_share("Entries 06:00 to 10:00")


def departures(service_date: date) -> pd.DataFrame:
    """Scheduled departures per station in the AM (6-10am) and PM (3-7pm) peaks."""
    deps = pd.concat([feed_departures(f, service_date) for f in ("sydneytrains", "metro")])
    deps["station"] = clean_station(deps["stop_name"])
    hours = deps["dep_s"] / 3600
    deps["am"] = ((hours >= AM_START) & (hours < AM_END)).astype(int)
    deps["pm"] = ((hours >= PM_START) & (hours < PM_END)).astype(int)
    return deps.groupby("station").agg(
        am_departures=("am", "sum"), pm_departures=("pm", "sum"),
        lines=("route_short_name", lambda s: " ".join(sorted(set(s)))),
    ).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="AM peak entries per scheduled train, by station.")
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    service_date = args.date or next_tuesday()

    demand = monthly_entries()
    share = am_peak_share()
    pm_share = peak_share("Entries 15:00 to 19:00")
    demand["am_peak_share"] = demand["station"].map(share)
    demand["pm_peak_share"] = demand["station"].map(pm_share)
    demand["share_source"] = np.where(demand["am_peak_share"].isna(), "network median", "2023 typical day")
    demand["am_peak_share"] = demand["am_peak_share"].fillna(share.median())
    demand["pm_peak_share"] = demand["pm_peak_share"].fillna(pm_share.median())
    demand["am_peak_entries_est"] = (demand["avg_daily_entries"] * demand["am_peak_share"]).round(0)
    demand["pm_peak_entries_est"] = (demand["avg_daily_entries"] * demand["pm_peak_share"]).round(0)

    out = demand.merge(departures(service_date), on="station", how="inner")
    out = out[out["am_departures"] + out["pm_departures"] > 0]
    out["entries_per_departure"] = (out["am_peak_entries_est"] / out["am_departures"].replace(0, np.nan)).round(1)
    out["pm_entries_per_departure"] = (out["pm_peak_entries_est"] / out["pm_departures"].replace(0, np.nan)).round(1)
    out["service_date"] = service_date.isoformat()
    out["demand_period"] = demand.attrs["period"]
    out = out.sort_values("entries_per_departure", ascending=False)

    MARTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(MARTS_DIR / "station_demand_supply.csv", index=False)
    unmatched = sorted(set(demand["station"]) - set(out["station"]))
    print(f"{len(out)} stations matched; demand {demand.attrs['period']}; unmatched patronage stations: {len(unmatched)}")
    cols = ["station", "lines", "avg_daily_entries", "am_peak_entries_est", "am_departures", "entries_per_departure", "share_source"]
    ranked = out[out["share_source"] == "2023 typical day"]
    print(ranked[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
