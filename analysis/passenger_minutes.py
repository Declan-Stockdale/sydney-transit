"""
Passenger minutes lost: which stations' delays affect the most people?

    passenger_minutes_per_day = average daily entries x average delay (minutes)
                                of the trains serving that station

Early or on-time trains count as 0 delay (being early doesn't give anyone time
back). A station with small delays but huge crowds can matter more than a
quiet station with long delays, which a plain on-time percentage hides.

Inputs: data/marts/stop_delays.parquet (reliability.py) and
        data/marts/station_demand_supply.csv (demand_supply.py)
Output: data/marts/station_passenger_minutes.csv

Preliminary until the collector has several full weekdays, including peaks:
the window covered is printed and saved with the results.

Usage:
    python -m analysis.reliability
    python -m analysis.passenger_minutes
"""

import pandas as pd

from analysis.reliability import MARTS_DIR

ON_TIME_S = 300  # "on time" = less than 5 minutes late


def main() -> None:
    d = pd.read_parquet(MARTS_DIR / "stop_delays.parquet")
    served = d[~d["cancelled"] & d["delay_s"].notna() & d["station"].notna()].copy()
    served["late_min"] = served["delay_s"].clip(lower=0) / 60
    window = f"{served['sched_local'].min():%d %b %H:%M} to {served['sched_local'].max():%d %b %H:%M}"

    by_station = served.groupby("station").agg(
        trains=("trip_id", "nunique"),
        avg_late_min=("late_min", "mean"),
        on_time_pct=("delay_s", lambda s: 100 * (s < ON_TIME_S).mean()),
    ).reset_index()

    demand = pd.read_csv(MARTS_DIR / "station_demand_supply.csv", usecols=["station", "lines", "avg_daily_entries"])
    out = by_station.merge(demand, on="station", how="inner")
    out["passenger_minutes_per_day"] = (out["avg_daily_entries"] * out["avg_late_min"]).round(0)
    out["window"] = window
    out = out.sort_values("passenger_minutes_per_day", ascending=False)
    out.to_csv(MARTS_DIR / "station_passenger_minutes.csv", index=False)

    print(f"Window: {window} Sydney time; {len(served):,} train-stop observations at {len(out)} stations")
    cols = ["station", "lines", "trains", "avg_late_min", "on_time_pct", "avg_daily_entries", "passenger_minutes_per_day"]
    print(out[cols].head(15).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
