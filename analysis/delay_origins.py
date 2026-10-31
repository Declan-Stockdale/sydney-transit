"""
Where delays start along a line: the stretch where trains lose time against the
timetable, and the station where each trip first became 5+ minutes late.

For every pair of consecutive stations on a trip:
    delay gained = delay at the second station - delay at the first
(positive = the train lost time between leaving the first station and leaving
the second, including its stop at the second). Only pairs where both delays
were measured from GPS are used, so a switch between GPS and the feed's
prediction can't masquerade as a delay.

Late starts: trips already 2+ minutes late leaving their first station. That
departure is the feed's prediction (GPS isn't used at origins, see
reliability.py), so treat it as indicative.

Outputs (app/data/):
    delay_gains.csv    mode, line, from_station, station, trains, avg_gain_s, pct_gain_2min, became_late5
    late_starts.csv    mode, line, station, trips, pct_start_late2, avg_start_delay_s

Usage:
    python -m analysis.reliability
    python -m analysis.delay_origins
"""

import pandas as pd

from analysis.stop_events import rail_events
from tfnsw import PROJECT_DIR

OUT = PROJECT_DIR / "app" / "data"
LATE_S = 300
GAIN_S = 120
START_LATE_S = 120
MIN_TRAINS = 10  # stretches with fewer measured trains are left out


def delay_gains(rail: pd.DataFrame) -> pd.DataFrame:
    """One row per consecutive station pair on a trip where both delays came from GPS."""
    d = rail.sort_values(["service_date", "trip_id", "sched_time"])
    g = d.groupby(["service_date", "trip_id"], sort=False)
    d = d.assign(prev_delay=g["delay_s"].shift(), prev_source=g["delay_source"].shift())
    seg = d[d["prev_station"].notna() & (d["delay_source"] == "gps") & (d["prev_source"] == "gps")].copy()
    seg["gain_s"] = seg["delay_s"] - seg["prev_delay"]
    return seg


def first_late(rail: pd.DataFrame, late_s: int = LATE_S) -> pd.DataFrame:
    """The station where each trip first reached late_s, for trips that weren't that late at their first stop."""
    d = rail.sort_values(["service_date", "trip_id", "sched_time"])
    late = d["delay_s"] >= late_s
    first_row = ~d.duplicated(["service_date", "trip_id"])
    started_late = d.loc[first_row & late, ["service_date", "trip_id"]]
    hit = d[late & ~first_row].drop_duplicates(["service_date", "trip_id"])
    hit = hit.merge(started_late, on=["service_date", "trip_id"], how="left", indicator=True)
    return hit[hit["_merge"] == "left_only"].drop(columns="_merge")


def summarise(rail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seg = delay_gains(rail)
    gains = seg.groupby(["mode", "line", "prev_station", "station"]).agg(
        trains=("gain_s", "size"), avg_gain_s=("gain_s", "mean"),
        pct_gain_2min=("gain_s", lambda s: 100 * (s >= GAIN_S).mean())).reset_index()
    became = first_late(rail).groupby(["mode", "line", "prev_station", "station"]).size().rename("became_late5")
    gains = gains.merge(became.reset_index(), on=["mode", "line", "prev_station", "station"], how="left")
    gains["became_late5"] = gains["became_late5"].fillna(0).astype(int)
    gains = gains[gains["trains"] >= MIN_TRAINS].rename(columns={"prev_station": "from_station"})

    first = rail.sort_values("sched_time").drop_duplicates(["service_date", "trip_id"])
    starts = first.groupby(["mode", "line", "station"]).agg(
        trips=("delay_s", "size"), pct_start_late2=("delay_s", lambda s: 100 * (s >= START_LATE_S).mean()),
        avg_start_delay_s=("delay_s", "mean")).reset_index()
    starts = starts[starts["trips"] >= MIN_TRAINS]
    return gains.round(1), starts.round(1)


def main() -> None:
    rail = rail_events()
    gains, starts = summarise(rail)
    gains.to_csv(OUT / "delay_gains.csv", index=False)
    starts.to_csv(OUT / "late_starts.csv", index=False)
    print(f"delay_gains.csv: {len(gains):,} stretches; late_starts.csv: {len(starts):,} origins")
    print("Stretches where most trains first became 5+ min late:")
    print(gains.nlargest(10, "became_late5")[["line", "from_station", "station", "trains", "avg_gain_s", "became_late5"]])
    print("Most time gained per train:")
    print(gains[gains["trains"] >= 50].nlargest(10, "avg_gain_s")[["line", "from_station", "station", "trains", "avg_gain_s", "pct_gain_2min"]])


if __name__ == "__main__":
    main()
