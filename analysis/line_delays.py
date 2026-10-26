"""
Delays by line and hour of each day, and how many people ride each line, for the
"Line delays" dashboard page.

Reads data/marts/vehicle_minutes.parquet (from analysis/replay.py: every minute of
every vehicle running, with how late it was) and summarises it per Sydney date,
hour, mode and line:
    vehicle_minutes   minutes of service observed (a measure of how much ran)
    avg_late          average minutes late of vehicles running (negative = early)
    avg_late_pos      the same with early running counted as 0 (what passengers lose)
    pct_late5         share of vehicle-minutes running 5+ minutes late

Lateness is "running lateness": how late each vehicle was while it was out on the
line, not only at the end of its trip. Trains and Metro use delays measured from
GPS at stations; buses, ferries and light rail use the operator feed's own
estimate. The two are close but not identical, so compare lines within a mode.

Daily trips per line (Opal, average day over the last 12 months) are added where
TfNSW publishes them by line: Sydney Trains / TrainLink lines, Metro, light rail
lines and ferry routes. Buses are only published by contract region, so bus routes
have no passenger counts.

Outputs (app/data/, committed with the dashboard):
    line_hour.csv     date, mode, line, hour, vehicle_minutes, avg_late, avg_late_pos, pct_late5
    line_trips.csv    mode, line, daily_trips

Usage:
    python -m analysis.replay
    python -m analysis.line_delays
"""

import pandas as pd

from analysis.reliability import MARTS_DIR, SYDNEY
from tfnsw import DATA_DIR, PROJECT_DIR

MINUTES = MARTS_DIR / "vehicle_minutes.parquet"
PATRONAGE_DIR = DATA_DIR / "raw" / "patronage"
OUT = PROJECT_DIR / "app" / "data"

# Opal line names -> the line codes used in the realtime feeds
TRAIN_NAMES = {"Blue Mountains Line": "BMT", "Central Coast & Newcastle Line": "CCN", "South Coast Line": "SCO",
               "Southern Highlands Line": "SHL", "Hunter Line": "HUN"}
LIGHT_RAIL_NAMES = {"Dulwich Hill Line": "L1", "Randwick Line": "L2", "Kingsford Line": "L3",
                    "Westmead & Carlingford Line": "L4", "Newcastle Light Rail": "NLR"}


def line_hours(vm: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(vm["vp_ts"], unit="s", utc=True).dt.tz_convert(SYDNEY)
    vm = vm.assign(date=t.dt.strftime("%Y-%m-%d"), hour=t.dt.hour, late_pos=vm["late"].clip(lower=0),
                   late5=(vm["late"] >= 5).where(vm["late"].notna()))
    out = vm.groupby(["date", "mode", "line", "hour"]).agg(
        vehicle_minutes=("late", "size"), avg_late=("late", "mean"), avg_late_pos=("late_pos", "mean"),
        pct_late5=("late5", "mean")).reset_index()
    out["pct_late5"] = 100 * out["pct_late5"]
    return out.round({"avg_late": 2, "avg_late_pos": 2, "pct_late5": 1})


def daily_trips() -> pd.DataFrame:
    """Average daily Opal trips per line over the latest 12 months, keyed by mode and line code."""
    frames = []
    for name, mode, code in [
        ("train", "Trains", lambda s: s.str.extract(r"^(T\d)")[0].fillna(s.map(TRAIN_NAMES))),
        ("metro", "Metro", lambda s: s.map({"Sydney Metro": "M1"})),
        ("lightrail", "Light rail", lambda s: s.map(LIGHT_RAIL_NAMES)),
        ("ferry", "Ferry", lambda s: s.str.extract(r"^(F\d+)\b")[0]),
    ]:
        f = PATRONAGE_DIR / f"trips_{name}_monthly.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        t["month"] = pd.to_datetime(t["Year_Month"], format="%b-%y", errors="coerce").fillna(
            pd.to_datetime(t["Year_Month"], format="%b-%Y", errors="coerce"))
        t["trips"] = pd.to_numeric(t["Trip"], errors="coerce")
        recent = t[t["month"] > t["month"].max() - pd.DateOffset(months=12)]
        per_line = recent.groupby("Line")["trips"].sum().reset_index()
        per_line["line"] = code(per_line["Line"])
        per_line = per_line.dropna(subset=["line"]).groupby("line")["trips"].sum() / 365
        frames.append(pd.DataFrame({"mode": mode, "line": per_line.index, "daily_trips": per_line.round(0).values}))
    trips = pd.concat(frames, ignore_index=True)
    # Intercity lines (BMT, CCN, SCO, SHL, HUN) run in the TrainLink feed too: give both modes the count
    intercity = trips[trips["line"].isin(list(TRAIN_NAMES.values()))].assign(mode="TrainLink")
    return pd.concat([trips, intercity], ignore_index=True)


def main() -> None:
    vm = pd.read_parquet(MINUTES)
    lh = line_hours(vm)
    lh.to_csv(OUT / "line_hour.csv", index=False)
    trips = daily_trips()
    trips.to_csv(OUT / "line_trips.csv", index=False)
    print(f"line_hour.csv: {len(lh):,} rows, {lh['date'].nunique()} days, {lh['line'].nunique():,} lines; "
          f"line_trips.csv: {len(trips)} lines with Opal trips")


if __name__ == "__main__":
    main()
