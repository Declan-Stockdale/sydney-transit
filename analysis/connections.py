"""
Missed connections: how often a late arrival makes someone miss the connection
the timetable offered them, and how long they then wait.

A "planned connection" is, for every arrival at a station and every line and
direction leaving it (other than the line you arrived on, and not straight back
the way you came), the first departure the timetable lets you catch: scheduled
at least the transfer time after your scheduled arrival, within MAX_WAIT_S.
It is missed when that departure actually left before you could get there
(actual arrival + transfer time). You then take the next departure on the same
line and direction that you can catch; the extra wait is the difference. An
extra wait over an hour means the next service wasn't collected, so it's dropped.
On lines every few minutes (Metro) a miss costs little; the extra wait shows that.

Three kinds:
    Train to train  at every station two or more lines meet, 3 minute change
    Bus to train    buses arriving at a stop within 250 m of a station
    Train to bus    the reverse (e.g. Metro to the bus home at Tallawong)
    Bus changes allow 2 minutes plus the walk (straight line x 1.3 at 1.2 m/s).

Counts are per connection the timetable offers, not per passenger: nobody
publishes how many people change where. Train times are measured from GPS;
bus times are the operator's own predictions (see analysis/stop_events.py).

Outputs:
    data/marts/connections.parquet   every planned connection, missed or made
    app/data/connections.csv         kind, station, from_line, to_line: connections, missed, extra_wait_min (total)
    app/data/connections_hour.csv    kind, station, hour: the same, all lines

Usage:
    python -m analysis.reliability
    python -m analysis.stop_events
    python -m analysis.connections
"""

import numpy as np
import pandas as pd

from analysis.reliability import MARTS_DIR, SYDNEY
from analysis.stop_events import BUS_EVENTS, bus_events, rail_events
from tfnsw import PROJECT_DIR

OUT = PROJECT_DIR / "app" / "data"
RAIL_TRANSFER_S = 180
BUS_TRANSFER_BASE_S = 120
WALK_DETOUR, WALK_SPEED = 1.3, 1.2  # straight line -> path, metres per second
MAX_WAIT_S = 20 * 60  # longer planned waits aren't a "connection" the timetable is offering
MAX_EXTRA_WAIT_S = 60 * 60  # a longer "extra wait" means the next service wasn't collected (a gap in the data)


def bus_transfer_s(walk_m) -> np.ndarray:
    return BUS_TRANSFER_BASE_S + np.asarray(walk_m, dtype=float) * WALK_DETOUR / WALK_SPEED


def plan_connections(arr: pd.DataFrame, dep: pd.DataFrame, max_wait_s: int = MAX_WAIT_S) -> pd.DataFrame:
    """Planned and actual connections.

    arr: hub, from_line, sched_arr, arr_actual, transfer_s, optionally prev_station
    dep: hub, to_line, line_dir, sched_dep, dep_actual, optionally next_station and dir_transfer_s
         (a transfer time that belongs to the departure stop, e.g. the walk to a bus stop; overrides transfer_s)
    Returns one row per arrival x departing line_dir with planned_sched_dep, planned_dep_actual, missed,
    caught_dep_actual, planned_wait_s and extra_wait_s (0 when made).
    """
    extra = [c for c in ["to_mode", "next_station", "dir_transfer_s"] if c in dep]
    dirs = dep[["hub", "line_dir", "to_line", *extra]].drop_duplicates(["hub", "line_dir"])
    a = arr.reset_index(drop=True).assign(arr_id=lambda x: x.index).merge(dirs, on="hub")
    a = a[a["to_line"] != a["from_line"]]
    if "prev_station" in a and "next_station" in a:
        a = a[a["next_station"].isna() | (a["next_station"] != a["prev_station"])]  # not straight back
    if "dir_transfer_s" in a:
        a["transfer_s"] = a["dir_transfer_s"].fillna(a["transfer_s"])
        a = a.drop(columns="dir_transfer_s")
    a = a.assign(ready_sched=(a["sched_arr"] + a["transfer_s"]).astype(float),
                 ready_actual=(a["arr_actual"] + a["transfer_s"]).astype(float))
    d = dep[["hub", "line_dir", "sched_dep", "dep_actual"]].dropna().astype({"sched_dep": float, "dep_actual": float})

    planned = pd.merge_asof(a.sort_values("ready_sched"), d.sort_values("sched_dep"), left_on="ready_sched",
                            right_on="sched_dep", by=["hub", "line_dir"], direction="forward", tolerance=float(max_wait_s))
    planned = planned.dropna(subset=["sched_dep"]).rename(columns={"sched_dep": "planned_sched_dep",
                                                                   "dep_actual": "planned_dep_actual"})
    caught = pd.merge_asof(planned.sort_values("ready_actual"), d[["hub", "line_dir", "dep_actual"]].sort_values("dep_actual"),
                           left_on="ready_actual", right_on="dep_actual", by=["hub", "line_dir"], direction="forward")
    caught = caught.rename(columns={"dep_actual": "caught_dep_actual"}).dropna(subset=["caught_dep_actual"])
    caught["missed"] = caught["planned_dep_actual"] < caught["ready_actual"]
    caught["planned_wait_s"] = caught["planned_sched_dep"] - caught["sched_arr"]
    caught["extra_wait_s"] = np.where(caught["missed"], caught["caught_dep_actual"] - caught["planned_dep_actual"], 0.0)
    caught = caught[caught["extra_wait_s"] <= MAX_EXTRA_WAIT_S]
    return caught.drop(columns=["ready_sched", "ready_actual"]).sort_values("arr_id").reset_index(drop=True)


def rail_sides(rail: pd.DataFrame):
    arr = rail[rail["prev_station"].notna()].rename(columns={"station": "hub", "line": "from_line", "sched_time": "sched_arr"})
    arr = arr.assign(transfer_s=float(RAIL_TRANSFER_S), from_mode=arr["mode"])
    dep = rail[rail["next_station"].notna()].rename(columns={"station": "hub", "line": "to_line", "sched_time": "sched_dep"})
    return arr, dep.assign(to_mode=dep["mode"])


def bus_sides(bus: pd.DataFrame):
    b = bus.rename(columns={"station": "hub"})
    b = b.assign(transfer_s=bus_transfer_s(b["walk_m"]), bus_line="Bus " + b["line"])
    arr = b[~b["first_stop"]].assign(from_line=lambda x: x["bus_line"], from_mode="Bus")
    dep = b[~b["last_stop"]].assign(to_line=lambda x: x["bus_line"], to_mode="Bus",
                                    line_dir=lambda x: x["bus_line"] + "@" + x["stop_id"],  # a kerbside stop serves one direction
                                    dir_transfer_s=lambda x: x["transfer_s"])
    return arr, dep


ARR_COLS = ["hub", "from_line", "from_mode", "sched_arr", "arr_actual", "transfer_s"]
DEP_COLS = ["hub", "to_line", "to_mode", "line_dir", "sched_dep", "dep_actual"]


def build() -> pd.DataFrame:
    r_arr, r_dep = rail_sides(rail_events())
    parts = [plan_connections(r_arr[ARR_COLS + ["prev_station"]], r_dep[DEP_COLS + ["next_station"]]).assign(kind="Train to train")]
    if BUS_EVENTS.exists():
        b_arr, b_dep = bus_sides(bus_events())
        parts.append(plan_connections(b_arr[ARR_COLS], r_dep[DEP_COLS]).assign(kind="Bus to train"))
        parts.append(plan_connections(r_arr[ARR_COLS], b_dep[DEP_COLS + ["dir_transfer_s"]]).assign(kind="Train to bus"))
    else:
        print(f"{BUS_EVENTS.name} not found: run python -m analysis.stop_events for bus connections")
    con = pd.concat(parts, ignore_index=True)
    t = pd.to_datetime(con["sched_arr"], unit="s", utc=True).dt.tz_convert(SYDNEY)
    con["hour"], con["date"] = t.dt.hour, t.dt.strftime("%Y-%m-%d")
    return con


def summarise(con: pd.DataFrame, by: list[str], min_connections: int = 1) -> pd.DataFrame:
    """Counts and total extra minutes waited, so the dashboard can re-aggregate before dividing."""
    s = con.groupby(by).agg(connections=("missed", "size"), missed=("missed", "sum"),
                            extra_wait_min=("extra_wait_s", "sum")).reset_index()
    s["extra_wait_min"] = (s["extra_wait_min"] / 60).round(1)
    return s[s["connections"] >= min_connections].rename(columns={"hub": "station"})


def main() -> None:
    con = build()
    con.to_parquet(MARTS_DIR / "connections.parquet", index=False)
    pairs = summarise(con, ["kind", "hub", "from_line", "to_line"], min_connections=10)
    pairs.to_csv(OUT / "connections.csv", index=False)
    hours = summarise(con, ["kind", "hub", "hour"])
    hours.to_csv(OUT / "connections_hour.csv", index=False)
    print(f"connections.csv: {len(pairs):,} line pairs, connections_hour.csv: {len(hours):,} rows, "
          f"from {len(con):,} planned connections")
    k = con.groupby("kind").agg(n=("missed", "size"), missed=("missed", "mean"),
                                wait_if_missed=("extra_wait_s", lambda x: x[x > 0].median() / 60))
    print(k.round(3))
    top = con.groupby(["kind", "hub"]).agg(n=("missed", "size"), missed=("missed", "mean")).query("n >= 200")
    print(top.sort_values("missed", ascending=False).head(15).round(3))


if __name__ == "__main__":
    main()
