"""
Crowding from the occupancy field in the vehicle positions feeds.

What the feeds carry (checked 30 Oct 2026):
    Metro          every fix: 1 many seats, 2 few seats, 3 standing room only
    Sydney Trains  about half of fixes (others leave it out): 1, 2, 3
    Buses          every fix: 1, 2, 3
    No feed reports crushed (4) or full (5), and none gives carriage-by-carriage loads.
These are the operators' own categories, from load sensors or ticketing; TfNSW
doesn't publish how they are set, so treat them as a guide.

Trains and Metro: the occupancy as each train left each station (the fix used for
its measured departure, reliability.py). Buses: every GPS fix of every bus in service.
    pct_standing  share of departures (or bus fixes) at "standing room only"
    pct_few_seats share at "few seats" or worse

Outputs (app/data/):
    crowding.csv          mode, line, towards (the train's last stop), station, hour, reports, pct_few_seats,
                          pct_standing, order (station's position along the line)  (trains and Metro)
    crowding_bus.csv      line, hour, reports, pct_few_seats, pct_standing

Usage:
    python -m analysis.reliability
    python -m analysis.crowding
"""

from collections import Counter
from datetime import datetime

import pandas as pd
from google.transit import gtfs_realtime_pb2

from analysis.reliability import SYDNEY, iter_vp_snapshots
from analysis.replay import route_names
from analysis.stop_events import rail_events
from tfnsw import PROJECT_DIR

OUT = PROJECT_DIR / "app" / "data"
FEW_SEATS, STANDING = 2, 3
MIN_REPORTS = 10


def shares(df: pd.DataFrame, by: list[str], occ: str = "occupancy") -> pd.DataFrame:
    d = df.dropna(subset=[occ])
    s = d.groupby(by).agg(reports=(occ, "size"), pct_few_seats=(occ, lambda o: 100 * (o >= FEW_SEATS).mean()),
                          pct_standing=(occ, lambda o: 100 * (o >= STANDING).mean())).reset_index()
    return s[s["reports"] >= MIN_REPORTS].round(1)


def with_direction(rail: pd.DataFrame) -> pd.DataFrame:
    """Adds "towards": the last station of each train's trip (e.g. Metro towards Tallawong or Sydenham)."""
    last = rail.sort_values("sched_time").groupby(["service_date", "trip_id"])["station"].last().rename("towards")
    return rail.merge(last.reset_index(), on=["service_date", "trip_id"])


def station_order(rail: pd.DataFrame) -> pd.DataFrame:
    """Each station's position along its line and direction, from the trip that called at the most stations."""
    n = rail.groupby(["service_date", "trip_id"])["station"].transform("size")
    longest = rail.assign(n=n).sort_values(["n", "sched_time"], ascending=[False, True])
    longest = longest[longest.groupby(["mode", "line", "towards"])["trip_id"].transform("first") == longest["trip_id"]]
    longest = longest.drop_duplicates(["mode", "line", "towards", "station"])
    longest["order"] = longest.groupby(["mode", "line", "towards"]).cumcount()
    return longest[["mode", "line", "towards", "station", "order"]]


def bus_occupancy(dates=None) -> pd.DataFrame:
    """Bus-minutes per route, Sydney hour and occupancy level: every bus in service in every snapshot
    (one a minute), counted as it reads so ~2,500 buses a minute stay small in memory."""
    counts = Counter()
    for raw in iter_vp_snapshots("buses", dates):
        msg = gtfs_realtime_pb2.FeedMessage()
        msg.ParseFromString(raw)
        hour = datetime.fromtimestamp(msg.header.timestamp, SYDNEY).hour
        counts.update((e.vehicle.trip.route_id, hour, e.vehicle.occupancy_status) for e in msg.entity
                      if e.vehicle.trip.trip_id and e.vehicle.HasField("occupancy_status"))
    df = pd.DataFrame([(*k, n) for k, n in counts.items()], columns=["route_id", "hour", "occupancy", "n"])
    df["line"] = df["route_id"].map(route_names("buses")).fillna(df["route_id"].str.split("_").str[-1])
    return df.loc[df.index.repeat(df["n"])].drop(columns="n")


def main() -> None:
    rail = with_direction(rail_events())
    rail = rail[rail["next_station"].notna()]  # departures
    key = ["mode", "line", "towards", "station"]
    rail_s = shares(rail, [*key, "hour"]).merge(station_order(rail), on=key, how="left")
    rail_s.to_csv(OUT / "crowding.csv", index=False)
    print(f"crowding.csv: {len(rail_s):,} rows. Departures with occupancy reported:")
    print(rail.groupby("mode")["occupancy"].apply(lambda o: o.notna().mean()).round(2))
    print(shares(rail, ["mode", "line"]).sort_values("pct_standing", ascending=False).head(10))

    bus = bus_occupancy()
    bus_s = shares(bus, ["line", "hour"])
    bus_s.to_csv(OUT / "crowding_bus.csv", index=False)
    print(f"crowding_bus.csv: {len(bus_s):,} rows from {len(bus):,} bus fixes; levels: "
          f"{bus['occupancy'].value_counts(normalize=True).round(3).to_dict()}")
    print(shares(bus, ["line"]).query("reports >= 500").sort_values("pct_standing", ascending=False).head(10))


if __name__ == "__main__":
    main()
