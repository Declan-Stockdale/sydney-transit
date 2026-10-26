"""
Catchment conversion: how many station entries a day for every resident who
lives within walking distance (1.5 km) and has that station as their nearest.

    entries_per_resident = average daily entries / walk-up catchment residents

High values mean a station draws far beyond its walk-up catchment (bus
interchanges, park-and-ride, destinations like the CBD or a hospital). Low
values mean many people live close by but few use the station, which can point
to poor service, a hard walk, or trips rail doesn't serve.

Only suburban "origin" stations are worth ranking at the low end, so stations
with a small catchment (< 2,000 residents) are flagged rather than ranked.

Inputs: data/marts/mesh_block_access.parquet (service_equity.py) and
        data/marts/station_demand_supply.csv (demand_supply.py)
Output: data/marts/station_catchment.csv

Usage:
    python -m analysis.catchment
"""

import pandas as pd

from analysis.service_equity import MARTS_DIR, WALK_M

MIN_CATCHMENT = 2000


def main() -> None:
    mb = pd.read_parquet(MARTS_DIR / "mesh_block_access.parquet")
    walk = mb[mb["nearest_m"] <= WALK_M]
    catchment = walk.groupby("nearest_station")["residents"].sum().rename("catchment_residents")

    ds = pd.read_csv(MARTS_DIR / "station_demand_supply.csv")
    out = ds.merge(catchment, left_on="station", right_index=True, how="inner")
    out["entries_per_resident"] = (out["avg_daily_entries"] / out["catchment_residents"]).round(3)
    out["am_entries_per_resident"] = (out["am_peak_entries_est"] / out["catchment_residents"]).round(3)
    out["ranked"] = out["catchment_residents"] >= MIN_CATCHMENT
    out = out.sort_values("entries_per_resident", ascending=False)
    out.to_csv(MARTS_DIR / "station_catchment.csv", index=False)

    cols = ["station", "lines", "catchment_residents", "avg_daily_entries", "entries_per_resident", "am_departures"]
    r = out[out["ranked"]]
    print(f"{len(r)} stations ranked; median {r['entries_per_resident'].median():.3f} entries per resident per day")
    print("HIGHEST\n", r[cols].head(12).to_string(index=False))
    print("LOWEST\n", r[cols].tail(15).iloc[::-1].to_string(index=False))


if __name__ == "__main__":
    main()
