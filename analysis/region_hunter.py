"""
Newcastle and the Hunter, as a separate region next to Greater Sydney.

Newcastle and the Hunter Valley are served by NSW TrainLink (Central Coast &
Newcastle line, Hunter line) and Newcastle buses. They are reported as their own
region so that they don't change the Greater Sydney equity statistics.

Region = ABS SA4s "Newcastle and Lake Macquarie" (111) and "Hunter Valley exc
Newcastle" (106). For every residential mesh block it computes:
    - straight-line distance to the nearest station (x 1.28 for walking, the
      median detour measured in Sydney; no footpath network is built here)
    - PTAL access, rail only vs rail + bus (same method as access_index.py)
and summarises by SA2, plus station-level service and demand.

Outputs (data/marts/):
    region_hunter_sa2.csv        PTAL and walk-to-station per SA2
    region_hunter_stations.csv   stations in the region with service and demand
    region_hunter_summary.json   headline numbers

Usage (after station_service, demand_supply and access_index):
    python -m analysis.region_hunter
"""

import json

import geopandas as gpd
import numpy as np
import pandas as pd

from analysis.access_index import DETOUR, ptal_for, sa2_summary
from analysis.service_equity import CENSUS_DIR, CRS_METRES, MARTS_DIR

SA4 = ("106", "111")
WALK_M = 1500


def load_region_mesh_blocks() -> gpd.GeoDataFrame:
    shp = f"zip://{CENSUS_DIR / 'MB_2021_AUST_SHP_GDA2020.zip'}!MB_2021_AUST_GDA2020.shp"
    mb = gpd.read_file(shp, where="SA4_CODE21 IN ('106', '111')")
    counts = pd.concat(
        pd.read_excel(CENSUS_DIR / "Mesh_Block_Counts_2021.xlsx", sheet_name=s, header=6, dtype={"MB_CODE_2021": str})
        for s in ("Table 1", "Table 1.1"))
    counts = counts.dropna(subset=["Person"])[["MB_CODE_2021", "Person"]]
    mb = mb.merge(counts, left_on="MB_CODE21", right_on="MB_CODE_2021")
    mb = mb[mb["Person"] > 0].to_crs(CRS_METRES)
    mb["geometry"] = mb.geometry.representative_point()
    mb["x"], mb["y"] = mb.geometry.x, mb.geometry.y
    return mb.rename(columns={"Person": "residents"})[["MB_CODE21", "SA2_CODE21", "SA2_NAME21", "SA4_NAME21", "residents", "x", "y", "geometry"]]


def main() -> None:
    mb = load_region_mesh_blocks()
    st = pd.read_csv(MARTS_DIR / "station_service.csv")
    sg = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st["lon"], st["lat"]), crs="EPSG:4326").to_crs(CRS_METRES)

    near = gpd.sjoin_nearest(mb[["MB_CODE21", "geometry"]], sg[["station", "geometry"]], distance_col="nearest_m")
    near = near.drop_duplicates("MB_CODE21")
    mb = mb.merge(near[["MB_CODE21", "station", "nearest_m"]].rename(columns={"station": "nearest_station"}), on="MB_CODE21")
    mb["walk_est_m"] = mb["nearest_m"] * DETOUR

    freq = pd.read_parquet(MARTS_DIR / "stop_frequency.parquet")
    ptal = ptal_for(mb.drop(columns="geometry"), freq)
    sa2 = sa2_summary(ptal)
    walk = mb.groupby("SA2_CODE21").apply(
        lambda g: 100 * g.loc[g["walk_est_m"] <= WALK_M, "residents"].sum() / g["residents"].sum(), include_groups=False
    ).rename("pct_within_1500m_walk").round(1)
    sa2 = sa2.merge(walk, on="SA2_CODE21")
    sa2.to_csv(MARTS_DIR / "region_hunter_sa2.csv", index=False)

    # Stations inside the region
    region = gpd.read_file(f"zip://{CENSUS_DIR / 'SA2_2021_AUST_SHP_GDA2020.zip'}!SA2_2021_AUST_GDA2020.shp",
                           where="SA4_CODE21 IN ('106', '111')").to_crs(CRS_METRES)
    inside = gpd.sjoin(sg, region[["SA2_NAME21", "geometry"]], predicate="within")
    ds = pd.read_csv(MARTS_DIR / "station_demand_supply.csv")[
        ["station", "avg_daily_entries", "entries_per_departure", "pm_entries_per_departure"]]
    stations = inside.drop(columns=["geometry", "index_right"]).merge(ds, on="station", how="left")
    stations.to_csv(MARTS_DIR / "region_hunter_stations.csv", index=False)

    w = ptal["residents"]
    summary = {
        "residents": int(w.sum()),
        "poor_rail_pct": round(100 * w[ptal["ai_rail"] <= 5].sum() / w.sum(), 1),
        "poor_all_pct": round(100 * w[ptal["ai_all"] <= 5].sum() / w.sum(), 1),
        "within_1500m_walk_pct": round(100 * mb.loc[mb["walk_est_m"] <= WALK_M, "residents"].sum() / mb["residents"].sum(), 1),
        "stations": int(len(stations)),
        "daily_departures": int(stations["daily_departures"].sum()),
    }
    (MARTS_DIR / "region_hunter_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(stations.sort_values("daily_departures", ascending=False)[["station", "lines", "daily_departures", "am_peak_tph", "avg_daily_entries"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
