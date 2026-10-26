"""
Service equity: how much rail service can each Greater Sydney resident walk to?

Joins scheduled service per station (station_service.py) to where people live,
using the 2021 Census at mesh block level (the smallest Census unit, roughly a
street block, with a resident count).

For every residential mesh block in Greater Sydney:
    - straight-line distance from its centre to the nearest station
    - the best service at any station within 800 m and within 1.5 km
      (daily departures, and AM peak trains per hour)

Then aggregated to SA2 (suburb-sized areas, ~3,000 to 25,000 people) and SA3
(district-sized, e.g. "Manly", "Hornsby"):
    - residents, density
    - % of residents within 800 m / 1.5 km of a station
    - residents more than 1.5 km from any station
    - population-weighted average of the best walkable service
      (0 for residents with no station within 1.5 km)

Known limitations (document alongside any result):
    - Rail only (Trains, Metro, intercity). Buses, light rail and ferries are out
      of scope, so areas served well by buses (e.g. the Northern Beaches B-Line)
      will look worse than residents' real access.
    - Straight-line distance. Walking distance is usually 20 to 40% longer.
    - 2021 population against a 2026 timetable. Fast-growing areas (e.g. the
      north-west and south-west growth areas) have more people now.

Usage:
    python -m analysis.station_service --date 2026-10-27   # run first
    python -m analysis.service_equity
"""

import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd

from tfnsw import DATA_DIR

CENSUS_DIR = DATA_DIR / "raw" / "census"
MARTS_DIR = DATA_DIR / "marts"
CRS_METRES = "EPSG:7856"  # GDA2020 / MGA zone 56, metres, covers Sydney
NEAR_M, WALK_M = 800, 1500
GREATER_SYDNEY = "1GSYD"


def load_mesh_blocks() -> gpd.GeoDataFrame:
    """Greater Sydney mesh blocks with their 2021 resident counts, as centre points."""
    shp = f"zip://{CENSUS_DIR / 'MB_2021_AUST_SHP_GDA2020.zip'}!MB_2021_AUST_GDA2020.shp"
    mb = gpd.read_file(shp, where=f"GCC_CODE21 = '{GREATER_SYDNEY}'")

    counts = pd.concat(
        pd.read_excel(CENSUS_DIR / "Mesh_Block_Counts_2021.xlsx", sheet_name=sheet, header=6, dtype={"MB_CODE_2021": str})
        for sheet in ("Table 1", "Table 1.1")  # NSW is split over two sheets
    )
    counts = counts.dropna(subset=["Person"])[["MB_CODE_2021", "Person"]]
    mb = mb.merge(counts, left_on="MB_CODE21", right_on="MB_CODE_2021")
    mb = mb[mb["Person"] > 0].to_crs(CRS_METRES)
    mb["geometry"] = mb.geometry.representative_point()  # a point guaranteed inside the block
    keep = ["MB_CODE21", "SA2_CODE21", "SA2_NAME21", "SA3_NAME21", "SA4_NAME21", "AREASQKM21", "Person", "geometry"]
    return mb[keep].rename(columns={"Person": "residents"})


def load_stations() -> gpd.GeoDataFrame:
    st = pd.read_csv(MARTS_DIR / "station_service.csv")
    gdf = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st["lon"], st["lat"]), crs="EPSG:7844")
    return gdf.to_crs(CRS_METRES)


def best_within(mb: gpd.GeoDataFrame, stations: gpd.GeoDataFrame, radius: int) -> pd.DataFrame:
    """Best daily departures and AM peak trains/hour at any station within radius of each mesh block."""
    zones = stations[["station", "daily_departures", "am_peak_tph", "geometry"]].copy()
    zones["geometry"] = zones.buffer(radius)
    hits = gpd.sjoin(mb[["MB_CODE21", "geometry"]], zones, predicate="within")
    best = hits.groupby("MB_CODE21").agg(daily=("daily_departures", "max"), am_tph=("am_peak_tph", "max"))
    return best.add_suffix(f"_{radius}m")


def mesh_block_access() -> gpd.GeoDataFrame:
    mb = load_mesh_blocks()
    stations = load_stations()

    nearest = gpd.sjoin_nearest(
        mb[["MB_CODE21", "geometry"]], stations[["station", "geometry"]], distance_col="nearest_m"
    ).drop_duplicates("MB_CODE21")  # ties at equal distance
    mb = mb.merge(nearest[["MB_CODE21", "station", "nearest_m"]].rename(columns={"station": "nearest_station"}), on="MB_CODE21")

    for radius in (NEAR_M, WALK_M):
        mb = mb.merge(best_within(mb, stations, radius), left_on="MB_CODE21", right_index=True, how="left")
    fill = [c for c in mb.columns if c.startswith(("daily_", "am_tph_"))]
    mb[fill] = mb[fill].fillna(0)
    return mb


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values, weights=weights)) if weights.sum() else np.nan


def summarise(mb: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        r = g["residents"]
        return pd.Series({
            "residents": int(r.sum()),
            "area_sqkm": g["AREASQKM21"].sum(),
            "pct_within_800m": round(100 * r[g["nearest_m"] <= NEAR_M].sum() / r.sum(), 1),
            "pct_within_1500m": round(100 * r[g["nearest_m"] <= WALK_M].sum() / r.sum(), 1),
            "residents_beyond_1500m": int(r[g["nearest_m"] > WALK_M].sum()),
            "avg_distance_km": round(weighted_mean(g["nearest_m"], r) / 1000, 2),
            "avg_walkable_daily_departures": round(weighted_mean(g[f"daily_{WALK_M}m"], r), 0),
            "avg_walkable_am_peak_tph": round(weighted_mean(g[f"am_tph_{WALK_M}m"], r), 1),
            "main_nearest_station": g.groupby("nearest_station")["residents"].sum().idxmax(),
        })

    out = mb.groupby(by).apply(agg, include_groups=False).reset_index()
    out["density_per_sqkm"] = (out["residents"] / out["area_sqkm"]).round(0)
    return out


def gini(values: np.ndarray, weights: np.ndarray) -> float:
    """Gini coefficient of service across residents (0 = everyone equal, 1 = all service to one person)."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum_w = np.cumsum(w) / w.sum()
    cum_v = np.cumsum(v * w) / (v * w).sum()
    return float(1 - np.sum((cum_w - np.concatenate([[0], cum_w[:-1]])) * (cum_v + np.concatenate([[0], cum_v[:-1]]))))


def main() -> None:
    mb = mesh_block_access()
    MARTS_DIR.mkdir(parents=True, exist_ok=True)
    # Keep the point as plain x/y (metres, EPSG:7856) so later steps can reuse it
    mb.assign(x=mb.geometry.x, y=mb.geometry.y).drop(columns="geometry").to_parquet(
        MARTS_DIR / "mesh_block_access.parquet", index=False
    )

    sa2 = summarise(mb, ["SA2_CODE21", "SA2_NAME21", "SA3_NAME21", "SA4_NAME21"])
    sa3 = summarise(mb, ["SA3_NAME21", "SA4_NAME21"])
    sa2.to_csv(MARTS_DIR / "sa2_service_equity.csv", index=False)
    sa3.to_csv(MARTS_DIR / "sa3_service_equity.csv", index=False)

    r = mb["residents"]
    print(f"Greater Sydney residents: {r.sum():,}")
    print(f"  within 800 m of a station:  {100 * r[mb['nearest_m'] <= NEAR_M].sum() / r.sum():.1f}%")
    print(f"  within 1.5 km of a station: {100 * r[mb['nearest_m'] <= WALK_M].sum() / r.sum():.1f}%")
    print(f"  Gini of walkable daily departures: {gini(mb[f'daily_{WALK_M}m'].to_numpy(), r.to_numpy()):.2f}")
    print(f"Wrote {len(sa2)} SA2s and {len(sa3)} SA3s to {MARTS_DIR}")


if __name__ == "__main__":
    main()
