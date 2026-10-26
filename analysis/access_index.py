"""
Public transport access for every home, rail only vs rail + bus, using the
PTAL method (Public Transport Accessibility Level) that Transport for London
uses for planning.

For each residential mesh block and each route-direction it can walk to:

    walk time      = straight-line distance x 1.28 (the median detour measured
                     in walk_access.py) / 80 m per minute
    scheduled wait = half the gap between services, AM peak 7 to 9am
    reliability    = 2 min for bus, 0.75 min for rail (TfL's allowances)
    access time    = walk + wait + reliability
    EDF            = 30 / access time   ("equivalent doorstep frequency")

Only the nearest stop of each route-direction counts. Walking limits are TfL's:
640 m for bus and 960 m for rail, as walking distance.

Per mode, the access index = best EDF + half the sum of all the others, so a
second route adds value but less than the first. The total index = rail + bus.
Bands follow TfL: 0, 1a (up to 2.5), 1b (5), 2 (10), 3 (15), 4 (20), 5 (25),
6a (40), 6b (above 40). "Poor access" here means band 0 to 1b (index 5 or less).

Also records the quickest time to be on any service (minimum access time),
rail only and with buses, which feeds the wait penalty visual.

Bus timetable: pull_static.py --feeds buses. Light rail and ferries are not
included (out of scope).

Outputs:
    data/marts/stop_frequency.parquet       route-direction frequency at each stop
    data/marts/mesh_block_ptal.parquet      access index per mesh block
    data/marts/sa2_ptal.csv                 resident-weighted results per SA2

Usage:
    python -m analysis.access_index --date 2026-10-27
"""

import argparse
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd

from analysis.service_equity import CRS_METRES, MARTS_DIR
from analysis.station_service import active_service_ids, gtfs_seconds, next_tuesday, rail_routes, read

DETOUR = 1.28
WALK_M_PER_MIN = 80
MAX_WALK = {"bus": 640, "rail": 960}
RELIABILITY = {"bus": 2.0, "rail": 0.75}
AM = (7 * 3600, 9 * 3600)
BBOX = (149.9, -34.9, 152.2, -31.9)  # Greater Sydney plus Newcastle and the Hunter
INTERCITY = {"BMT", "CCN", "SCO", "SHL"}
BANDS = [(0, "0"), (2.5, "1a"), (5, "1b"), (10, "2"), (15, "3"), (20, "4"), (25, "5"), (40, "6a"), (np.inf, "6b")]


def ptal_band(ai: pd.Series) -> pd.Series:
    edges = [-np.inf, 0] + [b for b, _ in BANDS[1:-1]] + [np.inf]
    return pd.cut(ai, edges, labels=[l for _, l in BANDS], right=True).astype(str)


def stop_frequency(service_date: date) -> pd.DataFrame:
    """Departures per hour, 7 to 9am, for each route-direction at each stop (rail: per station)."""
    frames = []
    for feed in ("sydneytrains", "metro", "buses"):
        mode = "bus" if feed == "buses" else "rail"
        routes = read(feed, "routes", usecols=["route_id", "route_short_name", "agency_id"])
        if feed == "sydneytrains":
            routes = rail_routes(routes, feed)
        trips = read(feed, "trips", usecols=["route_id", "service_id", "trip_id", "direction_id"])
        trips = trips[trips["service_id"].isin(active_service_ids(feed, service_date))].merge(routes, on="route_id")
        st = pd.read_csv(
            f"data/static/{feed}/stop_times.txt", dtype=str, engine="pyarrow",
            usecols=["trip_id", "stop_id", "departure_time", "pickup_type"],
        )
        st = st[st["trip_id"].isin(set(trips["trip_id"])) & (st["pickup_type"].fillna("0") != "1")]
        st["dep_s"] = gtfs_seconds(st["departure_time"])
        st = st[st["dep_s"].between(*AM)].merge(trips[["trip_id", "route_short_name", "direction_id"]], on="trip_id")
        stops = read(feed, "stops", usecols=["stop_id", "stop_name", "stop_lat", "stop_lon", "parent_station"])
        if mode == "rail":  # roll platforms up to the station
            parent = stops["parent_station"].fillna("")
            st["stop_id"] = st["stop_id"].map(dict(zip(stops["stop_id"], parent.where(parent != "", stops["stop_id"]))))
        f = st.groupby(["stop_id", "route_short_name", "direction_id"]).size().rename("trips").reset_index()
        f["per_hour"] = f["trips"] / ((AM[1] - AM[0]) / 3600)
        f = f.merge(stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]], on="stop_id")
        f["mode"] = mode
        f["route"] = feed + ":" + f["route_short_name"].astype(str) + ":" + f["direction_id"].fillna("0").astype(str)
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out[["stop_lat", "stop_lon"]] = out[["stop_lat", "stop_lon"]].astype(float)
    out = out[out["stop_lon"].between(BBOX[0], BBOX[2]) & out["stop_lat"].between(BBOX[1], BBOX[3])]
    out["key"] = out["mode"] + ":" + out["stop_id"]
    return out


def mode_index(edf: pd.Series) -> float:
    s = np.sort(edf.to_numpy())[::-1]
    return s[0] + 0.5 * s[1:].sum() if len(s) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="PTAL-style access index, rail only vs rail + bus.")
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    service_date = args.date or next_tuesday()

    freq = stop_frequency(service_date)
    freq.to_parquet(MARTS_DIR / "stop_frequency.parquet", index=False)
    print(f"{freq['key'].nunique():,} stops, {freq['route'].nunique():,} route-directions in the AM peak")

    mb = pd.read_parquet(MARTS_DIR / "mesh_block_access.parquet")
    out = ptal_for(mb, freq)
    out.to_parquet(MARTS_DIR / "mesh_block_ptal.parquet", index=False)
    sa2 = sa2_summary(out)
    sa2.to_csv(MARTS_DIR / "sa2_ptal.csv", index=False)

    w = out["residents"]
    for label, col in (("rail only", "ai_rail"), ("rail + bus", "ai_all")):
        dist = out.groupby(ptal_band(out[col]))["residents"].sum() / w.sum() * 100
        print(f"{label:>10}: poor access (PTAL 0-1b) {100 * w[out[col] <= 5].sum() / w.sum():.1f}% | "
              + "  ".join(f"{b}:{dist.get(b, 0):.0f}%" for _, b in BANDS))


def ptal_for(mb: pd.DataFrame, freq: pd.DataFrame) -> pd.DataFrame:
    """PTAL index (rail only, bus, all) and minutes to board for each mesh block (needs x, y in metres)."""
    stops = freq.drop_duplicates("key")[["key", "mode", "stop_lon", "stop_lat"]]
    stops_g = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]), crs="EPSG:4326").to_crs(CRS_METRES)
    mb_g = gpd.GeoDataFrame(mb[["MB_CODE21", "residents"]], geometry=gpd.points_from_xy(mb["x"], mb["y"]), crs=CRS_METRES)

    pairs = []
    for mode in ("bus", "rail"):
        radius = MAX_WALK[mode] / DETOUR
        s = stops_g[stops_g["mode"] == mode]
        j = gpd.sjoin(mb_g, s[["key", "geometry"]], predicate="dwithin", distance=radius)
        j["dist"] = j.geometry.distance(s.set_index("key").loc[j["key"], "geometry"].set_axis(j.index))
        pairs.append(j[["MB_CODE21", "key", "dist"]].assign(mode=mode))
    pairs = pd.concat(pairs, ignore_index=True)

    r = pairs.merge(freq[["key", "route", "per_hour"]], on="key")
    r["walk_min"] = r["dist"] * DETOUR / WALK_M_PER_MIN
    r["wait_min"] = 0.5 * 60 / r["per_hour"]
    r["rel_min"] = np.where(r["mode"] == "bus", RELIABILITY["bus"], RELIABILITY["rail"])
    r["tat"] = r["walk_min"] + r["wait_min"] + r["rel_min"]
    best = r.sort_values("tat").drop_duplicates(["MB_CODE21", "route"])  # nearest stop per route
    best["edf"] = 30 / best["tat"]

    ai = best.groupby(["MB_CODE21", "mode"])["edf"].apply(mode_index).unstack(fill_value=0)
    ai = ai.reindex(columns=["rail", "bus"], fill_value=0)
    first = best.groupby(["MB_CODE21", "mode"])["tat"].min().unstack()
    out = mb[["MB_CODE21", "SA2_CODE21", "SA2_NAME21", "residents"]].merge(ai, on="MB_CODE21", how="left").fillna({"rail": 0, "bus": 0})
    out = out.rename(columns={"rail": "ai_rail", "bus": "ai_bus"})
    out["ai_all"] = out["ai_rail"] + out["ai_bus"]
    out["ptal_rail"] = ptal_band(out["ai_rail"])
    out["ptal_all"] = ptal_band(out["ai_all"])
    out = out.merge(first.rename(columns={"rail": "board_min_rail", "bus": "board_min_bus"}), on="MB_CODE21", how="left")
    out["board_min_all"] = out[["board_min_rail", "board_min_bus"]].min(axis=1)
    return out


def sa2_summary(out: pd.DataFrame) -> pd.DataFrame:
    """Resident-weighted PTAL results per SA2."""
    def agg(g: pd.DataFrame) -> pd.Series:
        w = g["residents"]
        poor = lambda col: 100 * w[g[col] <= 5].sum() / w.sum()
        return pd.Series({
            "residents": w.sum(),
            "ai_rail": np.average(g["ai_rail"], weights=w),
            "ai_all": np.average(g["ai_all"], weights=w),
            "pct_poor_rail": poor("ai_rail"),
            "pct_poor_all": poor("ai_all"),
            "board_min_rail": np.average(g["board_min_rail"].fillna(60), weights=w),
            "board_min_all": np.average(g["board_min_all"].fillna(60), weights=w),
        })

    sa2 = out.groupby(["SA2_CODE21", "SA2_NAME21"]).apply(agg, include_groups=False).round(2).reset_index()
    sa2["ptal_rail"] = ptal_band(sa2["ai_rail"])
    sa2["ptal_all"] = ptal_band(sa2["ai_all"])
    return sa2


if __name__ == "__main__":
    main()
