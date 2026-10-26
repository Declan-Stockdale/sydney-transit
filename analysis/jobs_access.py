"""
Jobs reachable within 45 minutes, leaving home at 8am: rail only vs rail + bus.

For every SA2 (suburb) in Greater Sydney:
    - Start at the SA2's resident-weighted centre (from Census mesh blocks).
    - Walk to any stop within 800 m (walking distance = straight line x 1.28,
      at 80 m per minute).
    - Ride the timetable (Tuesday 27 Oct 2026), changing between services at
      the same stop, or walking up to 300 m to another stop, with 1 extra
      minute allowed for each change.
    - A destination SA2 counts as reached if any of its stops is reached
      within the 45 minutes; all of its jobs (2021 Census place of work) count.

Uses the connection scan algorithm (earliest arrival over a time-sorted list of
every stop-to-stop hop), compiled with numba so all origins run in seconds.

Run twice: rail only (Trains, Metro, intercity) and rail + bus. Light rail and
ferries are out of scope.

Output: data/marts/sa2_jobs_access.csv (jobs_rail, jobs_all, gain)

Usage:
    python -m analysis.access_index --date 2026-10-27    # builds stop data first
    python -m analysis.jobs_access --date 2026-10-27
"""

import argparse
import zipfile
from datetime import date

import geopandas as gpd
import numba
import numpy as np
import pandas as pd

from analysis.access_index import BBOX, DETOUR, INTERCITY, WALK_M_PER_MIN
from analysis.service_equity import CENSUS_DIR, CRS_METRES, GREATER_SYDNEY, MARTS_DIR
from analysis.station_service import active_service_ids, gtfs_seconds, next_tuesday, read
from tfnsw import DATA_DIR

DEPART = 8 * 3600
BUDGET = 45 * 60
ACCESS_M = 800
TRANSFER_M = 300
CHANGE_S = 60


def connections(service_date: date, feeds, depart: int = DEPART, budget: int = BUDGET) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stop-to-stop hops departing within the time budget, plus stop coordinates."""
    hops, stops_all = [], []
    for feed in feeds:
        trips = read(feed, "trips", usecols=["route_id", "service_id", "trip_id"])
        routes = read(feed, "routes", usecols=["route_id", "route_short_name", "agency_id"])
        if feed == "sydneytrains":
            routes = routes[(routes["agency_id"] == "SydneyTrains") | routes["route_short_name"].isin(INTERCITY)]
        trips = trips[trips["service_id"].isin(active_service_ids(feed, service_date))].merge(routes, on="route_id")
        st = pd.read_csv(f"data/static/{feed}/stop_times.txt", dtype=str, engine="pyarrow",
                         usecols=["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"])
        st = st[st["trip_id"].isin(set(trips["trip_id"]))]
        st["seq"] = st["stop_sequence"].astype(int)
        st["arr"] = gtfs_seconds(st["arrival_time"].fillna(st["departure_time"]))
        st["dep"] = gtfs_seconds(st["departure_time"].fillna(st["arrival_time"]))
        st = st.sort_values(["trip_id", "seq"])
        nxt = st.groupby("trip_id").shift(-1)
        h = pd.DataFrame({"from": feed + ":" + st["stop_id"], "to": feed + ":" + nxt["stop_id"],
                          "dep": st["dep"], "arr": nxt["arr"], "trip": feed + ":" + st["trip_id"]}).dropna()
        h = h[(h["dep"] >= depart) & (h["dep"] <= depart + budget)]
        hops.append(h)
        s = read(feed, "stops", usecols=["stop_id", "stop_lat", "stop_lon"])
        s["key"] = feed + ":" + s["stop_id"]
        stops_all.append(s[["key", "stop_lat", "stop_lon"]])
    hops = pd.concat(hops).sort_values("dep")
    stops = pd.concat(stops_all)
    stops[["stop_lat", "stop_lon"]] = stops[["stop_lat", "stop_lon"]].astype(float)
    stops = stops[stops["key"].isin(set(hops["from"]) | set(hops["to"]))]
    stops = stops[stops["stop_lon"].between(BBOX[0], BBOX[2]) & stops["stop_lat"].between(BBOX[1], BBOX[3])]
    return hops, stops.reset_index(drop=True)


@numba.njit(cache=True)
def scan(start_stops, start_times, c_from, c_to, c_dep, c_arr, c_trip, n_trips, fp_ptr, fp_to, fp_t, n_stops):
    """Earliest arrival at every stop (connection scan with footpaths)."""
    best = np.full(n_stops, np.inf)
    for i in range(len(start_stops)):
        if start_times[i] < best[start_stops[i]]:
            best[start_stops[i]] = start_times[i]
    # walking transfers from the start stops
    for i in range(len(start_stops)):
        s = start_stops[i]
        for k in range(fp_ptr[s], fp_ptr[s + 1]):
            t = best[s] + fp_t[k]
            if t < best[fp_to[k]]:
                best[fp_to[k]] = t
    on = np.zeros(n_trips, dtype=np.bool_)
    for k in range(len(c_dep)):
        tr = c_trip[k]
        if on[tr] or best[c_from[k]] <= c_dep[k]:
            on[tr] = True
            a = c_arr[k]
            if a < best[c_to[k]]:
                best[c_to[k]] = a
                for f in range(fp_ptr[c_to[k]], fp_ptr[c_to[k] + 1]):
                    t = a + fp_t[f]
                    if t < best[fp_to[f]]:
                        best[fp_to[f]] = t
    return best


def travel_times(feeds, service_date, origins, sa2_shapes, depart: int = DEPART, budget: int = BUDGET):
    """Minutes from each origin SA2 centre to each SA2 (first stop reached inside it).

    Returns (matrix, origin codes, destination codes); unreachable = inf. An
    origin's own SA2 is 0 when it has a stop within the access walk.
    """
    hops, stops = connections(service_date, feeds, depart, budget)
    idx = pd.Series(np.arange(len(stops)), index=stops["key"])
    hops = hops[hops["from"].isin(idx.index) & hops["to"].isin(idx.index)]
    g = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]), crs="EPSG:4326").to_crs(CRS_METRES)

    # footpaths between nearby stops (CSR arrays); includes a change penalty
    fp = gpd.sjoin(g[["geometry"]], g[["geometry"]], predicate="dwithin", distance=TRANSFER_M / DETOUR)
    fp = fp[fp.index != fp["index_right"]]
    d = g.geometry.iloc[fp.index].reset_index(drop=True).distance(g.geometry.iloc[fp["index_right"]].reset_index(drop=True))
    fp = pd.DataFrame({"a": fp.index.to_numpy(), "b": fp["index_right"].to_numpy(),
                       "t": (d.to_numpy() * DETOUR / WALK_M_PER_MIN * 60 + CHANGE_S)}).sort_values("a")
    fp_ptr = np.searchsorted(fp["a"].to_numpy(), np.arange(len(stops) + 1)).astype(np.int64)

    c_from = hops["from"].map(idx).to_numpy(np.int64)
    c_to = hops["to"].map(idx).to_numpy(np.int64)
    c_trip, _ = pd.factorize(hops["trip"])
    c_dep, c_arr = hops["dep"].to_numpy(np.float64), hops["arr"].to_numpy(np.float64)

    # which SA2 each stop is in
    stop_sa2 = gpd.sjoin(g[["geometry"]], sa2_shapes[["SA2_CODE21", "geometry"]], predicate="within")["SA2_CODE21"]
    stop_sa2 = stop_sa2[~stop_sa2.index.duplicated()].reindex(range(len(stops)))
    sa2_codes = sa2_shapes["SA2_CODE21"].to_numpy()
    sa2_pos = pd.Series(np.arange(len(sa2_codes)), index=sa2_codes)
    stop_sa2_idx = stop_sa2.map(sa2_pos).fillna(-1).to_numpy(np.int64)
    has = stop_sa2_idx >= 0

    o_codes = origins["SA2_CODE21"].to_numpy()
    mat = np.full((len(o_codes), len(sa2_codes)), np.inf)
    near = gpd.sjoin(origins, g[["geometry"]], predicate="dwithin", distance=ACCESS_M / DETOUR)
    for code, grp in near.groupby("SA2_CODE21"):
        oi = np.flatnonzero(o_codes == code)[0]
        o = origins.geometry.iloc[oi]
        s_idx = grp["index_right"].to_numpy(np.int64)
        walk_s = g.geometry.iloc[s_idx].distance(o).to_numpy() * DETOUR / WALK_M_PER_MIN * 60
        best = scan(s_idx, depart + walk_s, c_from, c_to, c_dep, c_arr, c_trip, c_trip.max() + 1,
                    fp_ptr, fp["b"].to_numpy(np.int64), fp["t"].to_numpy(np.float64), len(stops))
        mins = (best[has] - depart) / 60
        row = np.full(len(sa2_codes), np.inf)
        np.minimum.at(row, stop_sa2_idx[has], mins)
        if code in sa2_pos:
            row[sa2_pos[code]] = 0  # your own suburb
        mat[oi] = row
    return mat, o_codes, sa2_codes


def run(feeds, service_date, origins, sa2_jobs, sa2_shapes):
    mat, o_codes, d_codes = travel_times(feeds, service_date, origins, sa2_shapes)
    jobs_vec = pd.Series(sa2_jobs).reindex(d_codes).fillna(0).to_numpy()
    reached = mat <= BUDGET / 60
    return pd.Series((reached * jobs_vec).sum(axis=1), index=o_codes).where(np.isfinite(mat).any(axis=1), 0)


def load_inputs():
    """SA2 shapes, jobs per SA2 (2021 Census place of work) and resident-weighted SA2 centres."""
    shp = f"zip://{CENSUS_DIR / 'SA2_2021_AUST_SHP_GDA2020.zip'}!SA2_2021_AUST_GDA2020.shp"
    shapes = gpd.read_file(shp, where=f"GCC_CODE21 = '{GREATER_SYDNEY}'").to_crs(CRS_METRES)
    with zipfile.ZipFile(DATA_DIR / "raw" / "context" / "2021_WPP_SA2_for_NSW_short-header.zip") as z:
        w = pd.read_csv(z.open("2021 Census WPP Statistical Area 2 for NSW/2021Census_W01A_NSW_POW_SA2.csv"), dtype={"POW_SA2_CODE_2021": str})
    tot = [c for c in w.columns if c.startswith("P_") and c.endswith("_Tot")]
    jobs = pd.Series(w[tot].sum(axis=1).to_numpy(), index=w["POW_SA2_CODE_2021"])
    mb = pd.read_parquet(MARTS_DIR / "mesh_block_access.parquet")
    cen = mb.groupby("SA2_CODE21").apply(
        lambda g: pd.Series({"x": np.average(g["x"], weights=g["residents"]), "y": np.average(g["y"], weights=g["residents"])}),
        include_groups=False).reset_index()
    origins = gpd.GeoDataFrame(cen[["SA2_CODE21"]], geometry=gpd.points_from_xy(cen["x"], cen["y"]), crs=CRS_METRES)
    return shapes, jobs, mb, cen, origins


def main() -> None:
    parser = argparse.ArgumentParser(description="Jobs reachable within 45 minutes, rail only vs rail + bus.")
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    service_date = args.date or next_tuesday()
    shapes, jobs, mb, cen, origins = load_inputs()

    rail = run(["sydneytrains", "metro"], service_date, origins, jobs, shapes)
    allm = run(["sydneytrains", "metro", "buses"], service_date, origins, jobs, shapes)
    out = pd.DataFrame({"jobs_rail": rail, "jobs_all": allm}).reindex(cen["SA2_CODE21"]).fillna(0)
    out.index.name = "SA2_CODE21"
    out["gain"] = out["jobs_all"] - out["jobs_rail"]
    out.to_csv(MARTS_DIR / "sa2_jobs_access.csv")

    res = mb.groupby("SA2_CODE21")["residents"].sum()
    w8 = res.reindex(out.index).fillna(0)
    print(f"Resident-weighted jobs within 45 min: rail only {np.average(out['jobs_rail'], weights=w8):,.0f}, "
          f"rail + bus {np.average(out['jobs_all'], weights=w8):,.0f} (of {jobs.reindex(shapes['SA2_CODE21']).sum():,.0f} jobs in Greater Sydney)")


if __name__ == "__main__":
    main()
