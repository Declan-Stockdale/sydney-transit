"""
Jobs reachable within 45 minutes, leaving home at 8am: rail only vs rail + bus.

For every SA2 (suburb) in Greater Sydney:
    - Start from up to four resident-weighted points in the SA2 (one per
      quadrant, weighted by residents), and average the results.
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

from analysis.access_index import BBOX, DETOUR, WALK_M_PER_MIN
from analysis.service_equity import CENSUS_DIR, CRS_METRES, GREATER_SYDNEY, MARTS_DIR
from analysis.station_service import active_service_ids, gtfs_seconds, next_tuesday, rail_routes, read
from tfnsw import DATA_DIR

DEPART = 8 * 3600
BUDGET = 45 * 60
ACCESS_M = 800
TRANSFER_M = 300
CHANGE_S = 60


def connections(service_date: date, feeds) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every stop-to-stop hop on the service date (sorted by departure), plus stop coordinates."""
    hops, stops_all = [], []
    for feed in feeds:
        trips = read(feed, "trips", usecols=["route_id", "service_id", "trip_id"])
        routes = read(feed, "routes", usecols=["route_id", "route_short_name", "agency_id"])
        if feed == "sydneytrains":
            routes = rail_routes(routes, feed)
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


_NETWORKS: dict = {}


def network(feeds, service_date, sa2_shapes) -> dict:
    """The day's timetable as scan-ready arrays, built once per (feeds, date) and reused
    for every departure time (loading the bus stop_times is the slow part)."""
    key = (tuple(feeds), service_date)
    if key in _NETWORKS:
        return _NETWORKS[key]
    hops, stops = connections(service_date, feeds)
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
    net = {
        "g": g, "n_stops": len(stops), "sa2_codes": sa2_codes, "sa2_pos": sa2_pos, "stop_sa2_idx": stop_sa2_idx,
        "c_from": c_from, "c_to": c_to, "c_dep": c_dep, "c_arr": c_arr, "c_trip": c_trip.astype(np.int64),
        "n_trips": int(c_trip.max()) + 1, "fp_ptr": fp_ptr, "fp_to": fp["b"].to_numpy(np.int64), "fp_t": fp["t"].to_numpy(np.float64),
    }
    _NETWORKS[key] = net
    return net


def travel_times(feeds, service_date, origins, sa2_shapes, depart: int = DEPART, budget: int = BUDGET):
    """Minutes from each origin point to each SA2 (first stop reached inside it).

    origins: one row per origin point (several per SA2, see load_inputs), with
    SA2_CODE21 and weight. Returns (matrix, origins, destination codes);
    unreachable = inf. An origin's own SA2 is 0 when it has a stop within the
    access walk.
    """
    net = network(feeds, service_date, sa2_shapes)
    lo, hi = np.searchsorted(net["c_dep"], [depart, depart + budget], side="left")
    sl = slice(lo, hi + 1)
    has = net["stop_sa2_idx"] >= 0
    g = net["g"]
    mat = np.full((len(origins), len(net["sa2_codes"])), np.inf)
    near = gpd.sjoin(origins.reset_index(drop=True), g[["geometry"]], predicate="dwithin", distance=ACCESS_M / DETOUR)
    for oi, grp in near.groupby(level=0):
        o = origins.geometry.iloc[oi]
        s_idx = grp["index_right"].to_numpy(np.int64)
        walk_s = g.geometry.iloc[s_idx].distance(o).to_numpy() * DETOUR / WALK_M_PER_MIN * 60
        best = scan(s_idx, depart + walk_s, net["c_from"][sl], net["c_to"][sl], net["c_dep"][sl], net["c_arr"][sl],
                    net["c_trip"][sl], net["n_trips"], net["fp_ptr"], net["fp_to"], net["fp_t"], net["n_stops"])
        row = np.full(len(net["sa2_codes"]), np.inf)
        np.minimum.at(row, net["stop_sa2_idx"][has], (best[has] - depart) / 60)
        code = origins["SA2_CODE21"].iloc[oi]
        if code in net["sa2_pos"]:
            row[net["sa2_pos"][code]] = 0  # your own suburb
        mat[oi] = row
    return mat, origins, net["sa2_codes"]


def by_sa2(values: np.ndarray, origins: pd.DataFrame) -> pd.Series:
    """Resident-weighted average of a per-origin-point value, per SA2."""
    df = pd.DataFrame({"SA2_CODE21": origins["SA2_CODE21"].to_numpy(), "v": values * origins["weight"].to_numpy()})
    return df.groupby("SA2_CODE21")["v"].sum()


def run(feeds, service_date, origins, sa2_jobs, sa2_shapes, depart: int = DEPART, budget: int = BUDGET):
    mat, origins, d_codes = travel_times(feeds, service_date, origins, sa2_shapes, depart, budget)
    jobs_vec = pd.Series(sa2_jobs).reindex(d_codes).fillna(0).to_numpy()
    per_point = np.where(np.isfinite(mat).any(axis=1), ((mat <= budget / 60) * jobs_vec).sum(axis=1), 0)
    return by_sa2(per_point, origins)


def origin_points(mb: pd.DataFrame) -> gpd.GeoDataFrame:
    """Up to four resident-weighted origin points per SA2 (one per quadrant around the
    SA2's centre), each weighted by its share of residents. A single centre point
    misrepresents suburbs whose centre falls in a park or between stops (e.g.
    Gosford - Springfield read as zero access although most residents can walk to a stop)."""
    rows = []
    for code, g in mb[mb["residents"] > 0].groupby("SA2_CODE21"):
        mx, my = np.average(g["x"], weights=g["residents"]), np.average(g["y"], weights=g["residents"])
        quad = (g["x"] >= mx).astype(int) * 2 + (g["y"] >= my).astype(int)
        total = g["residents"].sum()
        for _, q in g.groupby(quad):
            w = q["residents"].sum()
            rows.append((code, np.average(q["x"], weights=q["residents"]), np.average(q["y"], weights=q["residents"]), w / total))
    o = pd.DataFrame(rows, columns=["SA2_CODE21", "x", "y", "weight"])
    return gpd.GeoDataFrame(o, geometry=gpd.points_from_xy(o["x"], o["y"]), crs=CRS_METRES)


def load_inputs():
    """SA2 shapes, jobs per SA2 (2021 Census place of work), mesh blocks, SA2 centres and origin points."""
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
    return shapes, jobs, mb, cen, origin_points(mb)


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
