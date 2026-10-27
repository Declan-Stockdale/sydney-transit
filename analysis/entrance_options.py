"""
Station entrance shortlist: at which stations would one extra entrance bring the
most residents within a 10-minute walk?

Builds on walk_access.py (the OpenStreetMap walking network and the TfNSW 2020
station entrances). For every train and Metro station with entrance data:
    1. Candidate entrance spots = walking-network nodes within 200 m of the
       station's platforms (its stop location), thinned to one per 25 m, and
       skipping spots already within 100 m walk of an existing entrance.
    2. For each candidate, a shortest-path search from that spot gives the walk
       every nearby home would have if the entrance were there. A home's new
       walk is the shorter of that and its walk today.
    3. The station's best candidate is the one that brings the most residents
       (2021 Census, mesh blocks) from over 800 m to within 800 m walk.
       Residents brought within 1.5 km and the average walk saved are reported
       too.

800 m is about a 10-minute walk, the usual planning catchment for a station.

What this can't tell: whether a spot can physically reach the platforms (a
200 m radius covers a typical 8-car platform plus approaches, but footbridges
and lifts aren't modelled), who owns the land, or the cost. It's a shortlist for
feasibility work, not a design. Paths missing from OpenStreetMap, or
entrances opened since 2020, can make a gain look bigger than it is.

Output: data/marts/entrance_options.csv, and for the dashboard
app/data/entrance_options.csv (one row per station) and
app/data/entrance_gain_points.csv (homes brought within 800 m, top stations).

Usage:
    python -m analysis.walk_access         # builds the walking network first
    python -m analysis.entrance_options
"""

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer

from analysis.service_equity import CRS_METRES, MARTS_DIR, NEAR_M, WALK_M
from analysis.walk_access import EDGES_PATH, nearest_node, station_sources
from tfnsw import PROJECT_DIR

APP_DATA = PROJECT_DIR / "app" / "data"
CANDIDATE_RADIUS_M = 200  # around the station's platforms
CANDIDATE_GRID_M = 25  # keep one candidate spot per 25 m square
MIN_FROM_EXISTING_M = 100  # a spot this close (walking) to an existing entrance adds nothing new
TOP_N_POINTS = 25  # stations whose newly-served homes are exported for the dashboard map


def current_walk(g: nx.Graph, nodes: gpd.GeoDataFrame, sources: gpd.GeoDataFrame) -> dict:
    """Walk from every network node to its nearest existing entrance (as walk_access.py, entrances only)."""
    src = nearest_node(sources, nodes)
    h = g.copy()
    for node, hop in zip(src["node"], src["hop_m"]):
        old = h.get_edge_data("SRC", node, {"length": np.inf})["length"]
        h.add_edge("SRC", node, length=min(float(hop), old))
    return nx.single_source_dijkstra_path_length(h, "SRC", cutoff=WALK_M + 1000, weight="length")


def candidates(station_xy: tuple[float, float], nodes: gpd.GeoDataFrame, now: dict) -> pd.DataFrame:
    x, y = station_xy
    nx_, ny_ = nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy()
    near = np.hypot(nx_ - x, ny_ - y) <= CANDIDATE_RADIUS_M
    c = pd.DataFrame({"node": nodes["node"].to_numpy()[near], "x": nx_[near], "y": ny_[near]})
    c = c[c["node"].map(now).fillna(np.inf) >= MIN_FROM_EXISTING_M]
    c["cell"] = (c["x"] // CANDIDATE_GRID_M).astype(int).astype(str) + "_" + (c["y"] // CANDIDATE_GRID_M).astype(int).astype(str)
    return c.drop_duplicates("cell")


def evaluate(g: nx.Graph, cand: pd.DataFrame, homes: pd.DataFrame) -> pd.DataFrame:
    """Residents newly within 800 m / 1.5 km, and walk saved, for each candidate spot."""
    rows = []
    for c in cand.itertuples():
        d = nx.single_source_dijkstra_path_length(g, c.node, cutoff=WALK_M, weight="length")
        new = homes["node"].map(d) + homes["hop_m"]
        after = np.fmin(homes["walk_m"].fillna(np.inf), new.fillna(np.inf))
        before = homes["walk_m"].fillna(np.inf)
        r = homes["residents"]
        gain800 = (before > NEAR_M) & (after <= NEAR_M)
        rows.append({"node": c.node, "x": c.x, "y": c.y,
                     "residents_new_800m": int(r[gain800].sum()),
                     "residents_new_1500m": int(r[(before > WALK_M) & (after <= WALK_M)].sum()),
                     "residents_shorter_walk": int(r[after < before - 50].sum()),
                     "walk_saved_m": float(((before - after).clip(lower=0) * r)[after <= WALK_M].sum()
                                           / max(r[(after < before - 50) & (after <= WALK_M)].sum(), 1))})
    return pd.DataFrame(rows)


def export_entrances() -> None:
    """Today's entrances (TfNSW 2020) for the dashboard map."""
    e = station_sources()
    e = e[e["entrance"]].to_crs("EPSG:4326")
    pd.DataFrame({"station": e["station"], "lon": e.geometry.x.round(6), "lat": e.geometry.y.round(6)}).to_csv(
        APP_DATA / "station_entrances.csv", index=False)


def main() -> None:
    edges = pd.read_parquet(EDGES_PATH)
    g = nx.Graph()
    g.add_weighted_edges_from(zip(edges["u"], edges["v"], edges["length"]), weight="length")
    node_xy = pd.concat([edges[["u", "ux", "uy"]].set_axis(["node", "x", "y"], axis=1),
                         edges[["v", "vx", "vy"]].set_axis(["node", "x", "y"], axis=1)]).drop_duplicates("node")
    nodes = gpd.GeoDataFrame(node_xy[["node"]], geometry=gpd.points_from_xy(node_xy["x"], node_xy["y"]), crs=CRS_METRES)

    sources = station_sources()
    entrances = sources[sources["entrance"]]
    now = current_walk(g, nodes, entrances)

    # Homes within reach of a station, with their nearest network node and today's walk
    mb = pd.read_parquet(MARTS_DIR / "mesh_block_access.parquet")
    mb = mb[(mb["nearest_m"] <= WALK_M + 500) & (mb["residents"] > 0)]
    pts = gpd.GeoDataFrame(mb[["MB_CODE21"]], geometry=gpd.points_from_xy(mb["x"], mb["y"]), crs=CRS_METRES)
    homes = mb[["MB_CODE21", "SA2_NAME21", "residents", "x", "y"]].join(nearest_node(pts, nodes))
    homes["walk_m"] = homes["node"].map(now) + homes["hop_m"]

    stations = pd.read_csv(MARTS_DIR / "station_service.csv")
    stations = stations[stations["station"].isin(entrances["station"])]
    to_m = Transformer.from_crs("EPSG:4326", CRS_METRES, always_xy=True)
    stations["x"], stations["y"] = to_m.transform(stations["lon"].to_numpy(), stations["lat"].to_numpy())
    n_entrances = entrances.groupby("station").size()

    results, best_points = [], []
    for i, s in enumerate(stations.itertuples(), 1):
        local = homes[np.hypot(homes["x"] - s.x, homes["y"] - s.y) <= WALK_M + CANDIDATE_RADIUS_M]
        cand = candidates((s.x, s.y), nodes, now)
        if local.empty or cand.empty:
            continue
        ev = evaluate(g, cand, local).sort_values(["residents_new_800m", "residents_new_1500m"], ascending=False)
        best = ev.iloc[0]
        results.append({"station": s.station, "lines": s.lines, "entrances_now": int(n_entrances.get(s.station, 0)),
                        "candidates_tested": len(ev), **best.drop("node").to_dict(),
                        "residents_within_800m_now": int(local.loc[local["walk_m"] <= NEAR_M, "residents"].sum())})
        best_points.append((s.station, best["node"]))
        if i % 50 == 0:
            print(f"{i} of {len(stations)} stations")

    out = pd.DataFrame(results).sort_values("residents_new_800m", ascending=False)
    lon, lat = Transformer.from_crs(CRS_METRES, "EPSG:4326", always_xy=True).transform(out["x"].to_numpy(), out["y"].to_numpy())
    out["entrance_lon"], out["entrance_lat"] = np.round(lon, 6), np.round(lat, 6)
    out["pct_increase_800m"] = (100 * out["residents_new_800m"] / out["residents_within_800m_now"].replace(0, np.nan)).round(0)
    out = out.drop(columns=["x", "y"]).round({"walk_saved_m": 0})
    out.to_csv(MARTS_DIR / "entrance_options.csv", index=False)
    out.to_csv(APP_DATA / "entrance_options.csv", index=False)

    # Homes newly within 800 m of the proposed entrance, for the top stations' map
    from_lonlat = Transformer.from_crs(CRS_METRES, "EPSG:4326", always_xy=True)
    pts_out = []
    top = out.head(TOP_N_POINTS)["station"]
    for station, node in best_points:
        if station not in set(top):
            continue
        s = stations[stations["station"] == station].iloc[0]
        local = homes[np.hypot(homes["x"] - s.x, homes["y"] - s.y) <= WALK_M + CANDIDATE_RADIUS_M]
        d = nx.single_source_dijkstra_path_length(g, node, cutoff=NEAR_M, weight="length")
        new = local["node"].map(d) + local["hop_m"]
        hit = local[(local["walk_m"].fillna(np.inf) > NEAR_M) & (new <= NEAR_M)]
        lo, la = from_lonlat.transform(hit["x"].to_numpy(), hit["y"].to_numpy())
        pts_out.append(pd.DataFrame({"station": station, "lon": np.round(lo, 5), "lat": np.round(la, 5),
                                     "residents": hit["residents"].to_numpy(), "walk_now_m": hit["walk_m"].round(0).to_numpy(),
                                     "walk_new_m": new[hit.index].round(0).to_numpy()}))
    pd.concat(pts_out, ignore_index=True).to_csv(APP_DATA / "entrance_gain_points.csv", index=False)
    export_entrances()

    print(out[["station", "entrances_now", "residents_within_800m_now", "residents_new_800m", "pct_increase_800m",
               "residents_new_1500m", "walk_saved_m"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
