"""
Walking distance to the nearest station entrance, along real paths, for every
residential mesh block within reach of rail.

service_equity.py uses straight-line distance, which flatters stations behind
rivers, motorways, rail corridors or cul-de-sac estates. This replaces it with
the shortest walk on the OpenStreetMap street and footpath network to the
nearest station entrance.

Steps:
    1. Build a walking network from the Geofabrik Australia OpenStreetMap
       extract (data/raw/context/osm/australia-latest.osm.pbf, ~1 GB, download
       it with the command below). Only walkable ways inside the Greater Sydney
       box are kept. The public Overpass API refuses bulk downloads like this,
       so a local extract is used instead. The network is cached as
       data/raw/context/osm/walk_edges.parquet.
    2. Sources = TfNSW station entrances (2020). Stations missing from that file
       (e.g. Erskineville and Metro stations opened since) use every path node
       within 150 m of the station centre instead.
    3. One multi-source shortest-path pass from all entrances, stopping at
       2.5 km, gives each network node its walk to the nearest entrance.
    4. Each mesh block takes the value of its nearest network node, plus the
       straight-line hop to it.

Walkable = highway ways except motorways and their ramps, minus ways tagged
foot=no or access=private/no (unless foot=yes/designated/permissive).

Output: data/marts/mesh_block_walk.parquet (MB_CODE21, residents, nearest_m,
walk_m) and a printed comparison of straight-line vs walking coverage.

Data: (c) OpenStreetMap contributors, ODbL; TfNSW Open Data.

Usage:
    curl.exe -L -o data/raw/context/osm/australia-latest.osm.pbf https://download.geofabrik.de/australia-oceania/australia-latest.osm.pbf
    python -m analysis.walk_access
"""

import geopandas as gpd
import networkx as nx
import numpy as np
import osmium
import pandas as pd
from pyproj import Transformer

from analysis.service_equity import CRS_METRES, MARTS_DIR, NEAR_M, WALK_M
from tfnsw import DATA_DIR

CONTEXT_DIR = DATA_DIR / "raw" / "context"
PBF = CONTEXT_DIR / "osm" / "australia-latest.osm.pbf"
EDGES_PATH = CONTEXT_DIR / "osm" / "walk_edges.parquet"
REACH_M = 2500
BBOX = (149.9, -34.9, 152.0, -32.7)  # lon/lat box around Greater Sydney and the intercity lines
NOT_WALKABLE = {"motorway", "motorway_link", "construction", "proposed", "raceway", "bus_guideway", "abandoned"}
FOOT_OK = {"yes", "designated", "permissive"}


def in_box(lon: float, lat: float) -> bool:
    return BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]


def build_edges() -> pd.DataFrame:
    """One row per walkable segment between consecutive OSM nodes, with lengths in metres."""
    to_m = Transformer.from_crs("EPSG:4326", CRS_METRES, always_xy=True)
    rows = []
    fp = osmium.FileProcessor(str(PBF)).with_locations().with_filter(osmium.filter.KeyFilter("highway"))
    for way in fp:
        if not way.is_way():
            continue
        tags = way.tags
        foot = tags.get("foot", "")
        if tags.get("highway") in NOT_WALKABLE or foot == "no":
            continue
        if tags.get("access", "") in ("private", "no") and foot not in FOOT_OK:
            continue
        pts = [(n.ref, n.lon, n.lat) for n in way.nodes if n.location.valid()]
        if len(pts) < 2 or not any(in_box(lon, lat) for _, lon, lat in pts):
            continue
        refs, lons, lats = zip(*pts)
        xs, ys = to_m.transform(lons, lats)
        for i in range(len(refs) - 1):
            rows.append((refs[i], refs[i + 1], xs[i], ys[i], xs[i + 1], ys[i + 1]))
    e = pd.DataFrame(rows, columns=["u", "v", "ux", "uy", "vx", "vy"])
    e["length"] = np.hypot(e["vx"] - e["ux"], e["vy"] - e["uy"])
    return e


def station_sources() -> gpd.GeoDataFrame:
    stations = pd.read_csv(MARTS_DIR / "station_service.csv")
    ent = pd.read_csv(CONTEXT_DIR / "station_entrances_2020.csv")
    ent = ent.rename(columns={"Train_Station": "station", "LAT": "lat", "LONG": "lon"})[["station", "lat", "lon"]]
    missing = stations[~stations["station"].isin(ent["station"])][["station", "lat", "lon"]].assign(entrance=False)
    pts = pd.concat([ent[ent["station"].isin(stations["station"])].assign(entrance=True), missing], ignore_index=True)
    return gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["lon"], pts["lat"]), crs="EPSG:4326").to_crs(CRS_METRES)


def nearest_node(points: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame) -> pd.DataFrame:
    j = gpd.sjoin_nearest(points[["geometry"]], nodes, distance_col="hop_m")
    return j[~j.index.duplicated()][["node", "hop_m"]]


def main() -> None:
    if EDGES_PATH.exists():
        edges = pd.read_parquet(EDGES_PATH)
    else:
        if not PBF.exists():
            raise SystemExit(f"Download the OpenStreetMap extract to {PBF} first (see the docstring)")
        edges = build_edges()
        edges.to_parquet(EDGES_PATH, index=False)
    print(f"Walk network: {len(edges):,} segments")

    g = nx.Graph()
    g.add_weighted_edges_from(zip(edges["u"], edges["v"], edges["length"]), weight="length")
    node_xy = pd.concat([
        edges[["u", "ux", "uy"]].set_axis(["node", "x", "y"], axis=1),
        edges[["v", "vx", "vy"]].set_axis(["node", "x", "y"], axis=1),
    ]).drop_duplicates("node")
    nodes = gpd.GeoDataFrame(node_xy[["node"]], geometry=gpd.points_from_xy(node_xy["x"], node_xy["y"]), crs=CRS_METRES)

    # Multi-source search: a virtual source joined to every entrance's nearest node
    sources = station_sources()
    src = nearest_node(sources, nodes)
    # Stations missing from the 2020 entrances file (e.g. Erskineville, new Metro stations)
    # only have a centre point, whose single nearest path node can sit on the wrong side of
    # the tracks. Treat every path node within 150 m of the centre as a way in instead.
    centre = sources[~sources["entrance"]]
    near = gpd.sjoin(nodes, centre[["geometry"]].assign(geometry=centre.buffer(150)), predicate="within")
    near["hop_m"] = near.geometry.distance(centre.geometry.loc[near["index_right"]].set_axis(near.index))
    src = pd.concat([src, near[["node", "hop_m"]]])
    for node, hop in zip(src["node"], src["hop_m"]):
        g.add_edge("SRC", node, length=min(float(hop), g.get_edge_data("SRC", node, {"length": np.inf})["length"]))
    dist = nx.single_source_dijkstra_path_length(g, "SRC", cutoff=REACH_M, weight="length")

    mb = pd.read_parquet(MARTS_DIR / "mesh_block_access.parquet")
    if "x" not in mb:
        raise SystemExit("mesh_block_access.parquet has no x/y; rerun python -m analysis.service_equity first")
    near = mb[mb["nearest_m"] <= REACH_M]
    pts = gpd.GeoDataFrame(near[["MB_CODE21"]], geometry=gpd.points_from_xy(near["x"], near["y"]), crs=CRS_METRES)
    hit = nearest_node(pts, nodes)
    near = near.join(hit)
    near["walk_m"] = near["node"].map(dist) + near["hop_m"]

    out = mb[["MB_CODE21", "SA2_CODE21", "residents", "nearest_m"]].merge(
        near[["MB_CODE21", "walk_m"]], on="MB_CODE21", how="left"
    )
    out.to_parquet(MARTS_DIR / "mesh_block_walk.parquet", index=False)

    r = out["residents"]
    for label, col in (("straight line", "nearest_m"), ("walking", "walk_m")):
        within = lambda m: 100 * r[out[col] <= m].sum() / r.sum()
        print(f"{label:>13}: within {NEAR_M} m {within(NEAR_M):.1f}%, within {WALK_M} m {within(WALK_M):.1f}%")
    ratio = (out["walk_m"] / out["nearest_m"]).replace([np.inf], np.nan)
    print(f"Median walk / straight-line ratio: {ratio[out['nearest_m'] > 200].median():.2f}")


if __name__ == "__main__":
    main()
