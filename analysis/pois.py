"""
Everyday destinations from OpenStreetMap, for the cumulative opportunities
measure: hospitals, tertiary education (universities and TAFE/colleges) and
supermarkets in the Greater Sydney box.

    hospital     amenity=hospital
    education    amenity=university or amenity=college (TAFE NSW campuses are
                 mostly tagged college)
    supermarket  shop=supermarket

Points and building outlines (ways) are both read; an outline becomes the
average of its corner points. Multipolygon relations are skipped, and the same
place mapped twice (as a point and an outline) is merged when the name matches
within 300 m. OSM coverage is good in Sydney but not perfect, so counts are
indicative.

Output: data/marts/pois.csv (category, name, lon, lat, x, y in EPSG:7856)

Data: (c) OpenStreetMap contributors, ODbL.

Usage:
    python -m analysis.pois
"""

import numpy as np
import osmium
import pandas as pd
from pyproj import Transformer

from analysis.service_equity import CRS_METRES, MARTS_DIR
from analysis.walk_access import BBOX, PBF, in_box


def category(tags) -> str | None:
    amenity, shop = tags.get("amenity"), tags.get("shop")
    if amenity == "hospital":
        return "hospital"
    if amenity in ("university", "college"):
        return "education"
    if shop == "supermarket":
        return "supermarket"
    return None


def main() -> None:
    rows = []
    fp = (osmium.FileProcessor(str(PBF)).with_locations()
          .with_filter(osmium.filter.KeyFilter("amenity", "shop")))
    for obj in fp:
        if obj.is_relation():
            continue
        cat = category(obj.tags)
        if cat is None:
            continue
        if obj.is_node():
            lon, lat = obj.location.lon, obj.location.lat
        else:
            pts = [(n.lon, n.lat) for n in obj.nodes if n.location.valid()]
            if not pts:
                continue
            lon, lat = np.mean(pts, axis=0)
        if in_box(lon, lat):
            rows.append((cat, obj.tags.get("name", ""), lon, lat))
    p = pd.DataFrame(rows, columns=["category", "name", "lon", "lat"])
    p["x"], p["y"] = Transformer.from_crs("EPSG:4326", CRS_METRES, always_xy=True).transform(p["lon"].to_numpy(), p["lat"].to_numpy())
    # merge duplicates: same category and name within 300 m
    p["gx"], p["gy"] = (p["x"] // 300).astype(int), (p["y"] // 300).astype(int)
    named = p[p["name"] != ""].drop_duplicates(["category", "name", "gx", "gy"])
    p = pd.concat([named, p[p["name"] == ""]]).drop(columns=["gx", "gy"])
    p.to_csv(MARTS_DIR / "pois.csv", index=False)
    print(p["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
