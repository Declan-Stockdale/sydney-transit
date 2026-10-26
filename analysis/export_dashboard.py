"""
Export small, precomputed tables for the Streamlit dashboard into app/data/.

Everything the app shows comes from here, so the app needs no raw data, no API
key and no heavy libraries. Files are kept small enough to commit (a few MB in
total): suburb shapes are simplified to about 60 m.

    sa2.geojson        suburb shapes + every suburb-level metric
    stations.csv       station-level supply, demand (AM and PM peak), catchment, delays;
                       Greater Sydney stations only
    hourly.csv         jobs within 45 min for each hour, weekday and Sunday, per suburb
    hunter_sa2.geojson / hunter_stations.csv
                       Newcastle and the Hunter, reported as a separate region
    segments.csv       track sections with AM peak trains per hour (network map)
    timeline.csv       monthly station entries 2016 to now
    bankstown.csv      closure before/after by station
    lines.csv          card type profile by line
    modes.csv          Trains vs Metro comparison
    heatmap.csv        average lateness by line and hour (preview)
    excess_wait.csv    excess wait time by platform (preview)
    pois.csv           hospitals, campuses, supermarkets
    meta.json          headline numbers and data dates

Usage:
    python -m analysis.export_dashboard
"""

import json

import geopandas as gpd
import numpy as np
import pandas as pd

from analysis.opal_report import CLOSED, all_monthly_entries
from analysis.service_equity import MARTS_DIR
from analysis.visuals import SYDNEY, delays, sa2_shapes, segments, trip_stops
from tfnsw import PROJECT_DIR

OUT = PROJECT_DIR / "app" / "data"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- suburbs -----------------------------------------------------------
    ctx = pd.read_csv(MARTS_DIR / "sa2_equity_context.csv", dtype={"SA2_CODE21": str})
    meth = pd.read_csv(MARTS_DIR / "sa2_access_methods.csv", dtype={"SA2_CODE21": str})
    keep_ctx = ["SA2_CODE21", "SA2_NAME21", "SA3_NAME21", "SA4_NAME21", "residents", "erp_2025", "erp_growth_pct",
                "tzp_pop_2041", "growth_to_2041", "irsd_decile", "no_car_pct", "pct_within_800m", "pct_within_1500m",
                "avg_walkable_am_peak_tph", "ai_rail", "ai_all", "ptal_rail", "ptal_all", "pct_poor_rail", "pct_poor_all",
                "board_min_rail", "board_min_all", "jobs_rail", "jobs_all", "transport_disadvantaged",
                "transport_disadvantaged_with_bus"]
    keep_meth = ["SA2_CODE21", "jobs_30", "jobs_60", "hospital_45", "education_45", "supermarket_20", "gravity_index",
                 "jobs45_wk7am", "jobs45_wk9pm", "jobs45_sun10am", "need_z", "supply_z", "transit_gap",
                 "pct_65plus", "pct_15_24", "unemployment_rate"]
    sa2 = ctx[keep_ctx].merge(meth[keep_meth], on="SA2_CODE21", how="left")
    sa2["keep_9pm_pct"] = 100 * sa2["jobs45_wk9pm"] / sa2["jobs45_wk7am"].replace(0, np.nan)
    sa2["keep_sun_pct"] = 100 * sa2["jobs45_sun10am"] / sa2["jobs45_wk7am"].replace(0, np.nan)
    g = sa2_shapes()[["SA2_CODE21", "geometry"]].merge(sa2, on="SA2_CODE21")
    g["geometry"] = g.geometry.simplify(60)
    g = g.to_crs("EPSG:4326")
    num = g.select_dtypes("number").columns
    g[num] = g[num].round(2)
    g.to_file(OUT / "sa2.geojson", driver="GeoJSON", COORDINATE_PRECISION=5)

    # --- stations ----------------------------------------------------------
    st = pd.read_csv(MARTS_DIR / "station_service.csv")
    st["mode"] = np.where(st["networks"] == "metro", "Metro", np.where(st["networks"] == "sydneytrains", "Trains", "Both"))
    ds = pd.read_csv(MARTS_DIR / "station_demand_supply.csv")[
        ["station", "avg_daily_entries", "am_peak_entries_est", "am_departures", "entries_per_departure",
         "pm_peak_entries_est", "pm_departures", "pm_entries_per_departure", "share_source"]]
    cat = pd.read_csv(MARTS_DIR / "station_catchment.csv")[["station", "catchment_residents", "entries_per_resident"]]
    pm = pd.read_csv(MARTS_DIR / "station_passenger_minutes.csv")[["station", "avg_late_min", "on_time_pct", "passenger_minutes_per_day"]]
    jr = pd.read_csv(MARTS_DIR / "station_jobs_reach.csv").rename(columns={"Unnamed: 0": "station", "jobs_45min": "jobs_45_rail"})
    stations = (st[["station", "lat", "lon", "lines", "mode", "daily_departures", "am_peak_tph", "interpeak_tph", "pm_peak_tph"]]
                .merge(ds, on="station", how="left").merge(cat, on="station", how="left")
                .merge(pm, on="station", how="left").merge(jr, on="station", how="left"))
    # Keep stations inside Greater Sydney only. Intercity stations beyond it (Newcastle,
    # Southern Highlands, Bathurst) only show the Sydney-bound part of their service:
    # e.g. Hunter line trains at Newcastle are outside this project's feeds.
    shapes = sa2_shapes()
    pts = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations["lon"], stations["lat"]), crs="EPSG:4326").to_crs(shapes.crs)
    inside = gpd.sjoin(pts, shapes[["SA2_CODE21", "geometry"]], predicate="within").index.unique()
    stations = stations.loc[stations.index.isin(inside)]
    stations.round(3).to_csv(OUT / "stations.csv", index=False)

    # --- Newcastle and the Hunter (separate region) --------------------------
    from analysis.service_equity import CENSUS_DIR
    rshp = gpd.read_file(f"zip://{CENSUS_DIR / 'SA2_2021_AUST_SHP_GDA2020.zip'}!SA2_2021_AUST_GDA2020.shp",
                         where="SA4_CODE21 IN ('106', '111')")[["SA2_CODE21", "SA4_NAME21", "geometry"]]
    rsa2 = pd.read_csv(MARTS_DIR / "region_hunter_sa2.csv", dtype={"SA2_CODE21": str})
    rg = rshp.to_crs("EPSG:7856").merge(rsa2, on="SA2_CODE21")
    rg["geometry"] = rg.geometry.simplify(80)
    rg = rg.to_crs("EPSG:4326")
    rnum = rg.select_dtypes("number").columns
    rg[rnum] = rg[rnum].round(2)
    rg.to_file(OUT / "hunter_sa2.geojson", driver="GeoJSON", COORDINATE_PRECISION=5)
    rst = pd.read_csv(MARTS_DIR / "region_hunter_stations.csv")
    rst = rst.merge(pm, on="station", how="left")
    rst.round(3).to_csv(OUT / "hunter_stations.csv", index=False)

    hourly = pd.read_csv(MARTS_DIR / "sa2_hourly_access.csv", dtype={"SA2_CODE21": str})
    hourly["jobs45"] = hourly["jobs45"].round(0)
    hourly.to_csv(OUT / "hourly.csv", index=False)

    # --- network segments (lon/lat) -----------------------------------------
    seg = segments(trip_stops())
    pos = st.set_index("station")[["lon", "lat"]]
    seg = seg.join(pos.add_prefix("a_"), on="a").join(pos.add_prefix("b_"), on="b")
    seg[["a", "b", "mode", "tph", "a_lon", "a_lat", "b_lon", "b_lat"]].round(5).to_csv(OUT / "segments.csv", index=False)

    # --- demand ------------------------------------------------------------
    m = all_monthly_entries()
    metro_city = ["Gadigal", "Victoria Cross", "Crows Nest", "Barangaroo", "Waterloo"]
    tl = pd.DataFrame({
        "network": m.groupby("month")["trips"].sum(),
        "sydenham_bankstown_closed": m[m["station"].isin(CLOSED)].groupby("month")["trips"].sum(),
        "metro_city_new": m[m["station"].isin(metro_city)].groupby("month")["trips"].sum(),
    }).fillna(0).reset_index()
    tl.to_csv(OUT / "timeline.csv", index=False)
    pd.read_csv(MARTS_DIR / "bankstown_closure.csv").to_csv(OUT / "bankstown.csv", index=False)
    pd.read_csv(MARTS_DIR / "line_card_profile.csv").to_csv(OUT / "lines.csv", index=False)
    pd.read_csv(MARTS_DIR / "trains_vs_metro.csv", index_col=0).rename(index=lambda s: s.replace("\n", " ")).to_csv(OUT / "modes.csv")

    # --- reliability previews ----------------------------------------------
    d = delays()
    d["hour"] = pd.to_datetime(d["sched_time"], unit="s", utc=True).dt.tz_convert(SYDNEY).dt.hour
    hm = d.groupby(["route_short_name", "mode", "hour"]).agg(late_min=("late_min", "mean"), n=("late_min", "size")).reset_index()
    hm[hm["n"] >= 10].round(2).to_csv(OUT / "heatmap.csv", index=False)
    pd.read_csv(MARTS_DIR / "excess_wait.csv").round(3).to_csv(OUT / "excess_wait.csv", index=False)
    pd.read_csv(MARTS_DIR / "pois.csv")[["category", "name", "lon", "lat"]].round(5).to_csv(OUT / "pois.csv", index=False)

    # --- headline numbers --------------------------------------------------
    mb = pd.read_parquet(MARTS_DIR / "mesh_block_ptal.parquet")
    w = mb["residents"]
    walk = pd.read_parquet(MARTS_DIR / "mesh_block_walk.parquet")
    ww = walk["residents"]
    j = pd.read_csv(MARTS_DIR / "sa2_jobs_access.csv", dtype={"SA2_CODE21": str}).merge(
        sa2[["SA2_CODE21", "residents"]], on="SA2_CODE21")
    td_bus = sa2[sa2["transport_disadvantaged_with_bus"] == True]
    td_rail = sa2[sa2["transport_disadvantaged"] == True]
    meta = {
        "residents": int(w.sum()),
        "poor_rail_pct": round(100 * w[mb["ai_rail"] <= 5].sum() / w.sum(), 1),
        "poor_all_pct": round(100 * w[mb["ai_all"] <= 5].sum() / w.sum(), 1),
        "walk_1500_pct": round(100 * ww[walk["walk_m"] <= 1500].sum() / ww.sum(), 1),
        "jobs45_rail": int(np.average(j["jobs_rail"], weights=j["residents"])),
        "jobs45_all": int(np.average(j["jobs_all"], weights=j["residents"])),
        "td_rail_suburbs": int(len(td_rail)), "td_rail_people": int(td_rail["erp_2025"].sum()),
        "td_bus_suburbs": int(len(td_bus)), "td_bus_people": int(td_bus["erp_2025"].sum()),
        "timetable_date": "2026-10-27",
        "reliability_window": f"{d['sched_local'].min():%d %b %Y %H:%M} to {d['sched_local'].max():%H:%M}",
        "reliability_obs": int(len(d)),
        "hunter": json.loads((MARTS_DIR / "region_hunter_summary.json").read_text()),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    sizes = {p.name: round(p.stat().st_size / 1e6, 2) for p in OUT.iterdir()}
    print(json.dumps(meta, indent=2), sizes, f"total {sum(sizes.values()):.1f} MB")


if __name__ == "__main__":
    main()
