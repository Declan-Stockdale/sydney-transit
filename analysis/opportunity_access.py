"""
Five access methods used in transport planning internationally, applied to
Greater Sydney by rail + bus (timetable only; light rail and ferries excluded).

1. Cumulative opportunities (Wachs and Kumagai, 1973; the basis of the UK
   Department for Transport's journey time statistics): how many jobs,
   hospitals, tertiary campuses and supermarkets are reachable within a set time.
2. Gravity-based access (Hansen, 1959): every job counts, weighted down the
   longer it takes to reach: sum of jobs x exp(-beta x minutes), with beta set
   so a job 20 minutes away counts half. No hard cut-off.
3. Temporal coverage (time-of-day accessibility, e.g. Owen and Levinson's
   Access Across America): the same jobs measure at weekday 7am, weekday 9pm
   and Sunday 10am (each averaged over departures at :00, :10 and :20), to
   show who loses access outside commuter hours.
4. Transit gap index (Currie, 2010, Melbourne): transport need minus supply per
   suburb. Need = average z-score of disadvantage (inverse IRSD), households
   without a car, residents 65+, and residents 15 to 24. Supply = z-score of the
   PTAL access index (rail + bus). Positive gap = more need than service.
6. Social exclusion lens (UK Social Exclusion Unit, 2003, "Making the
   Connections"; Lucas, 2012): how access lines up with unemployment by suburb.
   Correlation only; access and unemployment share causes (income, housing
   costs, distance from the centre), so this is not evidence that one causes
   the other.

(Method 5, excess wait time, is in excess_wait.py: it needs realtime data.)

Travel times: earliest arrival from each suburb's resident-weighted centre to
any stop in each other suburb (jobs_access.py), walking up to 800 m to start.

Output: data/marts/sa2_access_methods.csv

Usage:
    python -m analysis.pois
    python -m analysis.opportunity_access
"""

import zipfile
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd

from analysis.jobs_access import load_inputs, travel_times
from analysis.service_equity import CENSUS_DIR, CRS_METRES, MARTS_DIR

WEEKDAY, SUNDAY = date(2026, 10, 27), date(2026, 11, 1)
ALL_FEEDS = ["sydneytrains", "metro", "buses"]
HALF_LIFE_MIN = 20
CUTOFFS = {"jobs": 45, "hospital": 45, "education": 45, "supermarket": 20}


def census_needs() -> pd.DataFrame:
    base = "2021 Census GCP Statistical Area 2 for NSW/2021Census_"
    with zipfile.ZipFile(CENSUS_DIR / "2021_GCP_SA2_for_NSW_short-header.zip") as z:
        g01 = pd.read_csv(z.open(base + "G01_NSW_SA2.csv"), dtype={"SA2_CODE_2021": str})
        g46 = pd.read_csv(z.open(base + "G46B_NSW_SA2.csv"), dtype={"SA2_CODE_2021": str})
    pop = g01["Tot_P_P"]
    return pd.DataFrame({
        "SA2_CODE21": g01["SA2_CODE_2021"],
        "pct_65plus": 100 * (g01["Age_65_74_yr_P"] + g01["Age_75_84_yr_P"] + g01["Age_85ov_P"]) / pop,
        "pct_15_24": 100 * (g01["Age_15_19_yr_P"] + g01["Age_20_24_yr_P"]) / pop,
        "unemployment_rate": 100 * g46["P_Tot_Unemp_Tot"] / g46["P_Tot_LF_Tot"],
    })


def z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def main() -> None:
    shapes, jobs, mb, cen, origins = load_inputs()
    d_codes = shapes["SA2_CODE21"].to_numpy()
    jobs_vec = jobs.reindex(d_codes).fillna(0).to_numpy()

    # Destinations per SA2
    pois = pd.read_csv(MARTS_DIR / "pois.csv")
    pg = gpd.GeoDataFrame(pois, geometry=gpd.points_from_xy(pois["x"], pois["y"]), crs=CRS_METRES)
    pj = gpd.sjoin(pg, shapes[["SA2_CODE21", "geometry"]], predicate="within")
    poi_count = pj.groupby(["category", "SA2_CODE21"]).size().unstack(fill_value=0).reindex(columns=d_codes, fill_value=0)

    out = pd.DataFrame({"SA2_CODE21": origins["SA2_CODE21"]}).set_index("SA2_CODE21")

    # Weekday 8am, 60-minute horizon: cumulative + gravity
    mat, o_codes, _ = travel_times(ALL_FEEDS, WEEKDAY, origins, shapes, depart=8 * 3600, budget=60 * 60)
    out["jobs_30"] = ((mat <= 30) * jobs_vec).sum(axis=1)
    out["jobs_45"] = ((mat <= 45) * jobs_vec).sum(axis=1)
    out["jobs_60"] = ((mat <= 60) * jobs_vec).sum(axis=1)
    for cat in ("hospital", "education", "supermarket"):
        out[f"{cat}_{CUTOFFS[cat]}"] = ((mat <= CUTOFFS[cat]) * poi_count.loc[cat].to_numpy()).sum(axis=1)
    beta = np.log(2) / HALF_LIFE_MIN
    decay = np.where(np.isfinite(mat), np.exp(-beta * np.where(np.isfinite(mat), mat, 0)), 0)
    out["gravity_jobs"] = (decay * jobs_vec).sum(axis=1)

    # Temporal coverage: jobs within 45 minutes at three times
    # averaged over departures at :00, :10 and :20, so one lucky or unlucky connection doesn't decide it
    for label, d, hour in (("wk7am", WEEKDAY, 7), ("wk9pm", WEEKDAY, 21), ("sun10am", SUNDAY, 10)):
        runs = []
        for offset in (0, 10, 20):
            m, _, _ = travel_times(ALL_FEEDS, d, origins, shapes, depart=hour * 3600 + offset * 60, budget=45 * 60)
            runs.append(((m <= 45) * jobs_vec).sum(axis=1))
        out[f"jobs45_{label}"] = np.mean(runs, axis=0)
    out = out.reset_index()

    # Residents per SA2 and need inputs
    res = mb.groupby("SA2_CODE21")["residents"].sum().rename("residents")
    ctx = pd.read_csv(MARTS_DIR / "sa2_equity_context.csv", dtype={"SA2_CODE21": str})[
        ["SA2_CODE21", "SA2_NAME21", "SA3_NAME21", "irsd_score", "irsd_decile", "no_car_pct", "ai_all", "erp_2025"]]
    out = out.merge(res, on="SA2_CODE21").merge(ctx, on="SA2_CODE21", how="left").merge(census_needs(), on="SA2_CODE21", how="left")
    syd_avg = np.average(out["gravity_jobs"], weights=out["residents"])
    out["gravity_index"] = (100 * out["gravity_jobs"] / syd_avg).round(1)

    # Transit gap index (Currie 2010), suburbs with 1,000+ residents
    m = out["residents"] >= 1000
    need = pd.concat([z(-out.loc[m, "irsd_score"]), z(out.loc[m, "no_car_pct"]),
                      z(out.loc[m, "pct_65plus"]), z(out.loc[m, "pct_15_24"])], axis=1).mean(axis=1)
    supply = z(np.log1p(out.loc[m, "ai_all"]))
    out.loc[m, "need_z"], out.loc[m, "supply_z"] = need, supply
    out.loc[m, "transit_gap"] = need - supply
    out.to_csv(MARTS_DIR / "sa2_access_methods.csv", index=False)

    w = out["residents"]
    print("Resident-weighted averages:")
    for c in ["jobs_30", "jobs_45", "jobs_60", "hospital_45", "education_45", "supermarket_20",
              "jobs45_wk7am", "jobs45_wk9pm", "jobs45_sun10am"]:
        print(f"  {c:<16} {np.average(out[c], weights=w):>12,.1f}")
    for cat, c in (("hospital", "hospital_45"), ("education", "education_45"), ("supermarket", "supermarket_20")):
        print(f"  residents with no {cat} in reach: {100 * w[out[c] == 0].sum() / w.sum():.1f}%")
    g = out[m].sort_values("transit_gap", ascending=False)
    print(g[["SA2_NAME21", "SA3_NAME21", "residents", "need_z", "supply_z", "transit_gap"]].head(12).round(2).to_string(index=False))
    s = out[m]
    for col in ("jobs_45", "gravity_index", "ai_all"):
        rho = s[col].rank().corr(s["unemployment_rate"].rank())
        print(f"Spearman rho, {col} vs unemployment: {rho:.2f}")
    # partial: rank residuals after removing IRSD
    r = lambda a, b: a.rank() - np.polyval(np.polyfit(b.rank(), a.rank(), 1), b.rank())
    print(f"Partial rho (controlling for IRSD), gravity vs unemployment: "
          f"{r(s['gravity_index'], s['irsd_score']).corr(r(s['unemployment_rate'], s['irsd_score'])):.2f}")


if __name__ == "__main__":
    main()
