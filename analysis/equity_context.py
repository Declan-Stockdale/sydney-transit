"""
Who is let down by poor rail access? Adds disadvantage, car ownership and
up-to-date population to the SA2 service equity results.

Joins to data/marts/sa2_service_equity.csv:
    - SEIFA 2021 IRSD (Index of Relative Socio-economic Disadvantage) score and
      national decile: 1 = most disadvantaged 10%.
    - % of dwellings with no car (Census 2021 table G34).
    - ABS Estimated Resident Population 2021 and latest year (2025), so the
      counts of people beyond walking distance reflect today's population.
      Mesh block shares from 2021 are scaled to the latest ERP.

Then flags "transport-disadvantaged" areas: IRSD decile 1 to 3 OR at least 15%
of dwellings without a car, AND fewer than half of residents within 1.5 km of a
station or under 4 AM peak trains an hour on average.

Outputs: data/marts/sa2_equity_context.csv; prints the largest such areas.

Usage:
    python -m analysis.equity_context
"""

import zipfile

import pandas as pd

from analysis.service_equity import CENSUS_DIR, MARTS_DIR
from tfnsw import DATA_DIR

CONTEXT_DIR = DATA_DIR / "raw" / "context"
NO_CAR_FLAG = 15  # % of dwellings
LOW_DECILE = 3


def seifa() -> pd.DataFrame:
    t = pd.read_excel(CONTEXT_DIR / "seifa_2021_sa2.xlsx", sheet_name="Table 1", header=None, skiprows=6)
    t = t.iloc[:, [0, 2, 3]].dropna()
    t.columns = ["SA2_CODE21", "irsd_score", "irsd_decile"]
    t["SA2_CODE21"] = t["SA2_CODE21"].astype(str).str.replace(r"\.0$", "", regex=True)
    # Areas with too few residents have "-" instead of a score
    t[["irsd_score", "irsd_decile"]] = t[["irsd_score", "irsd_decile"]].apply(pd.to_numeric, errors="coerce")
    return t[t["SA2_CODE21"].str.fullmatch(r"\d{9}")]


def no_car() -> pd.DataFrame:
    with zipfile.ZipFile(CENSUS_DIR / "2021_GCP_SA2_for_NSW_short-header.zip") as z:
        g = pd.read_csv(z.open("2021 Census GCP Statistical Area 2 for NSW/2021Census_G34_NSW_SA2.csv"), dtype={"SA2_CODE_2021": str})
    stated = g["Num_MVs_per_dweling_Tot"]
    return pd.DataFrame({
        "SA2_CODE21": g["SA2_CODE_2021"],
        "no_car_pct": (100 * g["Num_MVs_per_dweling_0_MVs"] / stated).where(stated > 0).round(1),
    })


def erp() -> pd.DataFrame:
    e = pd.read_csv(CONTEXT_DIR / "erp_asgs2021.csv")
    e = e[e["REGION_TYPE: Geography Level"].str.startswith("SA2")]
    e["SA2_CODE21"] = e["ASGS_2021: Region"].str.split(":").str[0].str.strip()
    latest = e["TIME_PERIOD: Time Period"].max()
    piv = e.pivot_table(index="SA2_CODE21", columns="TIME_PERIOD: Time Period", values="OBS_VALUE")
    out = pd.DataFrame({"erp_2021": piv[2021], f"erp_{latest}": piv[latest]}).reset_index()
    out["erp_growth_pct"] = (100 * (out[f"erp_{latest}"] / out["erp_2021"] - 1)).round(1)
    out.attrs["latest"] = latest
    return out


def tzp_forecast(year: int = 2041) -> pd.DataFrame:
    """TfNSW TZP24 projected population per SA2 (travel zones carry their SA2 code)."""
    header = pd.read_csv(CONTEXT_DIR / "tzp24_population_by_tz.csv", nrows=0).columns
    cols = [c for c in header if c.endswith(f"_{year}")]
    t = pd.read_csv(CONTEXT_DIR / "tzp24_population_by_tz.csv", usecols=["SA2_CODE21", *cols], dtype={"SA2_CODE21": str})
    t[f"tzp_pop_{year}"] = t[cols].sum(axis=1)
    return t.groupby("SA2_CODE21")[f"tzp_pop_{year}"].sum().round(0).reset_index()


def main() -> None:
    sa2 = pd.read_csv(MARTS_DIR / "sa2_service_equity.csv", dtype={"SA2_CODE21": str})
    pop = erp()
    latest = pop.attrs["latest"]
    out = (
        sa2.merge(seifa(), on="SA2_CODE21", how="left")
        .merge(no_car(), on="SA2_CODE21", how="left")
        .merge(pop, on="SA2_CODE21", how="left")
        .merge(tzp_forecast(), on="SA2_CODE21", how="left")
    )
    out["growth_to_2041"] = (out["tzp_pop_2041"] - out[f"erp_{latest}"]).round(0)

    # Scale 2021 mesh-block counts to today's population
    scale = out[f"erp_{latest}"] / out["residents"]
    out[f"residents_beyond_1500m_{latest}"] = (out["residents_beyond_1500m"] * scale).round(0)

    needy = (out["irsd_decile"] <= LOW_DECILE) | (out["no_car_pct"] >= NO_CAR_FLAG)
    poor_rail = (out["pct_within_1500m"] < 50) | (out["avg_walkable_am_peak_tph"] < 4)
    out["transport_disadvantaged"] = needy & poor_rail & (out["residents"] >= 2000)

    # With buses (access_index.py, jobs_access.py): poor = most residents at PTAL 0 to 1b
    ptal_path, jobs_path = MARTS_DIR / "sa2_ptal.csv", MARTS_DIR / "sa2_jobs_access.csv"
    if ptal_path.exists():
        ptal = pd.read_csv(ptal_path, dtype={"SA2_CODE21": str}).drop(columns=["SA2_NAME21", "residents"])
        out = out.merge(ptal, on="SA2_CODE21", how="left")
        out["transport_disadvantaged_with_bus"] = needy & (out["pct_poor_all"] >= 50) & (out["residents"] >= 2000)
    if jobs_path.exists():
        out = out.merge(pd.read_csv(jobs_path, dtype={"SA2_CODE21": str}), on="SA2_CODE21", how="left")
    out.to_csv(MARTS_DIR / "sa2_equity_context.csv", index=False)

    td = out[out["transport_disadvantaged"]].sort_values(f"erp_{latest}", ascending=False)
    print(f"{len(td)} transport-disadvantaged SA2s, {td[f'erp_{latest}'].sum():,.0f} people ({latest} ERP)")
    cols = ["SA2_NAME21", "SA3_NAME21", f"erp_{latest}", "erp_growth_pct", "irsd_decile", "no_car_pct",
            "pct_within_1500m", "avg_walkable_am_peak_tph"]
    print(td[cols].head(20).to_string(index=False))
    print("\nFastest-growing SA2s since 2021 (people beyond 1.5 km today):")
    g = out[out["erp_2021"] >= 1000].sort_values("erp_growth_pct", ascending=False)
    print(g[["SA2_NAME21", "erp_2021", f"erp_{latest}", "erp_growth_pct", "pct_within_1500m", f"residents_beyond_1500m_{latest}"]].head(12).to_string(index=False))
    print("\nMost projected growth to 2041 in SA2s where under half of residents are near rail:")
    f = out[out["pct_within_1500m"] < 50].sort_values("growth_to_2041", ascending=False)
    print(f[["SA2_NAME21", "SA3_NAME21", f"erp_{latest}", "tzp_pop_2041", "growth_to_2041", "pct_within_1500m"]].head(12).to_string(index=False))
    print(f"Total projected growth to 2041 in such SA2s: {f['growth_to_2041'].clip(lower=0).sum():,.0f}; "
          f"all Greater Sydney SA2s: {out['growth_to_2041'].clip(lower=0).sum():,.0f}")
    if "transport_disadvantaged_with_bus" in out:
        tb = out[out["transport_disadvantaged_with_bus"]].sort_values(f"erp_{latest}", ascending=False)
        print(f"\nWith buses counted: {len(tb)} transport-disadvantaged SA2s, {tb[f'erp_{latest}'].sum():,.0f} people")
        print(tb[["SA2_NAME21", "SA3_NAME21", f"erp_{latest}", "irsd_decile", "no_car_pct", "pct_poor_rail",
                  "pct_poor_all", "jobs_rail", "jobs_all"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
