"""
Charts for the service equity analysis. Reads the marts written by
service_equity.py and saves PNGs to outputs/figures/.

    1. equity_map.png        SA2 map: AM peak trains per hour a typical resident can walk to
    2. beyond_walking.png    Districts (SA3) with the most residents beyond 1.5 km of a station
    3. lorenz.png            How unevenly service is shared across residents

Usage:
    python -m analysis.plot_equity
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from analysis.service_equity import CENSUS_DIR, CRS_METRES, GREATER_SYDNEY, MARTS_DIR, WALK_M
from tfnsw import PROJECT_DIR

FIG_DIR = PROJECT_DIR / "outputs" / "figures"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NO_SERVICE = "#d9d8d3"  # neutral gray: no station within walking distance
BLUES = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # sequential, light -> dark
BAR = "#2a78d6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def equity_map(sa2: pd.DataFrame, stations: pd.DataFrame) -> None:
    shp = f"zip://{CENSUS_DIR / 'SA2_2021_AUST_SHP_GDA2020.zip'}!SA2_2021_AUST_GDA2020.shp"
    shapes = gpd.read_file(shp, where=f"GCC_CODE21 = '{GREATER_SYDNEY}'").to_crs(CRS_METRES)
    gdf = shapes.merge(sa2.assign(SA2_CODE21=sa2["SA2_CODE21"].astype(str)), on="SA2_CODE21", how="left")

    bins = [0.05, 4, 8, 15, 30, 60, 200]
    labels = ["Under 4", "4 to 8", "8 to 15", "15 to 30", "30 to 60", "60+"]
    cmap = ListedColormap(BLUES)
    norm = BoundaryNorm(bins, cmap.N)

    fig, ax = plt.subplots(figsize=(9, 10))
    served = gdf[gdf["avg_walkable_am_peak_tph"] >= 0.05]
    unserved = gdf[~(gdf["avg_walkable_am_peak_tph"] >= 0.05)]
    unserved.plot(ax=ax, color=NO_SERVICE, edgecolor=SURFACE, linewidth=0.3)
    served.plot(ax=ax, column="avg_walkable_am_peak_tph", cmap=cmap, norm=norm, edgecolor=SURFACE, linewidth=0.3)
    st = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations["lon"], stations["lat"]), crs="EPSG:7844")
    st.to_crs(CRS_METRES).clip(shapes.total_bounds).plot(ax=ax, color=INK, markersize=2)

    # Focus on the urban area; the far rural west and north are mostly unpopulated
    ax.set_xlim(265_000, 360_000)
    ax.set_ylim(6_210_000, 6_330_000)
    ax.set_axis_off()
    handles = [Patch(color=c, label=l) for c, l in zip(BLUES, labels)]
    handles.append(Patch(color=NO_SERVICE, label="No station within 1.5 km"))
    ax.legend(handles=handles, title="AM peak trains per hour", loc="lower right", frameon=False, fontsize=9, title_fontsize=9)
    ax.set_title(
        "Rail service a typical resident can walk to\nAverage AM peak trains per hour at the best station within 1.5 km, by SA2",
        loc="left", fontsize=12,
    )
    fig.text(0.02, 0.02, "Sources: TfNSW GTFS (timetable for Tue 27 Oct 2026), ABS Census 2021. Rail only; buses not included.",
             fontsize=8, color=INK_2)
    fig.savefig(FIG_DIR / "equity_map.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def beyond_walking(sa3: pd.DataFrame) -> None:
    top = sa3.sort_values("residents_beyond_1500m", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["SA3_NAME21"], top["residents_beyond_1500m"] / 1000, color=BAR, height=0.6)
    for y, (v, pct) in enumerate(zip(top["residents_beyond_1500m"], 100 - top["pct_within_1500m"])):
        ax.text(v / 1000 + 2, y, f"{v / 1000:.0f}k  ({pct:.0f}% of residents)", va="center", fontsize=9, color=INK_2)
    ax.set_xlabel("Residents more than 1.5 km from a station (thousands)")
    ax.set_xlim(0, top["residents_beyond_1500m"].max() / 1000 * 1.35)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Districts with the most residents beyond walking distance of rail", loc="left", fontsize=12)
    fig.savefig(FIG_DIR / "beyond_walking.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def lorenz(mb: pd.DataFrame) -> None:
    v = mb[f"daily_{WALK_M}m"].to_numpy()
    w = mb["residents"].to_numpy()
    order = np.argsort(v)
    x = np.concatenate([[0], np.cumsum(w[order]) / w.sum()])
    y = np.concatenate([[0], np.cumsum((v * w)[order]) / (v * w).sum()])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle="--")
    ax.plot(x, y, color=BAR, linewidth=2)
    ax.text(0.62, 0.55, "Equal share", color=INK_2, fontsize=9, rotation=45)
    ax.set_xlabel("Share of Greater Sydney residents (least served first)")
    ax.set_ylabel("Share of walkable train departures")
    ax.set_title("How evenly is rail service shared?", loc="left", fontsize=12)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(FIG_DIR / "lorenz.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sa2 = pd.read_csv(MARTS_DIR / "sa2_service_equity.csv", dtype={"SA2_CODE21": str})
    sa3 = pd.read_csv(MARTS_DIR / "sa3_service_equity.csv")
    stations = pd.read_csv(MARTS_DIR / "station_service.csv")
    mb = pd.read_parquet(MARTS_DIR / "mesh_block_access.parquet")
    equity_map(sa2, stations)
    beyond_walking(sa3)
    lorenz(mb)
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
