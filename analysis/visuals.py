"""
The visual gallery: 15 charts and maps built from the marts and raw data.

Each function writes one file to outputs/figures/ (PNG, or GIF for the
timelapse). Charts marked PREVIEW use the realtime, weather and car park data
collected so far; rerun them once a few weeks are in.

    python -m analysis.visuals              # all
    python -m analysis.visuals marey heatmap

Run first: station_service, service_equity, walk_access, equity_context,
demand_supply, catchment, reliability, opal_report (see README).
"""

import sys
import zipfile
from datetime import date, datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Polygon
from PIL import Image
from zoneinfo import ZoneInfo

from analysis.demand_supply import PATRONAGE_DIR
from analysis.opal_report import BEFORE, CLOSED, all_monthly_entries
from analysis.plot_equity import BAR, BLUES, FIG_DIR, GRID, INK, INK_2, NO_SERVICE, SURFACE
from analysis.service_equity import CENSUS_DIR, CRS_METRES, GREATER_SYDNEY, MARTS_DIR
from analysis.station_service import active_service_ids, gtfs_seconds, read
from tfnsw import DATA_DIR

SYDNEY = ZoneInfo("Australia/Sydney")
SERVICE_DATE = date(2026, 10, 27)  # timetable date used for supply-side charts
METRO = "#168388"  # Sydney Metro's own route colour, used wherever mode is the point
TRAINS = BAR
EXTENT = (265_000, 360_000, 6_210_000, 6_330_000)  # urban Sydney in EPSG:7856
DARK_BG, DARK_INK = "#0e1116", "#c9d1d9"


def save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / name, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved", name)


def preview_note(ax, text: str) -> None:
    ax.text(0.0, -0.08, f"PREVIEW: {text}", transform=ax.transAxes, fontsize=8, color="#b5470b", va="top")


def to_m(lon, lat):
    from pyproj import Transformer
    return Transformer.from_crs("EPSG:4326", CRS_METRES, always_xy=True).transform(lon, lat)


# --- shared data -------------------------------------------------------------

def sa2_shapes() -> gpd.GeoDataFrame:
    shp = f"zip://{CENSUS_DIR / 'SA2_2021_AUST_SHP_GDA2020.zip'}!SA2_2021_AUST_GDA2020.shp"
    return gpd.read_file(shp, where=f"GCC_CODE21 = '{GREATER_SYDNEY}'").to_crs(CRS_METRES)


def stations() -> pd.DataFrame:
    st = pd.read_csv(MARTS_DIR / "station_service.csv")
    st["x"], st["y"] = to_m(st["lon"].to_numpy(), st["lat"].to_numpy())
    st["mode"] = np.where(st["networks"].str.contains("sydneytrains"), "Trains", "Metro")
    st.loc[st["networks"] == "metro sydneytrains", "mode"] = "Both"
    return st


def trip_stops(service_date: date = SERVICE_DATE) -> pd.DataFrame:
    """Every scheduled boardable stop of every trip on the date, in order, with station coordinates."""
    frames = []
    for feed in ("sydneytrains", "metro"):
        routes = read(feed, "routes", usecols=["route_id", "route_short_name", "agency_id"])
        if feed == "sydneytrains":
            routes = routes[(routes["agency_id"] == "SydneyTrains") | routes["route_short_name"].isin({"BMT", "CCN", "SCO", "SHL"})]
        trips = read(feed, "trips", usecols=["route_id", "service_id", "trip_id"])
        trips = trips[trips["service_id"].isin(active_service_ids(feed, service_date))].merge(routes, on="route_id")
        stt = read(feed, "stop_times", usecols=["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time", "pickup_type"])
        stt = stt[stt["trip_id"].isin(set(trips["trip_id"]))]
        stops = read(feed, "stops", usecols=["stop_id", "stop_name", "parent_station"])
        parent = stops["parent_station"].fillna("")
        stops["station_id"] = parent.where(parent != "", stops["stop_id"])
        names = stops.set_index("stop_id")["stop_name"]
        stt["station"] = stt["stop_id"].map(stops.set_index("stop_id")["station_id"]).map(names)
        stt["station"] = stt["station"].str.replace(r"\s+Station$", "", regex=True)
        stt = stt.merge(trips[["trip_id", "route_short_name"]], on="trip_id")
        stt["seq"] = stt["stop_sequence"].astype(int)
        stt["arr_s"] = gtfs_seconds(stt["arrival_time"].fillna(stt["departure_time"]))
        stt["dep_s"] = gtfs_seconds(stt["departure_time"].fillna(stt["arrival_time"]))
        stt["mode"] = "Metro" if feed == "metro" else "Trains"
        frames.append(stt)
    ts = pd.concat(frames).sort_values(["trip_id", "seq"])
    xy = stations().set_index("station")[["x", "y"]]
    return ts.join(xy, on="station").dropna(subset=["x"])


def segments(ts: pd.DataFrame, start_h=6.5, end_h=10) -> pd.DataFrame:
    """Undirected station-to-station segments with AM peak trains per hour, by mode."""
    nxt = ts.groupby("trip_id").shift(-1)
    seg = pd.DataFrame({"a": ts["station"], "b": nxt["station"], "dep_s": ts["dep_s"], "mode": ts["mode"],
                        "ax": ts["x"], "ay": ts["y"], "bx": nxt["x"], "by": nxt["y"]}).dropna()
    seg = seg[seg["a"] != seg["b"]]
    swap = seg["a"] > seg["b"]
    seg.loc[swap, ["a", "b", "ax", "ay", "bx", "by"]] = seg.loc[swap, ["b", "a", "bx", "by", "ax", "ay"]].to_numpy()
    am = seg["dep_s"].between(start_h * 3600, end_h * 3600)
    out = seg[am].groupby(["a", "b", "mode"]).agg(trains=("dep_s", "size"), ax=("ax", "first"), ay=("ay", "first"),
                                                  bx=("bx", "first"), by=("by", "first")).reset_index()
    out["tph"] = out["trains"] / (end_h - start_h)
    return out


def draw_network(ax, seg, color="#9aa4b1", lw=0.6, alpha=0.8):
    lines = [[(r.ax, r.ay), (r.bx, r.by)] for r in seg.itertuples()]
    ax.add_collection(LineCollection(lines, colors=color, linewidths=lw, alpha=alpha, zorder=2))


def frame_axes(ax, extent=EXTENT):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_axis_off()


def delays() -> pd.DataFrame:
    d = pd.read_parquet(MARTS_DIR / "stop_delays.parquet")
    d = d[~d["cancelled"] & d["delay_s"].notna() & d["station"].notna()].copy()
    d["mode"] = np.where(d["feed"] == "metro", "Metro", "Trains")
    d["late_min"] = d["delay_s"] / 60
    return d


# --- 1. bivariate disadvantage x access ---------------------------------------

def bivariate() -> None:
    ctx = pd.read_csv(MARTS_DIR / "sa2_equity_context.csv", dtype={"SA2_CODE21": str})
    shapes = sa2_shapes().merge(ctx, on="SA2_CODE21", how="left")
    # Disadvantage: IRSD decile 1-3 high, 4-7 mid, 8-10 low. Access: walkable AM trains/hour
    shapes["dis"] = pd.cut(shapes["irsd_decile"], [0, 3, 7, 10], labels=[2, 1, 0]).astype(float)
    shapes["acc"] = pd.cut(shapes["avg_walkable_am_peak_tph"], [-0.1, 4, 12, 1000], labels=[2, 1, 0]).astype(float)
    # 3x3 palette: rows = disadvantage (low->high), cols = poor access (low->high)
    pal = [["#e8e8e8", "#e4acac", "#c85a5a"],
           ["#b0d5df", "#ad9ea5", "#985356"],
           ["#64acbe", "#627f8c", "#574249"]]
    shapes["color"] = [pal[int(d)][int(a)] if not (np.isnan(d) or np.isnan(a)) else "#f4f4f2"
                       for d, a in zip(shapes["dis"], shapes["acc"])]
    fig, ax = plt.subplots(figsize=(9, 10))
    shapes.plot(ax=ax, color=shapes["color"], edgecolor=SURFACE, linewidth=0.3)
    frame_axes(ax)
    ax.set_title("Disadvantage meets poor rail access\nSA2s by IRSD disadvantage and walkable AM peak trains per hour",
                 loc="left", fontsize=12)
    # legend grid
    lax = fig.add_axes([0.66, 0.16, 0.16, 0.16])
    for i in range(3):
        for j in range(3):
            lax.add_patch(plt.Rectangle((j, i), 1, 1, color=pal[i][j]))
    lax.set_xlim(0, 3); lax.set_ylim(0, 3); lax.set_xticks([]); lax.set_yticks([])
    lax.set_xlabel("Poorer rail access →", fontsize=8, color=INK_2)
    lax.set_ylabel("More disadvantaged →", fontsize=8, color=INK_2)
    for s in lax.spines.values():
        s.set_visible(False)
    fig.text(0.02, 0.03, "Darkest = most disadvantaged (IRSD decile 1-3) and under 4 walkable AM trains/hour. "
             "Sources: ABS SEIFA 2021, TfNSW GTFS.", fontsize=8, color=INK_2)
    save(fig, "v01_bivariate_disadvantage_access.png")


# --- 2. Marey / string diagrams --------------------------------------------------

def marey(line: str = "T1", through: str = "Penrith", fname: str = "v02_marey_t1_western.png", title=None) -> None:
    d = delays()
    ts = trip_stops(date.fromisoformat(pd.to_datetime(d["service_date"].max()).strftime("%Y-%m-%d")))
    cand = ts[(ts["route_short_name"] == line)]
    trip_has = cand.groupby("trip_id")["station"].apply(set)
    long_trip = trip_has[trip_has.apply(lambda s: through in s)].apply(len).idxmax()
    order = ts[ts["trip_id"] == long_trip].sort_values("seq")
    xy = order[["x", "y"]].to_numpy()
    km = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]) / 1000
    pos = dict(zip(order["station"], km))

    obs = d[(d["route_short_name"] == line) & d["station"].isin(pos)].copy()
    obs["km"] = obs["station"].map(pos)
    obs["t"] = pd.to_datetime(obs["obs_time"], unit="s", utc=True).dt.tz_convert(SYDNEY)
    obs["s"] = pd.to_datetime(obs["sched_time"], unit="s", utc=True).dt.tz_convert(SYDNEY)
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.get_cmap("RdYlBu_r")
    for _, g in obs.groupby("trip_id"):
        if len(g) < 3:
            continue
        g = g.sort_values("s")
        ax.plot(g["s"], g["km"], color="#c8c8c4", linewidth=0.7, zorder=1)
        late = g["late_min"].clip(-2, 10).mean()
        ax.plot(g["t"], g["km"], color=cmap((late + 2) / 12), linewidth=1.3, zorder=2)
    ax.set_yticks(list(pos.values())[::2])
    ax.set_yticklabels(list(pos.keys())[::2], fontsize=7)
    ax.set_ylabel("")
    ax.grid(axis="x", color=GRID)
    ax.set_title(title or f"{line}: every train today, scheduled (grey) vs measured (coloured by average lateness)",
                 loc="left", fontsize=12)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-2, 10))
    fig.colorbar(sm, ax=ax, shrink=0.6, label="Average minutes late")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=SYDNEY))
    preview_note(ax, f"one day ({obs['t'].min():%d %b}); times from GPS positions, or the last prediction where no GPS fix matched")
    save(fig, fname)


# --- 3. frequency network map ------------------------------------------------------

def frequency_network() -> None:
    seg = segments(trip_stops())
    fig, ax = plt.subplots(figsize=(10, 11), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    shapes = sa2_shapes()
    shapes.plot(ax=ax, color="#151a21", edgecolor="#1c222b", linewidth=0.3)
    for mode, color in (("Trains", "#f2c14e"), ("Metro", "#4fd1c5")):
        s = seg[seg["mode"] == mode].sort_values("tph")
        for glow, alpha in ((6, 0.06), (3, 0.15), (1, 0.9)):
            lines = [[(r.ax, r.ay), (r.bx, r.by)] for r in s.itertuples()]
            widths = np.clip(s["tph"] / 6, 0.4, 8) * glow
            ax.add_collection(LineCollection(lines, colors=color, linewidths=widths, alpha=alpha, zorder=3, capstyle="round"))
    frame_axes(ax)
    ax.set_title("How often trains run: AM peak trains per hour on each section of track",
                 loc="left", fontsize=12, color=DARK_INK)
    ax.legend(handles=[Patch(color="#f2c14e", label="Sydney Trains + intercity"), Patch(color="#4fd1c5", label="Sydney Metro")],
              loc="lower right", frameon=False, labelcolor=DARK_INK, fontsize=9)
    fig.text(0.02, 0.03, "Line width = trains per hour, both directions, 6:30 to 10am on Tue 27 Oct 2026. Source: TfNSW GTFS.",
             fontsize=8, color=DARK_INK)
    save(fig, "v03_frequency_network.png")


# --- 4. ridership timeline ------------------------------------------------------------

def ridership_timeline() -> None:
    m = all_monthly_entries()
    total = m.groupby("month")["trips"].sum() / 1e6
    closed = m[m["station"].isin(CLOSED)].groupby("month")["trips"].sum() / 1e6
    metro_new = m[m["station"].isin(["Gadigal", "Victoria Cross", "Crows Nest", "Barangaroo", "Waterloo"])].groupby("month")["trips"].sum() / 1e6
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(total.index, total, color=BAR, linewidth=2)
    ax1.set_ylabel("Station entries per month (millions)")
    ax1.set_ylim(0, total.max() * 1.15)
    for when, text, ha, level in (("2020-03-01", "COVID lockdowns ", "right", 1.10), ("2021-07-01", " Delta lockdown", "left", 1.10),
                                  ("2024-08-01", "Metro City opens ", "right", 1.10), ("2024-09-30", " Sydenham-Bankstown closes", "left", 1.03)):
        ax1.axvline(pd.Timestamp(when), color=INK_2, linewidth=0.8, linestyle="--")
        ax1.text(pd.Timestamp(when), total.max() * level, text, fontsize=8, color=INK_2, ha=ha)
    ax1.set_title("Sydney rail station entries, July 2016 to September 2026", loc="left", fontsize=12)
    ax2.plot(closed.index, closed, color="#b5470b", linewidth=1.8, label="9 Sydenham-Bankstown stations")
    ax2.plot(metro_new.index, metro_new, color=METRO, linewidth=1.8, label="5 new Metro City stations")
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_ylabel("Millions per month")
    for a in (ax1, ax2):
        a.grid(color=GRID)
    fig.text(0.02, 0.0, "Train and Metro stations; light rail excluded. Source: TfNSW station entries and exits (Opal).",
             fontsize=8, color=INK_2)
    save(fig, "v04_ridership_timeline.png")


# --- 5. growth vs rail ---------------------------------------------------------------------

def growth_vs_rail() -> None:
    ctx = pd.read_csv(MARTS_DIR / "sa2_equity_context.csv", dtype={"SA2_CODE21": str})
    shapes = sa2_shapes()
    pts = shapes.merge(ctx, on="SA2_CODE21")
    pts["geometry"] = pts.geometry.representative_point()
    pts = pts[pts["growth_to_2041"] > 2000]
    seg = segments(trip_stops())
    fig, ax = plt.subplots(figsize=(9, 10))
    shapes.plot(ax=ax, color="#f1f0ec", edgecolor=SURFACE, linewidth=0.3)
    draw_network(ax, seg, color=INK, lw=0.9)
    near = pts["pct_within_1500m"] >= 50
    for mask, color, label in ((near, "#9aa4b1", "Half or more of residents near rail"),
                               (~near, "#c85a5a", "Under half near rail")):
        p = pts[mask]
        ax.scatter(p.geometry.x, p.geometry.y, s=p["growth_to_2041"] / 150, color=color, alpha=0.65,
                   edgecolor="white", linewidth=0.5, label=label, zorder=3)
    frame_axes(ax, (255_000, 360_000, 6_200_000, 6_330_000))
    ax.legend(loc="lower right", frameon=False, fontsize=9, title="Projected extra residents 2025 to 2041\n(bubble size)",
              title_fontsize=8)
    ax.set_title("Where Sydney is growing, and where the trains go\nTfNSW TZP24 projected growth by SA2 over the rail network",
                 loc="left", fontsize=12)
    save(fig, "v05_growth_vs_rail.png")


# --- 6. walking detours ----------------------------------------------------------------------

def walking_detours() -> None:
    walk = pd.read_parquet(MARTS_DIR / "mesh_block_walk.parquet")
    mb = pd.read_parquet(MARTS_DIR / "mesh_block_access.parquet")[["MB_CODE21", "nearest_station"]]
    w = walk.merge(mb, on="MB_CODE21")
    w = w[(w["nearest_m"].between(300, 1500)) & w["walk_m"].notna()]
    w["ratio"] = w["walk_m"] / w["nearest_m"]
    agg = w.groupby("nearest_station").apply(
        lambda g: pd.Series({"ratio": np.average(g["ratio"], weights=g["residents"]), "residents": g["residents"].sum()}),
        include_groups=False).reset_index()
    agg = agg[agg["residents"] >= 1000]
    st = stations().merge(agg, left_on="station", right_on="nearest_station")
    fig, ax = plt.subplots(figsize=(9, 10))
    sa2_shapes().plot(ax=ax, color="#f1f0ec", edgecolor=SURFACE, linewidth=0.3)
    draw_network(ax, segments(trip_stops()), color="#9aa4b1", lw=0.7)
    norm = plt.Normalize(1.1, 1.8)
    sc = ax.scatter(st["x"], st["y"], c=st["ratio"], cmap="YlOrRd", norm=norm, s=st["residents"] / 400 + 8,
                    edgecolor="white", linewidth=0.4, zorder=3)
    for r in st.nlargest(8, "ratio").itertuples():
        ax.annotate(f"{r.station} ({r.ratio:.1f}x)", (r.x, r.y), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK)
    frame_axes(ax)
    fig.colorbar(sc, ax=ax, shrink=0.5, label="Walk distance / straight-line distance")
    ax.set_title("Where the walk to the station is much longer than it looks\nResident-weighted detour ratio, homes 300 m to 1.5 km away",
                 loc="left", fontsize=12)
    save(fig, "v06_walking_detours.png")


# --- 7. delay timelapse ---------------------------------------------------------------------------

def timelapse(step_min: int = 5) -> None:
    from analysis.reliability import vehicle_points
    d = delays()
    frames_vp = []
    for feed, mode in (("sydneytrains", "Trains"), ("metro", "Metro")):
        vp = vehicle_points(feed, None)
        vp["mode"] = mode
        frames_vp.append(vp)
    vp = pd.concat(frames_vp).sort_values("vp_ts")
    # lateness of each train = its most recent measured delay at or before the fix
    dd = d[["trip_id", "obs_time", "late_min"]].dropna().astype({"obs_time": "int64"}).sort_values("obs_time")
    vp["vp_ts"] = vp["vp_ts"].astype("int64")
    vp = pd.merge_asof(vp, dd.rename(columns={"obs_time": "vp_ts"}), on="vp_ts", by="trip_id", direction="backward")
    vp["late_min"] = vp["late_min"].fillna(0)
    seg = segments(trip_stops())
    # A few vehicles report stale GPS clocks (days old), so take the window from the bulk of fixes
    t0, t1 = vp["vp_ts"].quantile(0.02), vp["vp_ts"].quantile(0.999)
    times = np.arange(t0 + 60, t1, step_min * 60)
    cmap = plt.get_cmap("RdYlBu_r")
    fig, ax = plt.subplots(figsize=(8, 8.5))
    draw_network(ax, seg, color="#c8c8c4", lw=0.6)
    frame_axes(ax)
    scat = ax.scatter([], [], s=18, edgecolor=INK, linewidth=0.3, zorder=3)
    title = ax.set_title("", loc="left", fontsize=11)
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-2, 10)), ax=ax, shrink=0.5, label="Minutes late")

    def update(t):
        now = vp[(vp["vp_ts"] > t - 90) & (vp["vp_ts"] <= t)].drop_duplicates("trip_id", keep="last")
        scat.set_offsets(np.c_[now["x"], now["y"]])
        scat.set_color(cmap((now["late_min"].clip(-2, 10) + 2) / 12))
        stamp = datetime.fromtimestamp(t, SYDNEY)
        title.set_text(f"Every train in Sydney, coloured by lateness  {stamp:%a %d %b %H:%M}  ({len(now)} trains)")
        return scat, title

    anim = animation.FuncAnimation(fig, update, frames=times, blit=False)
    anim.save(FIG_DIR / "v07_delay_timelapse.gif", writer=animation.PillowWriter(fps=6), dpi=90)
    update(times[len(times) // 2])
    save(fig, "v07_delay_timelapse_still.png")


# --- 8. car park fill curves (PREVIEW) ---------------------------------------------------------------

def carpark_curves() -> None:
    import gzip
    import json
    rows = []
    for f in sorted((DATA_DIR / "raw" / "carpark").glob("*/*.json.gz")):
        snap = json.loads(gzip.decompress(f.read_bytes()))
        t = pd.Timestamp(snap["fetched_utc"]).tz_convert(SYDNEY)
        for fac in snap["facilities"]:
            spots = pd.to_numeric(fac.get("spots"), errors="coerce")
            occ = pd.to_numeric((fac.get("occupancy") or {}).get("total"), errors="coerce")
            if spots and spots > 0 and pd.notna(occ):
                rows.append((t, fac["facility_name"].replace("Park&Ride - ", ""), 100 * min(occ / spots, 1)))
    df = pd.DataFrame(rows, columns=["t", "carpark", "pct_full"])
    fig, ax = plt.subplots(figsize=(10, 6))
    if df.empty:
        ax.text(0.5, 0.5, "No car park data yet", ha="center")
    else:
        top = df.groupby("carpark")["pct_full"].max().nlargest(5).index
        for name, g in df.groupby("carpark"):
            hi = name in top
            ax.plot(g["t"], g["pct_full"], color=BAR if hi else "#d5d4cf", linewidth=1.6 if hi else 0.8, marker="o" if len(g) < 5 else None)
            if hi:
                ax.text(g["t"].iloc[-1], g["pct_full"].iloc[-1], " " + name, fontsize=7, color=INK_2, va="center")
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M", tz=SYDNEY))
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of spaces full")
    ax.grid(color=GRID)
    ax.set_title("When do Park&Ride car parks fill up?", loc="left", fontsize=12)
    preview_note(ax, f"{df['t'].nunique() if not df.empty else 0} snapshots so far; weekday morning fill curves need a few days of collection")
    save(fig, "v08_carpark_fill.png")


# --- 9. rain radar over delays (PREVIEW) ---------------------------------------------------------------

# BOM radar colour scale -> rain rate (mm/h), lower bound of each class
BOM_RAIN = {(245, 245, 255): 0.2, (180, 180, 255): 0.5, (120, 120, 255): 1.5, (20, 20, 255): 2.5,
            (0, 216, 195): 4, (0, 150, 144): 6, (0, 102, 102): 10, (255, 255, 0): 15, (255, 200, 0): 20,
            (255, 150, 0): 35, (255, 100, 0): 50, (255, 0, 0): 80, (200, 0, 0): 120, (120, 0, 0): 200, (40, 0, 0): 300}
RADAR_LAT, RADAR_LON, RADAR_KM_PER_PX = -33.701, 151.210, 0.5  # Terrey Hills, IDR713 (128 km range, 512 px)


def radar_rain(path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im)
    rain = np.zeros(arr.shape[:2])
    for rgb, mm in BOM_RAIN.items():
        rain[(arr == rgb).all(axis=2)] = mm
    return rain


def rain_radar() -> None:
    files = sorted((DATA_DIR / "raw" / "weather" / "radar" / "IDR713").glob("20*/*.png"))
    st = stations()
    cx, cy = to_m(RADAR_LON, RADAR_LAT)
    st["px"] = (256 + (st["x"] - cx) / 1000 / RADAR_KM_PER_PX).round().astype(int).clip(0, 511)
    st["py"] = (256 - (st["y"] - cy) / 1000 / RADAR_KM_PER_PX).round().astype(int).clip(0, 511)
    rows, max_rain = [], None
    for f in files:
        r = radar_rain(f)
        max_rain = r if max_rain is None else np.maximum(max_rain, r)
        t = pd.Timestamp(datetime.strptime(f.stem, "%Y%m%d%H%M"), tz="UTC")
        win = np.array([r[max(py - 2, 0):py + 3, max(px - 2, 0):px + 3].max() for px, py in zip(st["px"], st["py"])])
        rows.append(pd.DataFrame({"station": st["station"], "t": t, "rain": win}))
    rain = pd.concat(rows) if rows else pd.DataFrame(columns=["station", "t", "rain"])
    d = delays()
    d["t"] = pd.to_datetime(d["obs_time"], unit="s", utc=True).dt.floor("5min")
    rain["t"] = rain["t"].dt.floor("5min")
    j = d.merge(rain, on=["station", "t"], how="inner")
    j["wet"] = np.where(j["rain"] >= 0.5, "Raining at station", "Dry")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.3, 1]})
    if max_rain is not None:
        half = 256 * RADAR_KM_PER_PX * 1000
        ax1.imshow(np.ma.masked_less(max_rain, 0.2), extent=(cx - half, cx + half, cy - half, cy + half),
                   cmap="Blues", norm=plt.Normalize(0, 10), zorder=1)
    draw_network(ax1, segments(trip_stops()), color=INK, lw=0.6)
    frame_axes(ax1)
    ax1.set_title("Heaviest radar rain rate in the period, over the rail network", loc="left", fontsize=11)
    summary = j.groupby("wet")["late_min"].agg(["mean", "count"])
    ax2.bar(summary.index, summary["mean"], color=[NO_SERVICE if i == "Dry" else BAR for i in summary.index], width=0.5)
    for i, (m, n) in enumerate(zip(summary["mean"], summary["count"])):
        ax2.text(i, m, f"{m:.1f} min\n(n={n:,})", ha="center", va="bottom", fontsize=9, color=INK_2)
    ax2.set_ylabel("Average minutes late")
    ax2.grid(axis="y", color=GRID)
    ax2.set_title("Train delays when it's raining vs dry", loc="left", fontsize=11)
    preview_note(ax2, f"{len(files)} radar frames so far; needs several wet days to compare fairly (by line and hour)")
    save(fig, "v09_rain_radar_delays.png")


# --- 10. delay heatmap (PREVIEW) ----------------------------------------------------------------------------

def delay_heatmap() -> None:
    d = delays()
    d["hour"] = pd.to_datetime(d["sched_time"], unit="s", utc=True).dt.tz_convert(SYDNEY).dt.hour
    keep = d["route_short_name"].value_counts()
    d = d[d["route_short_name"].isin(keep[keep >= 30].index)]
    piv = d.pivot_table(index="route_short_name", columns="hour", values="late_min", aggfunc="mean")
    n = d.pivot_table(index="route_short_name", columns="hour", values="late_min", aggfunc="size")
    piv = piv.where(n >= 10)  # hide cells with too few train-stops to mean anything
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(piv.clip(-2, 8), cmap="RdYlBu_r", norm=plt.Normalize(-2, 8), aspect="auto")
    ax.set_yticks(range(len(piv)), piv.index)
    ax.set_xticks(range(len(piv.columns)), [f"{h}:00" for h in piv.columns], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Average minutes late")
    ax.set_title("Average lateness by line and hour (M1 = Metro)", loc="left", fontsize=12)
    preview_note(ax, "one day of data; peaks and weekly patterns need a few weeks")
    save(fig, "v10_delay_heatmap.png")


# --- 11. jobs reachable within 45 minutes ---------------------------------------------------------------

def jobs_reach(depart_h: float = 8.0, budget_min: int = 45) -> None:
    """Earliest-arrival search (connection scan) over the timetable from every station,
    departing 8:00am, then counts jobs (2021 Census place of work) at reachable stations."""
    ts = trip_stops()
    nxt = ts.groupby("trip_id").shift(-1)
    con = pd.DataFrame({"from": ts["station"], "to": nxt["station"], "dep": ts["dep_s"], "arr": nxt["arr_s"],
                        "trip": ts["trip_id"]}).dropna()
    t0 = depart_h * 3600
    con = con[(con["dep"] >= t0) & (con["dep"] <= t0 + budget_min * 60)].sort_values("dep")
    names = sorted(set(ts["station"]))
    idx = {s: i for i, s in enumerate(names)}
    c_from, c_to = con["from"].map(idx).to_numpy(), con["to"].map(idx).to_numpy()
    c_dep, c_arr = con["dep"].to_numpy(), con["arr"].to_numpy()
    c_trip = pd.factorize(con["trip"])[0]
    TRANSFER = 180  # seconds allowed to change trains

    # Jobs per station: each SA2's jobs split across the stations inside it
    with zipfile.ZipFile(DATA_DIR / "raw" / "context" / "2021_WPP_SA2_for_NSW_short-header.zip") as z:
        w = pd.read_csv(z.open("2021 Census WPP Statistical Area 2 for NSW/2021Census_W01A_NSW_POW_SA2.csv"), dtype={"POW_SA2_CODE_2021": str})
    tot_cols = [c for c in w.columns if c.startswith("P_") and c.endswith("_Tot")]
    jobs = pd.Series(w[tot_cols].sum(axis=1).to_numpy(), index=w["POW_SA2_CODE_2021"])
    st = stations()
    shapes = sa2_shapes()
    sp = gpd.sjoin(gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st["x"], st["y"]), crs=CRS_METRES),
                   shapes[["SA2_CODE21", "geometry"]], predicate="within")
    per = sp.groupby("SA2_CODE21")["station"].transform("count")
    sp["jobs"] = sp["SA2_CODE21"].map(jobs).fillna(0) / per
    station_jobs = sp.groupby("station")["jobs"].sum().reindex(names).fillna(0).to_numpy()

    reach_jobs = {}
    for origin in names:
        best = np.full(len(names), np.inf)
        best[idx[origin]] = t0
        on_trip = np.zeros(c_trip.max() + 1, dtype=bool)
        for k in range(len(c_dep)):
            tr = c_trip[k]
            if on_trip[tr] or best[c_from[k]] + (0 if c_from[k] == idx[origin] else TRANSFER) <= c_dep[k]:
                on_trip[tr] = True
                if c_arr[k] < best[c_to[k]]:
                    best[c_to[k]] = c_arr[k]
        ok = best <= t0 + budget_min * 60
        reach_jobs[origin] = station_jobs[ok].sum()
    rj = pd.Series(reach_jobs, name="jobs_45min")
    rj.to_csv(MARTS_DIR / "station_jobs_reach.csv")

    ctx = pd.read_csv(MARTS_DIR / "sa2_equity_context.csv", dtype={"SA2_CODE21": str})
    ctx["jobs_45min"] = ctx["main_nearest_station"].map(rj)
    ctx.loc[ctx["pct_within_1500m"] < 50, "jobs_45min"] = 0
    g = shapes.merge(ctx[["SA2_CODE21", "jobs_45min"]], on="SA2_CODE21", how="left")
    bins = [1, 100_000, 300_000, 600_000, 900_000, 1_200_000, 5_000_000]
    fig, ax = plt.subplots(figsize=(9, 10))
    g[~(g["jobs_45min"] > 0)].plot(ax=ax, color=NO_SERVICE, edgecolor=SURFACE, linewidth=0.3)
    g[g["jobs_45min"] > 0].plot(ax=ax, column="jobs_45min", cmap=ListedColormap(BLUES), norm=BoundaryNorm(bins, 6),
                                edgecolor=SURFACE, linewidth=0.3)
    frame_axes(ax)
    labels = ["Under 100k", "100k-300k", "300k-600k", "600k-900k", "900k-1.2m", "1.2m+"]
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(BLUES, labels)] + [Patch(color=NO_SERVICE, label="Most residents not near rail")],
              title="Jobs reachable", loc="lower right", frameon=False, fontsize=9, title_fontsize=9)
    ax.set_title(f"Jobs reachable by rail within {budget_min} minutes, leaving 8am\nFrom each suburb's main station; includes waiting, 3 min to change trains",
                 loc="left", fontsize=12)
    fig.text(0.02, 0.03, "Jobs = 2021 Census place of work, by SA2, split across its stations. Walking to the first station not counted.",
             fontsize=8, color=INK_2)
    save(fig, "v11_jobs_reach_45min.png")


# --- 12. delay ripple from Central (PREVIEW) -------------------------------------------------------------

def delay_ripple() -> None:
    d = delays()
    st = stations().set_index("station")
    cx, cy = st.loc["Central", ["x", "y"]]
    by = d.groupby(["station", "mode"])["late_min"].agg(["mean", "count"]).reset_index()
    by = by[by["count"] >= 10].join(st[["x", "y"]], on="station")
    by["km"] = np.hypot(by["x"] - cx, by["y"] - cy) / 1000
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode, color in (("Trains", TRAINS), ("Metro", METRO)):
        b = by[by["mode"] == mode]
        ax.scatter(b["km"], b["mean"], s=b["count"] / 2 + 8, color=color, alpha=0.6, edgecolor="white", label=mode)
        binned = b.groupby(pd.cut(b["km"], np.arange(0, 130, 10)), observed=True)["mean"].mean()
        ax.plot([i.mid for i in binned.index], binned.to_numpy(), color=color, linewidth=2)
    for r in by.nlargest(6, "mean").itertuples():
        ax.annotate(r.station, (r.km, r.mean), xytext=(4, 3), textcoords="offset points", fontsize=8, color=INK_2)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel("Distance from Central (km)")
    ax.set_ylabel("Average minutes late")
    ax.grid(color=GRID)
    ax.legend(frameon=False)
    ax.set_title("Does lateness build up with distance from Central?", loc="left", fontsize=12)
    preview_note(ax, "one day; line shows the average for each 10 km band")
    save(fig, "v12_delay_ripple.png")


# --- 13. wait penalty for underserved suburbs ------------------------------------------------------------

def wait_penalty(walk_speed_m_per_min: float = 80) -> None:
    ctx = pd.read_csv(MARTS_DIR / "sa2_equity_context.csv", dtype={"SA2_CODE21": str})
    walk = pd.read_parquet(MARTS_DIR / "mesh_block_walk.parquet")
    walk["can_walk"] = walk["walk_m"].notna()  # a station entrance within 2.5 km on foot
    walk["walk_min"] = walk["walk_m"] / walk_speed_m_per_min
    per = walk.groupby("SA2_CODE21").apply(lambda g: pd.Series({
        "walk_min": np.average(g.loc[g["can_walk"], "walk_min"], weights=g.loc[g["can_walk"], "residents"]) if g["can_walk"].any() else np.nan,
        "pct_too_far": 100 * g.loc[~g["can_walk"], "residents"].sum() / g["residents"].sum(),
    }), include_groups=False)
    ctx = ctx.join(per, on="SA2_CODE21")
    st = pd.read_csv(MARTS_DIR / "station_service.csv").set_index("station")
    ctx["tph"] = ctx["main_nearest_station"].map(st["am_peak_tph"]) / 2  # one direction
    ctx["wait_min"] = (60 / (2 * ctx["tph"])).clip(upper=30)
    ctx["total"] = ctx["walk_min"] + ctx["wait_min"]
    ok = ctx["total"].notna()
    med = np.average(ctx.loc[ok, "total"], weights=ctx.loc[ok, "residents"] * (100 - ctx.loc[ok, "pct_too_far"]) / 100)
    td = ctx[ctx["transport_disadvantaged"]].nlargest(25, "erp_2025").sort_values("pct_too_far")
    fig, ax = plt.subplots(figsize=(11, 9))
    y = np.arange(len(td))
    ax.barh(y, td["walk_min"].fillna(0), color=BAR, height=0.65, label="Walk to station entrance (OSM paths)")
    ax.barh(y, td["wait_min"].where(td["walk_min"].notna(), 0), left=td["walk_min"].fillna(0), color="#86b6ef", height=0.65,
            label="Average wait (half the gap between trains)")
    for i, (t, p) in enumerate(zip(td["total"], td["pct_too_far"])):
        ax.text((0 if pd.isna(t) else t) + 0.5, i, f"{p:.0f}% live beyond a 2.5 km walk",
                va="center", fontsize=7.5, color="#b5470b" if p >= 50 else INK_2)
    ax.set_yticks(y, td["SA2_NAME21"])
    ax.axvline(med, color=INK, linestyle="--", linewidth=1)
    ax.text(med, len(td) - 0.3, f"  Sydney average {med:.0f} min", fontsize=8, color=INK)
    ax.set_xlim(0, max(td["total"].max(), med) + 38)
    ax.set_xlabel("Minutes from front door to being on a train, AM peak (residents who can walk to a station)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="x", color=GRID)
    ax.tick_params(axis="y", length=0, labelsize=8)
    ax.set_title("The access penalty in transport-disadvantaged suburbs\n25 largest of the 57: walk plus wait for those who can walk, and how many can't",
                 loc="left", fontsize=12)
    save(fig, "v13_wait_penalty.png")


# --- 14. Bankstown closure flow ------------------------------------------------------------------------------

def bankstown_flow() -> None:
    bk = pd.read_csv(MARTS_DIR / "bankstown_closure.csv")
    m = all_monthly_entries()
    b = m[m["month"].between(*BEFORE)]
    a = m[m["month"] > m["month"].max() - pd.DateOffset(months=12)]
    others = set(CLOSED) | set(bk["station"])
    growth = a[~a["station"].isin(others)]["trips"].sum() / b[~b["station"].isin(others)]["trips"].sum()
    alt = bk[bk["group"] == "Alternative"].copy()
    alt["above_trend"] = alt["entries_after"] - alt["entries_before"] * growth
    lost = bk.loc[bk["group"] == "Closed", "entries_before"].sum() / 1e6
    syd = alt.loc[alt["station"] == "Sydenham", "above_trend"].sum() / 1e6
    other_gain = alt.loc[(alt["station"] != "Sydenham") & (alt["above_trend"] > 0), "above_trend"].sum() / 1e6
    other_loss = -alt.loc[(alt["station"] != "Sydenham") & (alt["above_trend"] < 0), "above_trend"].sum() / 1e6
    unaccounted = lost + other_loss - syd - other_gain
    flows = [("Sydenham (Metro interchange,\nreplacement buses end here)", syd, METRO),
             ("Other nearby stations", other_gain, BAR),
             ("Not seen at any station\n(replacement bus, other modes, or not travelling)", unaccounted, "#c85a5a")]
    flows = [f for f in flows if f[1] >= 0.05]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.add_patch(plt.Rectangle((0, 0), 0.6, lost, color="#8a8a86"))
    ax.add_patch(plt.Rectangle((0, lost), 0.6, other_loss, color="#b9b8b2"))
    ax.text(-0.1, lost / 2, f"Entries at the 9 closed\nstations, year before\n{lost:.1f} million", ha="right", va="center", fontsize=10)
    ax.text(-0.1, lost + other_loss / 2, f"Below-trend fall at Bankstown,\nBirrong and Yagoona\n{other_loss:.1f} million",
            ha="right", va="center", fontsize=9, color=INK_2)
    lost = lost + other_loss
    y_left, y_right, gap = 0, 0, 0.25
    for label, val, color in flows:
        if val <= 0:
            continue
        ax.add_patch(Polygon([(0.6, y_left), (3.4, y_right), (3.4, y_right + val), (0.6, y_left + val)], color=color, alpha=0.35))
        ax.add_patch(plt.Rectangle((3.4, y_right), 0.6, val, color=color))
        ax.text(4.1, y_right + val / 2, f"{label}: {val:.1f}m", va="center", fontsize=9)
        y_left += val
        y_right += val + gap
    ax.set_xlim(-2.2, 8)
    ax.set_ylim(-0.3, y_right + 0.3)
    ax.set_axis_off()
    ax.set_title(f"Where did the Sydenham-Bankstown riders go?\nChange above the {100 * (growth - 1):.0f}% network growth, "
                 "12 months before closure vs latest 12 months (yearly entries)", loc="left", fontsize=12)
    fig.text(0.02, 0.02, "Left = riders lost from closed or shrinking stations; right = where extra entries above trend appeared. "
             "Source: TfNSW station entries (Opal).", fontsize=8, color=INK_2)
    save(fig, "v14_bankstown_flow.png")


# --- 15. Trains vs Metro --------------------------------------------------------------------------------------

def trains_vs_metro() -> None:
    st = pd.read_csv(MARTS_DIR / "station_service.csv")
    st["mode"] = np.where(st["networks"] == "metro", "Metro", np.where(st["networks"] == "sydneytrains", "Trains", "Both"))
    ds = pd.read_csv(MARTS_DIR / "station_demand_supply.csv")
    cat = pd.read_csv(MARTS_DIR / "station_catchment.csv")
    d = delays()
    pure = st[st["mode"] != "Both"][["station", "mode", "am_peak_tph", "lat"]]
    pure = pure[pure["lat"].between(-34.2, -33.5)]  # Sydney basin, so intercity outskirts don't skew Trains
    j = pure.merge(ds[["station", "avg_daily_entries", "entries_per_departure", "share_source"]], on="station", how="left")
    j = j.merge(cat[["station", "entries_per_resident"]], on="station", how="left")
    rel = d.groupby("mode").agg(on_time=("delay_s", lambda s: 100 * (s < 300).mean()),
                                late=("delay_s", lambda s: 100 * (s >= 120).mean()))
    rows = {
        "AM peak trains per hour\n(median station)": j.groupby("mode")["am_peak_tph"].median(),
        "Daily entries\n(median station)": j.groupby("mode")["avg_daily_entries"].median(),
        "AM boardings per train\n(median station)": j[j["share_source"] == "2023 typical day"].groupby("mode")["entries_per_departure"].median(),
        "Entries per walk-up resident\n(median station)": j.groupby("mode")["entries_per_resident"].median(),
        "On time, < 5 min late (%)\n(today, preview)": rel["on_time"],
        "2+ minutes late (%)\n(today, preview)": rel["late"],
    }
    table = pd.DataFrame(rows).T
    table.to_csv(MARTS_DIR / "trains_vs_metro.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (name, vals) in zip(axes.flat, table.iterrows()):
        vals = vals.reindex(["Trains", "Metro"])
        ax.bar(vals.index, vals.to_numpy(), color=[TRAINS, METRO], width=0.55)
        for i, v in enumerate(vals):
            if pd.notna(v):
                ax.text(i, v, f"{v:,.2f}" if abs(v) < 1 else f"{v:,.1f}" if abs(v) < 100 else f"{v:,.0f}",
                        ha="center", va="bottom", fontsize=9)
        ax.set_title(name, fontsize=10, loc="left")
        ax.set_yticks([])
        for s in ("left", "top", "right"):
            ax.spines[s].set_visible(False)
    n = j["mode"].value_counts()
    fig.suptitle(f"Sydney Trains vs Sydney Metro, station by station ({n.get('Trains', 0)} Trains-only and "
                 f"{n.get('Metro', 0)} Metro-only stations in the Sydney basin; shared interchanges excluded)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "v15_trains_vs_metro.png")
    print(table.round(2))


# --- rail only vs rail + bus (static timetable; buses are not in the reliability work) ----------

PTAL_COLORS = {"0": "#f1efe9", "1a": "#d9e6f5", "1b": "#b7d3f6", "2": "#86b6ef", "3": "#5598e7",
               "4": "#2a78d6", "5": "#1c5cab", "6a": "#104281", "6b": "#0d366b"}


def two_maps(title: str, fname: str, draw, labels=("Rail only (Trains + Metro)", "Rail + bus"), legend=None, note=""):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9.5))
    for ax, which, label in zip(axes, ("rail", "all"), labels):
        draw(ax, which)
        frame_axes(ax)
        ax.set_title(label, loc="left", fontsize=11)
    if legend:
        axes[1].legend(handles=legend[0], title=legend[1], loc="lower right", frameon=False, fontsize=8, title_fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13)
    if note:
        fig.text(0.01, 0.02, note, fontsize=8, color=INK_2)
    save(fig, fname)


def ptal_maps() -> None:
    sa2 = pd.read_csv(MARTS_DIR / "sa2_ptal.csv", dtype={"SA2_CODE21": str})
    g = sa2_shapes().merge(sa2, on="SA2_CODE21", how="left")

    def draw(ax, which):
        col = g[f"ptal_{which}"].fillna("0")
        g.plot(ax=ax, color=col.map(PTAL_COLORS).fillna("#f1efe9"), edgecolor=SURFACE, linewidth=0.25)

    mb = pd.read_parquet(MARTS_DIR / "mesh_block_ptal.parquet")  # headline from homes, not suburb averages
    w = mb["residents"]
    poor = {k: 100 * w[mb[f"ai_{k}"] <= 5].sum() / w.sum() for k in ("rail", "all")}
    two_maps(f"Public transport access level (PTAL) by suburb: residents with poor access (PTAL 0-1b) drop from "
             f"{poor['rail']:.0f}% to {poor['all']:.0f}% once buses are counted",
             "v16_ptal_rail_vs_bus.png", draw,
             legend=([Patch(color=c, label=k) for k, c in PTAL_COLORS.items()], "PTAL band (0 = none, 6b = excellent)"),
             note="Resident-weighted average access index per SA2, AM peak 7 to 9am, Tue 27 Oct 2026. TfL PTAL method; "
                  "walk limits 640 m bus, 960 m rail. Light rail and ferries not included.")


def bivariate_bus() -> None:
    ctx = pd.read_csv(MARTS_DIR / "sa2_equity_context.csv", dtype={"SA2_CODE21": str})
    g = sa2_shapes().merge(ctx, on="SA2_CODE21", how="left")
    pal = [["#e8e8e8", "#e4acac", "#c85a5a"], ["#b0d5df", "#ad9ea5", "#985356"], ["#64acbe", "#627f8c", "#574249"]]
    dis = pd.cut(g["irsd_decile"], [0, 3, 7, 10], labels=[2, 1, 0]).astype(float)

    def draw(ax, which):
        poor = g[f"pct_poor_{which}"]
        acc = pd.cut(poor, [-1, 20, 50, 101], labels=[0, 1, 2]).astype(float)
        colors = [pal[int(d)][int(a)] if not (np.isnan(d) or np.isnan(a)) else "#f4f4f2" for d, a in zip(dis, acc)]
        g.plot(ax=ax, color=colors, edgecolor=SURFACE, linewidth=0.25)

    handles = [Patch(color=pal[2][2], label="Most disadvantaged + poor access"),
               Patch(color=pal[2][0], label="Most disadvantaged, good access"),
               Patch(color=pal[0][2], label="Least disadvantaged, poor access"),
               Patch(color=pal[1][1], label="Middle on both")]
    two_maps("Disadvantage meets poor access: counting buses shrinks the problem to the outer south-west, Mount Druitt and Wyong",
             "v01b_bivariate_rail_vs_bus.png", draw, legend=(handles, "Disadvantage (IRSD) x share of residents at PTAL 0-1b"),
             note="Poor access = share of residents at PTAL 0 to 1b: under 20% / 20-50% / over 50%. Disadvantage = IRSD decile 1-3 / 4-7 / 8-10.")


def jobs_maps() -> None:
    j = pd.read_csv(MARTS_DIR / "sa2_jobs_access.csv", dtype={"SA2_CODE21": str})
    g = sa2_shapes().merge(j, on="SA2_CODE21", how="left")
    bins = [1, 25_000, 100_000, 300_000, 600_000, 1_000_000, 5_000_000]
    labels = ["Under 25k", "25k-100k", "100k-300k", "300k-600k", "600k-1m", "1m+"]

    def draw(ax, which):
        v = g[f"jobs_{which}"]
        g[~(v > 0)].plot(ax=ax, color=NO_SERVICE, edgecolor=SURFACE, linewidth=0.25)
        g[v > 0].plot(ax=ax, column=f"jobs_{which}", cmap=ListedColormap(BLUES), norm=BoundaryNorm(bins, 6),
                      edgecolor=SURFACE, linewidth=0.25)

    two_maps("Jobs reachable within 45 minutes by public transport, leaving home at 8am",
             "v11b_jobs_rail_vs_bus.png", draw,
             legend=([Patch(color=c, label=l) for c, l in zip(BLUES, labels)] + [Patch(color=NO_SERVICE, label="No stop within 800 m")], "Jobs"),
             note="From each suburb's resident-weighted centre: walk up to 800 m, timetable on Tue 27 Oct 2026, changes allowed (300 m walk + 1 min). "
                  "Jobs = 2021 Census place of work.")


def board_penalty() -> None:
    ctx = pd.read_csv(MARTS_DIR / "sa2_equity_context.csv", dtype={"SA2_CODE21": str})
    td = ctx[ctx["transport_disadvantaged"] | ctx.get("transport_disadvantaged_with_bus", False)]
    td = td.nlargest(25, "erp_2025").sort_values("board_min_all")
    w = ctx["residents"]
    avg = {k: np.average(ctx[f"board_min_{k}"].fillna(60), weights=w) for k in ("rail", "all")}
    fig, ax = plt.subplots(figsize=(11, 9))
    y = np.arange(len(td))
    ax.barh(y + 0.2, td["board_min_rail"].clip(upper=60), height=0.38, color="#c8c8c4", label="Rail only")
    ax.barh(y - 0.2, td["board_min_all"].clip(upper=60), height=0.38, color=BAR, label="Rail + bus")
    ax.set_yticks(y, td["SA2_NAME21"])
    for k, ls in (("rail", ":"), ("all", "--")):
        ax.axvline(avg[k], color=INK, linestyle=ls, linewidth=1)
        ax.text(avg[k], len(td) - 0.2, f" Sydney avg, {'rail only' if k == 'rail' else 'rail + bus'}: {avg[k]:.0f} min",
                fontsize=8, color=INK, rotation=0)
    ax.set_xlabel("Minutes from front door to boarding the quickest service, AM peak (walk + half the gap + reliability allowance; 60 = none in reach)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color=GRID)
    ax.tick_params(axis="y", length=0, labelsize=8)
    ax.set_title("Time to get on public transport in transport-disadvantaged suburbs: rail only vs rail + bus\n"
                 "25 largest suburbs flagged on either measure", loc="left", fontsize=12)
    save(fig, "v13b_board_time_rail_vs_bus.png")


# --- international access methods (rail + bus) -------------------------------------------------

def methods_data() -> gpd.GeoDataFrame:
    m = pd.read_csv(MARTS_DIR / "sa2_access_methods.csv", dtype={"SA2_CODE21": str})
    return sa2_shapes().merge(m.drop(columns=["SA2_NAME21"]), on="SA2_CODE21", how="left")


def quantile_map(ax, g, col, cmap="Blues", k=6, fmt="{:,.0f}", title=""):
    v = g[col]
    edges = np.unique(np.nanquantile(v[v > 0], np.linspace(0, 1, k + 1)))
    zero = ~(v > 0)
    g[zero].plot(ax=ax, color=NO_SERVICE, edgecolor=SURFACE, linewidth=0.2)
    cm = plt.get_cmap(cmap)
    colors = [cm(0.25 + 0.75 * i / (len(edges) - 2)) for i in range(len(edges) - 1)]
    g[~zero].plot(ax=ax, column=col, cmap=ListedColormap(colors), norm=BoundaryNorm(edges, len(colors)),
                  edgecolor=SURFACE, linewidth=0.2)
    frame_axes(ax)
    ax.set_title(title, loc="left", fontsize=10)
    labels = [f"{fmt.format(a)} to {fmt.format(b)}" for a, b in zip(edges[:-1], edges[1:])]
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors, labels)] + [Patch(color=NO_SERVICE, label="None in reach")],
              loc="lower right", frameon=False, fontsize=6.5)


def cumulative_opportunities() -> None:
    g = methods_data()
    fig, axes = plt.subplots(2, 2, figsize=(15, 16))
    specs = [("jobs_45", "Jobs within 45 min"), ("hospital_45", "Hospitals within 45 min"),
             ("education_45", "University and TAFE campuses within 45 min"), ("supermarket_20", "Supermarkets within 20 min")]
    for ax, (col, title) in zip(axes.flat, specs):
        quantile_map(ax, g, col, title=title)
    fig.suptitle("Cumulative opportunities by public transport (rail + bus), leaving 8am on a weekday", x=0.01, ha="left", fontsize=13)
    fig.text(0.01, 0.01, "Method: Wachs & Kumagai (1973); used in the UK DfT journey time statistics. Destinations: 2021 Census jobs, "
             "OpenStreetMap hospitals, campuses, supermarkets. Quantile colour classes.", fontsize=8, color=INK_2)
    save(fig, "v17_cumulative_opportunities.png")


def gravity_map() -> None:
    g = methods_data()
    fig, ax = plt.subplots(figsize=(9, 10))
    bins = [0, 25, 50, 75, 100, 150, 250, 1000]
    cols = ["#c85a5a", "#e4acac", "#f1e3dc", "#d9e6f5", "#86b6ef", "#2a78d6", "#0d366b"]
    g.plot(ax=ax, column="gravity_index", cmap=ListedColormap(cols), norm=BoundaryNorm(bins, len(cols)),
           edgecolor=SURFACE, linewidth=0.2, missing_kwds={"color": NO_SERVICE})
    frame_axes(ax)
    labels = ["Under 25", "25-50", "50-75", "75-100", "100-150", "150-250", "250+"]
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, labels)], title="Index (Sydney average = 100)",
              loc="lower right", frameon=False, fontsize=8, title_fontsize=8)
    ax.set_title("Gravity-based job access by public transport (rail + bus)\nEvery job counts, weighted down by travel time "
                 "(a job 20 min away counts half)", loc="left", fontsize=12)
    fig.text(0.02, 0.03, "Method: Hansen (1959), negative exponential decay, half-life 20 min, 60 min horizon. Weekday 8am.",
             fontsize=8, color=INK_2)
    save(fig, "v18_gravity_access.png")


def temporal_coverage() -> None:
    g = methods_data()
    g["ret_9pm"] = 100 * g["jobs45_wk9pm"] / g["jobs45_wk7am"].replace(0, np.nan)
    g["ret_sun"] = 100 * g["jobs45_sun10am"] / g["jobs45_wk7am"].replace(0, np.nan)
    bins = [0, 25, 50, 75, 90, 110, 1000]
    cols = ["#c85a5a", "#e4acac", "#f1e3dc", "#d9e6f5", "#86b6ef", "#2a78d6"]

    def draw(ax, which):
        col = "ret_9pm" if which == "rail" else "ret_sun"
        g.plot(ax=ax, column=col, cmap=ListedColormap(cols), norm=BoundaryNorm(bins, len(cols)),
               edgecolor=SURFACE, linewidth=0.2, missing_kwds={"color": NO_SERVICE})

    labels = ["Under 25%", "25-50%", "50-75%", "75-90%", "90-110%", "Over 110%"]
    w = g["residents"].fillna(0)
    avg = {c: np.average(g[c].fillna(0), weights=w) for c in ("jobs45_wk7am", "jobs45_wk9pm", "jobs45_sun10am")}
    two_maps(f"Access outside commuter hours: jobs reachable in 45 min, as a share of weekday 7am "
             f"(Sydney average: 9pm {100 * avg['jobs45_wk9pm'] / avg['jobs45_wk7am']:.0f}%, Sunday {100 * avg['jobs45_sun10am'] / avg['jobs45_wk7am']:.0f}%)",
             "v19_temporal_coverage.png", draw, labels=("Weekday 9pm vs 7am", "Sunday 10am vs weekday 7am"),
             legend=([Patch(color=c, label=l) for c, l in zip(cols, labels)] + [Patch(color=NO_SERVICE, label="No service at 7am")], "Share kept"),
             note="Method: time-of-day accessibility (e.g. Owen & Levinson, Access Across America). Rail + bus timetables for Tue 27 Oct and Sun 1 Nov 2026.")


def transit_gap() -> None:
    g = methods_data()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.1, 1]})
    hi = g["need_z"] > 0.5
    g.plot(ax=ax1, color="#ecebe6", edgecolor=SURFACE, linewidth=0.2)
    g[hi].plot(ax=ax1, column="transit_gap", cmap="RdBu_r", norm=plt.Normalize(-2.5, 2.5), edgecolor=SURFACE, linewidth=0.2)
    frame_axes(ax1)
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-2.5, 2.5)), ax=ax1, shrink=0.6,
                 location="bottom", pad=0.02, label="Gap: need minus supply (z-scores); red = under-served")
    ax1.set_title("High-need suburbs (need above +0.5 sd), coloured by gap", loc="left", fontsize=10)
    top = g[hi & (g["residents"] >= 2000)].nlargest(15, "transit_gap").sort_values("transit_gap")
    ax2.barh(top["SA2_NAME21_x"] if "SA2_NAME21_x" in top else top["SA2_NAME21"], top["transit_gap"], color="#c85a5a", height=0.6)
    ax2.set_xlabel("Transit gap (need z minus supply z)")
    ax2.grid(axis="x", color=GRID)
    ax2.tick_params(axis="y", length=0, labelsize=8)
    ax2.set_title("Largest gaps among high-need suburbs", loc="left", fontsize=10)
    fig.suptitle("Transit gap index: where need for public transport outstrips supply (rail + bus)", x=0.01, ha="left", fontsize=13)
    fig.text(0.01, 0.02, "Method: Currie (2010), Melbourne. Need = mean z-score of disadvantage (IRSD), no-car households, age 65+, age 15-24. "
             "Supply = z-score of log PTAL index.", fontsize=8, color=INK_2)
    save(fig, "v20_transit_gap.png")


def excess_wait_chart() -> None:
    e = pd.read_csv(MARTS_DIR / "excess_wait.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.6]})
    by = e.groupby("mode").apply(lambda g: pd.Series({"Scheduled wait": np.average(g["swt_min"], weights=g["trains"]),
                                                      "Actual wait": np.average(g["awt_min"], weights=g["trains"])}),
                                 include_groups=False)
    x = np.arange(len(by))
    ax1.bar(x - 0.18, by["Scheduled wait"], width=0.34, color="#c8c8c4", label="Scheduled wait")
    ax1.bar(x + 0.18, by["Actual wait"], width=0.34, color=[METRO if m == "Metro" else TRAINS for m in by.index], label="Actual wait")
    for i, (s, a) in enumerate(zip(by["Scheduled wait"], by["Actual wait"])):
        ax1.text(i + 0.18, a, f"+{(a - s) * 60:.0f} s", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x, by.index)
    ax1.set_ylabel("Average wait for a turn-up-and-go passenger (min)")
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_title("Scheduled vs actual average wait", loc="left", fontsize=10)
    top = e.nlargest(12, "ewt_min").sort_values("ewt_min")
    ax2.barh(top["station"] + " (" + top["lines"] + ")", top["ewt_min"] * 60,
             color=[METRO if m == "Metro" else TRAINS for m in top["mode"]], height=0.6)
    ax2.set_xlabel("Excess wait time (seconds)")
    ax2.tick_params(axis="y", labelsize=8, length=0)
    ax2.grid(axis="x", color=GRID)
    ax2.set_title("Platforms with the most excess wait", loc="left", fontsize=10)
    fig.suptitle("Excess wait time on frequent Trains and Metro platforms (a train at least every 10 min)", x=0.01, ha="left", fontsize=12)
    preview_note(ax1, "about 9 hours on one weekday, mostly off-peak. Method: Transport for London's EWT for high-frequency routes")
    save(fig, "v21_excess_wait.png")


def social_exclusion() -> None:
    m = pd.read_csv(MARTS_DIR / "sa2_access_methods.csv", dtype={"SA2_CODE21": str})
    m = m[(m["residents"] >= 1000) & m["unemployment_rate"].notna() & (m["gravity_index"] > 0)]
    grp = pd.cut(m["irsd_decile"], [0, 3, 7, 10], labels=["Most disadvantaged (IRSD 1-3)", "Middle (4-7)", "Least disadvantaged (8-10)"])
    colors = {"Most disadvantaged (IRSD 1-3)": "#c85a5a", "Middle (4-7)": "#9aa4b1", "Least disadvantaged (8-10)": "#2a78d6"}
    fig, ax = plt.subplots(figsize=(10, 7))
    for k, c in colors.items():
        s = m[grp == k]
        ax.scatter(s["gravity_index"], s["unemployment_rate"], s=s["residents"] / 600, color=c, alpha=0.6, edgecolor="white", label=k)
    ax.set_xscale("log")
    ax.set_xlabel("Gravity job access index, rail + bus (Sydney average = 100, log scale)")
    ax.set_ylabel("Unemployment rate, 2021 Census (%)")
    rho = m["gravity_index"].rank().corr(m["unemployment_rate"].rank())
    rho_low = m[grp == "Most disadvantaged (IRSD 1-3)"]
    rho_low = rho_low["gravity_index"].rank().corr(rho_low["unemployment_rate"].rank())
    ax.text(0.01, 0.97, f"Spearman rho, all suburbs: {rho:+.2f}\nWithin the most disadvantaged: {rho_low:+.2f}",
            transform=ax.transAxes, va="top", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(color=GRID)
    ax.set_title("Access and unemployment by suburb: a correlation, not a cause", loc="left", fontsize=12)
    fig.text(0.01, 0.0, "Lens: UK Social Exclusion Unit (2003), Lucas (2012). Both measures share causes such as income, housing and "
             "distance from the centre, so neither explains the other here.", fontsize=8, color=INK_2)
    save(fig, "v22_social_exclusion.png")


ALL = {"bivariate": bivariate, "marey": marey,
       "marey_metro": lambda: marey("M1", "Tallawong", "v02b_marey_metro.png",
                                    "Metro M1: every train today, scheduled (grey) vs measured (coloured by average lateness)"),
       "network": frequency_network, "timeline": ridership_timeline, "growth": growth_vs_rail,
       "detours": walking_detours, "timelapse": timelapse, "carpark": carpark_curves, "radar": rain_radar,
       "heatmap": delay_heatmap, "jobs": jobs_reach, "ripple": delay_ripple, "wait": wait_penalty,
       "bankstown": bankstown_flow, "modes": trains_vs_metro,
       "ptal": ptal_maps, "bivariate_bus": bivariate_bus, "jobs_bus": jobs_maps, "board_bus": board_penalty,
       "cumulative": cumulative_opportunities, "gravity": gravity_map, "temporal": temporal_coverage,
       "gap": transit_gap, "ewt": excess_wait_chart, "exclusion": social_exclusion}


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    for name in sys.argv[1:] or ALL:
        try:
            ALL[name]()
        except Exception as exc:  # keep going so one failure doesn't stop the gallery
            import traceback
            traceback.print_exc()
            print(f"FAILED {name}: {exc}")
