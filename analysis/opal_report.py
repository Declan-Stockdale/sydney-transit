"""
Tables and charts for the first Opal patronage report.

    1. Busy stations, thin service: AM peak boardings per scheduled train
       (from demand_supply.py's mart).
    2. Sydenham to Bankstown closure (from 30 Sep 2024): station entries in the
       12 months before vs the latest 12 months, for the closed stations and the
       nearby stations riders could switch to.
    3. Who rides each line: concession and senior/pensioner share of trips by
       line (last 12 months), against how frequent the line's service is.

Writes CSVs to data/marts/ and PNGs to outputs/figures/.

Usage:
    python -m analysis.demand_supply --date 2026-10-27   # run first
    python -m analysis.opal_report
"""

import matplotlib.pyplot as plt
import pandas as pd

from analysis.demand_supply import PATRONAGE_DIR, clean_station
from analysis.plot_equity import BAR, FIG_DIR, GRID, INK, INK_2, NO_SERVICE  # shared chart style
from tfnsw import DATA_DIR

MARTS_DIR = DATA_DIR / "marts"
CLOSED = ["Marrickville", "Dulwich Hill", "Hurlstone Park", "Canterbury", "Campsie",
          "Belmore", "Lakemba", "Wiley Park", "Punchbowl"]
# Stations riders could switch to: Bankstown (still on T6), T2/T3 Inner West, T8 East Hills, Sydenham
ALTERNATIVES = ["Bankstown", "Sydenham", "Ashfield", "Croydon", "Burwood", "Summer Hill",
                "Lewisham", "Petersham", "Stanmore", "Kingsgrove", "Beverly Hills", "Narwee",
                "Bexley North", "Bardwell Park", "Yagoona", "Birrong"]
BEFORE = ("2023-10-01", "2024-09-30")  # 12 months before the closure
AFTER_MONTHS = 12


def all_monthly_entries() -> pd.DataFrame:
    """Monthly entries per station, 2016 to now, stitched from the three published files."""
    files = ["station_entry_exit_monthly_to_jun2024.csv", "station_entry_exit_monthly_jul2024_oct2024.csv",
             "station_entry_exit_monthly.csv"]
    frames = []
    for f in files:
        d = pd.read_csv(PATRONAGE_DIR / f)
        d.columns = d.columns.str.strip()  # some files pad header names with spaces
        frames.append(d[["MonthYear", "Station", "Station_Type", "Entry_Exit", "Trip"]])
    df = pd.concat(frames)
    df = df[(df["Entry_Exit"] == "Entry") & (df["Station_Type"] != "Light rail")]
    df["month"] = pd.to_datetime(df["MonthYear"].astype(str).str[:7])
    df["trips"] = pd.to_numeric(df["Trip"].replace("Less than 50", "25"), errors="coerce")
    df["station"] = clean_station(df["Station"])
    # Overlapping months between files: keep the newest file's figure
    df = df.drop_duplicates(["month", "station"], keep="last")
    return df[["month", "station", "trips"]]


def bankstown_closure() -> pd.DataFrame:
    m = all_monthly_entries()
    after_start = m["month"].max() - pd.DateOffset(months=AFTER_MONTHS - 1)
    before = m[m["month"].between(*BEFORE)].groupby("station")["trips"].sum()
    after = m[m["month"] >= after_start].groupby("station")["trips"].sum()
    out = pd.DataFrame({"entries_before": before, "entries_after": after}).reset_index()
    out = out[out["station"].isin(CLOSED + ALTERNATIVES)].copy()
    out["group"] = out["station"].map(lambda s: "Closed" if s in CLOSED else "Alternative")
    out["change"] = out["entries_after"] - out["entries_before"]
    out["pct_change"] = (100 * out["change"] / out["entries_before"]).round(1)
    out.attrs["after"] = f"{after_start:%b %Y} to {m['month'].max():%b %Y}"
    return out.sort_values(["group", "change"])


def line_profile() -> pd.DataFrame:
    t = pd.read_csv(PATRONAGE_DIR / "trips_train_monthly.csv")
    t["month"] = pd.to_datetime(t["Year_Month"], format="%b-%y", errors="coerce").fillna(
        pd.to_datetime(t["Year_Month"], format="%b-%Y", errors="coerce"))
    t["trips"] = pd.to_numeric(t["Trip"], errors="coerce")
    recent = t[t["month"] > t["month"].max() - pd.DateOffset(months=12)]
    piv = recent.pivot_table(index="Line", columns="Card_type", values="trips", aggfunc="sum").fillna(0)
    total = piv.sum(axis=1)
    out = pd.DataFrame({
        "annual_trips": total,
        "concession_pct": 100 * piv.get("Concession", 0) / total,
        "senior_pct": 100 * piv.get("Senior/Pensioner", 0) / total,
        "child_school_pct": 100 * (piv.get("Child/Youth", 0) + piv.get("School Student", 0)) / total,
    }).round(1)
    out["concession_senior_pct"] = out["concession_pct"] + out["senior_pct"]
    out = out[out["annual_trips"] > 1_000_000].reset_index()
    out["code"] = out["Line"].str.extract(r"^(T\d)")[0]
    names = {"Blue Mountains Line": "BMT", "Central Coast & Newcastle Line": "CCN",
             "South Coast Line": "SCO", "Southern Highlands Line": "SHL"}
    out["code"] = out["code"].fillna(out["Line"].map(names))
    return out.dropna(subset=["code"])


def line_frequency() -> pd.Series:
    """Median AM peak trains/hour across the stations each line serves."""
    st = pd.read_csv(MARTS_DIR / "station_service.csv")
    rows = st.assign(line=st["lines"].str.split()).explode("line")
    return rows.groupby("line")["am_peak_tph"].median()


def chart_busy(ds: pd.DataFrame) -> None:
    top = ds[ds["share_source"] == "2023 typical day"].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["station"], top["entries_per_departure"], color=BAR, height=0.6)
    for y, (v, d, lines) in enumerate(zip(top["entries_per_departure"], top["am_departures"], top["lines"])):
        ax.text(v + 1.5, y, f"{v:.0f}   ({d} trains, {lines})", va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, top["entries_per_departure"].max() * 1.45)
    ax.set_xlabel("Estimated AM peak (6 to 10am) boardings per scheduled departure")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Most boardings per train in the morning peak", loc="left", fontsize=12)
    fig.savefig(FIG_DIR / "boardings_per_train.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_bankstown(bk: pd.DataFrame) -> None:
    df = bk.sort_values("change")
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = [NO_SERVICE if g == "Closed" else BAR for g in df["group"]]
    ax.barh(df["station"], df["change"] / 1000, color=colors, height=0.6)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    for y, (c, p) in enumerate(zip(df["change"], df["pct_change"])):
        ax.text(c / 1000 + (20 if c >= 0 else -20), y, f"{p:+.0f}%", va="center",
                ha="left" if c >= 0 else "right", fontsize=8, color=INK_2)
    ax.set_xlabel("Change in yearly station entries (thousands)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(df["change"].min() / 1000 * 1.45,max(df["change"].max() / 1000 * 1.4, 300))
    ax.set_title("Sydenham to Bankstown closure: entries 12 months before vs latest 12 months", loc="left", fontsize=11)
    ax.text(0.99, 0.02, "Grey: closed stations   Blue: nearby open stations", transform=ax.transAxes,
            ha="right", fontsize=8, color=INK_2)
    fig.savefig(FIG_DIR / "bankstown_closure.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_lines(lp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(lp["am_peak_tph"], lp["concession_senior_pct"], s=lp["annual_trips"] / 1e6 * 4 + 20,
               color=BAR, alpha=0.8, edgecolor="white", linewidth=1)
    for _, r in lp.iterrows():
        # T2 and T3 sit almost on top of each other; put T3's label on the left
        offset, ha = ((-8, -12), "right") if r["code"] == "T3" else ((6, 4), "left")
        ax.annotate(r["code"], (r["am_peak_tph"], r["concession_senior_pct"]), xytext=offset,
                    textcoords="offset points", fontsize=9, color=INK, ha=ha)
    ax.set_xlabel("Median AM peak trains per hour at the line's stations")
    ax.set_ylabel("Trips on concession or senior cards (%)")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Lines used most by concession and senior card holders,\nagainst how often trains come", loc="left", fontsize=11)
    fig.savefig(FIG_DIR / "concession_by_line.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ds = pd.read_csv(MARTS_DIR / "station_demand_supply.csv")
    bk = bankstown_closure()
    lp = line_profile()
    lp["am_peak_tph"] = lp["code"].map(line_frequency())
    bk.to_csv(MARTS_DIR / "bankstown_closure.csv", index=False)
    lp.to_csv(MARTS_DIR / "line_card_profile.csv", index=False)

    chart_busy(ds)
    chart_bankstown(bk)
    chart_lines(lp.dropna(subset=["am_peak_tph"]))

    pd.set_option("display.width", 220)
    print("Bankstown closure, after =", bk.attrs["after"])
    print(bk.to_string(index=False))
    print(bk.groupby("group")[["entries_before", "entries_after", "change"]].sum())
    print(lp.sort_values("concession_senior_pct", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
