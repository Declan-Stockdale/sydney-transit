"""
Sydney Public Transport: Who Gets Let Down? Interactive dashboard.

Reads only the small precomputed files in app/data/ (made by
analysis/export_dashboard.py), so it runs anywhere without raw data or an API key.

    streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

DATA = Path(__file__).parent / "data"
BLUE, METRO, RED, GREY = "#2a78d6", "#168388", "#c85a5a", "#9aa4b1"
SEQ = ["#d9e6f5", "#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
VIEW = pdk.ViewState(latitude=-33.84, longitude=150.98, zoom=8.6, pitch=0)

st.set_page_config(page_title="Sydney transport: who gets let down?", page_icon="🚆", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px;}
  [data-testid="stMetricValue"] {font-size: 1.9rem;}
  [data-testid="stMetric"] {background: #ffffff; border: 1px solid #e4e3df; border-radius: 10px; padding: 14px 16px;}
  .lead {font-size: 1.1rem; color: #33322f; line-height: 1.55;}
  .tag {display:inline-block; padding:2px 8px; border-radius:6px; background:#fff1e6; color:#b5470b; font-size:0.8rem; font-weight:600;}
  .src {color:#6b6a66; font-size:0.82rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data ---

@st.cache_data
def load():
    gj = json.loads((DATA / "sa2.geojson").read_text())
    props = pd.DataFrame([f["properties"] for f in gj["features"]])
    return {
        "geo": gj, "sa2": props,
        "stations": pd.read_csv(DATA / "stations.csv"),
        "segments": pd.read_csv(DATA / "segments.csv"),
        "timeline": pd.read_csv(DATA / "timeline.csv", parse_dates=["month"]),
        "bankstown": pd.read_csv(DATA / "bankstown.csv"),
        "lines": pd.read_csv(DATA / "lines.csv"),
        "modes": pd.read_csv(DATA / "modes.csv", index_col=0),
        "heatmap": pd.read_csv(DATA / "heatmap.csv"),
        "ewt": pd.read_csv(DATA / "excess_wait.csv"),
        "pois": pd.read_csv(DATA / "pois.csv"),
        "hourly": pd.read_csv(DATA / "hourly.csv", dtype={"SA2_CODE21": str}),
        "hunter_geo": json.loads((DATA / "hunter_sa2.geojson").read_text()),
        "hunter_stations": pd.read_csv(DATA / "hunter_stations.csv"),
        "meta": json.loads((DATA / "meta.json").read_text()),
    }


D = load()
META = D["meta"]

# Plain-English definitions: shown as tooltips, in "What these measures mean" boxes, and on the Glossary page.
GLOSSARY = {
    "Suburb (SA2)": "ABS Statistical Area Level 2, roughly a suburb (3,000 to 25,000 people). Greater Sydney has about 370. "
                    "Names with a dash cover several suburbs (e.g. Erskineville - Alexandria).",
    "PTAL / access level": "Public Transport Accessibility Level, Transport for London's standard score for how easy it is to get on "
                           "public transport from home. For each route you can walk to: access time = walk (up to 640 m for bus, 960 m "
                           "for rail) + half the average gap between services + a reliability allowance (2 min bus, 0.75 min rail). "
                           "Each route scores 30 / access time; your best route counts in full and others count half. Weekday 7 to 9am.",
    "PTAL index": "The number behind the PTAL bands. 0 = no service in walking range; around 5 = a couple of routes every 15 to 20 "
                  "minutes; 25+ = several frequent routes; 40+ = city centre.",
    "Poor access": "PTAL band 0, 1a or 1b (index 5 or below): no service in walking range, or only infrequent routes.",
    "Residents with poor access (%)": "Share of a suburb's residents (2021 Census mesh blocks) whose home has poor access.",
    "Minutes to board": "From the front door to getting on the quickest service in the morning peak: walking time + half the gap "
                        "between services + reliability allowance. Blank or 60 = nothing within walking range.",
    "Jobs within 45 minutes": "How many jobs (2021 Census place of work) you can reach by public transport within 45 minutes of "
                              "leaving home at 8am on a weekday, including walking up to 800 m to the first stop, waiting and changes. "
                              "Calculated from up to four points per suburb, weighted by residents.",
    "Transport-disadvantaged suburb": "A suburb where people are more likely to depend on public transport (SEIFA IRSD decile 1 to 3, "
                                      "or at least 15% of households without a car) AND most residents have poor access (rail + bus "
                                      "version). At least 2,000 residents.",
    "IRSD decile": "ABS SEIFA Index of Relative Socio-economic Disadvantage, ranked into tenths nationally: 1 = most "
                   "disadvantaged 10% of areas in Australia, 10 = least disadvantaged.",
    "No car %": "Share of a suburb's households with no motor vehicle (2021 Census).",
    "Within a 1.5 km walk of a station": "Share of residents whose walk to a train or Metro station entrance, along real footpaths "
                                         "(OpenStreetMap), is 1.5 km or less.",
    "Hospitals / campuses / supermarkets in reach": "Number of hospitals or university/TAFE campuses within 45 minutes, or supermarkets "
                                                    "within 20 minutes, by public transport from 8am (OpenStreetMap locations).",
    "Gravity index": "Every job counts, but less the longer it takes to reach (a job 20 minutes away counts half, 40 minutes a quarter). "
                     "Scaled so the Sydney resident average = 100. Hansen (1959).",
    "Jobs kept at 9pm / Sunday": "Jobs reachable in 45 minutes at that time, as a % of the same suburb at 7am on a weekday. Can exceed "
                                 "100% because timetabled buses are slower in peak traffic.",
    "Transit gap": "Need minus supply, in standard deviations (Currie 2010). Need = average of disadvantage, households without a car, "
                   "residents aged 65+ and 15 to 24. Supply = PTAL. Positive = more need than service. Shown for high-need suburbs only.",
    "Daily entries": "Average Opal tap-ons at a station per day over the last 12 months (weekends included).",
    "AM / PM boardings per train": "Estimated tap-ons in the morning peak (6 to 10am) or evening peak (3 to 7pm), divided by the "
                                   "scheduled departures from that station in the same period (both directions). The peak share of "
                                   "each station's entries comes from TfNSW's 2023 typical-day data.",
    "Entries per walk-up resident": "Daily entries divided by the residents whose nearest station it is within 1.5 km. High = draws "
                                    "people from far away (buses, parking, a destination); low = locals don't use it much.",
    "Trains per hour (AM peak)": "Scheduled departures from a station between 6:30 and 10am, both directions, per hour.",
    "Minutes late": "Measured from train GPS positions: departure time at each station minus the timetable. Negative = early. "
                    "The feed's last prediction is used where no GPS fix matched.",
    "On time": "Less than 5 minutes late (Sydney Trains' own definition for suburban services).",
    "Passenger minutes lost": "Daily entries at a station x average minutes late of trains there (early counts as 0). Weights "
                              "delay by how many people it affects.",
    "Excess wait time": "Transport for London's measure for turn-up-and-go services: how much longer the average passenger waits "
                        "than the timetable implies, because trains bunch or leave gaps.",
}


def define(term: str) -> str:
    return GLOSSARY[term]


def explain(*terms: str):
    with st.expander("What these measures mean"):
        for t in terms:
            st.markdown(f"**{t}.** {GLOSSARY[t]}")


def hex_rgb(h: str, a: int = 200) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [a]


def colour_geo(values: pd.Series, palette=SEQ, bins=None, reverse=False, none_colour="#e7e5df", zero_is_none=True):
    """Quantile (or given) bins -> RGBA per feature, plus legend rows.

    zero_is_none: a value of 0 means "nothing in reach" (jobs, counts) and is greyed out;
    set False for measures where 0 is a real value (percentages, minutes).
    """
    v = values.astype(float)
    present = v.notna() & ((v > 0) if zero_is_none else True)
    pal = palette[::-1] if reverse else palette
    if bins is None:
        bins = np.unique(np.nanquantile(v[present], np.linspace(0, 1, len(pal) + 1))) if present.any() else np.array([0, 1])
    pal = pal[: len(bins) - 1] if len(bins) - 1 < len(pal) else pal
    idx = np.clip(np.digitize(v, bins[1:-1]), 0, len(pal) - 1)
    cols = [hex_rgb(pal[i]) if p else hex_rgb(none_colour, 160) for i, p in zip(idx, present)]
    legend = [(pal[i], bins[i], bins[i + 1]) for i in range(min(len(pal), len(bins) - 1))]
    return cols, legend


def choropleth(values: pd.Series, tooltip_html: str, palette=SEQ, bins=None, reverse=False, highlight=None, height=560,
               zero_is_none=True, geo=None, view=None):
    gj = json.loads(json.dumps(geo or D["geo"]))  # copy
    cols, legend = colour_geo(values, palette, bins, reverse, zero_is_none=zero_is_none)
    for f, c, v in zip(gj["features"], cols, values):
        f["properties"]["fill"] = c
        f["properties"]["value_fmt"] = "n/a" if pd.isna(v) else f"{v:,.1f}" if abs(v) < 100 else f"{v:,.0f}"
        f["properties"]["line"] = [11, 11, 11, 230] if highlight is not None and f["properties"]["SA2_CODE21"] in highlight else [255, 255, 255, 90]
        f["properties"]["lw"] = 180 if highlight is not None and f["properties"]["SA2_CODE21"] in highlight else 8
    layer = pdk.Layer("GeoJsonLayer", gj, pickable=True, stroked=True, filled=True, get_fill_color="properties.fill",
                      get_line_color="properties.line", get_line_width="properties.lw", line_width_min_pixels=0.5,
                      auto_highlight=True, highlight_color=[255, 200, 0, 120])
    deck = pdk.Deck(layers=[layer], initial_view_state=view or VIEW, map_provider="carto", map_style="light",
                    tooltip={"html": tooltip_html, "style": {"backgroundColor": "#0b0b0b", "color": "white", "fontSize": "12px"}})
    st.pydeck_chart(deck, height=height, width="stretch")
    return legend


def legend_bar(legend, fmt="{:,.0f}", none_label="None / not in reach"):
    items = "".join(
        f"<span style='display:inline-flex;align-items:center;margin-right:12px;font-size:0.8rem'>"
        f"<span style='width:14px;height:14px;background:{c};border-radius:3px;margin-right:5px;display:inline-block'></span>"
        f"{fmt.format(a)} to {fmt.format(b)}</span>" for c, a, b in legend)
    items += (f"<span style='display:inline-flex;align-items:center;font-size:0.8rem'><span style='width:14px;height:14px;"
              f"background:#e7e5df;border-radius:3px;margin-right:5px;display:inline-block'></span>{none_label}</span>")
    st.markdown(items, unsafe_allow_html=True)


def preview(text: str):
    st.markdown(f"<span class='tag'>PREVIEW</span> <span class='src'>{text}</span>", unsafe_allow_html=True)


def source(text: str):
    st.markdown(f"<div class='src'>{text}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------- pages ---

def overview():
    st.title("Sydney public transport: who gets let down?")
    st.markdown(
        f"<p class='lead'>Once buses are counted alongside trains and Metro, about <b>one in three residents "
        f"({META['poor_all_pct']:.0f}%)</b> has poor public transport access. The gap is sharpest where disadvantage "
        f"and poor access overlap: <b>{META['td_bus_suburbs']} suburbs with about {META['td_bus_people'] / 1000:,.0f},000 people</b>, "
        f"mostly in Campbelltown, Mount Druitt, parts of Bankstown and the Wyong lakes.</p>", unsafe_allow_html=True)

    c = st.columns(4)
    c[0].metric("Residents with poor access", f"{META['poor_all_pct']:.0f}%", f"{META['poor_all_pct'] - META['poor_rail_pct']:.0f} pts vs rail only", delta_color="inverse",
                help=define("Poor access") + " " + define("PTAL / access level"))
    c[1].metric("Jobs within 45 min (average resident)", f"{META['jobs45_all'] / 1000:,.0f}k", f"+{(META['jobs45_all'] - META['jobs45_rail']) / 1000:,.0f}k vs rail only",
                help=define("Jobs within 45 minutes"))
    c[2].metric("Transport-disadvantaged suburbs", f"{META['td_bus_suburbs']}", f"{META['td_bus_suburbs'] - META['td_rail_suburbs']} vs rail only", delta_color="inverse",
                help=define("Transport-disadvantaged suburb"))
    c[3].metric("Within a 1.5 km walk of a station", f"{META['walk_1500_pct']:.0f}%", help=define("Within a 1.5 km walk of a station"))
    explain("PTAL / access level", "Poor access", "Jobs within 45 minutes", "Transport-disadvantaged suburb", "IRSD decile",
            "No car %", "Minutes to board", "Suburb (SA2)")

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Where disadvantage meets poor access")
        sa2 = D["sa2"]
        td = sa2["transport_disadvantaged_with_bus"].fillna(False).astype(bool)
        v = sa2["pct_poor_all"].where(sa2["residents"] >= 500)
        legend = choropleth(v, "<b>{SA2_NAME21}</b><br/>Residents with poor access: {value_fmt}%<br/>IRSD decile: {irsd_decile}",
                            palette=["#fbe9e7", "#f4c3bd", "#e4938a", "#c85a5a", "#9b3434"],
                            bins=np.array([0, 10, 25, 50, 75, 100.1]), highlight=set(sa2.loc[td, "SA2_CODE21"]), height=500,
                            zero_is_none=False)
        legend_bar(legend, fmt="{:,.0f}%", none_label="Under 500 residents")
        source("Share of residents at PTAL 0 to 1b with rail + bus. Black outline: transport-disadvantaged suburbs.")
    with right:
        st.subheader("The 17 transport-disadvantaged suburbs")
        t = sa2[td].sort_values("erp_2025", ascending=False)[
            ["SA2_NAME21", "SA3_NAME21", "erp_2025", "irsd_decile", "no_car_pct", "pct_poor_all", "board_min_all"]]
        st.dataframe(t, hide_index=True, width="stretch", height=500, column_config={
            "SA2_NAME21": "Suburb", "SA3_NAME21": "District",
            "erp_2025": st.column_config.NumberColumn("Residents (2025)", format="%,d"),
            "irsd_decile": st.column_config.NumberColumn("IRSD decile", format="%d"),
            "no_car_pct": st.column_config.NumberColumn("No car %", format="%.1f"),
            "pct_poor_all": st.column_config.ProgressColumn("Poor access %", min_value=0, max_value=100, format="%.0f"),
            "board_min_all": st.column_config.NumberColumn("Min to board", format="%.0f"),
        })

    st.subheader("So what")
    st.markdown("""
1. **More frequent buses in the transport-disadvantaged suburbs.** Most are within reach of a bus already; frequency is the gap.
2. **New station entrances where the walk is much longer than it looks** (Valley Heights, Cheltenham, Birrong, Bella Vista, Turrella: walks 1.6 to 1.8 times the straight line).
3. **Rapid bus in the north-west and south-west growth areas before rail arrives.** Almost half of growth to 2041 is heading to suburbs without good rail access.
4. **Plan the Bankstown transition around the riders who left.** Most of the 5.8 million yearly entries at the closed stations didn't move to another station.
5. **Treat evening and weekend service as an equity issue.** Disadvantaged suburbs lose the most access outside commuter hours.
""")
    source("Directions for further study, not costed proposals. Committed projects (Metro West, Western Sydney Airport line, Bankstown Metro) are not yet modelled.")


METRICS = {
    "Access level (PTAL index)": ("ai_rail", "ai_all", "{:,.1f}", "PTAL access index (higher is better); Transport for London method", False),
    "Jobs within 45 minutes": ("jobs_rail", "jobs_all", "{:,.0f}", "Jobs reachable within 45 min leaving 8am", False),
    "Residents with poor access (%)": ("pct_poor_rail", "pct_poor_all", "{:,.0f}%", "Share of residents at PTAL 0 to 1b", True),
    "Minutes to board a service": ("board_min_rail", "board_min_all", "{:,.0f}", "Walk + half the gap between services + reliability allowance", True),
}


def access_map():
    st.title("Access to public transport")
    st.markdown("<p class='lead'>Flip between rail only and rail + bus to see how much buses change the picture.</p>", unsafe_allow_html=True)
    a, b, c = st.columns([1.4, 1, 1])
    metric = a.selectbox("Measure", list(METRICS), help="Hover the map for values. Definitions are below the map.")
    with_bus = b.toggle("Include buses", value=True, help="Off = trains and Metro only. Light rail and ferries are never included.")
    outline = c.toggle("Outline transport-disadvantaged suburbs", value=True, help=define("Transport-disadvantaged suburb"))
    rail_col, all_col, fmt, desc, reverse = METRICS[metric]
    sa2 = D["sa2"]
    col = all_col if with_bus else rail_col
    v = sa2[col].where(sa2["residents"] >= 500)
    if col.startswith("board"):
        v = v.where(v < 60)
    flag = "transport_disadvantaged_with_bus" if with_bus else "transport_disadvantaged"
    hl = set(sa2.loc[sa2[flag].fillna(False).astype(bool), "SA2_CODE21"]) if outline else None

    left, right = st.columns([2, 1])
    with left:
        legend = choropleth(v, f"<b>{{SA2_NAME21}}</b><br/>{metric}: {{value_fmt}}<br/>Rail only: {{{rail_col}}} | Rail + bus: {{{all_col}}}",
                            reverse=reverse, highlight=hl, height=600, zero_is_none=not reverse)
        legend_bar(legend, fmt=fmt.replace("%", ""))
        source(f"{desc}. Weekday AM peak, timetable for {META['timetable_date']}. Light rail and ferries not included.")
        explain("PTAL / access level", "PTAL index", "Jobs within 45 minutes", "Residents with poor access (%)", "Minutes to board",
                "Transport-disadvantaged suburb", "Hospitals / campuses / supermarkets in reach", "Jobs kept at 9pm / Sunday")
    with right:
        st.subheader("Look up a suburb")
        names = sa2.sort_values("SA2_NAME21")["SA2_NAME21"].tolist()
        pick = st.selectbox("Suburb (SA2)", names, index=names.index("Lethbridge Park - Tregear") if "Lethbridge Park - Tregear" in names else 0)
        r = sa2[sa2["SA2_NAME21"] == pick].iloc[0]
        st.caption(f"{r['SA3_NAME21']} · {r['erp_2025']:,.0f} residents (2025) · IRSD decile {r['irsd_decile']:.0f} · {r['no_car_pct']:.1f}% households without a car")
        comp = pd.DataFrame({
            "Measure": ["Jobs within 45 min", "PTAL index", "Poor access %", "Minutes to board"],
            "Rail only": [r["jobs_rail"], r["ai_rail"], r["pct_poor_rail"], r["board_min_rail"] if r["board_min_rail"] < 60 else None],
            "Rail + bus": [r["jobs_all"], r["ai_all"], r["pct_poor_all"], r["board_min_all"] if r["board_min_all"] < 60 else None],
        })
        st.caption("Blank minutes to board = no service within walking distance.")
        st.dataframe(comp, hide_index=True, width="stretch",
                     column_config={"Rail only": st.column_config.NumberColumn(format="%,.1f"),
                                    "Rail + bus": st.column_config.NumberColumn(format="%,.1f")})
        k1, k2 = st.columns(2)
        k1.metric("Hospitals in 45 min", f"{r['hospital_45']:.0f}", help=define("Hospitals / campuses / supermarkets in reach"))
        k2.metric("Supermarkets in 20 min", f"{r['supermarket_20']:.0f}", help=define("Hospitals / campuses / supermarkets in reach"))
        k1.metric("Jobs kept at 9pm", f"{r['keep_9pm_pct']:.0f}%" if pd.notna(r["keep_9pm_pct"]) else "n/a", help=define("Jobs kept at 9pm / Sunday"))
        k2.metric("Jobs kept on Sunday", f"{r['keep_sun_pct']:.0f}%" if pd.notna(r["keep_sun_pct"]) else "n/a", help=define("Jobs kept at 9pm / Sunday"))
        if pd.notna(r["growth_to_2041"]):
            st.caption(f"Projected growth to 2041: {r['growth_to_2041']:+,.0f} residents (TfNSW TZP24)")


def methods():
    st.title("Six international access methods")
    st.markdown("<p class='lead'>Standard planning methods applied to Sydney's rail and bus timetable. They all point to the same places.</p>",
                unsafe_allow_html=True)
    sa2 = D["sa2"]
    t1, t2, t3, t4, t5 = st.tabs(["Cumulative opportunities", "Gravity access", "Time of day", "Transit gap", "Access and unemployment"])
    with t1:
        dest = st.radio("Destination", ["Jobs (45 min)", "Hospitals (45 min)", "University and TAFE (45 min)", "Supermarkets (20 min)"], horizontal=True)
        col = {"Jobs (45 min)": "jobs_all", "Hospitals (45 min)": "hospital_45",
               "University and TAFE (45 min)": "education_45", "Supermarkets (20 min)": "supermarket_20"}[dest]
        legend = choropleth(sa2[col], f"<b>{{SA2_NAME21}}</b><br/>{dest}: {{value_fmt}}")
        legend_bar(legend)
        w = sa2["residents"].fillna(0)
        none = 100 * w[(sa2[col].fillna(0) == 0)].sum() / w.sum()
        st.caption(f"{none:.1f}% of residents have none within reach. Method: Wachs & Kumagai (1973), the basis of the UK DfT journey time statistics. "
                   "Destinations: 2021 Census jobs and OpenStreetMap.")
        explain("Jobs within 45 minutes", "Hospitals / campuses / supermarkets in reach")
    with t2:
        bins = np.array([0, 25, 50, 75, 100, 150, 250, 2000])
        pal = ["#9b3434", "#e4938a", "#f6dcd6", "#d9e6f5", "#86b6ef", "#2a78d6", "#0d366b"]
        legend = choropleth(sa2["gravity_index"], "<b>{SA2_NAME21}</b><br/>Gravity index: {value_fmt}", palette=pal, bins=bins)
        legend_bar(legend)
        g = sa2.dropna(subset=["irsd_decile"]).copy()
        g["group"] = pd.cut(g["irsd_decile"], [0, 3, 7, 10], labels=["Most disadvantaged (1-3)", "Middle (4-7)", "Least disadvantaged (8-10)"])
        avg = g.groupby("group", observed=True).apply(lambda x: np.average(x["gravity_index"].fillna(0), weights=x["residents"]), include_groups=False)
        st.caption("Sydney average = 100. Every job counts, weighted down by travel time (a job 20 minutes away counts half). "
                   + " · ".join(f"{k}: {v:.0f}" for k, v in avg.items()) + ". Method: Hansen (1959).")
        explain("Gravity index", "IRSD decile")
    with t3:
        hr = D["hourly"]
        res = sa2.set_index("SA2_CODE21")["residents"]
        hr = hr.assign(w=hr["SA2_CODE21"].map(res).fillna(0))
        avg = hr.groupby(["day", "hour"]).apply(lambda x: np.average(x["jobs45"], weights=x["w"]), include_groups=False).rename("jobs45").reset_index()
        avg["series"] = avg["day"].map({"weekday": "Sydney average, weekday", "sunday": "Sydney average, Sunday"})
        names = sa2.sort_values("SA2_NAME21")["SA2_NAME21"].tolist()
        picks = st.multiselect("Compare suburbs", names, default=["Lethbridge Park - Tregear", "Parramatta - South"]
                               if "Parramatta - South" in names else names[:1], max_selections=4)
        sub = hr.merge(sa2[["SA2_CODE21", "SA2_NAME21"]], on="SA2_CODE21")
        sub = sub[sub["SA2_NAME21"].isin(picks)].assign(series=lambda x: x["SA2_NAME21"] + ", " + x["day"])
        lines = pd.concat([avg[["hour", "jobs45", "series", "day"]], sub[["hour", "jobs45", "series", "day"]]])
        lines["jobs_k"] = lines["jobs45"] / 1000
        chart = alt.Chart(lines).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("hour:O", title="Leaving home at (hour of day)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("jobs_k:Q", title="Jobs reachable within 45 minutes (thousands)"),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom", columns=2)),
            strokeDash=alt.StrokeDash("day:N", scale=alt.Scale(domain=["weekday", "sunday"], range=[[1, 0], [5, 3]]), legend=None),
            tooltip=["series", "hour", alt.Tooltip("jobs45:Q", format=",.0f", title="Jobs")],
        ).properties(height=380)
        st.altair_chart(chart, width="stretch")
        c1, c2 = st.columns([1, 2])
        day = c1.radio("Map: day", ["weekday", "sunday"], horizontal=True)
        hour = c2.slider("Map: hour", 5, 23, 21)
        snap = hr[(hr["day"] == day) & (hr["hour"] == hour)].set_index("SA2_CODE21")["jobs45"]
        peak = hr[(hr["day"] == "weekday") & (hr["hour"] == 8)].set_index("SA2_CODE21")["jobs45"]
        v = sa2["SA2_CODE21"].map(snap)
        bins = np.unique(np.nanquantile(peak[peak > 0], np.linspace(0, 1, 8)))  # same colours at every hour
        legend = choropleth(v, f"<b>{{SA2_NAME21}}</b><br/>Jobs within 45 min at {hour}:00 ({day}): {{value_fmt}}", bins=bins)
        legend_bar(legend)
        st.caption("Each hour is averaged over departures at :00 and :30. Colours use the same scale at every hour (weekday 8am "
                   "quantiles), so the map fades as service thins. Method: time-of-day accessibility (e.g. Owen & Levinson, Access Across America).")
        explain("Jobs within 45 minutes", "Jobs kept at 9pm / Sunday")
    with t4:
        hi = sa2["need_z"] > 0.5
        v = sa2["transit_gap"].where(hi)
        bins = np.array([-5, -1, -0.5, 0, 0.5, 1, 1.5, 5])
        pal = ["#0d366b", "#5598e7", "#d9e6f5", "#f6dcd6", "#e4938a", "#c85a5a", "#7a1f1f"]
        vv = v + 5  # shift so the colouring helper treats all values as present
        legend = choropleth(vv.where(hi), "<b>{SA2_NAME21}</b><br/>Transit gap: {transit_gap}<br/>Need: {need_z} · Supply: {supply_z}",
                            palette=pal, bins=bins + 5)
        legend_bar([(c, a - 5, b - 5) for c, a, b in legend], fmt="{:+.1f}", none_label="Not high-need")
        top = sa2[hi & (sa2["residents"] >= 2000)].nlargest(10, "transit_gap")[["SA2_NAME21", "SA3_NAME21", "need_z", "supply_z", "transit_gap"]]
        st.dataframe(top.round(2), hide_index=True, width="stretch",
                     column_config={"SA2_NAME21": "Suburb", "SA3_NAME21": "District", "need_z": "Need (z)", "supply_z": "Supply (z)", "transit_gap": "Gap"})
        st.caption("High-need suburbs only (need above +0.5 sd). Need = disadvantage, no-car households, ages 65+ and 15 to 24; supply = PTAL. Method: Currie (2010).")
        explain("Transit gap", "PTAL index")
    with t5:
        s = sa2[(sa2["residents"] >= 1000) & sa2["unemployment_rate"].notna() & (sa2["gravity_index"] > 0)].copy()
        s["group"] = pd.cut(s["irsd_decile"], [0, 3, 7, 10], labels=["Most disadvantaged (1-3)", "Middle (4-7)", "Least disadvantaged (8-10)"]).astype(str)
        rho = s["gravity_index"].rank().corr(s["unemployment_rate"].rank())
        chart = alt.Chart(s).mark_circle(opacity=0.65, stroke="white", strokeWidth=0.5).encode(
            x=alt.X("gravity_index:Q", scale=alt.Scale(type="log"), title="Gravity job access index (Sydney = 100, log scale)"),
            y=alt.Y("unemployment_rate:Q", title="Unemployment rate, 2021 (%)"),
            size=alt.Size("residents:Q", legend=None, scale=alt.Scale(range=[15, 300])),
            color=alt.Color("group:N", scale=alt.Scale(domain=["Most disadvantaged (1-3)", "Middle (4-7)", "Least disadvantaged (8-10)"],
                                                         range=[RED, GREY, BLUE]), title="IRSD"),
            tooltip=["SA2_NAME21", alt.Tooltip("gravity_index:Q", format=".0f"), alt.Tooltip("unemployment_rate:Q", format=".1f"), "group"],
        ).properties(height=480).interactive()
        st.altair_chart(chart, width="stretch")
        st.info(f"Spearman ρ = {rho:+.2f}. Better access goes with slightly higher unemployment. This is a correlation, not a cause: "
                "both reflect where lower-income households can find housing. Lens: UK Social Exclusion Unit (2003), Lucas (2012).")
        explain("Gravity index", "IRSD decile")
        st.caption("Unemployment rate = unemployed people as a % of the labour force, 2021 Census. Spearman ρ compares rankings: "
                   "+1 = perfectly matched order, 0 = no relationship.")


def demand():
    st.title("Demand vs supply")
    stn = D["stations"]
    t1, t2, t3 = st.tabs(["Busy stations, thin service", "Ten years of ridership", "Bankstown line closure"])
    with t1:
        peak = st.radio("Peak", ["AM peak (6 to 10am)", "PM peak (3 to 7pm)"], horizontal=True,
                        help=define("AM / PM boardings per train"))
        pm = peak.startswith("PM")
        epd_col, dep_col = ("pm_entries_per_departure", "pm_departures") if pm else ("entries_per_departure", "am_departures")
        st.caption("In the morning the busiest boarding stations are suburban hubs; in the evening they are the CBD and job centres, "
                   "as people head home. Greater Sydney stations only; Newcastle and the Hunter have their own page.")
        left, right = st.columns([1.6, 1])
        with left:
            s = stn.dropna(subset=["avg_daily_entries"]).copy()
            s["radius"] = np.sqrt(s["avg_daily_entries"]) * 18
            s["epd"] = s[epd_col].fillna(0)
            mx = s["epd"].quantile(0.97)
            lo, hi = np.array(hex_rgb("#cfe0f4")[:3]), np.array(hex_rgb("#b3261e")[:3])
            s["color"] = s["epd"].apply(lambda x: [*(lo + (hi - lo) * min(x / mx, 1)).astype(int).tolist(), 210])
            seg = D["segments"]
            lines = pdk.Layer("LineLayer", seg, get_source_position=["a_lon", "a_lat"], get_target_position=["b_lon", "b_lat"],
                              get_color=[150, 150, 150, 120], get_width=1.2)
            dots = pdk.Layer("ScatterplotLayer", s, get_position=["lon", "lat"], get_radius="radius", get_fill_color="color",
                             pickable=True, stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=0.5)
            st.pydeck_chart(pdk.Deck(layers=[lines, dots], initial_view_state=VIEW, map_provider="carto", map_style="light",
                                     tooltip={"html": "<b>{station}</b> ({lines})<br/>Entries per day: {avg_daily_entries}<br/>"
                                                      "AM boardings per train: {entries_per_departure}<br/>PM boardings per train: "
                                                      "{pm_entries_per_departure}<br/>AM trains/hour: {am_peak_tph}"}),
                            height=560, width="stretch")
            source(f"Circle size = daily entries; colour = {'evening' if pm else 'morning'}-peak boardings per scheduled train (red = most per train).")
        with right:
            top = stn[stn["share_source"] == "2023 typical day"].nlargest(15, epd_col)
            chart = alt.Chart(top).mark_bar(color=RED if pm else BLUE, cornerRadiusEnd=3).encode(
                x=alt.X(f"{epd_col}:Q", title=f"{'PM' if pm else 'AM'} peak boardings per scheduled train"),
                y=alt.Y("station:N", sort="-x", title=None),
                tooltip=["station", "lines", alt.Tooltip(f"{epd_col}:Q", format=".0f", title="Boardings per train"),
                         alt.Tooltip(f"{dep_col}:Q", title="Departures in the peak")],
            ).properties(height=520)
            st.altair_chart(chart, width="stretch")
        explain("AM / PM boardings per train", "Daily entries", "Trains per hour (AM peak)")
    with t2:
        tl = D["timeline"].melt("month", var_name="series", value_name="entries")
        tl["series"] = tl["series"].map({"network": "All stations", "sydenham_bankstown_closed": "9 Sydenham-Bankstown stations",
                                         "metro_city_new": "5 new Metro City stations"})
        tl["entries_m"] = tl["entries"] / 1e6
        which = st.radio("Show", ["All stations", "Closure vs new Metro"], horizontal=True)
        data = tl[tl["series"] == "All stations"] if which == "All stations" else tl[tl["series"] != "All stations"]
        events = pd.DataFrame({"month": pd.to_datetime(["2020-03-01", "2021-07-01", "2024-08-01", "2024-09-30"]),
                               "label": ["COVID", "Delta lockdown", "Metro City opens", "Bankstown line closes"]})
        line = alt.Chart(data).mark_line(strokeWidth=2.2).encode(
            x=alt.X("month:T", title=None), y=alt.Y("entries_m:Q", title="Station entries per month (millions)"),
            color=alt.Color("series:N", scale=alt.Scale(range=[BLUE, RED, METRO]), title=None),
            tooltip=[alt.Tooltip("month:T", format="%b %Y"), "series", alt.Tooltip("entries_m:Q", format=".2f")])
        rules = alt.Chart(events).mark_rule(strokeDash=[4, 4], color="#6b6a66").encode(x="month:T")
        text = alt.Chart(events).mark_text(align="left", dx=4, dy=-140, fontSize=11, color="#6b6a66").encode(x="month:T", text="label")
        st.altair_chart((line + rules + text).properties(height=420), width="stretch")
        source("TfNSW station entries and exits (Opal), trains and Metro; light rail excluded.")
    with t3:
        bk = D["bankstown"].copy()
        bk["change_k"] = bk["change"] / 1000
        chart = alt.Chart(bk).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("change_k:Q", title="Change in yearly entries (thousands), 12 months before vs latest 12"),
            y=alt.Y("station:N", sort="x", title=None),
            color=alt.Color("group:N", scale=alt.Scale(domain=["Closed", "Alternative"], range=[GREY, BLUE]), title=None),
            tooltip=["station", "group", alt.Tooltip("entries_before:Q", format=",.0f"), alt.Tooltip("entries_after:Q", format=",.0f"),
                     alt.Tooltip("pct_change:Q", format="+.0f")],
        ).properties(height=600)
        st.altair_chart(chart, width="stretch")
        st.caption("The nine closed stations had 5.8 million entries a year. About 3.1 million reappeared above network trend at Sydenham; "
                   "about 4.7 million aren't seen at any station (replacement buses, other modes, or not travelling).")


def trains_vs_metro():
    st.title("Sydney Trains vs Sydney Metro")
    m = D["modes"].reset_index().rename(columns={"index": "measure"}).melt("measure", var_name="mode", value_name="value")
    m["label"] = m["value"].apply(lambda v: f"{v:,.2f}" if abs(v) < 1 else f"{v:,.1f}" if abs(v) < 100 else f"{v:,.0f}")
    base = alt.Chart(m).encode(
        x=alt.X("mode:N", title=None, sort=["Trains", "Metro"], axis=alt.Axis(labelAngle=0)),
        y=alt.Y("value:Q", title=None, axis=None),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, width=46).encode(
        color=alt.Color("mode:N", scale=alt.Scale(domain=["Trains", "Metro"], range=[BLUE, METRO]), legend=None))
    text = base.mark_text(dy=-8).encode(text="label:N")
    st.altair_chart((bars + text).properties(width=190, height=170).facet(
        facet=alt.Facet("measure:N", title=None, header=alt.Header(labelFontSize=12, labelLimit=260)), columns=3
    ).resolve_scale(y="independent"), width="content")
    source("Median station, Sydney basin, shared interchanges excluded. Reliability measures are a one-day preview.")
    explain("Trains per hour (AM peak)", "Daily entries", "AM / PM boardings per train", "Entries per walk-up resident", "On time", "Minutes late")
    stn = D["stations"]
    s = stn[stn["mode"].isin(["Trains", "Metro"]) & stn["lat"].between(-34.2, -33.5)].copy()
    a, b = st.columns(2)
    with a:
        st.subheader("Trains per hour, AM peak")
        st.altair_chart(alt.Chart(s).mark_tick(thickness=2, size=26, opacity=0.6).encode(
            x=alt.X("am_peak_tph:Q", title="Trains per hour at the station (both directions)"),
            y=alt.Y("mode:N", title=None), color=alt.Color("mode:N", scale=alt.Scale(domain=["Trains", "Metro"], range=[BLUE, METRO]), legend=None),
            tooltip=["station", "lines", "am_peak_tph"]).properties(height=200), width="stretch")
    with b:
        st.subheader("Entries per walk-up resident")
        st.altair_chart(alt.Chart(s.dropna(subset=["entries_per_resident"])).mark_tick(thickness=2, size=26, opacity=0.6).encode(
            x=alt.X("entries_per_resident:Q", scale=alt.Scale(type="log"), title="Daily entries per resident within 1.5 km (log)"),
            y=alt.Y("mode:N", title=None), color=alt.Color("mode:N", scale=alt.Scale(domain=["Trains", "Metro"], range=[BLUE, METRO]), legend=None),
            tooltip=["station", alt.Tooltip("entries_per_resident:Q", format=".2f")]).properties(height=200), width="stretch")
    st.subheader("Who rides each line")
    lines = D["lines"].dropna(subset=["am_peak_tph"])
    st.altair_chart(alt.Chart(lines).mark_circle(opacity=0.8, color=BLUE).encode(
        x=alt.X("am_peak_tph:Q", title="Median AM peak trains per hour at the line's stations"),
        y=alt.Y("concession_senior_pct:Q", title="Trips on concession or senior cards (%)", scale=alt.Scale(zero=False)),
        size=alt.Size("annual_trips:Q", legend=None, scale=alt.Scale(range=[60, 900])),
        tooltip=["Line", alt.Tooltip("annual_trips:Q", format=",.0f"), "concession_senior_pct", "am_peak_tph"])
        + alt.Chart(lines).mark_text(dx=12, align="left").encode(x="am_peak_tph:Q", y="concession_senior_pct:Q", text="code:N"),
        width="stretch")


def reliability():
    st.title("Reliability")
    preview(f"Built from {META['reliability_obs']:,} measured train arrivals, {META['reliability_window']}. "
            "Patterns need a few weeks of collection, including peaks and wet days.")
    st.markdown("Delays come from train GPS positions (every 30 seconds), with the feed's last prediction as fallback.")
    explain("Minutes late", "On time", "Passenger minutes lost", "Excess wait time")
    hm = D["heatmap"]
    st.subheader("Average lateness by line and hour")
    st.altair_chart(alt.Chart(hm).mark_rect().encode(
        x=alt.X("hour:O", title="Hour (Sydney time)", axis=alt.Axis(labelAngle=0)), y=alt.Y("route_short_name:N", title=None),
        color=alt.Color("late_min:Q", scale=alt.Scale(scheme="redyellowblue", reverse=True, domain=[-2, 8]), title="Minutes late"),
        tooltip=["route_short_name", "hour", alt.Tooltip("late_min:Q", format=".1f"), "n"]).properties(height=420),
        width="stretch")
    st.subheader("Busiest stations: passenger minutes lost")
    stn = D["stations"].dropna(subset=["passenger_minutes_per_day"]).nlargest(15, "passenger_minutes_per_day")
    st.altair_chart(alt.Chart(stn).mark_bar(color=BLUE, cornerRadiusEnd=3).encode(
        x=alt.X("passenger_minutes_per_day:Q", title="Passenger minutes lost per day (entries x average minutes late)"),
        y=alt.Y("station:N", sort="-x", title=None),
        tooltip=["station", alt.Tooltip("avg_late_min:Q", format=".1f"), alt.Tooltip("on_time_pct:Q", format=".0f")]).properties(height=420),
        width="stretch")
    st.subheader("Excess wait time on frequent platforms")
    e = D["ewt"]
    by = e.groupby("mode").apply(lambda g: pd.Series({"Scheduled wait (min)": np.average(g["swt_min"], weights=g["trains"]),
                                                      "Actual wait (min)": np.average(g["awt_min"], weights=g["trains"])}),
                                 include_groups=False).round(2)
    st.dataframe(by, width="stretch")
    source("Transport for London's excess wait time: the extra average wait caused by uneven gaps between trains. Platforms with a train at least every 10 minutes.")


def hunter():
    st.title("Newcastle and the Hunter")
    h = META["hunter"]
    st.markdown("<p class='lead'>Reported as its own region, separate from the Greater Sydney figures. Served by NSW TrainLink "
                "(Central Coast & Newcastle and Hunter lines) and Newcastle buses. Light rail and ferries are not included.</p>",
                unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Residents (2021)", f"{h['residents']:,}", help="ABS SA4s Newcastle and Lake Macquarie, and Hunter Valley exc Newcastle")
    c[1].metric("Residents with poor access", f"{h['poor_all_pct']:.0f}%", f"{h['poor_all_pct'] - h['poor_rail_pct']:.0f} pts vs rail only",
                delta_color="inverse", help=define("Poor access"))
    c[2].metric("Within a 1.5 km walk of a station", f"{h['within_1500m_walk_pct']:.0f}%",
                help="Straight-line distance x 1.28 (the median detour measured in Sydney); footpaths are not modelled here.")
    c[3].metric("Stations / daily departures", f"{h['stations']} / {h['daily_departures']:,}")
    props = pd.DataFrame([f["properties"] for f in D["hunter_geo"]["features"]])
    a, b = st.columns([1.4, 1])
    with a:
        with_bus = st.toggle("Include buses", value=True, key="hunter_bus")
        col = "pct_poor_all" if with_bus else "pct_poor_rail"
        view = pdk.ViewState(latitude=-32.85, longitude=151.55, zoom=8.3)
        legend = choropleth(props[col], "<b>{SA2_NAME21}</b><br/>Residents with poor access: {value_fmt}%<br/>Walk to station within "
                            "1.5 km: {pct_within_1500m_walk}%", palette=["#fbe9e7", "#f4c3bd", "#e4938a", "#c85a5a", "#9b3434"],
                            bins=np.array([0, 10, 25, 50, 75, 100.1]), zero_is_none=False, geo=D["hunter_geo"], view=view, height=520)
        legend_bar(legend, fmt="{:,.0f}%")
    with b:
        stn = D["hunter_stations"].sort_values("daily_departures", ascending=False)
        st.dataframe(stn[["station", "lines", "daily_departures", "am_peak_tph", "avg_daily_entries", "avg_late_min"]], hide_index=True,
                     width="stretch", height=520, column_config={
                         "station": "Station", "lines": "Lines",
                         "daily_departures": st.column_config.NumberColumn("Departures / day", help="Both directions, Tue 27 Oct 2026"),
                         "am_peak_tph": st.column_config.NumberColumn("AM trains / hour", format="%.1f", help=define("Trains per hour (AM peak)")),
                         "avg_daily_entries": st.column_config.NumberColumn("Entries / day", format="%.0f", help=define("Daily entries")),
                         "avg_late_min": st.column_config.NumberColumn("Avg min late (preview)", format="%.1f", help=define("Minutes late")),
                     })
    source("HUN = Hunter line, CCN = Central Coast & Newcastle line, NRC/NRW = regional XPT and Xplorer trains. "
           "Hunter line delays come from NSW TrainLink's own realtime feed (collected from 26 Oct 2026).")
    explain("Poor access", "PTAL / access level", "Trains per hour (AM peak)", "Daily entries", "Minutes late")


def glossary():
    st.title("Glossary")
    st.markdown("<p class='lead'>Every measure in the dashboard, in plain English.</p>", unsafe_allow_html=True)
    q = st.text_input("Search", placeholder="e.g. PTAL, gravity, boardings")
    for term, text in GLOSSARY.items():
        if not q or q.lower() in (term + text).lower():
            st.markdown(f"**{term}**  \n{text}")


def about():
    st.title("About this project")
    st.markdown("""
**Question:** Where does Sydney's public transport let the most people down, and why?

**Data:** Transport for NSW GTFS and GTFS-realtime (trains, Metro, buses), Opal patronage, ABS Census 2021, SEIFA,
ABS population estimates, TfNSW TZP24 projections, OpenStreetMap, Bureau of Meteorology.

**Methods:** Transport for London's PTAL; cumulative opportunities (Wachs & Kumagai 1973); gravity access (Hansen 1959);
time-of-day accessibility; transit gap index (Currie 2010); excess wait time (TfL). Travel times use a connection-scan search over
the full timetable. Delays are measured from train GPS positions.

**Scope:** reliability is Sydney Trains and Metro only; buses are timetable only; light rail and ferries are not included.
Committed projects (Metro West, the Western Sydney Airport line, Bankstown Metro) are not yet modelled.

Code, decisions log and data pipeline: see the project README.
""")


pg = st.navigation([
    st.Page(overview, title="Overview", icon="📊", default=True),
    st.Page(access_map, title="Access map", icon="🗺️"),
    st.Page(methods, title="Access methods", icon="🧭"),
    st.Page(demand, title="Demand vs supply", icon="👥"),
    st.Page(trains_vs_metro, title="Trains vs Metro", icon="🚇"),
    st.Page(reliability, title="Reliability (preview)", icon="⏱️"),
    st.Page(hunter, title="Newcastle & Hunter", icon="🚂"),
    st.Page(glossary, title="Glossary", icon="📖"),
    st.Page(about, title="About", icon="ℹ️"),
])
pg.run()
