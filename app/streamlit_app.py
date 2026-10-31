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
import streamlit.components.v1 as components

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
    "Running lateness": "How late a vehicle was while out on its route, checked every minute, averaged over every vehicle running "
                        "on the line in that hour. Trains and Metro: the latest delay measured from GPS at a station. Buses, "
                        "ferries and light rail: the delay the operator's realtime feed was reporting for the next stop (its own "
                        "estimate). Early running counts as 0 in the commuter figures.",
    "Running 5+ min late": "Share of the minutes vehicles on the line spent running 5 or more minutes behind the timetable.",
    "Passenger hours lost per day": "Average daily Opal trips on the line (last 12 months) x the line's average running lateness "
                                    "(early = 0), assuming passengers are spread across the day in proportion to the service "
                                    "running. A rough size of the problem, not a measured total. Not available for buses: TfNSW "
                                    "only publishes bus trips by contract region, not by route.",
    "New entrance gain": "Residents (2021 Census) whose walk to the station would drop from over 800 m to 800 m or less "
                         "(about 10 minutes) if one extra entrance were added at the best spot within 200 m of the platforms. "
                         "Walks follow real streets and paths (OpenStreetMap) to today's entrances (TfNSW, 2020).",
    "Excess wait time": "Transport for London's measure for turn-up-and-go services: how much longer the average passenger waits "
                        "than the timetable implies, because trains bunch or leave gaps.",
    "Planned connection": "For every arrival at a station and every other line (or bus route) leaving it, the first departure "
                          "the timetable lets you catch: at least 3 minutes after you arrive for a train-to-train change, or 2 "
                          "minutes plus the walk for a bus. Waits over 20 minutes aren't counted as a connection.",
    "Missed connection": "The planned departure actually left before you could reach it, because your train or bus arrived late "
                         "(or the other one left early). Counted per connection the timetable offers, not per passenger.",
    "Extra wait": "When a connection is missed: how much later the next departure you can catch on the same line and direction "
                  "actually left, compared with the one you planned on.",
    "Delay gained": "How much later a train is leaving a station than it was leaving the one before: the time it lost on that "
                    "stretch, including its stop. Negative = it made time up. Only where both times were measured from GPS.",
    "Became 5+ min late": "The stretch where a train that started on time first ran 5 or more minutes late.",
    "Late start": "A train leaving its first station 2 or more minutes late (from the feed's prediction, as GPS isn't used at origins).",
    "Running time": "Time from a train leaving one station (last GPS fix near it) to arriving at the next (first fix near it). "
                    "Speed = straight-line distance / running time, so curving track reads slower than it is.",
    "Time lost to slow running": "A train's running time minus that stretch's best runs (the fastest 10% of trains): time lost to "
                                 "congestion, signals, speed restrictions or waiting for a platform.",
    "Train-minutes lost per day": "Time lost to slow running, added up over every train on the stretch, per day collected. Shows "
                                  "where the most total time goes, so busy stretches rank high.",
    "Bunched": "Two services arriving less than half the timetabled gap apart (e.g. under 5 minutes apart on a 10-minute "
               "service). Only for services every 15 minutes or more often.",
    "Big gap": "A gap between services more than 1.5 times the timetabled one (e.g. over 15 minutes on a 10-minute service).",
    "Regularity": "How uneven the gaps are: the spread (standard deviation) of actual gap / timetabled gap. 0 = perfectly "
                  "even; 1 = gaps typically off by a whole timetabled gap.",
    "Platform time (dwell)": "How long a train spends at a platform. Metro: first GPS fix stopped at the platform to the first fix "
                             "after leaving. Sydney Trains: time in the platform's signalling location, including braking and "
                             "pulling away. A trip's first and last stops are left out.",
    "Extra in peak": "A station's median platform time at 7 to 9am and 5 to 7pm minus its median at 10am to 3pm. Comparing a "
                     "station with itself cancels out the measurement differences between stations.",
    "Occupancy level": "The operator's category for how full a vehicle is: many seats available, few seats available, or "
                       "standing room only. From load sensors on newer trains and Metro; TfNSW doesn't publish how buses set it.",
    "% standing": "Share of departures (trains, Metro) or bus-minutes reported as standing room only.",
    "Fill time": "The first time a Park&Ride car park reached 95% of its spaces that weekday morning, interpolated between the "
                 "15-minute readings. Typical = median over the weekdays collected.",
    "Peak trains after full": "Share of the station's scheduled departures from 6 to 10am (all lines, both directions) that leave "
                              "after the car park is full: the service a commuter who drives in later can't park for.",
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


REPLAY_DIR, REPLAY_TEMPLATE = Path(__file__).parent / "static" / "replay", Path(__file__).parent / "replay_map.html"


@st.cache_data
def replay_html(version: float) -> str:
    """The animated map with the rail network inlined. Vehicle data isn't inlined: the page fetches
    app/static/replay/<mode>.json itself, so a rebuilt replay shows up on reload."""
    seg = D["segments"][["mode", "a_lon", "a_lat", "b_lon", "b_lat"]].to_json(orient="records")
    return REPLAY_TEMPLATE.read_text(encoding="utf-8").replace("__SEGMENTS__", seg)


def replay():
    st.title("Live replay")
    if not any(REPLAY_DIR.glob("*.json")):
        st.info("No replay data yet. Run `python -m analysis.replay` to build app/static/replay/.")
        return
    preview("Every train, bus, ferry and light rail vehicle's GPS position from the realtime feeds, replayed for the "
            "latest 24 hours collected. Buses, ferries and light rail were added on 27 Oct 2026; gaps are times the "
            "collector wasn't running.")
    st.markdown("Press **Play**, or drag the slider. Click the buttons along the top to show or hide each kind of "
                "transport. Each dot is a vehicle, coloured by how late it was running at that moment. Hover a dot for "
                "its line or route.")
    components.html(replay_html(REPLAY_TEMPLATE.stat().st_mtime), height=650)
    source("Trains and Metro: latest delay measured at a station from train GPS. Buses, ferries and light rail: the delay "
           "the operator's feed was reporting for the next stop (its own estimate). GPS fixes arrive every 30 to 60 seconds; "
           "the replay keeps one per vehicle per minute and moves the dot in a straight line between fixes, so it can cut "
           "corners. A vehicle is hidden across gaps of 5+ minutes. Buses are a large download (tens of MB for a full day).")


LINE_HOUR, LINE_TRIPS = DATA / "line_hour.csv", DATA / "line_trips.csv"
MODE_COLOURS = {"Trains": BLUE, "Metro": METRO, "TrainLink": "#e0a526", "Light rail": RED, "Ferry": "#3a9a5b", "Bus": "#7a5bb5"}
WORK_DAYS = 230  # working days a year, for "time lost a year" if you commute at the same hour


@st.cache_data
def load_line_hour(version: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    lh = pd.read_csv(LINE_HOUR, dtype={"line": str})
    lh["label"] = np.where(lh["mode"] == "Bus", "Bus " + lh["line"], lh["line"])
    trips = (pd.read_csv(LINE_TRIPS, dtype={"line": str}) if LINE_TRIPS.exists()
             else pd.DataFrame(columns=["mode", "line", "daily_trips"]))
    return lh, trips


def weighted(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Combine rows (days, or hours) weighting each by the minutes of service it covers."""
    cols = ["avg_late", "avg_late_pos", "pct_late5"]
    w = df.assign(**{c: df[c] * df["vehicle_minutes"] for c in cols})
    g = w.groupby(by, as_index=False)[["vehicle_minutes", *cols]].sum()
    for c in cols:
        g[c] = g[c] / g["vehicle_minutes"]
    return g


def line_delays():
    st.title("Line delays by hour")
    if not LINE_HOUR.exists():
        st.info("No data yet. Run `python -m analysis.replay` then `python -m analysis.line_delays`.")
        return
    lh, trips = load_line_hour(LINE_HOUR.stat().st_mtime)
    days = sorted(lh["date"].unique(), reverse=True)
    preview(f"{len(days)} day(s) of realtime data ({days[-1]} to {days[0]}); buses, ferries and light rail from 27 Oct 2026. "
            "Hour-by-hour patterns need a few weeks, including wet days, to be dependable.")
    st.markdown("<p class='lead'>When is each line late, and what does that cost the people who ride it?</p>",
                unsafe_allow_html=True)
    explain("Running lateness", "Running 5+ min late", "Passenger hours lost per day")

    present = [m for m in MODE_COLOURS if m in set(lh["mode"])]
    a, b, c = st.columns([2.2, 1, 1.2])
    modes = a.pills("Transport", present, selection_mode="multi", default=present) or present
    day = b.selectbox("Day", ["All days (average)"] + days)
    measure = c.radio("Colour by", ["Minutes late", "% running 5+ min late"], horizontal=True)
    if measure == "Minutes late":
        col, title = "avg_late", "Minutes late"
        scale = alt.Scale(scheme="redyellowblue", reverse=True, domain=[-2, 8], clamp=True)
    else:
        col, title = "pct_late5", "% 5+ min late"
        scale = alt.Scale(scheme="orangered", domain=[0, 40], clamp=True)

    d = lh[lh["mode"].isin(modes)]
    one_day = d if day.startswith("All") else d[d["date"] == day]
    by_hour = weighted(one_day, ["mode", "label", "hour"])
    totals = weighted(one_day, ["mode", "label"])
    totals = totals[totals["vehicle_minutes"] >= 60]  # at least an hour of service seen
    totals = totals.merge(trips.rename(columns={"line": "label"})[["mode", "label", "daily_trips"]], on=["mode", "label"], how="left")
    totals["pax_hours"] = totals["daily_trips"] * totals["avg_late_pos"] / 60
    tips = [alt.Tooltip("avg_late:Q", format=".1f", title="Minutes late"),
            alt.Tooltip("pct_late5:Q", format=".0f", title="% 5+ min late"),
            alt.Tooltip("vehicle_minutes:Q", title="Minutes of service seen")]

    st.subheader("Every line, hour by hour")
    order = st.radio("Show the 30 lines with", ["the most delay", "the most service", "the most passenger hours lost"],
                     horizontal=True)
    key = {"the most delay": "avg_late_pos", "the most service": "vehicle_minutes",
           "the most passenger hours lost": "pax_hours"}[order]
    top = totals.dropna(subset=[key]).nlargest(30, key)
    if top.empty:
        st.caption("No lines with Opal trip counts in the transport selected (bus routes don't have them).")
    else:
        hm = by_hour[by_hour["label"].isin(top["label"])]
        st.altair_chart(alt.Chart(hm).mark_rect().encode(
            x=alt.X("hour:O", title="Hour (Sydney time)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("label:N", sort=top["label"].tolist(), title=None, axis=alt.Axis(labelOverlap=False)),
            color=alt.Color(f"{col}:Q", scale=scale, title=title),
            tooltip=[alt.Tooltip("label:N", title="Line"), "mode", "hour", *tips]).properties(height=max(200, 20 * len(top))),
            width="stretch")

    st.subheader("One line, day by day")
    lines = totals.sort_values("vehicle_minutes", ascending=False)["label"].tolist()
    if not lines:
        st.info("Not enough service seen for the transport selected.")
        return
    line = st.selectbox("Line or route", lines)
    ld = d[d["label"] == line]
    st.altair_chart(alt.Chart(ld).mark_rect().encode(
        x=alt.X("hour:O", title="Hour (Sydney time)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("date:O", title=None, sort="descending"),
        color=alt.Color(f"{col}:Q", scale=scale, title=title),
        tooltip=["date", "hour", *tips]).properties(height=max(120, 34 * ld["date"].nunique())),
        width="stretch")

    st.subheader(f"What it means if you ride {line}")
    prof = weighted(ld, ["hour"])
    hours = prof["hour"].tolist()
    hour = st.select_slider("You usually travel at", hours, value=8 if 8 in hours else hours[0], format_func=lambda h: f"{h}:00")
    row = prof[prof["hour"] == hour].iloc[0]
    k = st.columns(4)
    k[0].metric("Typical delay", f"{row['avg_late_pos']:.1f} min", help="Average running lateness at that hour, early counted as 0.")
    odds = f" (1 in {100 / row['pct_late5']:.0f})" if row["pct_late5"] >= 1 else ""
    k[1].metric("Running 5+ min late", f"{row['pct_late5']:.0f}%{odds}")
    k[2].metric("Time lost a year", f"{row['avg_late_pos'] * WORK_DAYS / 60:.1f} hours",
                help=f"Typical delay x {WORK_DAYS} working days, one trip a day at this hour.")
    t = totals[totals["label"] == line]
    pax = t["pax_hours"].iloc[0] if len(t) else np.nan
    k[3].metric("Passenger hours lost a day", "n/a" if pd.isna(pax) else f"{pax:,.0f}", help=define("Passenger hours lost per day"))
    st.altair_chart(alt.Chart(prof).mark_bar(color=MODE_COLOURS.get(ld["mode"].iloc[0], BLUE), cornerRadiusEnd=2).encode(
        x=alt.X("hour:O", title="Hour (Sydney time)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("avg_late_pos:Q", title="Typical delay (min, early = 0)"),
        opacity=alt.condition(alt.datum.hour == hour, alt.value(1), alt.value(0.45)),
        tooltip=["hour", alt.Tooltip("avg_late_pos:Q", format=".1f", title="Typical delay"),
                 alt.Tooltip("pct_late5:Q", format=".0f", title="% 5+ min late")]).properties(height=220), width="stretch")

    st.subheader("Where delays cost passengers the most")
    pax = totals.dropna(subset=["pax_hours"]).nlargest(15, "pax_hours")
    if pax.empty:
        st.caption("No lines with Opal trip counts in the transport selected (bus routes don't have them).")
    else:
        st.altair_chart(alt.Chart(pax).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("pax_hours:Q", title="Passenger hours lost per day (estimate)"),
            y=alt.Y("label:N", sort="-x", title=None),
            color=alt.Color("mode:N", scale=alt.Scale(domain=list(MODE_COLOURS), range=list(MODE_COLOURS.values())), title=None),
            tooltip=[alt.Tooltip("label:N", title="Line"), "mode", alt.Tooltip("daily_trips:Q", format=",.0f", title="Daily trips"),
                     alt.Tooltip("avg_late_pos:Q", format=".2f", title="Typical delay (min)"),
                     alt.Tooltip("pax_hours:Q", format=",.0f", title="Passenger hours lost")]).properties(height=28 * len(pax) + 40),
            width="stretch")
    source("Running lateness from the realtime feeds (analysis/replay.py, analysis/line_delays.py); daily trips from TfNSW Opal "
           "trips by line, last 12 months. Compare lines within a mode: bus, ferry and light rail delays are the operators' own "
           "estimates, train and Metro delays are measured from GPS.")


ENTRANCES = DATA / "entrance_options.csv"
ESRI_STYLE = "/app/static/esri_light_style.json"  # muted Esri street map, no API key needed (Carto now needs one)


def entrances():
    st.title("Station entrance shortlist")
    if not ENTRANCES.exists():
        st.info("No results yet. Run `python -m analysis.entrance_options`.")
        return
    opts = pd.read_csv(ENTRANCES)
    gain = pd.read_csv(DATA / "entrance_gain_points.csv")
    ent = pd.read_csv(DATA / "station_entrances.csv")
    st.markdown("<p class='lead'>New lines take decades. A new station entrance takes a year or two, and can put "
                "thousands more people within a 10-minute walk of a train straight away. Where would one extra "
                "entrance help the most people?</p>", unsafe_allow_html=True)
    explain("New entrance gain", "Within a 1.5 km walk of a station")

    top = opts.head(10)
    k = st.columns(3)
    k[0].metric("Stations tested", f"{len(opts)}")
    k[1].metric("Top 10: residents brought within 800 m", f"{top['residents_new_800m'].sum():,.0f}")
    k[2].metric("Best single station", f"{top.iloc[0]['station']}", f"+{top.iloc[0]['residents_new_800m']:,.0f} residents",
                delta_color="off")

    st.subheader("The shortlist")
    n = st.slider("Stations shown", 5, 40, 15)
    show = opts.head(n)
    st.altair_chart(alt.Chart(show).mark_bar(color=BLUE, cornerRadiusEnd=3).encode(
        x=alt.X("residents_new_800m:Q", title="Residents brought within an 800 m walk"),
        y=alt.Y("station:N", sort="-x", title=None),
        tooltip=["station", "lines", alt.Tooltip("entrances_now:Q", title="Entrances today"),
                 alt.Tooltip("residents_within_800m_now:Q", format=",", title="Within 800 m today"),
                 alt.Tooltip("residents_new_800m:Q", format=",", title="Brought within 800 m"),
                 alt.Tooltip("pct_increase_800m:Q", title="% more than today"),
                 alt.Tooltip("residents_new_1500m:Q", format=",", title="Brought within 1.5 km"),
                 alt.Tooltip("walk_saved_m:Q", format=",.0f", title="Average walk saved (m)")]).properties(height=24 * n + 30),
        width="stretch")
    table = show[["station", "lines", "entrances_now", "residents_within_800m_now", "residents_new_800m", "pct_increase_800m",
                  "residents_new_1500m", "walk_saved_m"]].rename(columns={
        "station": "Station", "lines": "Lines", "entrances_now": "Entrances today", "residents_within_800m_now": "Within 800 m today",
        "residents_new_800m": "Brought within 800 m", "pct_increase_800m": "% more", "residents_new_1500m": "Brought within 1.5 km",
        "walk_saved_m": "Avg walk saved (m)"})
    st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("Where the new entrance would go")
    mapped = [s_ for s_ in opts["station"] if s_ in set(gain["station"])]
    station = st.selectbox("Station", mapped)
    row = opts[opts["station"] == station].iloc[0]
    g = gain[gain["station"] == station].assign(radius=lambda d: 3 + np.sqrt(d["residents"]) * 0.6)
    e = ent[ent["station"] == station]
    new = pd.DataFrame({"lon": [row["entrance_lon"]], "lat": [row["entrance_lat"]], "label": ["Possible new entrance"]})
    layers = [
        pdk.Layer("ScatterplotLayer", g, get_position=["lon", "lat"], get_radius="radius", radius_units="'pixels'",
                  get_fill_color=[42, 120, 214, 150], pickable=True),
        pdk.Layer("ScatterplotLayer", e.assign(label="Entrance today"), get_position=["lon", "lat"], get_radius=7,
                  radius_units="'pixels'", get_fill_color=[11, 11, 11, 230], get_line_color=[255, 255, 255], stroked=True,
                  line_width_min_pixels=2, pickable=True),
        pdk.Layer("ScatterplotLayer", new, get_position=["lon", "lat"], get_radius=10, radius_units="'pixels'",
                  get_fill_color=[200, 90, 90, 240], get_line_color=[255, 255, 255], stroked=True, line_width_min_pixels=2,
                  pickable=True),
    ]
    view = pdk.ViewState(latitude=row["entrance_lat"], longitude=row["entrance_lon"], zoom=14.3)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_provider="carto", map_style=ESRI_STYLE,
                             tooltip={"html": "{label}<br>{residents} residents: walk {walk_now_m} m today, {walk_new_m} m with the new entrance",
                                      "style": {"backgroundColor": "#0b0b0b", "color": "white", "fontSize": "12px"}}),
                    height=520)
    st.caption(f"Red: best spot found for a new entrance. Black: entrances today. Blue: homes (mesh blocks, sized by residents) "
               f"that would come within an 800 m walk: {row['residents_new_800m']:,.0f} residents, "
               f"{row['pct_increase_800m']:.0f}% more than the {row['residents_within_800m_now']:,.0f} within 800 m today.")
    source("Method: analysis/entrance_options.py. Candidate spots are path nodes within 200 m of the platforms; the best is the one "
           "that brings the most residents within 800 m. Not checked: whether a spot can physically reach the platforms (bridges, "
           "lifts), land ownership or cost. Paths missing from OpenStreetMap, or entrances opened since 2020, can overstate a gain. "
           "A shortlist for feasibility work, not a design.")


# ------------------------------------------------- operations pages ---
# Built by analysis/operations.py from the realtime data collected so far; re-run it as more days come in.

OPS_PREVIEW = ("Built from the realtime feeds collected so far (Sydney Trains and Metro from 26 Oct 2026, buses from 27 Oct). "
               "A few weekdays only, and collection pauses overnight (about 10:30pm to 7:45am, PC asleep), so the morning "
               "peak before 8am is thin. Read the patterns as a first look, not a verdict.")


@st.cache_data
def load_ops(name: str, version: float) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv", dtype={"line": str, "from_line": str, "to_line": str})


def ops(*names: str):
    """The operations CSVs, or None (with a note) if analysis/operations.py hasn't been run yet."""
    paths = [DATA / f"{n}.csv" for n in names]
    if not all(p.exists() for p in paths):
        st.info("No data yet. Run `python -m analysis.reliability` then `python -m analysis.operations`.")
        return None
    out = [load_ops(n, p.stat().st_mtime) for n, p in zip(names, paths)]
    return out if len(out) > 1 else out[0]


def clock(h: float) -> str:
    if pd.isna(h):
        return ""
    hh, mm = divmod(int(round(h * 60)), 60)
    return f"{hh}:{mm:02d}"


def rate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Missed share and waits from summed counts (so days, hours and line pairs combine correctly)."""
    g = df.groupby(by, as_index=False)[["connections", "missed", "extra_wait_min"]].sum()
    g["pct_missed"] = 100 * g["missed"] / g["connections"]
    g["wait_if_missed"] = g["extra_wait_min"] / g["missed"].where(g["missed"] > 0)
    g["extra_per_100"] = 100 * g["extra_wait_min"] / g["connections"]
    return g


def pooled(df: pd.DataFrame, by: list[str], weight: str, cols: list[str]) -> pd.DataFrame:
    """Weighted average of percentage columns over the rows being combined."""
    w = df.assign(**{c: df[c] * df[weight] for c in cols}).groupby(by, as_index=False)[[weight, *cols]].sum()
    for c in cols:
        w[c] = w[c] / w[weight]
    return w


def connections_page():
    st.title("Missed connections")
    data = ops("connections", "connections_hour")
    if data is None:
        return
    pairs, hours = data
    preview(OPS_PREVIEW)
    st.markdown("<p class='lead'>When a train or bus runs late, how often does it make people miss the connection the "
                "timetable promised them, and how long do they then wait?</p>", unsafe_allow_html=True)
    explain("Planned connection", "Missed connection", "Extra wait")
    kinds = [k for k in ["Train to train", "Bus to train", "Train to bus"] if k in set(pairs["kind"])]
    kind = st.radio("Change", kinds, horizontal=True)
    p, h = pairs[pairs["kind"] == kind], hours[hours["kind"] == kind]
    tot = rate(p, ["kind"]).iloc[0]
    k = st.columns(4)
    k[0].metric("Planned connections seen", f"{tot['connections']:,.0f}")
    k[1].metric("Missed", f"{tot['pct_missed']:.1f}%")
    k[2].metric("Average extra wait when missed", f"{tot['wait_if_missed']:.1f} min", help=define("Extra wait"))
    k[3].metric("Extra minutes per 100 connections", f"{tot['extra_per_100']:.0f}")

    st.subheader("Stations where connections are missed most")
    min_n = st.slider("Only stations with at least this many planned connections", 50, 2000, 300, step=50)
    by_st = rate(p, ["station"])
    by_st = by_st[by_st["connections"] >= min_n].nlargest(20, "pct_missed")
    st.altair_chart(alt.Chart(by_st).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("pct_missed:Q", title="% of planned connections missed"),
        y=alt.Y("station:N", sort="-x", title=None),
        color=alt.Color("wait_if_missed:Q", scale=alt.Scale(scheme="orangered"), title="Extra wait if missed (min)"),
        tooltip=["station", alt.Tooltip("connections:Q", format=","), alt.Tooltip("pct_missed:Q", format=".1f", title="% missed"),
                 alt.Tooltip("wait_if_missed:Q", format=".1f", title="Extra wait if missed (min)")]).properties(
        height=24 * len(by_st) + 30), width="stretch")
    st.caption("On lines every few minutes (Metro) a miss costs little: the colour shows the wait it caused.")

    st.subheader("By hour")
    hr = rate(h, ["hour"])
    st.altair_chart(alt.Chart(hr).mark_bar(color=RED, cornerRadiusEnd=2).encode(
        x=alt.X("hour:O", title="Hour of arrival (Sydney time)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("pct_missed:Q", title="% missed"),
        tooltip=["hour", alt.Tooltip("connections:Q", format=","), alt.Tooltip("pct_missed:Q", format=".1f"),
                 alt.Tooltip("wait_if_missed:Q", format=".1f", title="Extra wait if missed (min)")]).properties(height=220),
        width="stretch")

    st.subheader("One station, line by line")
    stations = rate(p, ["station"]).sort_values("connections", ascending=False)["station"].tolist()
    station = st.selectbox("Station", stations, index=stations.index("Tallawong") if "Tallawong" in stations else 0)
    one = rate(p[p["station"] == station], ["from_line", "to_line"]).sort_values("connections", ascending=False)
    st.dataframe(one.rename(columns={"from_line": "Arriving on", "to_line": "Changing to", "connections": "Connections",
                                     "pct_missed": "% missed", "wait_if_missed": "Extra wait if missed (min)"})[
        ["Arriving on", "Changing to", "Connections", "% missed", "Extra wait if missed (min)"]].round(1),
        hide_index=True, width="stretch")
    sh = rate(h[h["station"] == station], ["hour"])
    st.altair_chart(alt.Chart(sh).mark_bar(color=BLUE, cornerRadiusEnd=2).encode(
        x=alt.X("hour:O", title="Hour of arrival", axis=alt.Axis(labelAngle=0)), y=alt.Y("pct_missed:Q", title="% missed"),
        tooltip=["hour", "connections", alt.Tooltip("pct_missed:Q", format=".1f")]).properties(height=180), width="stretch")
    source("analysis/connections.py. Counts are per connection the timetable offers, not per passenger (nobody publishes how "
           "many people change where). Train times are measured from GPS; bus times are the operator's own predictions. "
           "Buses are counted at stops within 250 m of a station.")


def delay_origins_page():
    st.title("Where delays start")
    data = ops("delay_gains", "late_starts")
    if data is None:
        return
    gains, starts = data
    preview(OPS_PREVIEW)
    st.markdown("<p class='lead'>A late train is usually late everywhere after the place it lost the time. Find that place "
                "and you find what to fix.</p>", unsafe_allow_html=True)
    explain("Delay gained", "Became 5+ min late", "Late start")
    gains = gains.assign(stretch=gains["from_station"] + " → " + gains["station"])
    lines = gains.groupby("line")["trains"].sum().sort_values(ascending=False).index.tolist()
    line = st.selectbox("Line", ["All lines"] + lines)
    g = gains if line == "All lines" else gains[gains["line"] == line]
    min_n = st.slider("Only stretches with at least this many trains measured", 10, 300, 50, step=10)
    g = g[g["trains"] >= min_n]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Most time lost per train")
        top = g.nlargest(15, "avg_gain_s").assign(avg_gain_min=lambda d: d["avg_gain_s"] / 60)
        st.altair_chart(alt.Chart(top).mark_bar(color=RED, cornerRadiusEnd=3).encode(
            x=alt.X("avg_gain_min:Q", title="Average delay gained (min)"),
            y=alt.Y("stretch:N", sort="-x", title=None),
            tooltip=["line", "stretch", "trains", alt.Tooltip("avg_gain_min:Q", format=".2f", title="Minutes gained"),
                     alt.Tooltip("pct_gain_2min:Q", format=".0f", title="% of trains losing 2+ min")]).properties(
            height=24 * len(top) + 30), width="stretch")
    with c2:
        st.subheader("Where trains first became 5+ min late")
        late = g[g["became_late5"] > 0].nlargest(15, "became_late5")
        if late.empty:
            st.caption("No trains here became 5+ minutes late in the data so far.")
        else:
            st.altair_chart(alt.Chart(late).mark_bar(color=BLUE, cornerRadiusEnd=3).encode(
                x=alt.X("became_late5:Q", title="Trains"), y=alt.Y("stretch:N", sort="-x", title=None),
                tooltip=["line", "stretch", "trains", "became_late5"]).properties(height=24 * len(late) + 30), width="stretch")
    st.subheader("Trains that start late")
    s = starts if line == "All lines" else starts[starts["line"] == line]
    st.dataframe(s.sort_values("pct_start_late2", ascending=False).rename(columns={
        "line": "Line", "station": "Starts at", "trips": "Trips", "pct_start_late2": "% leaving 2+ min late",
        "avg_start_delay_s": "Average delay leaving (s)"})[["Line", "Starts at", "Trips", "% leaving 2+ min late",
                                                            "Average delay leaving (s)"]], hide_index=True, width="stretch")
    source("analysis/delay_origins.py. Delay gained uses only station pairs where both times were measured from GPS. "
           "Departures from a trip's first station are the feed's prediction, so late starts are indicative.")


def slow_segments_page():
    st.title("Slow stretches")
    seg = ops("slow_segments")
    if seg is None:
        return
    preview(OPS_PREVIEW)
    st.markdown("<p class='lead'>Between which stations do trains crawl, and where is the most time lost against what the "
                "track allows on a good run?</p>", unsafe_allow_html=True)
    explain("Running time", "Time lost to slow running", "Train-minutes lost per day")
    seg = seg.assign(stretch=seg["from_station"] + " → " + seg["station"])
    present = [m for m in ["Trains", "Metro", "TrainLink"] if m in set(seg["mode"])]
    modes = st.pills("Transport", present, selection_mode="multi", default=[m for m in present if m != "TrainLink"]) or present
    s = seg[seg["mode"].isin(modes)]
    worst = s["lost_min_per_day"].quantile(0.95) if len(s) else 1
    s = s.assign(colour=[hex_rgb(RED, 230) if v >= worst else hex_rgb(BLUE, 90 + int(140 * min(v / worst, 1)))
                         for v in s["lost_min_per_day"]], width=1 + 6 * (s["lost_min_per_day"] / worst).clip(0, 1))
    layer = pdk.Layer("LineLayer", s, get_source_position=["from_lon", "from_lat"], get_target_position=["lon", "lat"],
                      get_color="colour", get_width="width", width_units="'pixels'", pickable=True)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=VIEW, map_provider="carto", map_style=ESRI_STYLE,
                             tooltip={"html": "<b>{stretch}</b> ({lines})<br>{trains} trains, median {median_kmh} km/h<br>"
                                              "{lost_s} s lost per train, {lost_min_per_day} train-minutes a day",
                                      "style": {"backgroundColor": "#0b0b0b", "color": "white", "fontSize": "12px"}}),
                    height=520)
    st.caption("Thicker and darker = more train-minutes lost a day; red = the worst 5%. Drawn straight between stations, "
               "one line per direction (they overlap).")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Most train-minutes lost a day")
        t = s.nlargest(15, "lost_min_per_day")
        st.altair_chart(alt.Chart(t).mark_bar(color=RED, cornerRadiusEnd=3).encode(
            x=alt.X("lost_min_per_day:Q", title="Train-minutes lost per day"), y=alt.Y("stretch:N", sort="-x", title=None),
            tooltip=["stretch", "lines", "trains", alt.Tooltip("lost_s:Q", title="Seconds lost per train"),
                     alt.Tooltip("peak_lost_s:Q", title="... in peak hours"), "median_kmh"]).properties(
            height=24 * len(t) + 30), width="stretch")
    with c2:
        st.subheader("Slowest stretches (over 1 km)")
        t = s[s["dist_km"] >= 1].nsmallest(15, "median_kmh")
        st.altair_chart(alt.Chart(t).mark_bar(color=BLUE, cornerRadiusEnd=3).encode(
            x=alt.X("median_kmh:Q", title="Median speed, km/h (straight line)"), y=alt.Y("stretch:N", sort="x", title=None),
            tooltip=["stretch", "lines", "trains", "dist_km", "median_run_s", "best_run_s", "p90_run_s"]).properties(
            height=24 * len(t) + 30), width="stretch")
    source("analysis/slow_segments.py. Running time is from the last GPS fix near one station to the first near the next, so "
           "it leaves out the stops. Straight-line distance understates curving track: compare stretches with each other.")


def headways_page():
    st.title("Bunching and gaps")
    data = ops("headways", "headway_stops")
    if data is None:
        return
    hw, stops = data
    preview(OPS_PREVIEW)
    st.markdown("<p class='lead'>On a turn-up-and-go service the spacing matters more than the timetable. Two buses together "
                "followed by a long gap means a longer wait and a fuller vehicle.</p>", unsafe_allow_html=True)
    explain("Bunched", "Big gap", "Regularity", "Excess wait time")
    modes = [m for m in ["Bus", "Metro", "Trains", "TrainLink"] if m in set(hw["mode"])]
    mode = st.radio("Transport", modes, horizontal=True)
    h = hw[hw["mode"] == mode]
    lines = pooled(h, ["line"], "gaps", ["pct_bunched", "pct_big_gap"])
    k = st.columns(3)
    allm = pooled(h.assign(all=1), ["all"], "gaps", ["pct_bunched", "pct_big_gap"]).iloc[0]
    k[0].metric("Gaps between services measured", f"{allm['gaps']:,.0f}")
    k[1].metric("Bunched", f"{allm['pct_bunched']:.1f}%", help=define("Bunched"))
    k[2].metric("Big gaps", f"{allm['pct_big_gap']:.1f}%", help=define("Big gap"))
    min_n = st.slider("Only lines with at least this many gaps measured", 20, 1000, 100, step=10)
    lines = lines[lines["gaps"] >= min_n].nlargest(25, "pct_bunched")
    long = lines.melt(id_vars=["line", "gaps"], value_vars=["pct_bunched", "pct_big_gap"], var_name="kind", value_name="pct")
    long["kind"] = long["kind"].map({"pct_bunched": "Bunched", "pct_big_gap": "Big gap"})
    colours = alt.Scale(domain=["Bunched", "Big gap"], range=[RED, GREY])
    st.subheader("Least evenly spaced lines")
    st.altair_chart(alt.Chart(long).mark_bar().encode(
        x=alt.X("pct:Q", title="% of gaps between services"), y=alt.Y("line:N", sort=lines["line"].tolist(), title=None),
        color=alt.Color("kind:N", scale=colours, title=None), yOffset="kind:N",
        tooltip=["line", "kind", alt.Tooltip("pct:Q", format=".1f"), "gaps"]).properties(height=30 * len(lines) + 30),
        width="stretch")
    st.subheader("By hour")
    hr = pooled(h, ["hour"], "gaps", ["pct_bunched", "pct_big_gap"]).rename(columns={"pct_bunched": "Bunched", "pct_big_gap": "Big gap"})
    st.altair_chart(alt.Chart(hr.melt(id_vars=["hour"], value_vars=["Bunched", "Big gap"], var_name="kind", value_name="pct"))
                    .mark_line(point=True).encode(
        x=alt.X("hour:O", title="Hour (Sydney time)", axis=alt.Axis(labelAngle=0)), y=alt.Y("pct:Q", title="% of gaps"),
        color=alt.Color("kind:N", scale=colours, title=None),
        tooltip=["hour", "kind", alt.Tooltip("pct:Q", format=".1f")]).properties(height=220), width="stretch")
    st.subheader("Worst places")
    s = stops[(stops["mode"] == mode) & (stops["gaps"] >= 30)].nlargest(20, "pct_bunched")
    st.dataframe(s.rename(columns={"line": "Line", "station": "At or next to", "gaps": "Gaps measured",
                                   "pct_bunched": "% bunched", "pct_big_gap": "% big gap", "regularity": "Regularity",
                                   "sched_gap_min": "Timetabled gap (min)"}).drop(columns="mode"), hide_index=True, width="stretch")
    source("analysis/headways.py. Services every 15 minutes or more often only. Buses are measured at stops within 250 m of a "
           "station, from the operator's own predicted times; trains and Metro from GPS. See the Reliability page for excess "
           "wait time on frequent train platforms.")


def dwell_page():
    st.title("Dwell times")
    data = ops("dwell", "dwell_stations")
    if data is None:
        return
    dw, stn = data
    preview(OPS_PREVIEW)
    st.markdown("<p class='lead'>How long trains stand at each platform, and which stations slow down in the peak, a sign "
                "of crowded platforms and doors.</p>", unsafe_allow_html=True)
    explain("Platform time (dwell)", "Extra in peak")
    mode = st.radio("Transport", ["Trains", "Metro"], horizontal=True)
    s = stn[(stn["mode"] == mode) & (stn["stops"] >= 50)].dropna(subset=["extra_peak_s"])
    st.subheader("Stations that slow down most in the peak")
    t = s.nlargest(20, "extra_peak_s")
    st.altair_chart(alt.Chart(t).mark_bar(color=RED, cornerRadiusEnd=3).encode(
        x=alt.X("extra_peak_s:Q", title="Extra platform time, peak vs midday (seconds, median)"),
        y=alt.Y("station:N", sort="-x", title=None),
        tooltip=["station", "stops", alt.Tooltip("offpeak_median_s:Q", title="Midday (s)"),
                 alt.Tooltip("peak_median_s:Q", title="Peak (s)")]).properties(height=24 * len(t) + 30), width="stretch")
    st.subheader("Platform time by hour, busiest stations")
    busiest = s.nlargest(25, "stops")["station"].tolist()
    d = dw[(dw["mode"] == mode) & dw["station"].isin(busiest)]
    lo, hi = d["median_s"].quantile([0.05, 0.95]).tolist() if len(d) else (0, 1)
    st.altair_chart(alt.Chart(d).mark_rect().encode(
        x=alt.X("hour:O", title="Hour (Sydney time)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("station:N", sort=busiest, title=None),
        color=alt.Color("median_s:Q", scale=alt.Scale(scheme="orangered", domain=[lo, hi], clamp=True), title="Median (s)"),
        tooltip=["station", "hour", "stops", "median_s", "p90_s"]).properties(height=20 * len(busiest) + 30), width="stretch")
    source("analysis/dwell.py. Metro: from the first GPS fix stopped at the platform to the first fix after leaving (~30 s "
           "resolution). Sydney Trains: time spent in the platform's signalling location, which includes braking in and "
           "pulling out, so it reads longer than the doors-open time. Compare stations within a mode.")


def crowding_page():
    st.title("Crowding")
    data = ops("crowding", "crowding_bus")
    if data is None:
        return
    cr, bus = data
    preview(OPS_PREVIEW)
    st.markdown("<p class='lead'>Where and when are trains, Metro and buses full enough that people stand?</p>",
                unsafe_allow_html=True)
    explain("Occupancy level", "% standing")

    m = cr[cr["mode"] == "Metro"]
    if len(m):
        st.subheader("Metro: share of departures with standing room only")
        dirs = m.groupby("towards")["reports"].sum().sort_values(ascending=False)
        towards = st.radio("Towards", dirs.index[:2].tolist(), horizontal=True)
        d = m[m["towards"] == towards]
        order = d.drop_duplicates("station").sort_values("order")["station"].tolist()
        st.altair_chart(alt.Chart(d).mark_rect().encode(
            x=alt.X("hour:O", title="Hour of departure (Sydney time)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("station:N", sort=order, title=None),
            color=alt.Color("pct_standing:Q", scale=alt.Scale(scheme="orangered", domain=[0, 100]), title="% standing"),
            tooltip=["station", "hour", "reports", alt.Tooltip("pct_standing:Q", format=".0f", title="% standing room only"),
                     alt.Tooltip("pct_few_seats:Q", format=".0f", title="% few seats or worse")]).properties(
            height=22 * len(order) + 30), width="stretch")

    st.subheader("Train lines")
    r = cr[cr["mode"] != "Metro"]
    lines = pooled(r, ["line"], "reports", ["pct_few_seats", "pct_standing"])
    lines = lines[lines["reports"] >= 50].sort_values("pct_few_seats", ascending=False)
    st.dataframe(lines.rename(columns={"line": "Line", "reports": "Departures with occupancy", "pct_few_seats": "% few seats or worse",
                                       "pct_standing": "% standing"}).round(1), hide_index=True, width="stretch")
    if len(lines):
        line = st.selectbox("Line, station by station", lines["line"].tolist())
        d = r[r["line"] == line]
        towards = d.groupby("towards")["reports"].sum().idxmax()
        d = d[d["towards"] == towards]
        order = d.drop_duplicates("station").sort_values("order")["station"].tolist()
        st.altair_chart(alt.Chart(d).mark_rect().encode(
            x=alt.X("hour:O", title="Hour of departure", axis=alt.Axis(labelAngle=0)), y=alt.Y("station:N", sort=order, title=None),
            color=alt.Color("pct_few_seats:Q", scale=alt.Scale(scheme="orangered", domain=[0, 100]), title="% few seats or worse"),
            tooltip=["station", "hour", "reports", alt.Tooltip("pct_few_seats:Q", format=".0f"),
                     alt.Tooltip("pct_standing:Q", format=".0f")]).properties(height=18 * len(order) + 30), width="stretch")
        st.caption(f"Trains towards {towards}, the direction with the most reports.")

    st.subheader("Busiest bus routes")
    hours = st.slider("Hours", 0, 23, (6, 10))
    b = pooled(bus[bus["hour"].between(*hours)], ["line"], "reports", ["pct_few_seats", "pct_standing"])
    b = b[b["reports"] >= 200].nlargest(20, "pct_standing")
    st.altair_chart(alt.Chart(b).mark_bar(color=MODE_COLOURS["Bus"], cornerRadiusEnd=3).encode(
        x=alt.X("pct_standing:Q", title="% of bus-minutes with standing room only"),
        y=alt.Y("line:N", sort="-x", title=None),
        tooltip=["line", alt.Tooltip("reports:Q", title="Bus-minutes"), alt.Tooltip("pct_standing:Q", format=".1f"),
                 alt.Tooltip("pct_few_seats:Q", format=".1f", title="% few seats or worse")]).properties(
        height=24 * len(b) + 30), width="stretch")
    source("analysis/crowding.py. Trains and Metro: the occupancy as each train left each station. Buses: every bus in "
           "service, once a minute, so long routes count more. School routes (S...) run only at school times. Metro reports "
           "occupancy on every train, Sydney Trains on about half (trains with load sensors), buses on every bus.")


def carparks_page():
    st.title("Commuter car parks")
    data = ops("carpark_fill", "carpark_curves")
    if data is None:
        return
    fill, curves = data
    preview("Park&Ride occupancy collected every 15 minutes from 26 Oct 2026; weekdays only.")
    st.markdown("<p class='lead'>When does each commuter car park fill, and how much of the morning's train service is "
                "left for people who arrive after that?</p>", unsafe_allow_html=True)
    explain("Fill time", "Peak trains after full")
    seen = fill[fill["fill_seen"] & fill["fill_hour"].notna()]
    s = fill.groupby(["carpark", "station"], dropna=False).agg(
        spaces=("spaces", "max"), days=("date", "size"), days_full=("fill_hour", "count"),
        full_by=("fill_hour", "median"), am_departures=("am_departures", "max")).reset_index()
    s = s.merge(seen.groupby("carpark").agg(fill_hour=("fill_hour", "median"), pct_after=("pct_am_after_full", "median"))
                .reset_index(), on="carpark", how="left")
    s["observed"] = s["fill_hour"].notna()
    s["fill_hour"] = s["fill_hour"].fillna(s["full_by"])  # not seen filling: only known to be full by the first reading
    s["Typical fill time"] = [clock(h) if o or pd.isna(h) else f"by {clock(h)}" for h, o in zip(s["fill_hour"], s["observed"])]
    full = s.dropna(subset=["fill_hour"]).sort_values("fill_hour")
    k = st.columns(3)
    k[0].metric("Car parks", f"{len(s)}")
    k[1].metric("Full on most weekdays", f"{(s['days_full'] > s['days'] / 2).sum()}")
    if len(full):
        k[2].metric("First to fill", full.iloc[0]["carpark"], full.iloc[0]["Typical fill time"], delta_color="off")
    st.caption("The collector runs on a home PC that sleeps overnight, so readings usually start between 7 and 8am. A car park "
               "already full by then shows as \"by\" that time: it filled earlier, and its share of trains after full is unknown.")
    st.subheader("Fill time and the trains left after it")
    rail = full.dropna(subset=["station"]).assign(tph=lambda d: d["am_departures"] / 4)
    st.altair_chart(alt.Chart(rail).mark_circle(opacity=0.8).encode(
        x=alt.X("fill_hour:Q", title="Typical fill time (hour, Sydney)", scale=alt.Scale(zero=False)),
        color=alt.Color("observed:N", scale=alt.Scale(domain=[True, False], range=[BLUE, GREY]),
                        legend=alt.Legend(title="Filling seen", labelExpr="datum.value ? 'Yes' : 'No, full by then'")),
        y=alt.Y("tph:Q", title="Trains per hour, 6 to 10am (both directions)"),
        size=alt.Size("spaces:Q", title="Spaces"),
        tooltip=["carpark", "station", "spaces", "Typical fill time", alt.Tooltip("tph:Q", format=".1f", title="Trains per hour"),
                 alt.Tooltip("pct_after:Q", format=".0f", title="% of 6-10am trains after full")]).properties(height=380),
        width="stretch")
    st.dataframe(s.sort_values("fill_hour").rename(columns={
        "carpark": "Car park", "station": "Station", "spaces": "Spaces", "days": "Weekdays seen", "days_full": "Days it filled",
        "am_departures": "Trains 6-10am", "pct_after": "% of those after full"})[
        ["Car park", "Station", "Spaces", "Typical fill time", "Days it filled", "Weekdays seen", "Trains 6-10am",
         "% of those after full"]], hide_index=True, width="stretch")
    st.subheader("A morning, car park by car park")
    names = sorted(curves["carpark"].unique())
    default = [n for n in names if n.startswith(("Tallawong", "Kellyville"))][:4] or names[:3]
    pick = st.multiselect("Car parks", names, default=default)
    day = st.selectbox("Day", sorted(curves["date"].unique(), reverse=True))
    c = curves[curves["carpark"].isin(pick) & (curves["date"] == day) & curves["time_local"].between(4, 12)]
    st.altair_chart(alt.Chart(c).mark_line().encode(
        x=alt.X("time_local:Q", title="Time (hour, Sydney)", scale=alt.Scale(domain=[4, 12])),
        y=alt.Y("pct_full:Q", title="% full", scale=alt.Scale(domain=[0, 100])), color=alt.Color("carpark:N", title=None),
        tooltip=["carpark", alt.Tooltip("time_local:Q", format=".2f", title="Hour"), "pct_full"]).properties(height=300),
        width="stretch")
    source("analysis/carpark_fill.py. Full = 95% of spaces, interpolated between the 15-minute readings. Train counts are "
           "scheduled departures on a typical weekday (latest timetable). Car parks at B-Line bus stops have no station.")


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

**Scope:** reliability is Sydney Trains and Metro, measured from GPS; the operations pages (connections, bunching,
crowding) add buses from the operator's realtime feed; access analyses use the train, Metro and bus timetables; light
rail and ferries appear only on the Live replay and Line delays pages.
Committed projects (Metro West, the Western Sydney Airport line, Bankstown Metro) are not yet modelled.

Code, decisions log and data pipeline: see the project README.
""")


pg = st.navigation([
    st.Page(overview, title="Overview", icon="📊", default=True),
    st.Page(access_map, title="Access map", icon="🗺️"),
    st.Page(methods, title="Access methods", icon="🧭"),
    st.Page(entrances, title="Station entrances", icon="🚪"),
    st.Page(demand, title="Demand vs supply", icon="👥"),
    st.Page(trains_vs_metro, title="Trains vs Metro", icon="🚇"),
    st.Page(reliability, title="Reliability (preview)", icon="⏱️"),
    st.Page(line_delays, title="Line delays by hour", icon="🕒"),
    st.Page(delay_origins_page, title="Where delays start", icon="📍"),
    st.Page(slow_segments_page, title="Slow stretches", icon="🐢"),
    st.Page(connections_page, title="Missed connections", icon="🔁"),
    st.Page(headways_page, title="Bunching and gaps", icon="🚌"),
    st.Page(dwell_page, title="Dwell times", icon="⏸️"),
    st.Page(crowding_page, title="Crowding", icon="🧍"),
    st.Page(carparks_page, title="Car parks", icon="🅿️"),
    st.Page(replay, title="Live replay", icon="▶️"),
    st.Page(hunter, title="Newcastle & Hunter", icon="🚂"),
    st.Page(glossary, title="Glossary", icon="📖"),
    st.Page(about, title="About", icon="ℹ️"),
])
pg.run()
