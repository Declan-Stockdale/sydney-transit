"""
Excess wait time (EWT) for frequent Sydney Trains and Metro services: how much
longer passengers wait than the timetable implies, because trains bunch or
leave gaps. This is Transport for London's headline measure for frequent bus
routes, where people turn up without checking a timetable.

For each platform (one direction of travel, all lines that use it), over the
period collected:

    scheduled wait (SWT) = sum(h_sched^2) / (2 x sum(h_sched))
    actual wait (AWT)    = sum(h_actual^2) / (2 x sum(h_actual))
    EWT                  = AWT - SWT

where h are the gaps between consecutive trains (scheduled, and measured from
GPS or the last prediction). Uneven gaps raise the average wait even when the
number of trains is right: two trains 2 and 8 minutes apart average a 3.4
minute wait, against 2.5 minutes for even 5-minute gaps.

Only platforms averaging a train at least every 10 minutes are included
(turn-up-and-go), and gaps over 30 minutes are dropped as data gaps.

PREVIEW: needs a few weeks of collection, including peaks.

Output: data/marts/excess_wait.csv (per platform) and a per-line summary printed.

Usage:
    python -m analysis.reliability
    python -m analysis.excess_wait
"""

import numpy as np
import pandas as pd

from analysis.reliability import MARTS_DIR

MAX_SCHED_GAP_MIN = 10
MAX_GAP_MIN = 30


def wait(h: np.ndarray) -> float:
    return float((h ** 2).sum() / (2 * h.sum())) if len(h) and h.sum() > 0 else np.nan


def main() -> None:
    d = pd.read_parquet(MARTS_DIR / "stop_delays.parquet")
    d = d[~d["cancelled"] & d["obs_time"].notna() & d["sched_time"].notna() & d["station"].notna()]
    rows = []
    for (feed, stop), g in d.groupby(["feed", "stop_id"]):
        s = np.sort(g["sched_time"].to_numpy()) / 60
        a = np.sort(g["obs_time"].to_numpy()) / 60
        hs, ha = np.diff(s), np.diff(a)
        hs, ha = hs[hs <= MAX_GAP_MIN], ha[ha <= MAX_GAP_MIN]
        if len(hs) < 6 or np.median(hs) > MAX_SCHED_GAP_MIN:
            continue
        rows.append({
            "feed": feed, "stop_id": stop, "station": g["station"].iloc[0],
            "lines": " ".join(sorted(set(g["route_short_name"].dropna()))), "trains": len(g),
            "median_gap_min": np.median(hs), "swt_min": wait(hs), "awt_min": wait(ha),
        })
    e = pd.DataFrame(rows)
    e["ewt_min"] = e["awt_min"] - e["swt_min"]
    e["mode"] = np.where(e["feed"] == "metro", "Metro", "Trains")
    e.to_csv(MARTS_DIR / "excess_wait.csv", index=False)

    w = lambda g: pd.Series({"platforms": len(g), "swt": np.average(g["swt_min"], weights=g["trains"]),
                             "awt": np.average(g["awt_min"], weights=g["trains"]),
                             "ewt": np.average(g["ewt_min"], weights=g["trains"])})
    print(e.groupby("mode").apply(w, include_groups=False).round(2))
    e["line"] = e["lines"].str.split().str[0]
    print(e.groupby("line").apply(w, include_groups=False).round(2).sort_values("ewt", ascending=False))


if __name__ == "__main__":
    main()
