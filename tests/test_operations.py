"""Tests for the operations analyses: missed connections, delay origins, headways, dwell and car park fill.

    python -m pytest tests
"""

import numpy as np
import pandas as pd

from analysis.carpark_fill import as_clock, fill_time
from analysis.connections import plan_connections
from analysis.delay_origins import delay_gains, first_late
from analysis.dwell import platform_visits
from analysis.headways import gaps

MIN = 60


def arrivals(*rows):
    """(sched_arr, arr_actual) in minutes, from line A at hub X with a 3 minute change."""
    return pd.DataFrame([{"hub": "X", "from_line": "A", "sched_arr": s * MIN, "arr_actual": a * MIN, "transfer_s": 3 * MIN}
                         for s, a in rows])


def departures(*rows, line="B"):
    """(sched_dep, dep_actual) in minutes on line B from hub X."""
    return pd.DataFrame([{"hub": "X", "to_line": line, "line_dir": f"{line}>Y", "sched_dep": s * MIN, "dep_actual": a * MIN}
                         for s, a in rows])


def test_connection_made_when_on_time():
    c = plan_connections(arrivals((0, 0)), departures((5, 5), (15, 15)))
    assert len(c) == 1 and not c["missed"].iloc[0]
    assert c["extra_wait_s"].iloc[0] == 0
    assert c["planned_wait_s"].iloc[0] == 5 * MIN


def test_late_arrival_misses_and_waits_for_next():
    # Arrive 4 min late: ready at 7, the 5 o'clock train left on time, next is at 15
    c = plan_connections(arrivals((0, 4)), departures((5, 5), (15, 15)))
    assert c["missed"].iloc[0]
    assert c["extra_wait_s"].iloc[0] == 10 * MIN


def test_connection_held_by_late_departure():
    # Both late: the connecting train left at 8, we were ready at 7
    c = plan_connections(arrivals((0, 4)), departures((5, 8), (15, 15)))
    assert not c["missed"].iloc[0]


def test_planned_departure_respects_transfer_time():
    # A departure 2 min after arrival can't be planned with a 3 min change; the next one is
    c = plan_connections(arrivals((0, 0)), departures((2, 2), (6, 6)))
    assert c["planned_sched_dep"].iloc[0] == 6 * MIN


def test_same_line_and_long_waits_are_not_connections():
    assert plan_connections(arrivals((0, 0)), departures((5, 5), line="A")).empty
    assert plan_connections(arrivals((0, 0)), departures((40, 40))).empty  # over MAX_WAIT_S


def test_no_connection_back_the_way_you_came():
    arr = arrivals((0, 0)).assign(prev_station="Y")
    assert plan_connections(arr, departures((5, 5)).assign(next_station="Y")).empty
    assert len(plan_connections(arr, departures((5, 5)).assign(next_station="Z"))) == 1


def test_departure_stop_transfer_time_overrides():
    # A bus stop 6 minutes' walk away: the 5 o'clock bus can't be planned
    dep = departures((5, 5), (10, 10)).assign(dir_transfer_s=6 * MIN)
    assert plan_connections(arrivals((0, 0)), dep)["planned_sched_dep"].iloc[0] == 10 * MIN


def trip(delays, source="gps", trip_id="t1"):
    n = len(delays)
    return pd.DataFrame({"service_date": "20261029", "trip_id": trip_id, "sched_time": np.arange(n) * 120,
                         "station": [f"S{i}" for i in range(n)], "prev_station": [None] + [f"S{i}" for i in range(n - 1)],
                         "delay_s": delays, "delay_source": source})


def test_delay_gain_between_stations():
    seg = delay_gains(trip([0, 30, 200, 190]))
    assert seg["gain_s"].tolist() == [30, 170, -10]


def test_delay_gain_skips_prediction_pairs():
    t = trip([0, 30, 200, 190])
    t.loc[2, "delay_source"] = "prediction"
    assert delay_gains(t)["station"].tolist() == ["S1"]


def test_first_late_station():
    assert first_late(trip([0, 100, 320, 400]))["station"].tolist() == ["S2"]
    assert first_late(trip([400, 500]), late_s=300).empty  # already late at its origin


def test_headway_bunching_and_gaps():
    # Every 10 minutes scheduled; actual 0, 3, 25, 30: a bunch then a big gap
    ev = pd.DataFrame({"stop": "X", "sched": np.array([0, 10, 20, 30]) * MIN, "actual": np.array([0, 3, 25, 30]) * MIN})
    h = gaps(ev, ["stop"], "sched", "actual")
    assert h["bunched"].tolist() == [True, False, False]
    assert h["big_gap"].tolist() == [False, True, False]


def test_headways_ignore_infrequent_services():
    ev = pd.DataFrame({"stop": "X", "sched": np.array([0, 30, 60]) * MIN, "actual": np.array([0, 30, 60]) * MIN})
    assert gaps(ev, ["stop"], "sched", "actual").empty


def test_platform_time_runs_until_the_next_location():
    f = pd.DataFrame({"trip_id": ["a"] * 5, "ts": [0, 30, 60, 90, 120], "loc": ["sig1", "Plat 1", "Plat 1", "sig2", "sig3"],
                      "lon": 151.0, "lat": -33.8})
    v = platform_visits(f, f["loc"].str.startswith("Plat"))
    assert v["platform_s"].tolist() == [60]


def test_platform_time_needs_a_following_fix():
    f = pd.DataFrame({"trip_id": ["a", "a"], "ts": [0, 30], "loc": ["sig1", "Plat 1"], "lon": 151.0, "lat": -33.8})
    assert platform_visits(f, f["loc"].str.startswith("Plat")).empty


def test_car_park_fill_time_is_interpolated():
    hours = np.array([6.0, 6.25, 6.5])
    assert fill_time(hours, np.array([80.0, 90.0, 100.0])) == (6.375, True)  # 95% half way between 90 and 100
    assert np.isnan(fill_time(hours, np.array([10.0, 20.0, 30.0]))[0])
    assert as_clock(6.375) == "6:22" and as_clock(7.999) == "8:00"


def test_car_park_full_before_collection_is_only_full_by():
    # Collection resumed at 7:45 with the car park already full, or after a 2 hour gap
    assert fill_time(np.array([7.75, 8.0]), np.array([100.0, 100.0])) == (7.75, False)
    assert fill_time(np.array([5.0, 7.0]), np.array([20.0, 100.0])) == (7.0, False)
