"""Tests for the bad GPS filters in analysis/replay.py (trains, buses, light rail, ferries).

    python -m pytest tests
"""

import numpy as np
import pandas as pd

from analysis.replay import (MAX_OFF_TRACK_LR_M, SPEED_LIMITS, clean_track, frozen_fixes, impossible_moves,
                             off_network)


def straight_run(n, start_x=0.0, step_m=500.0, step_s=30.0, start_t=0.0):
    """A train moving steadily east at 60 km/h (500 m every 30 s)."""
    x = start_x + step_m * np.arange(n)
    return x, np.zeros(n), start_t + step_s * np.arange(n)


def test_normal_running_is_kept():
    x, y, t = straight_run(20)
    assert clean_track(x, y, t).all()


def test_single_fix_glitch_is_dropped():
    # One fix lands 10 km away (e.g. on a parallel line) then the train snaps back
    x, y, t = straight_run(20)
    y[8] = 10_000
    keep = clean_track(x, y, t)
    assert not keep[8]
    assert keep.sum() == 19


def test_short_burst_of_bad_fixes_is_dropped():
    x, y, t = straight_run(20)
    y[5:8] = 6_000  # three bad fixes in a row (~1.5 min)
    keep = clean_track(x, y, t)
    assert not keep[5:8].any()
    assert keep.sum() == 17


def test_glitch_at_start_of_trip_is_dropped():
    x, y, t = straight_run(20)
    x[0] = -15_000
    keep = clean_track(x, y, t)
    assert not keep[0] and keep[1:].all()


def test_small_move_with_clock_jitter_is_not_flagged():
    # 800 m in 5 s is "576 km/h", but under 1 km: GPS clock jitter, not a jump
    assert not impossible_moves(np.array([0, 800]), np.array([0, 0]), np.array([0, 5])).any()


def test_fast_intercity_train_is_not_flagged():
    # 150 km/h for a minute (2.5 km): fast but possible
    assert not impossible_moves(np.array([0, 2500]), np.array([0, 0]), np.array([0, 60])).any()


def test_long_gap_in_collection_is_not_a_jump():
    # PC asleep for an hour: the train moved 60 km, at 60 km/h
    x, y, t = straight_run(10)
    x2, y2, t2 = straight_run(10, start_x=x[-1] + 60_000, start_t=t[-1] + 3600)
    keep = clean_track(np.r_[x, x2], np.r_[y, y2], np.r_[t, t2])
    assert keep.all()


def test_all_short_runs_keeps_the_longest():
    # Track that flips between two places every couple of fixes: keep the biggest consistent run
    x = np.array([0, 100, 20_000, 20_100, 20_200, 0, 100], dtype=float)
    keep = clean_track(x, np.zeros(7), 30.0 * np.arange(7))
    assert keep.tolist() == [False, False, True, True, True, False, False]


def test_tiny_tracks_are_kept():
    assert clean_track([0.0], [0.0], [0.0]).tolist() == [True]
    assert clean_track([], [], []).tolist() == []


def test_off_network():
    seg = pd.DataFrame({"ax": [0.0], "ay": [0.0], "bx": [10_000.0], "by": [0.0]})  # one 10 km line
    x = np.array([5_000, 5_000, 12_500, -500])
    y = np.array([500, 2_500, 0, 0])
    # 500 m beside the line: on; 2.5 km beside: off; 2.5 km past the end: off; 500 m before the start: on
    assert off_network(x, y, seg).tolist() == [False, True, True, False]


# --- Buses, light rail and ferries -------------------------------------------------------------

def test_bus_speed_limit_is_lower_than_trains():
    # 2 km in 60 s = 120 km/h: possible for a train, not for a bus
    x, y, t = np.array([0, 2000]), np.array([0, 0]), np.array([0, 60])
    assert not impossible_moves(x, y, t, *SPEED_LIMITS["Trains"]).any()
    assert impossible_moves(x, y, t, *SPEED_LIMITS["Bus"]).all()


def test_motorway_bus_is_not_flagged():
    # 1.6 km in 60 s = 96 km/h on a motorway
    assert not impossible_moves(np.array([0, 1600]), np.array([0, 0]), np.array([0, 60]), *SPEED_LIMITS["Bus"]).any()


def test_light_rail_glitch_is_dropped():
    # Light rail at ~30 km/h (250 m a 30 s), with one fix landing 1 km away (e.g. on a nearby street)
    x, y, t = straight_run(20, step_m=250)
    y[10] = 1_000
    keep = clean_track(x, y, t, *SPEED_LIMITS["Light rail"])
    assert keep.sum() == 19 and not keep[10]


def test_frozen_bus_tracker_is_dropped():
    # The tracker resends one position for 5 minutes, then the bus "jumps" 2.5 km in a minute to catch up
    x = np.r_[0, 300, 600, [600] * 5, 3_100, 3_400, 3_700, 4_000, 4_300, 4_600]
    t = 60.0 * np.arange(len(x))
    keep = clean_track(x, np.zeros(len(x)), t, *SPEED_LIMITS["Bus"])
    assert keep[2] and not keep[3:8].any()  # first sighting kept, frozen repeats dropped
    assert keep[8:].all()  # 2.5 km in 6 min from the first sighting is a believable 25 km/h


def test_bus_waiting_at_terminus_is_kept():
    # Same position for 5 minutes, then drives off normally: a real stop, not a frozen tracker
    x = np.r_[0, 300, 600, [600] * 5, 900, 1_200, 1_500, 1_800, 2_100]
    t = 60.0 * np.arange(len(x))
    assert clean_track(x, np.zeros(len(x)), t, *SPEED_LIMITS["Bus"]).all()
    assert not frozen_fixes(x, np.zeros(len(x)), t, *SPEED_LIMITS["Bus"]).any()


def test_ferry_at_full_speed_is_kept():
    # A RiverCat at 50 km/h (833 m a minute)
    x, y, t = straight_run(20, step_m=833, step_s=60)
    assert clean_track(x, y, t, *SPEED_LIMITS["Ferry"]).all()


def test_light_rail_off_track():
    track = pd.DataFrame({"ax": [0.0], "ay": [0.0], "bx": [5_000.0], "by": [0.0]})
    x, y = np.array([2_000, 2_000]), np.array([100, 400])
    assert off_network(x, y, track, MAX_OFF_TRACK_LR_M).tolist() == [False, True]
