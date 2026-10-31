"""
Rebuild every operations analysis in order, for the dashboard pages "Missed
connections", "Where delays start", "Slow stretches", "Bunching and gaps",
"Dwell times", "Crowding" and "Car parks". Re-run as more data is collected.

    python -m analysis.reliability     # first: delays per train per station (~5 min)
    python -m analysis.operations      # then: all of the below (~15 min with buses)
    python -m analysis.operations --only connections headways

Steps: stop_events (bus times at stops next to stations, the slowest step) ->
connections, delay_origins, slow_segments, headways, dwell, crowding, carpark_fill.
"""

import argparse
import importlib
import sys
import time

STEPS = ["stop_events", "connections", "delay_origins", "slow_segments", "headways", "dwell", "crowding", "carpark_fill"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the operations analyses.")
    parser.add_argument("--only", nargs="+", choices=STEPS, help="Run just these steps")
    args = parser.parse_args()
    for step in args.only or STEPS:
        start = time.time()
        print(f"== {step}")
        sys.argv = [step]  # steps with their own options (stop_events --dates) run with their defaults
        importlib.import_module(f"analysis.{step}").main()
        print(f"   ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
