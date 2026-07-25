"""Choose and enforce one common A/B/C stopping iteration."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def read_metrics(run_dir):
    path = run_dir / "training_metrics.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def first_stable_crossing(records, window, threshold):
    levels = np.asarray(
        [
            record["terrain_level_mean"]
            for record in records
            if record.get("terrain_level_mean") is not None
        ],
        dtype=np.float64,
    )
    if len(levels) < window:
        return None
    rolling = np.convolve(levels, np.ones(window) / window, mode="valid")
    crossed = np.flatnonzero(rolling >= threshold)
    if len(crossed) == 0:
        return None
    # The rolling window ending at index window-1 corresponds to iteration
    # records[window-1]["iteration"].
    return int(records[int(crossed[0]) + window - 1]["iteration"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs=3)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--post_iterations", type=int, default=1500)
    parser.add_argument("--minimum", type=int, default=3000)
    parser.add_argument("--fallback_check", type=int, default=5000)
    parser.add_argument("--maximum", type=int, default=6000)
    parser.add_argument("--poll_seconds", type=int, default=60)
    args = parser.parse_args()

    run_dirs = [Path(value).resolve() for value in args.run_dirs]
    while True:
        records = {path.name: read_metrics(path) for path in run_dirs}
        crossings = {
            name: first_stable_crossing(
                values, args.window, args.threshold
            )
            for name, values in records.items()
        }
        latest = {
            name: values[-1]["iteration"] if values else 0
            for name, values in records.items()
        }
        print(
            json.dumps(
                {"crossings": crossings, "latest": latest},
                sort_keys=True,
            ),
            flush=True,
        )

        if all(value is not None for value in crossings.values()):
            common_stop = max(
                args.minimum,
                min(
                    args.maximum,
                    max(crossings.values()) + args.post_iterations,
                ),
            )
            reason = "all_variants_reached_threshold"
            break
        if min(latest.values()) >= args.fallback_check:
            common_stop = args.maximum
            reason = "at_least_one_variant_failed_by_fallback_check"
            break
        time.sleep(args.poll_seconds)

    decision = {
        "crossings": crossings,
        "common_stop_iteration": common_stop,
        "reason": reason,
        "rolling_window": args.window,
        "terrain_level_threshold": args.threshold,
        "post_iterations": args.post_iterations,
    }
    for run_dir in run_dirs:
        (run_dir / "stop_at_iteration.txt").write_text(
            f"{common_stop}\n", encoding="utf-8"
        )
        (run_dir / "common_stop_decision.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
