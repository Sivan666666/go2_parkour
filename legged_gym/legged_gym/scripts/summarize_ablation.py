"""Aggregate StairMaster ablation CSV files and paired effect sizes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_ablation import wilson_interval


def read_rows(paths):
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                row["success"] = int(row["success"])
                row["progress"] = float(row["progress"])
                row["fall"] = int(row["fall"])
                row["timeout_or_stall"] = int(row["timeout_or_stall"])
                rows.append(row)
    return rows


def bootstrap_difference(left, right, samples=10000, seed=20260725):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"Paired arrays differ: {left.shape} vs {right.shape}")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(left), size=(samples, len(left)))
    differences = (right[indices] - left[indices]).mean(axis=1)
    return {
        "mean": float((right - left).mean()),
        "ci95": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
    }


def aggregate(rows):
    grouped = {}
    for row in rows:
        key = (row["run"], row["condition"])
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (run, condition), group in sorted(grouped.items()):
        successes = sum(row["success"] for row in group)
        lower, upper = wilson_interval(successes, len(group))
        summaries.append(
            {
                "run": run,
                "condition": condition,
                "episodes": len(group),
                "success_rate": successes / len(group),
                "success_wilson_95": [lower, upper],
                "progress_mean": float(
                    np.mean([row["progress"] for row in group])
                ),
                "fall_rate": float(
                    np.mean([row["fall"] for row in group])
                ),
                "timeout_or_stall_rate": float(
                    np.mean(
                        [row["timeout_or_stall"] for row in group]
                    )
                ),
            }
        )
    return summaries, grouped


def paired_comparisons(grouped, comparisons):
    results = []
    conditions = sorted({condition for _, condition in grouped})
    for left_run, right_run in comparisons:
        for condition in conditions:
            left_rows = grouped.get((left_run, condition))
            right_rows = grouped.get((right_run, condition))
            if not left_rows or not right_rows:
                continue
            left_by_id = {
                int(row["episode_id"]): row for row in left_rows
            }
            right_by_id = {
                int(row["episode_id"]): row for row in right_rows
            }
            common = sorted(set(left_by_id) & set(right_by_id))
            result = {
                "left": left_run,
                "right": right_run,
                "condition": condition,
                "episodes": len(common),
            }
            for metric in ("success", "progress"):
                left = [left_by_id[index][metric] for index in common]
                right = [right_by_id[index][metric] for index in common]
                result[f"{metric}_difference"] = bootstrap_difference(
                    left, right
                )
            results.append(result)
    return results


def write_markdown(path, summaries, comparisons):
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(
            "| Run | Condition | N | Success (95% CI) | Progress | Fall | Stall |\n"
        )
        stream.write("|---|---|---:|---:|---:|---:|---:|\n")
        for item in summaries:
            lower, upper = item["success_wilson_95"]
            stream.write(
                f"| {item['run']} | {item['condition']} | {item['episodes']} | "
                f"{100 * item['success_rate']:.1f}% "
                f"[{100 * lower:.1f}, {100 * upper:.1f}] | "
                f"{100 * item['progress_mean']:.1f}% | "
                f"{100 * item['fall_rate']:.1f}% | "
                f"{100 * item['timeout_or_stall_rate']:.1f}% |\n"
            )
        stream.write("\n## Paired effects\n\n")
        for item in comparisons:
            effect = item["success_difference"]
            stream.write(
                f"- {item['condition']}: {item['right']} − {item['left']} = "
                f"{100 * effect['mean']:.1f} pp "
                f"(95% CI {100 * effect['ci95'][0]:.1f}, "
                f"{100 * effect['ci95'][1]:.1f}).\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="CSV files or directories")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="Paired comparison LEFT:RIGHT; may be repeated",
    )
    args = parser.parse_args()

    csv_paths = []
    for value in args.inputs:
        path = Path(value)
        if path.is_dir():
            csv_paths.extend(sorted(path.rglob("*.csv")))
        else:
            csv_paths.append(path)
    rows = read_rows(csv_paths)
    summaries, grouped = aggregate(rows)
    comparison_pairs = [
        tuple(value.split(":", 1)) for value in args.compare
    ]
    comparisons = paired_comparisons(grouped, comparison_pairs)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(
            {"summaries": summaries, "comparisons": comparisons},
            stream,
            indent=2,
            sort_keys=True,
        )
    write_markdown(output.with_suffix(".md"), summaries, comparisons)


if __name__ == "__main__":
    main()
