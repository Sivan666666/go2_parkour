"""Run each manifest condition in a fresh Isaac Gym process."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exptid", required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--reward_profile", required=True)
    parser.add_argument("--policy_variant", default=None)
    parser.add_argument("--suite", choices=("quick", "full"), default="quick")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--proj_name", default="aaai_ablation")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--use_camera", action="store_true")
    parser.add_argument("--delay", action="store_true")
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    conditions = [
        condition["name"]
        for condition in manifest["conditions"]
        if args.suite in condition.get("suites", [])
    ]

    evaluator = Path(__file__).with_name("evaluate_ablation.py")
    for condition in conditions:
        command = [
            sys.executable,
            str(evaluator),
            "--task",
            "go2",
            "--exptid",
            args.exptid,
            "--proj_name",
            args.proj_name,
            "--checkpoint",
            str(args.checkpoint),
            "--reward_profile",
            args.reward_profile,
            "--eval_manifest",
            args.manifest,
            "--eval_suite",
            args.suite,
            "--eval_condition",
            condition,
            "--output_dir",
            args.output_dir,
            "--device",
            args.device,
            "--headless",
            "--no_wandb",
        ]
        if args.episodes is not None:
            command.extend(["--episodes", str(args.episodes)])
        if args.policy_variant is not None:
            command.extend(["--policy_variant", args.policy_variant])
        if args.use_camera:
            command.append("--use_camera")
        if args.delay:
            command.append("--delay")
        print(f"Running {condition}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
