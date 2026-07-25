"""Deterministic, one-episode-per-environment StairMaster evaluation."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from statistics import NormalDist

import isaacgym  # noqa: F401 - must be imported before torch
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 - registers tasks
from legged_gym.utils import get_args, task_registry


TERRAIN_KEYS = (
    "smooth slope",
    "normal stairs down",
    "normal stairs up",
    "steep hollow stairs down",
    "steep hollow stairs up",
    "discrete",
    "stepping stones",
    "gaps",
    "flat",
    "pit",
    "wall",
    "platform",
    "hollow stairs down",
    "hollow stairs up",
    "parkour",
    "parkour_hurdle",
    "parkour_flat",
    "parkour_step",
    "parkour_gap",
    "demo",
)


def wilson_interval(successes, trials, confidence=0.95):
    if trials == 0:
        return 0.0, 0.0
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    probability = successes / trials
    denominator = 1 + z * z / trials
    center = (probability + z * z / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1 - probability) / trials
            + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return center - radius, center + radius


def stair_angle_deg(stair_type, difficulty):
    if difficulty == "mixed":
        return None
    width = 0.4 - 0.2 * float(difficulty)
    coefficient = 0.13 if stair_type == "steep" else 0.05
    height = 0.1 + coefficient * 9.0 / 7.0 * float(difficulty)
    return math.degrees(math.atan2(height, width))


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def select_condition(manifest, args):
    candidates = [
        condition
        for condition in manifest["conditions"]
        if args.eval_suite in condition.get("suites", [])
    ]
    if args.eval_condition is not None:
        candidates = [
            condition
            for condition in manifest["conditions"]
            if condition["name"] == args.eval_condition
        ]
    if len(candidates) != 1:
        names = ", ".join(condition["name"] for condition in candidates)
        raise ValueError(
            "evaluate_ablation.py runs one simulator condition per process. "
            f"Select one with --eval_condition. Candidates: {names}"
        )
    return candidates[0]


def configure_terrain(env_cfg, condition, episodes):
    terrain = {name: 0.0 for name in TERRAIN_KEYS}
    stair_type = condition["stair_type"]
    if stair_type == "regular":
        terrain["hollow stairs up"] = 1.0
    elif stair_type == "steep":
        terrain["steep hollow stairs up"] = 1.0
    elif stair_type == "mixed":
        terrain["hollow stairs up"] = 0.5
        terrain["steep hollow stairs up"] = 0.5
    else:
        raise ValueError(f"Unknown stair type: {stair_type}")

    env_cfg.env.num_envs = episodes
    env_cfg.env.episode_length_s = 20
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.terrain_dict = terrain
    env_cfg.terrain.terrain_proportions = list(terrain.values())
    env_cfg.terrain.num_cols = 10 if stair_type == "mixed" else 5
    if condition["difficulty"] == "mixed":
        env_cfg.terrain.fixed_difficulty = None
        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.curriculum = True
        env_cfg.terrain.max_init_terrain_level = 4
    else:
        env_cfg.terrain.fixed_difficulty = float(condition["difficulty"])
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.terrain.max_difficulty = False
    env_cfg.env.randomize_start_pos = False


def get_actions(runner, env, obs, depth_frame, depth_latent):
    if not runner.if_depth:
        policy = runner.get_inference_policy(device=env.device)
        with torch.inference_mode():
            return policy(
                obs.detach(), hist_encoding=True, scandots_latent=None
            ), depth_latent

    with torch.inference_mode():
        if depth_frame is not None:
            depth_encoder = runner.alg.depth_encoder
            observation = obs[:, : env.cfg.env.n_proprio].clone()
            observation[:, 6:8] = 0
            latent_and_yaw = depth_encoder(depth_frame, observation)
            depth_latent = latent_and_yaw[:, :-2]
        if depth_latent is None:
            raise RuntimeError("No initial depth latent was produced")
        actions = runner.alg.depth_actor(
            obs.detach(), hist_encoding=True, scandots_latent=depth_latent
        )
    return actions, depth_latent


def evaluate(args):
    manifest_path = args.eval_manifest
    if manifest_path is None:
        manifest_path = os.path.join(
            os.path.dirname(__file__),
            "manifests",
            "hollow_stairs_v1.json",
        )
    manifest = load_manifest(manifest_path)
    condition = select_condition(manifest, args)
    episodes = args.episodes or manifest["episodes_per_condition"]

    args.headless = True
    args.seed = int(condition["seed"])
    args.num_envs = episodes
    args.resumeid = args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    configure_terrain(env_cfg, condition, episodes)
    env, env_cfg = task_registry.make_env(
        name=args.task, args=args, env_cfg=env_cfg
    )
    obs = env.get_observations()

    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(
        log_root=None,
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        init_wandb=False,
    )
    runner.alg.actor_critic.eval()
    if runner.if_depth:
        runner.alg.depth_encoder.eval()
        runner.alg.depth_actor.eval()

    unresolved = torch.ones(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    edge_contacts = torch.zeros(
        env.num_envs, dtype=torch.float, device=env.device
    )
    hollow_events = torch.zeros_like(edge_contacts)
    rows = [None] * env.num_envs
    depth_frame = (
        env.depth_buffer[:, -1].clone() if runner.if_depth else None
    )
    depth_latent = None

    max_steps = int(env.max_episode_length) + 5
    for _ in range(max_steps):
        actions, depth_latent = get_actions(
            runner, env, obs, depth_frame, depth_latent
        )
        if runner.if_depth:
            depth_frame = None
        obs, _, _, dones, infos = env.step(actions.detach())

        active = unresolved.float()
        edge_contacts += (
            env.feet_at_edge.float().sum(dim=1) * active
        )
        hollow_events += (
            env.feet_under_hollow.float().sum(dim=1) * active
        )
        done_now = dones.bool() & unresolved
        done_ids = done_now.nonzero(as_tuple=False).flatten()
        for env_id_tensor in done_ids:
            env_id = int(env_id_tensor.item())
            success = bool(
                infos["terminal_success"][env_id].item()
            )
            fall = bool(infos["fall"][env_id].item())
            timeout = bool(
                infos["terminal_timeout"][env_id].item()
            )
            episode_steps = int(
                infos["terminal_episode_length"][env_id].item()
            )
            rows[env_id] = {
                "episode_id": env_id,
                "condition": condition["name"],
                "stair_type": condition["stair_type"],
                "difficulty": condition["difficulty"],
                "angle_deg": stair_angle_deg(
                    condition["stair_type"],
                    condition["difficulty"],
                ),
                "seed": condition["seed"],
                "success": int(success),
                "progress": float(
                    infos["terminal_progress"][env_id].item()
                ),
                "fall": int(fall),
                "timeout_or_stall": int(timeout),
                "completion_time_s": episode_steps * float(env.dt),
                "edge_contacts": float(edge_contacts[env_id].item()),
                "hollow_events": float(hollow_events[env_id].item()),
                "reward_profile": args.reward_profile,
                "policy_variant": (
                    args.policy_variant
                    if runner.if_depth
                    else "teacher"
                ),
                "checkpoint": args.checkpoint,
                "run": args.exptid,
            }
        unresolved[done_now] = False
        if runner.if_depth:
            runner.alg.depth_encoder.reset(dones)
            if infos.get("depth") is not None:
                depth_frame = infos["depth"].clone()
            elif torch.any(dones):
                # Reset environments receive a freshly initialized depth
                # buffer; refresh the batch so no episode reuses stale latent.
                depth_frame = env.depth_buffer[:, -1].clone()
        if not torch.any(unresolved):
            break

    if torch.any(unresolved):
        missing = unresolved.nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(
            f"{len(missing)} environments did not terminate: {missing[:10]}"
        )

    output_dir = Path(
        args.output_dir
        or os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "results",
            "aaai_ablation",
            args.exptid,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{condition['name']}_{args.reward_profile or 'default'}_"
        f"{args.policy_variant or 'teacher'}_ckpt{args.checkpoint}"
    )
    csv_path = output_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    successes = sum(row["success"] for row in rows)
    lower, upper = wilson_interval(successes, len(rows))
    summary = {
        "condition": condition,
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "success_wilson_95": [lower, upper],
        "progress_mean": float(np.mean([row["progress"] for row in rows])),
        "fall_rate": float(np.mean([row["fall"] for row in rows])),
        "timeout_or_stall_rate": float(
            np.mean([row["timeout_or_stall"] for row in rows])
        ),
        "completion_time_s_mean": float(
            np.mean([row["completion_time_s"] for row in rows])
        ),
        "edge_contacts_mean": float(
            np.mean([row["edge_contacts"] for row in rows])
        ),
        "hollow_events_mean": float(
            np.mean([row["hollow_events"] for row in rows])
        ),
        "csv": str(csv_path),
    }
    summary_path = output_dir / f"{stem}.summary.json"
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    evaluate(get_args())
