"""Frozen profiles for the AAAI StairMaster ablations.

The profiles live outside the default Go2 config on purpose: the working tree
contains tuning changes that belong to the user, while the remote experiment
checkout must receive an immutable, auditable copy of the values used here.
"""

from __future__ import annotations

from typing import Dict


REWARD_PROFILES = ("ep", "ep_plus_three", "current_full")


_ALL_REWARD_SCALES = {
    "tracking_goal_vel",
    "tracking_yaw",
    "tracking_ang_vel",
    "tracking_pitch",
    "tracking_lin_vel",
    "lin_vel_z",
    "ang_vel_xy",
    "orientation",
    "dof_acc",
    "collision",
    "action_rate",
    "delta_torques",
    "torques",
    "hip_pos",
    "smoothness",
    "dof_error",
    "feet_stumble",
    "feet_edge",
    "feet_hollow",
    "feet_air_time",
    "feet_contact_forces",
    "roll",
    "pitch",
    "base_height",
    "cur_goals",
}


_EP_SCALES: Dict[str, float] = {
    "tracking_goal_vel": 1.5,
    "tracking_yaw": 0.5,
    "lin_vel_z": -1.0,
    "ang_vel_xy": -0.05,
    "orientation": -1.0,
    "dof_acc": -2.5e-7,
    "collision": -10.0,
    "action_rate": -0.1,
    "delta_torques": -1.0e-7,
    "torques": -1.0e-5,
    "hip_pos": -0.5,
    "dof_error": -0.04,
    "feet_stumble": -1.0,
    # Extreme Parkour already contains this edge term. We keep it in T0 and
    # T1 rather than incorrectly counting it as a newly introduced reward.
    "feet_edge": -1.0,
}


_EP_PLUS_THREE_SCALES: Dict[str, float] = {
    **_EP_SCALES,
    "tracking_pitch": 0.5,
    "feet_hollow": -1.0,
}


_CURRENT_FULL_SCALES: Dict[str, float] = {
    "tracking_goal_vel": 2.5,
    "tracking_ang_vel": 0.5,
    "tracking_pitch": 0.5,
    "lin_vel_z": -1.0,
    "ang_vel_xy": -0.05,
    "orientation": -1.0,
    "dof_acc": -2.5e-7,
    "collision": -10.0,
    "action_rate": -0.1,
    "delta_torques": -1.0e-7,
    "torques": -1.0e-4,
    "hip_pos": -0.8,
    "smoothness": -0.005,
    "dof_error": -0.1,
    "feet_stumble": -1.0,
    "feet_edge": -1.0,
    "feet_hollow": -1.0,
    "feet_contact_forces": -0.01,
    "roll": -1.0,
    "pitch": -0.2,
    "base_height": -0.2,
    "cur_goals": 0.1,
}


_PROFILE_SCALES = {
    "ep": _EP_SCALES,
    "ep_plus_three": _EP_PLUS_THREE_SCALES,
    "current_full": _CURRENT_FULL_SCALES,
}


def apply_reward_profile(env_cfg, profile: str) -> None:
    """Apply one complete reward profile, explicitly zeroing other terms."""
    if profile not in _PROFILE_SCALES:
        raise ValueError(
            f"Unknown reward profile {profile!r}; expected one of {REWARD_PROFILES}"
        )

    scales = env_cfg.rewards.scales
    selected = _PROFILE_SCALES[profile]
    for name in _ALL_REWARD_SCALES:
        setattr(scales, name, selected.get(name, 0.0))

    if profile in ("ep", "ep_plus_three"):
        env_cfg.rewards.base_height_target = 1.0
        env_cfg.rewards.max_contact_force = 40.0
        env_cfg.rewards.soft_dof_pos_limit = 1.0
    else:
        env_cfg.rewards.base_height_target = 0.3
        env_cfg.rewards.max_contact_force = 200.0
        env_cfg.rewards.soft_dof_pos_limit = 0.9

    env_cfg.ablation_reward_profile = profile


def apply_frozen_training_profile(env_cfg) -> None:
    """Freeze non-reward settings shared by all teacher/student comparisons."""
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.friction_range = [0.5, 2.0]
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.added_mass_range = [0.0, 8.0]
    env_cfg.domain_rand.randomize_base_com = True
    env_cfg.domain_rand.added_com_range = [-0.2, 0.2]
    env_cfg.domain_rand.randomize_motor = True
    env_cfg.domain_rand.motor_strength_range = [0.5, 1.2]

    env_cfg.terrain.terrain_dict = {
        "smooth slope": 0.0,
        "normal stairs down": 0.0,
        "normal stairs up": 0.0,
        "steep hollow stairs down": 0.0,
        "steep hollow stairs up": 0.5,
        "discrete": 0.0,
        "stepping stones": 0.0,
        "gaps": 0.0,
        "flat": 0.2,
        "pit": 0.0,
        "wall": 0.0,
        "platform": 0.0,
        "hollow stairs down": 0.0,
        "hollow stairs up": 0.5,
        "parkour": 0.0,
        "parkour_hurdle": 0.0,
        "parkour_flat": 0.0,
        "parkour_step": 0.0,
        "parkour_gap": 0.0,
        "demo": 0.0,
    }
    env_cfg.terrain.terrain_proportions = list(
        env_cfg.terrain.terrain_dict.values()
    )
    env_cfg.ablation_training_profile = "stairmaster_current_2026_07_25"
