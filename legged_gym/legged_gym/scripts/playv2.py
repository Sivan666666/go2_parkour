# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 128
    env_cfg.env.episode_length_s = 20
    env_cfg.commands.resampling_time = 20
    
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]

    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "normal stairs down": 0.0,
                        "normal stairs up": 0.0,
                        "discrete": 0., 
                        "stepping stones": 0.0,
                        "gaps": 0., 
                        "flat": 0.0,
                        "pit": 0.0,
                        "wall": 0.0,
                        "platform": 0.,
                        "hollow stairs down": 0.0, 
                        "hollow stairs up": 1.0,
                        "parkour": 0.0,         # 0.2
                        "parkour_hurdle": 0.0,  # 0.2
                        "parkour_flat": 0.0,
                        "parkour_step": 0.0,    # 0.2
                        "parkour_gap": 0.0,     # 0.2
                        "demo": 0.0}            # 0.2
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True


    # env_cfg.depth.position = [0.35, 0, 0.147]  # front camera 
    # env_cfg.depth.angle = [59-1, 59+1]  # positive pitch down  #27-5,27+5

    # env_cfg.depth.position = [0.3, 0, 0.147]  # front camera 
    # env_cfg.depth.angle = [30-1, 30+1]  # positive pitch down  #27-5,27+5

    # for go2
        # position = [0.3, 0, 0.08] # front camera 002-g2-camera 
        # position = [0.355, 0, 0.065]
        # angle = [20, 25]

    #env_cfg.depth.position = [0.3, 0, 0.147]  # front camera
    #env_cfg.depth.angle = [10-1, 10+1]  # positive pitch down  #27-5,27+5

    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    
    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    
    
    # ==========================================
    # 🔥 初始化统计变量
    # ==========================================
    vel_tracking_error_x_buffer = deque(maxlen=1000000)  # X 方向误差
    
    # 成功/失败计数
    success_count = 0
    fail_count = 0
    episode_count = 0
    
    success_rate_buffer = [None] * env.num_envs

    # 获取最后一个目标点的索引
    num_goals = env.cfg.terrain.num_goals  # 通常是 8
    success_threshold = 0.5  # ±0.2m 范围内视为成功

    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    for i in range(1*int(1000)):
        # 1000 * 0.002 = 20s
        # episode length = env_cfg.env.episode_length_s / dt
        # # 强制设置 yaw 相关命令为0，保持直线行走
        # env.commands[:, 2] = 0.0  # yaw rate command 设为0
        
        # # 如果你想保持特定的前进方向，可以设置：
        env.commands[:, 0] = 0.5  # forward velocity
        env.commands[:, 2] = 0  # lateral velocity


        with torch.no_grad():
            if args.use_jit:
                if env.cfg.depth.use_camera:
                    if infos["depth"] is not None:
                        depth_latent = torch.ones((env_cfg.env.num_envs, 32), device=env.device)
                        actions, depth_latent = policy_jit(obs.detach(), True, infos["depth"], depth_latent)
                    else:
                        depth_buffer = torch.ones((env_cfg.env.num_envs, 58, 87), device=env.device)
                        actions, depth_latent = policy_jit(obs.detach(), False, depth_buffer, depth_latent)
                else:
                    obs_jit = torch.cat((obs.detach()[:, :env_cfg.env.n_proprio+env_cfg.env.n_priv], obs.detach()[:, -env_cfg.env.history_len*env_cfg.env.n_proprio:]), dim=1)
                    actions = policy(obs_jit)
            else:
                if env.cfg.depth.use_camera:
                    if infos["depth"] is not None:
                        obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                        obs_student[:, 6:8] = 0
                        depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                        depth_latent = depth_latent_and_yaw[:, :-2]
                        yaw = depth_latent_and_yaw[:, -2:] * 0
                    # 不使用 yaw 修正,保持原始观测
                    obs[:, 6:8] = 1.5*yaw  # 注释掉这行
                    # obs[:, 6:8] = 0  # 强制设为0
                    # obs[:, 6:8] = -env.yaw.unsqueeze(1)
                        
                else:
                    depth_latent = None
                obs[:, 6:8] = 0  # 强制设为0
                if hasattr(ppo_runner.alg, "depth_actor"):
                    actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
                else:
                    actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            
        # ==========================================
        # 🔥 在 step 前记录 lookat_id 的目标索引
        # ==========================================
        prev_goal_idx = env.cur_goal_idx[env.lookat_id].item()
        
        obs, _, rews, dones, infos = env.step(actions.detach())
        

        # ==========================================
        # 🔥 在循环内部实时检查所有环境
        # ==========================================
        for env_id in range(env.num_envs):
            # 只对未记录的环境进行检查
            if success_rate_buffer[env_id] is None:
                # 获取最后一个目标点位置
                last_goal = env.env_goals[env_id, num_goals-1, :2]  # [2] (x, y)

                # print(env.env_goals[env.lookat_id, num_goals-1, :2])
                
                # 获取机器人当前位置
                robot_pos = env.root_states[env_id, :2]  # [2] (x, y)
                # print(env.root_states[env.lookat_id, :2])
                # 计算距离
                distance_to_last_goal = torch.norm(robot_pos - last_goal).item()
                
                # 🔥 检查是否到达终点 OR done状态
                reached_goal = distance_to_last_goal <= success_threshold
                is_done = dones[env_id].item()
                
                # 如果到达终点或者done,记录结果
                if reached_goal or is_done:
                    success = distance_to_last_goal <= success_threshold
                    success_rate_buffer[env_id] = success
                    
                    # 打印该环境的首次完成信息
                    result = "✅ 成功" if success else "❌ 失败"
                    goal_idx = env.cur_goal_idx[env_id].item()
                    
                    print("\n" + "="*60)
                    print(f"环境 {env_id} 首次完成 - {result}")
                    print(f"  - 步数: {i}")
                    print(f"  - 当前目标索引: {goal_idx}/{num_goals-1}")
                    print(f"  - 到最后目标点的距离: {distance_to_last_goal:.3f} m")
                    print(f"  - 机器人位置: {robot_pos}")
                    print(f"  - 目标位置: {last_goal}")
                    print(f"  - 触发原因: {'到达终点' if reached_goal else 'Done状态'}")
                    
                    # 计算当前总体成功率
                    completed_envs = [x for x in success_rate_buffer if x is not None]
                    if completed_envs:
                        success_count = sum(completed_envs)
                        total_count = len(completed_envs)
                        print(f"  - 当前总体成功率: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
                    print("="*60 + "\n")


        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)

        print("time:", i / 50,
              "env id time:", env.episode_length_buf[env.lookat_id].item() / 50,
              "cmd vx", env.commands[env.lookat_id, 0].item(),
              "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )
        vx_error = env.commands[env.lookat_id, 0].item() - env.base_lin_vel[env.lookat_id, 0].item()
        vel_tracking_error_x_buffer.append(vx_error**2)  # 添加平方误差
        id = env.lookat_id
        print("当前观察的环境ID:", id, "X方向速度跟踪误差:", vx_error**2)
        vel_error_x_mean = np.mean(vel_tracking_error_x_buffer)
        vel_error_x_std = np.std(vel_tracking_error_x_buffer)
        print("X方向速度跟踪误差：均值 {:.4f}，标准差 {:.4f}".format(vel_error_x_mean, vel_error_x_std))

        

    

    vel_error_x_mean = np.mean(vel_tracking_error_x_buffer)
    vel_error_x_std = np.std(vel_tracking_error_x_buffer)
    print("X方向速度跟踪误差：均值 {:.4f}，标准差 {:.4f}".format(vel_error_x_mean, vel_error_x_std))

    # 最终统计
    print("\n" + "="*60)
    print("最终统计结果:")
    print("="*60)
    
    completed_envs = [x for x in success_rate_buffer if x is not None]
    if completed_envs:
        success_count = sum(completed_envs)
        fail_count = len(completed_envs) - success_count
        total_count = len(completed_envs)
        
        print(f"完成的环境数: {total_count}/{env.num_envs}")
        print(f"成功: {success_count}")
        print(f"失败: {fail_count}")
        success_rate = 100*success_count/env.num_envs
        print(f"总体成功率: {success_rate:.1f}%")
        
        # 追加写入 evaluation.log（先读再写末尾），地形按行记录，仅记录概率不为0的项，最后加分隔线
        log_file = os.path.join(log_pth, "evaluation.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        nonzero_terrain = {k: v for k, v in env_cfg.terrain.terrain_dict.items() if float(v) != 0.0}
        terrain_lines = "\n".join([f"  {k}: {v:.1f}" for k, v in nonzero_terrain.items()]) if nonzero_terrain else "  none"

        with open(log_file, "a+") as f:
            f.seek(0); _ = f.read()
            f.seek(0, os.SEEK_END)
            f.write(
                f"exptid={args.exptid}, step={i}, num_envs={env.num_envs}, "
                f"completed={total_count}, success={success_count}, fail={fail_count}, "
                f"success_rate={success_rate:.1f}%\n"
            )
            f.write("terrain:\n" + terrain_lines + "\n")
            f.write("="*60 + "\n")
    else:
        print("没有环境完成第一次 episode")
        # 也记录一次（成功率为0）
        log_file = os.path.join(log_pth, "evaluation.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a+") as f:
            f.seek(0); _ = f.read()
            f.seek(0, os.SEEK_END)
            f.write(f"exptid={args.exptid}, step={i}, num_envs={env.num_envs}, "
                    f"completed=0, success=0, fail=0, success_rate=0.0%\n")




if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
