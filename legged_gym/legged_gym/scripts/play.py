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
    env_cfg.env.num_envs = 16 if not args.save else 64
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.height = [0.02, 0.02]
    # env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
    #                                 "rough slope up": 0.0,
    #                                 "rough slope down": 0.0,
    #                                 "rough stairs up": 0., 
    #                                 "rough stairs down": 0., 
    #                                 "discrete": 0., 
    #                                 "stepping stones": 0.0,
    #                                 "gaps": 0., 
    #                                 "smooth flat": 0,
    #                                 "pit": 0.0,
    #                                 "wall": 0.0,
    #                                 "platform": 0.,
    #                                 "large stairs up": 0.,
    #                                 "large stairs down": 0.,
    #                                 "parkour": 0.2,
    #                                 "parkour_hurdle": 0.2,
    #                                 "parkour_flat": 0.2,
    #                                 "parkour_step": 0.2,
    #                                 "parkour_gap": 0.2, 
    #                                 "demo": 0.}
    # env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
    #                                 "rough slope up": 0.,
    #                                 "rough slope down": 0.0,
    #                                 "rough stairs up": 0., 
    #                                 "rough stairs down": 0., 
    #                                 "discrete": 0., 
    #                                 "stepping stones": 0.0,
    #                                 "gaps": 0., 
    #                                 "smooth flat": 0,
    #                                 "pit": 0.0,
    #                                 "wall": 0.0,
    #                                 "platform": 0.,
    #                                 "large stairs up": 0.,
    #                                 "large stairs down": 0.,
    #                                 "parkour": 0.2,
    #                                 "parkour_hurdle": 0.2,
    #                                 "parkour_flat": 0.2,
    #                                 "parkour_step": 0.2,
    #                                 "parkour_gap": 0.2, 
    #                                 "demo": 0.}
    
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "normal stairs up": 0.0,
                                    "normal stairs down": 0.5,
                                    "discrete": 0., 
                                    "stepping stones": 0.0,
                                    "gaps": 0., 
                                    "flat": 0.0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "hollow stairs up": 0.0, 
                                    "hollow stairs down": 0.5,
                                    "parkour": 0.0,         # 0.2
                                    "parkour_hurdle": 0.0,  # 0.2
                                    "parkour_flat": 0.,
                                    "parkour_step": 0.0,    # 0.2
                                    "parkour_gap": 0.0,     # 0.2
                                    "demo": 0.0}            # 0.2
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    # env_cfg.depth.angle = [0, 1]

    # env_cfg.depth.position =  [0.355, 0, 0.065]
    # env_cfg.depth.angle = [20, 21]

    # env_cfg.depth.position = [0.35, 0, 0.147]  # front camera 
    # env_cfg.depth.angle = [60-1, 30+1]  # positive pitch down  #27-5,27+5

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
    infos["rgb"] = env.rgb_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    show_plots = False # 改为 False 关闭所有绘图
    # 历史
    cmd_vx_hist, act_vx_hist, base_h_hist = [], [], []
    yaw_hist, yaw_cmd_hist, pos_x_hist, pos_y_hist = [], [], [], []

    # 子图初始化：两张图
    if show_plots:
        try:
            import matplotlib.pyplot as plt
            plt.ion()
            # 图1：速度 + 高度（两行）
            fig1, (ax_v, ax_h) = plt.subplots(2, 1, num=777, figsize=(8, 6), sharex=True)
            ax_v.set_title("Velocity (lookat env)"); ax_v.set_ylabel("vx (m/s)")
            line_cmd, = ax_v.plot([], [], label='cmd vx', color='tab:red')
            line_act, = ax_v.plot([], [], label='actual vx', color='tab:blue')
            ax_v.legend(); ax_v.grid(True, alpha=0.3)
            ax_h.set_title("Base height (lookat env)")
            ax_h.set_xlabel("Step"); ax_h.set_ylabel("height (m)")
            line_bh, = ax_h.plot([], [], label='base_height', color='tab:green')
            ax_h.legend(); ax_h.grid(True, alpha=0.3)

            # 图2：yaw + 位置（3行：yaw、x、y）
            fig2, (ax_yaw, ax_px, ax_py) = plt.subplots(3, 1, num=778, figsize=(8, 9), sharex=True)
            ax_yaw.set_title("Yaw angle & command (lookat env)")
            ax_yaw.set_ylabel("yaw (rad)")
            line_yaw, = ax_yaw.plot([], [], label='yaw', color='tab:purple')
            line_yaw_cmd, = ax_yaw.plot([], [], label='yaw_cmd', color='tab:orange')
            ax_yaw.legend(); ax_yaw.grid(True, alpha=0.3)
            ax_px.set_title("Base position X"); ax_px.set_ylabel("x (m)")
            line_px, = ax_px.plot([], [], label='x', color='tab:brown')
            ax_px.legend(); ax_px.grid(True, alpha=0.3)
            ax_py.set_title("Base position Y"); ax_py.set_xlabel("Step"); ax_py.set_ylabel("y (m)")
            line_py, = ax_py.plot([], [], label='y', color='tab:cyan')
            ax_py.legend(); ax_py.grid(True, alpha=0.3)
        except Exception:
            show_plots = False
            fig1 = fig2 = ax_v = ax_h = ax_yaw = ax_px = ax_py = None
            line_cmd = line_act = line_bh = line_yaw = line_yaw_cmd = line_px = line_py = None
    else:
        fig1 = fig2 = ax_v = ax_h = ax_yaw = ax_px = ax_py = None
        line_cmd = line_act = line_bh = line_yaw = line_yaw_cmd = line_px = line_py = None


    for i in range(10*int(env.max_episode_length)):
        # # 强制设置 yaw 相关命令为0，保持直线行走
        # env.commands[:, 2] = 0.0  # yaw rate command 设为0
        
        # # 如果你想保持特定的前进方向，可以设置：
        env.commands[:, 0] = 0.5  # forward velocity
        # env.commands[:, 2] = 0.5  # lateral velocity

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
                        depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student, infos["rgb"])
                    # depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                        depth_latent = depth_latent_and_yaw[:, :-2]
                        yaw = depth_latent_and_yaw[:, -2:] * 0
                    # 不使用 yaw 修正，保持原始观测
                    obs[:, 6:8] = 1.5*yaw  # 注释掉这行
                # obs[:, 6:8] = -env.yaw.unsqueeze(1)  # [num_envs, 2] 两列都填 -yaw
                    obs[:, 6:8] = -env.yaw.unsqueeze(1)  # [num_envs, 2] 两列都填 -yaw  # 强制设为0
                    # obs[:, 6:8] = 0
                
                        
                else:
                    depth_latent = None
                # obs[:, 6:8] = 0  # 强制设为0
                if hasattr(ppo_runner.alg, "depth_actor"):
                    actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
                else:
                    actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            
        obs, _, rews, dones, infos = env.step(actions.detach())
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
            
        # 记录
        look_id = env.lookat_id
        cmd_vx = env.commands[look_id, 0].item()
        act_vx = env.base_lin_vel[look_id, 0].item()
        base_h = env._get_base_heights()[look_id].item()
        cmd_vx_hist.append(cmd_vx); act_vx_hist.append(act_vx); base_h_hist.append(base_h)

        # 实际 yaw（由四元数转 yaw）
        q = env.base_quat[look_id]
        x, y, z, w = q[0].item(), q[1].item(), q[2].item(), q[3].item()
        denom = 1.0 - 2.0*(y*y + z*z)
        yaw = np.arctan2(2.0*(w*z + x*y), denom)
        yaw_hist.append(float(yaw))

        # 观测中的 delta_yaw（目标-当前），换成期望 yaw 便于对比
        delta_yaw = -env.yaw[look_id].item()  # 与 obs[:,6] 一致
        # print("delta_yaw (=-yaw):", delta_yaw)
        yaw_desired = yaw + delta_yaw
        yaw_cmd_hist.append(yaw_desired)

        # 位置
        pos_x_hist.append(env.root_states[look_id, 0].item())
        pos_y_hist.append(env.root_states[look_id, 1].item())

        # 刷新图1
        if show_plots and ax_v is not None:
            line_cmd.set_data(range(len(cmd_vx_hist)), cmd_vx_hist)
            line_act.set_data(range(len(act_vx_hist)), act_vx_hist)
            ax_v.relim(); ax_v.autoscale_view()
            line_bh.set_data(range(len(base_h_hist)), base_h_hist)
            ax_h.relim(); ax_h.autoscale_view()
            fig1.canvas.draw_idle(); fig1.canvas.flush_events()

        # 刷新图2（yaw + 位置）
        if show_plots and ax_yaw is not None:
            line_yaw.set_data(range(len(yaw_hist)), yaw_hist)
            line_yaw_cmd.set_data(range(len(yaw_cmd_hist)), yaw_cmd_hist)  # 这里是 obs[:,6] 的曲线
            ax_yaw.relim(); ax_yaw.autoscale_view()
            line_px.set_data(range(len(pos_x_hist)), pos_x_hist)
            line_py.set_data(range(len(pos_y_hist)), pos_y_hist)
            ax_px.relim(); ax_px.autoscale_view()
            ax_py.relim(); ax_py.autoscale_view()
            fig2.canvas.draw_idle(); fig2.canvas.flush_events()

        print("time:", env.episode_length_buf[env.lookat_id].item() / 50, 
              "cmd vx", env.commands[env.lookat_id, 0].item(),
              "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )
        
        id = env.lookat_id
        

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
