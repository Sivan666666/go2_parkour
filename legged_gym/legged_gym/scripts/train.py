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

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from shutil import copyfile
import torch
import wandb

def train(args):
    args.headless = True
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    try:
        os.makedirs(log_pth)
    except:
        pass

    # ==========================================
    # 🔥 备份配置文件到实验文件夹
    # ==========================================
    
    # 1️⃣ 备份任务特定的配置文件（如 go2_config.py）
    task_config_path = os.path.join(
        LEGGED_GYM_ENVS_DIR, 
        args.task,  # 例如 "go2"
        f"{args.task}_config.py"  # 例如 "go2_config.py"
    )
    
    if os.path.exists(task_config_path):
        task_config_backup = os.path.join(log_pth, f"{args.task}_config.py")
        copyfile(task_config_path, task_config_backup)
        print(f"✅ Backed up config: {task_config_path} -> {task_config_backup}")
    else:
        print(f"⚠️  Warning: Task config not found at {task_config_path}")
    
    # 2️⃣ 备份基础配置文件（保留这部分！）
    base_config_backup = os.path.join(log_pth, "legged_robot_config.py")
    base_robot_backup = os.path.join(log_pth, "legged_robot.py")
    copyfile(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", base_config_backup)
    copyfile(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", base_robot_backup)
    print(f"✅ Backed up base configs to {log_pth}")

    if args.debug:
        mode = "disabled"
        args.rows = 10
        args.cols = 8
        args.num_envs = 64
    else:
        mode = "online"
    
    if args.no_wandb:
        mode = "disabled"
    wandb.init(project=args.proj_name, name=args.exptid,  group=args.exptid[:3], mode=mode, dir="../../logs")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")

    # 上传任务配置
    if os.path.exists(task_config_path):
        wandb.save(task_config_path, policy="now")
        print(f"✅ Uploaded {args.task}_config.py to WandB")

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    # Log configs immediately
    args = get_args()
    train(args)
