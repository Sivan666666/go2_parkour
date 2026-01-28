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

import time
import os
from collections import deque
import statistics

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import wandb
# import ml_runlog
import datetime

from rsl_rl.algorithms import PPO
from rsl_rl.modules import *
from rsl_rl.env import VecEnv
import sys
from copy import copy, deepcopy
import warnings

import time
import os
from collections import deque
import statistics
import torch
import wandb
from copy import copy, deepcopy
import warnings

# 引用原文件中的类，假设该代码在同一个包或模块下
# 如果是在同一个文件中直接追加，则不需要import OnPolicyRunner
import time
import os
from collections import deque
import statistics
import torch
import torch.optim as optim
import wandb
from copy import copy, deepcopy
import warnings
import sys

# 假设这些模块存在于你的环境中
from rsl_rl.algorithms import PPO
from rsl_rl.algorithms import PPO_Student
from rsl_rl.modules import *
from rsl_rl.env import VecEnv

class Finetune_OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 init_wandb=True,
                 device='cpu', **kwargs):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg["estimator"]
        self.depth_encoder_cfg = train_cfg["depth_encoder"]
        self.device = device
        self.env = env
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]
        
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        print("Initializing Finetune_OnPolicyRunner for Vision Policy...")

        # 1. 初始化 ActorCritic
        # 注意：这里初始化的结构必须能承载 Depth Actor 的权重
        # 通常 Depth Actor 的输入维度是 proprio_dim + latent_dim
        # 我们假设 policy_cfg 已经适配了 Vision Actor 的配置
        self.actor_critic = ActorCriticRMA_Student(
            self.env.cfg.env.n_proprio,
            self.env.cfg.env.n_scan,
            self.env.num_obs,
            self.env.cfg.env.n_priv_latent,
            self.env.cfg.env.n_priv,
            self.env.cfg.env.history_len,
            self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)

        # 2. 初始化 Estimator (Critic 可能会用到，或者用于辅助 loss)
        self.estimator = Estimator(
            input_dim=env.cfg.env.n_proprio, 
            output_dim=env.cfg.env.n_priv, 
            hidden_dims=self.estimator_cfg["hidden_dims"]
        ).to(self.device)

        # 3. 初始化 Depth Encoder
        self.if_depth = self.depth_encoder_cfg["if_depth"]
        if self.if_depth:
            depth_backbone = DepthOnlyFCBackbone58x87(
                env.cfg.env.n_proprio, 
                self.policy_cfg["scan_encoder_dims"][-1], 
                self.depth_encoder_cfg["hidden_dims"],
            )
            self.depth_encoder = RecurrentDepthBackbone(depth_backbone, env.cfg).to(self.device)
        else:
            raise ValueError("Finetune_OnPolicyRunner requires 'if_depth' to be True.")

        # 4. 初始化 PPO 算法
        # 这里的关键是：必须确保 depth_encoder 的参数被包含在 PPO 的优化器中
        # rsl_rl 的 PPO 类通常接受额外的 modules 和 params
        # lg_class = PPO_Student # PPO
        
        # 我们不再需要单独的 depth_actor 对象，因为 actor_critic.actor 将加载它的权重
        depth_actor = None 
        
        self.alg: PPO_Student = PPO_Student(
            self.actor_critic, 
            self.estimator, self.estimator_cfg, 
            self.depth_encoder, self.depth_encoder_cfg, 
            depth_actor, # 传 None，因为我们直接训练主 policy
            device=self.device, 
            **self.alg_cfg
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        self.learn = self.learn_RL
        # 初始化 Storage
        # 注意：这里的 obs shape 可能需要根据 latent 调整，或者我们复用 num_obs 容器
        self.alg.init_storage(
            self.env.num_envs, 
            self.num_steps_per_env, 
            [self.env.num_obs], 
            [self.env.num_privileged_obs], 
            [self.env.num_actions],
        )

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print(f"Loading model for Finetuning from {path}...")
        loaded_dict = torch.load(path, map_location=self.device)


        # # 1. 加载 Critic 和其他基础结构 (从 model_state_dict)
        # # 修改：过滤掉 'critic' 相关的权重，让 Critic 保持当前的随机初始化状态
        # model_state_dict = loaded_dict['model_state_dict']
        # filtered_state_dict = {k: v for k, v in model_state_dict.items() if "critic" not in k}
        
        # self.alg.actor_critic.load_state_dict(filtered_state_dict, strict=False)
        # print("Loaded generic weights from 'model_state_dict' (excluding Critic). Critic is initialized from scratch.")

        # # 1. 加载 Critic 和其他基础结构 (从 model_state_dict)
        # # 这会先加载原来的 actor 参数，但马上会被覆盖
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        print("Loaded Critic and generic weights from 'model_state_dict'.")

        # 2. 关键修改：用 'depth_actor_state_dict' 覆盖 Actor
        if 'depth_actor_state_dict' in loaded_dict:
            print("Overwriting Actor weights with 'depth_actor_state_dict'...")
            self.alg.actor_critic.actor.load_state_dict(loaded_dict['depth_actor_state_dict'])
        else:
            warnings.warn("No 'depth_actor_state_dict' found in checkpoint! Using original actor weights.")

        # 3. 加载 Depth Encoder
        if 'depth_encoder_state_dict' in loaded_dict:
            print("Loading Depth Encoder weights...")
            self.alg.depth_encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'])
        else:
            print("Warning: No 'depth_encoder_state_dict' found, encoder is randomly initialized.")

        # 4. 加载 Estimator
        if 'estimator_state_dict' in loaded_dict:
             print("Loading Estimator weights...")
             self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])

        # 5. 优化器 (通常 Finetune 时建议重置优化器，这里保留选项)
        # if load_optimizer and 'optimizer_state_dict' in loaded_dict:
        #     print("Loading optimizer state...")
        #     self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        # else:
        #     print("Optimizer state not loaded (fresh start for finetuning).")

        print("*" * 80)
        print("Load 完成 没有问题")
        return loaded_dict.get('infos', None)

    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        print("Starting Vision Finetuning Loop...")
        
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        # 初始 Observation
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        
        infos = {}
        # 确保 Depth Buffer 准备好
        infos["depth"] = self.env.depth_buffer.clone().to(self.device)[:, -1] if self.if_depth else None
        depth_image = None
        depth_encoder_hidden_state = None
        # 切换到训练模式
        self.alg.actor_critic.train()
        self.alg.depth_encoder.train()

        # 统计数据容器
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0

            # Rollout Phase
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # print(f"Collecting step {i+1}/{self.num_steps_per_env} of iteration {it+1}/{tot_iter}")
                    # 1. 准备 Depth Encoder 的输入
                    # mean_std = self.alg.actor_critic.std.mean()
                    # print(f"Current Action Noise Std: {mean_std.item():.4f}")
                    obs_prop = obs[:, :self.env.cfg.env.n_proprio].clone()
                    # 深度编码器需要的 proprio 处理 (参考原代码逻辑，如置零 velocity/yaw 等)
                    obs_prop_encoder = obs_prop.clone()
                    obs_prop_encoder[:, 6:8] = 0 
                    
                    # 🔥 [修正] 初始化为 None
                    depth_encoder_hidden_state = None
                    if self.if_depth and hasattr(self.alg.depth_encoder, 'hidden_states'):
                        # 🔥 [修正] 必须检查值是否为 None
                        if self.alg.depth_encoder.hidden_states is not None:
                            depth_encoder_hidden_state = self.alg.depth_encoder.hidden_states.clone()

                    # 2. 运行 Depth Encoder 得到 Latent
                    if infos["depth"] is not None:
                        # print("infos['depth'].shape:", infos["depth"].shape)
                        depth_image = infos["depth"].clone()
                        depth_output = self.alg.depth_encoder(infos["depth"].clone(), obs_prop_encoder)
                        # print("depth_output.shape:", depth_output.shape)
                        depth_latent = depth_output[:, :-2] 

                    # 4. 获取 Action
                    # 使用 PPO 的 act 方法 (训练模式，带有随机性)
                    # 注意：这里传入的是拼接后的 visual_obs，而不是原始的 obs
                    # Critic 依然使用 critic_obs (privileged)
                    actions = self.alg.act(obs, critic_obs, infos, hist_encoding=True, depth_latent=depth_latent, depth_image=depth_image, depth_encoder_hidden_state=depth_encoder_hidden_state)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    total_rew = self.alg.process_env_step(rewards, dones, infos)
                    
                    # Log 统计
                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += total_rew
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # 计算 Returns (GAE)
                start = stop
                self.alg.compute_returns(critic_obs)
            
            # Update Phase            
            mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_disc_loss, mean_disc_acc, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
            # if hist_encoding:
            #     print("Updating dagger...")
            #     mean_hist_latent_loss = self.alg.update_dagger()

            # mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_disc_loss, mean_disc_acc, mean_priv_reg_loss, priv_reg_coef = 0, 0, 0, 0, 0, 0, 0
            mean_hist_latent_loss = 0

            stop = time.time()
            learn_time = stop - start
            
            # Logging
            if self.log_dir is not None:
                self.log(locals())
            
            # Save Logic
            if it < 2500:
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                    print(f"Saved model at iteration {it}")
            elif it < 5000:
                if it % (2*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            else:
                if it % (5*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss/estimator'] = locs['mean_estimator_loss']
        wandb_dict['Loss/hist_latent_loss'] = locs['mean_hist_latent_loss']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_dict['Loss/value_function'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])

        wandb.log(wandb_dict, step=locs['it'])

        str_art = f" \033[1m Finetune Iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        
        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_art.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_art.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s \n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        print(log_string)

    def save(self, path, infos=None):
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            'depth_encoder_state_dict': self.alg.depth_encoder.state_dict(),
            'estimator_state_dict': self.alg.estimator.state_dict()
        }
        torch.save(state_dict, path)

    def get_inference_policy(self, scandots_latent = None, device=None):
        # 这里的 inference policy 需要封装深度编码逻辑
        self.alg.actor_critic.eval()
        self.alg.depth_encoder.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
            self.alg.depth_encoder.to(device)
            
        # 返回一个闭包函数，处理 depth -> latent -> action
        def policy(obs, scandots_latent):
            with torch.no_grad():
                # obs_prop = obs[:, :self.env.cfg.env.n_proprio].clone()
                # obs_prop_encoder = obs_prop.clone()
                # obs_prop_encoder[:, 6:8] = 0
                
                # if infos["depth"] is not None:
                #     depth_output = self.alg.depth_encoder(infos["depth"], obs_prop_encoder)
                #     depth_latent = depth_output[:, :-2]
                # else:
                #     depth_latent = torch.zeros((obs.shape[0], self.env.cfg.env.n_priv_latent), device=self.device)
                 
                # visual_obs = torch.cat([obs_prop, depth_latent], dim=-1)
                return self.alg.actor_critic.act_inference(obs, hist_encoding=True, eval=True, scandots_latent = scandots_latent)
        
        return policy
    
    def get_estimator_inference_policy(self, device=None):
        self.alg.estimator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.estimator.to(device)
        return self.alg.estimator.inference
    
    def get_depth_encoder_inference_policy(self, device=None):
        self.alg.depth_encoder.eval()
        if device is not None:
            self.alg.depth_encoder.to(device)
        return self.alg.depth_encoder