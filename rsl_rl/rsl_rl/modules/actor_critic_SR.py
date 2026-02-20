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

import code
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU


class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output

class ScanAttentionEncoder(nn.Module):
    """
    根据论文图 8B 重构的扫描特征编码器 (适配 1 通道高度图输入)
    逻辑：
    1. Scan (L*W*1) -> 2D CNN -> Map Features ((L*W) * (d-1))
    2. Concat(Map Features, Scan) -> KV features ((L*W) * d)
    3. Proprioception -> MLP -> Proprio Embedding (1 * d) 作为 Query
    4. MHA(Q, K, V) -> 最终的 Map Encoding (1 * d)
    """
    def __init__(self, num_scan, scan_encoder_dims, proprio_dim, activation, if_scan_encode):
        super().__init__()
        # d 为注意力机制的嵌入维度，从配置中获取（如 64 或 256）
        self.d = scan_encoder_dims[-1] if scan_encoder_dims else 64
        self.num_scan = num_scan
        
        # 确定地图的网格尺寸 L 和 W
        # 注意：如果你的 num_scan 不是方阵（如 187=17x11），请手动在此指定 self.L 和 self.W
        self.L = 12
        self.W = 11
        if self.L * self.W != num_scan:
            # 兼容非方阵情况（常见于 legged-gym 的 187 点配置）
            if num_scan == 187: self.L, self.W = 17, 11
            else: raise ValueError(f"Cannot reshape num_scan {num_scan} into a rectangular grid.")

        # 1. Map CNN: 对应图 8B 的 CNN 模块
        # 输入通道为 1 (高度图)，输出通道设为 d-1，预留 1 维给原始高度值拼接
        self.map_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            activation,
            nn.Conv2d(16, self.d - 1, kernel_size=5, padding=2),
            activation
        )

        # 2. Proprioception 编码器: 生成 Query
        # 遵循你要求的“经过 MLP 编码”，将本体感知投影到维度 d
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            activation,
            nn.Linear(128, self.d)
        )

        # 3. Multi-Head Attention (MHA)
        # 将地图的所有点作为 Sequence 处理
        self.mha = nn.MultiheadAttention(embed_dim=self.d, num_heads=4, batch_first=True)
        
        # 归一化层，保证训练稳定
        self.norm = nn.LayerNorm(self.d)
        self.output_dim = self.d

    def forward(self, scan, proprioception):
        # batch_size
        batch_size = scan.shape[0]
        
        # 1. 还原高度图形状: [B, 1, L, W]
        h_map = scan.view(batch_size, 1, self.L, self.W)
        
        # 2. CNN 提取局部特征: [B, d-1, L, W]
        # 这里的卷积操作保持了原有的 L 和 W 维度
        map_geometry = self.map_cnn(h_map)
        
        # 3. 拼接原始高度坐标 (对应图 8B 的 Concat 逻辑): [B, L, W, d]
        # 调整特征维度顺序: [B, L, W, d-1]
        map_geometry = map_geometry.permute(0, 2, 3, 1)
        # 调整原始高度图顺序: [B, L, W, 1]
        h_map_raw = h_map.permute(0, 2, 3, 1)
        # 拼接后得到每个点的完整特征向量 (维度 d)
        map_combined = torch.cat([map_geometry, h_map_raw], dim=-1)
        
        # 4. 展平为 Token 序列作为 Key 和 Value: [B, L*W, d]
        kv = map_combined.reshape(batch_size, -1, self.d)
        
        # 5. 生成 Query (本体感知编码): [B, 1, d]
        q = self.proprio_encoder(proprioception).unsqueeze(1)
        
        # 6. Cross-Attention
        # 使用本体状态去检索地图中最重要的区域
        attn_out, _ = self.mha(query=q, key=kv, value=kv)
        
        # 7. 最终输出 [B, d]
        # 去掉 + q，仅保留 LayerNorm 处理后的注意力输出
        # squeeze(1) 是为了将 [B, 1, d] 变成 [B, d]
        scan_latent = self.norm(attn_out.squeeze(1))
        
        return scan_latent

class Actor(nn.Module):
    def __init__(self, num_prop, 
                 num_scan, 
                 num_actions, 
                 scan_encoder_dims,
                 actor_hidden_dims, 
                 priv_encoder_dims, 
                 num_priv_latent, 
                 num_priv_explicit, 
                 num_hist, activation, 
                 tanh_encoder_output=False) -> None:
        super().__init__()
        # prop -> scan -> priv_explicit -> priv_latent -> hist
        # actor input: prop -> scan -> priv_explicit -> latent
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0

        if len(priv_encoder_dims) > 0:
                    priv_encoder_layers = []
                    priv_encoder_layers.append(nn.Linear(num_priv_latent, priv_encoder_dims[0]))
                    priv_encoder_layers.append(activation)
                    for l in range(len(priv_encoder_dims) - 1):
                        priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                        priv_encoder_layers.append(activation)
                    self.priv_encoder = nn.Sequential(*priv_encoder_layers)
                    priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_output_dim)

        # 移除原本的 Sequential scan_encoder，改用 ScanAttentionEncoder
        if self.if_scan_encode:
            # 注意：这里需要传入本体感知的维度 num_prop
            self.scan_encoder = ScanAttentionEncoder(
                num_scan=num_scan,
                scan_encoder_dims=scan_encoder_dims,
                proprio_dim=num_prop, 
                activation=activation,
                if_scan_encode=self.if_scan_encode
            )
            self.scan_encoder_output_dim = self.scan_encoder.output_dim
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan
        
        actor_layers = []
        actor_layers.append(nn.Linear(num_prop+
                                      self.scan_encoder_output_dim+
                                      num_priv_explicit+
                                      priv_encoder_output_dim, 
                                      actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
        self.actor_backbone = nn.Sequential(*actor_layers)

    def forward(self, obs, hist_encoding: bool, eval=False, scandots_latent=None):
        proprio = obs[:, :self.num_prop]
        if not eval:
            if self.if_scan_encode:
                obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
                if scandots_latent is None:
                    scan_latent = self.scan_encoder(obs_scan, proprio)
                else:
                    scan_latent = scandots_latent
                obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
            else:
                obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
            obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
            if hist_encoding:
                latent = self.infer_hist_latent(obs)
            else:
                latent = self.infer_priv_latent(obs)
            backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
            backbone_output = self.actor_backbone(backbone_input)
            return backbone_output
        else:
            if self.if_scan_encode:
                obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
                if scandots_latent is None:
                    scan_latent = self.scan_encoder(obs_scan, proprio)
                else:
                    scan_latent = scandots_latent
                obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
            else:
                obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
            obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
            if hist_encoding:
                latent = self.infer_hist_latent(obs)
            else:
                latent = self.infer_priv_latent(obs)
            backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
            backbone_output = self.actor_backbone(backbone_input)
            return backbone_output
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan + self.num_priv_explicit: self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan, obs[:, :self.num_prop])

class ActorCriticRMA(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_scan,
                        num_critic_obs,
                        num_priv_latent, 
                        num_priv_explicit,
                        num_hist,
                        num_actions,
                        scan_encoder_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticRMA, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        activation = get_activation(activation)
        
        self.actor = Actor(num_prop, num_scan, num_actions, scan_encoder_dims, actor_hidden_dims, priv_encoder_dims, num_priv_latent, num_priv_explicit, num_hist, activation, tanh_encoder_output=kwargs['tanh_encoder_output'])
        

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, hist_encoding):
        mean = self.actor(observations, hist_encoding)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, hist_encoding=False, **kwargs):
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding=False, eval=False, scandots_latent=None, **kwargs):
        if not eval:
            actions_mean = self.actor(observations, hist_encoding, eval, scandots_latent)
            return actions_mean
        else:
            actions_mean, latent_hist, latent_priv = self.actor(observations, hist_encoding, eval=True)
            return actions_mean, latent_hist, latent_priv

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
