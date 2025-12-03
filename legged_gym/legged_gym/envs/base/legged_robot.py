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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict
from scipy.spatial.transform import Rotation as R
from .legged_robot_config import LeggedRobotCfg

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def euler_from_quaternion(quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

class PerlinNoise:
    """简化的柏林噪声生成器"""
    def __init__(self, scale=10.0, octaves=2):
        self.scale = scale
        self.octaves = octaves
        self.time_offset = 0.0
    
    def generate(self, shape, time_offset=0.0):
        """生成2D柏林噪声
        Args:
            shape: (height, width)
            time_offset: 时间偏移,用于生成连续的噪声
        Returns:
            noise: shape的噪声图,值范围[0, 1]
        """
        h, w = shape
        noise = np.zeros((h, w))
        
        # 使用多个八度叠加
        amplitude = 1.0
        frequency = 1.0
        
        for _ in range(self.octaves):
            # 生成随机梯度网格
            grid_h = int(h / self.scale / frequency) + 2
            grid_w = int(w / self.scale / frequency) + 2
            
            # 使用时间偏移作为随机种子
            np.random.seed(int(time_offset * 1000) % 2**31)
            gradients = np.random.randn(grid_h, grid_w, 2)
            
            # 插值生成噪声
            for i in range(h):
                for j in range(w):
                    # 网格坐标
                    x = i / self.scale / frequency
                    y = j / self.scale / frequency
                    
                    # 四个角点
                    x0, y0 = int(x), int(y)
                    x1, y1 = x0 + 1, y0 + 1
                    
                    # 双线性插值
                    sx = x - x0
                    sy = y - y0
                    
                    n00 = self._dot_grid_gradient(gradients, x0, y0, x, y)
                    n10 = self._dot_grid_gradient(gradients, x1, y0, x, y)
                    n01 = self._dot_grid_gradient(gradients, x0, y1, x, y)
                    n11 = self._dot_grid_gradient(gradients, x1, y1, x, y)
                    
                    nx0 = self._lerp(n00, n10, sx)
                    nx1 = self._lerp(n01, n11, sx)
                    value = self._lerp(nx0, nx1, sy)
                    
                    noise[i, j] += value * amplitude
            
            amplitude *= 0.5
            frequency *= 2.0
        
        # 归一化到[0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        return noise
    
    def _dot_grid_gradient(self, gradients, ix, iy, x, y):
        """计算梯度向量点积"""
        dx = x - ix
        dy = y - iy
        if ix >= gradients.shape[0] or iy >= gradients.shape[1]:
            return 0
        return gradients[ix, iy, 0] * dx + gradients[ix, iy, 1] * dy
    
    def _lerp(self, a, b, t):
        """线性插值"""
        return a + t * (b - a)

# class PerlinNoiseGPU:
#     """GPU 加速的柏林噪声生成器 (向量化八度循环)"""
#     def __init__(self, scale=10.0, octaves=2, device='cuda'):
#         self.scale = scale
#         self.octaves = octaves
#         self.device = device
        
#         # 🔥 预计算所有八度的频率和振幅 (避免循环中重复计算)
#         self.frequencies = torch.tensor(
#             [2.0**i for i in range(octaves)],
#             device=device,
#             dtype=torch.float32
#         )
#         self.amplitudes = torch.tensor(
#             [0.5**i for i in range(octaves)],
#             device=device,
#             dtype=torch.float32
#         )
        
#     def generate(self, shape, time_offset=0.0):
#         """生成2D柏林噪声 (批量处理所有八度)
#         Args:
#             shape: (height, width)
#             time_offset: 时间偏移,用于生成连续的噪声
#         Returns:
#             noise: shape 的噪声图 Tensor,值范围[0, 1]
#         """
#         h, w = shape
        
#         # 🔥 批量生成所有八度的噪声 (替代 for 循环)
#         # 为所有八度创建统一的坐标网格
#         octave_noise = torch.zeros(self.octaves, h, w, device=self.device)
        
#         for octave in range(self.octaves):
#             frequency = self.frequencies[octave]
#             amplitude = self.amplitudes[octave]
            
#             # 生成梯度网格
#             grid_h = int(h / self.scale / frequency) + 2
#             grid_w = int(w / self.scale / frequency) + 2
            
#             # 使用时间偏移作为随机种子
#             torch.manual_seed(int((time_offset + octave) * 1000) % 2**31)
#             gradients = torch.randn(grid_h, grid_w, 2, device=self.device)
            
#             # 🔥 坐标网格 (复用同一份内存)
#             y_grid = (torch.arange(h, device=self.device).float().unsqueeze(1) / self.scale / frequency).expand(h, w)
#             x_grid = (torch.arange(w, device=self.device).float().unsqueeze(0) / self.scale / frequency).expand(h, w)
            
#             # 四个角点坐标
#             x0 = torch.clamp(torch.floor(x_grid).long(), 0, grid_w - 1)
#             y0 = torch.clamp(torch.floor(y_grid).long(), 0, grid_h - 1)
#             x1 = torch.clamp(x0 + 1, 0, grid_w - 1)
#             y1 = torch.clamp(y0 + 1, 0, grid_h - 1)
            
#             # 插值系数
#             sx = x_grid - x0.float()
#             sy = y_grid - y0.float()
            
#             # 🔥 梯度点积 (向量化,避免重复计算)
#             # 预计算向量距离
#             dx0 = sx
#             dy0 = sy
#             dx1 = sx - 1.0
#             dy1 = sy - 1.0
            
#             # 四个角点的梯度
#             g00 = gradients[y0, x0]  # [h, w, 2]
#             g10 = gradients[y0, x1]
#             g01 = gradients[y1, x0]
#             g11 = gradients[y1, x1]
            
#             # 点积 (向量化)
#             n00 = g00[..., 0] * dx0 + g00[..., 1] * dy0
#             n10 = g10[..., 0] * dx1 + g10[..., 1] * dy0
#             n01 = g01[..., 0] * dx0 + g01[..., 1] * dy1
#             n11 = g11[..., 0] * dx1 + g11[..., 1] * dy1
            
#             # 双线性插值 (直接展开,避免函数调用)
#             nx0 = n00 + sx * (n10 - n00)
#             nx1 = n01 + sx * (n11 - n01)
#             value = nx0 + sy * (nx1 - nx0)
            
#             octave_noise[octave] = value * amplitude
        
#         # 🔥 所有八度求和 (GPU 并行)
#         noise = octave_noise.sum(dim=0)
        
#         # 归一化到 [0, 1]
#         noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
#         return noise

class PerlinNoiseGPU:
    """GPU 加速的柏林噪声生成器 (带性能分析)"""
    def __init__(self, scale=10.0, octaves=2, device='cuda'):
        self.scale = scale
        self.octaves = octaves
        self.device = device
        
        # 预计算所有八度的频率和振幅
        self.frequencies = torch.tensor(
            [2.0**i for i in range(octaves)],
            device=device,
            dtype=torch.float32
        )
        self.amplitudes = torch.tensor(
            [0.5**i for i in range(octaves)],
            device=device,
            dtype=torch.float32
        )
        
        # 🔥 性能统计
        self._timing_counter = 0
        self._timing_accumulator = {}
        
    def generate(self, shape, time_offset=0.0):
        """生成2D柏林噪声 (带详细性能分析)"""
        import time
        
        total_start = time.time()
        timings = {}
        
        h, w = shape
        octave_noise = torch.zeros(self.octaves, h, w, device=self.device)
        
        # 🔥 分八度统计
        octave_timings = []
        
        for octave in range(self.octaves):
            octave_start = time.time()
            octave_timing = {}
            
            frequency = self.frequencies[octave]
            amplitude = self.amplitudes[octave]
            
            # 1️⃣ 生成梯度网格
            t0 = time.time()
            grid_h = int(h / self.scale / frequency) + 2
            grid_w = int(w / self.scale / frequency) + 2
            torch.manual_seed(int((time_offset + octave) * 1000) % 2**31)
            gradients = torch.randn(grid_h, grid_w, 2, device=self.device)
            octave_timing['1_gradient_gen'] = (time.time() - t0) * 1000
            
            # 2️⃣ 生成坐标网格
            t0 = time.time()
            y_grid = (torch.arange(h, device=self.device).float().unsqueeze(1) / self.scale / frequency).expand(h, w)
            x_grid = (torch.arange(w, device=self.device).float().unsqueeze(0) / self.scale / frequency).expand(h, w)
            octave_timing['2_coord_grid'] = (time.time() - t0) * 1000
            
            # 3️⃣ 计算四个角点坐标
            t0 = time.time()
            x0 = torch.clamp(torch.floor(x_grid).long(), 0, grid_w - 1)
            y0 = torch.clamp(torch.floor(y_grid).long(), 0, grid_h - 1)
            x1 = torch.clamp(x0 + 1, 0, grid_w - 1)
            y1 = torch.clamp(y0 + 1, 0, grid_h - 1)
            octave_timing['3_corner_coords'] = (time.time() - t0) * 1000
            
            # 4️⃣ 计算插值系数
            t0 = time.time()
            sx = x_grid - x0.float()
            sy = y_grid - y0.float()
            octave_timing['4_interp_coeff'] = (time.time() - t0) * 1000
            
            # 5️⃣ 预计算向量距离
            t0 = time.time()
            dx0 = sx
            dy0 = sy
            dx1 = sx - 1.0
            dy1 = sy - 1.0
            octave_timing['5_vector_dist'] = (time.time() - t0) * 1000
            
            # 6️⃣ 获取四个角点的梯度
            t0 = time.time()
            g00 = gradients[y0, x0]
            g10 = gradients[y0, x1]
            g01 = gradients[y1, x0]
            g11 = gradients[y1, x1]
            octave_timing['6_gradient_fetch'] = (time.time() - t0) * 1000
            
            # 7️⃣ 梯度点积
            t0 = time.time()
            n00 = g00[..., 0] * dx0 + g00[..., 1] * dy0
            n10 = g10[..., 0] * dx1 + g10[..., 1] * dy0
            n01 = g01[..., 0] * dx0 + g01[..., 1] * dy1
            n11 = g11[..., 0] * dx1 + g11[..., 1] * dy1
            octave_timing['7_dot_product'] = (time.time() - t0) * 1000
            
            # 8️⃣ 双线性插值
            t0 = time.time()
            nx0 = n00 + sx * (n10 - n00)
            nx1 = n01 + sx * (n11 - n01)
            value = nx0 + sy * (nx1 - nx0)
            octave_timing['8_bilinear_interp'] = (time.time() - t0) * 1000
            
            # 9️⃣ 乘以振幅并存储
            t0 = time.time()
            octave_noise[octave] = value * amplitude
            octave_timing['9_amplitude_scale'] = (time.time() - t0) * 1000
            
            octave_timing['OCTAVE_TOTAL'] = (time.time() - octave_start) * 1000
            octave_timings.append(octave_timing)
        
        # 🔟 所有八度求和
        t0 = time.time()
        noise = octave_noise.sum(dim=0)
        timings['10_octave_sum'] = (time.time() - t0) * 1000
        
        # 1️⃣1️⃣ 归一化
        t0 = time.time()
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        timings['11_normalize'] = (time.time() - t0) * 1000
        
        timings['TOTAL'] = (time.time() - total_start) * 1000
        
        # 🔥 累积统计
        self._accumulate_timings(timings, octave_timings)
        self._timing_counter += 1
        
        # 每 100 次打印一次
        if self._timing_counter % 100 == 0:
            self._print_statistics()
        
        return noise
    
    def _accumulate_timings(self, timings, octave_timings):
        """累积时间统计"""
        # 累积总体时间
        for key, value in timings.items():
            if key not in self._timing_accumulator:
                self._timing_accumulator[key] = 0.0
            self._timing_accumulator[key] += value
        
        # 累积每个八度的时间 (取平均)
        for octave_idx, octave_timing in enumerate(octave_timings):
            for key, value in octave_timing.items():
                avg_key = f'octave_avg_{key}'
                if avg_key not in self._timing_accumulator:
                    self._timing_accumulator[avg_key] = 0.0
                self._timing_accumulator[avg_key] += value / len(octave_timings)
    
    def _print_statistics(self):
        """打印性能统计"""
        print("\n" + "="*90)
        print(f"🔥 柏林噪声生成器性能分析 (平均值, 基于 {self._timing_counter} 次调用)")
        print("="*90)
        
        # 打印每个八度的平均耗时
        print("\n📊 单个八度的详细耗时:")
        print("-"*90)
        
        octave_keys = [
            ('1_gradient_gen', '梯度网格生成'),
            ('2_coord_grid', '坐标网格生成'),
            ('3_corner_coords', '角点坐标计算'),
            ('4_interp_coeff', '插值系数计算'),
            ('5_vector_dist', '向量距离计算'),
            ('6_gradient_fetch', '梯度索引获取'),
            ('7_dot_product', '梯度点积计算'),
            ('8_bilinear_interp', '双线性插值'),
            ('9_amplitude_scale', '振幅缩放'),
            ('OCTAVE_TOTAL', '✅ 单八度总计'),
        ]
        
        octave_total = self._timing_accumulator.get('octave_avg_OCTAVE_TOTAL', 0) / self._timing_counter
        
        for key, desc in octave_keys:
            avg_key = f'octave_avg_{key}'
            if avg_key in self._timing_accumulator:
                avg_time = self._timing_accumulator[avg_key] / self._timing_counter
                if key == 'OCTAVE_TOTAL':
                    print("-"*90)
                    print(f"  {desc:.<50} {avg_time:>10.4f} ms")
                else:
                    percentage = (avg_time / octave_total * 100) if octave_total > 0 else 0
                    print(f"  {desc:.<50} {avg_time:>10.4f} ms  ({percentage:>5.1f}%)")
        
        # 打印总体耗时
        print("\n📊 总体耗时:")
        print("-"*90)
        
        total_keys = [
            ('10_octave_sum', '所有八度求和'),
            ('11_normalize', '归一化处理'),
            ('TOTAL', '⏱️  总计'),
        ]
        
        total_time = self._timing_accumulator.get('TOTAL', 0) / self._timing_counter
        
        for key, desc in total_keys:
            if key in self._timing_accumulator:
                avg_time = self._timing_accumulator[key] / self._timing_counter
                if key == 'TOTAL':
                    print("-"*90)
                    print(f"  {desc:.<50} {avg_time:>10.4f} ms")
                else:
                    percentage = (avg_time / total_time * 100) if total_time > 0 else 0
                    print(f"  {desc:.<50} {avg_time:>10.4f} ms  ({percentage:>5.1f}%)")
        
        # 八度数统计
        all_octaves_time = octave_total * self.octaves
        octave_overhead = total_time - all_octaves_time
        
        print("\n📊 八度统计:")
        print("-"*90)
        print(f"  {'八度数量':.<50} {self.octaves:>10d}")
        print(f"  {'单八度平均耗时':.<50} {octave_total:>10.4f} ms")
        print(f"  {'所有八度总耗时':.<50} {all_octaves_time:>10.4f} ms  ({all_octaves_time/total_time*100:>5.1f}%)")
        print(f"  {'其他开销(求和+归一化)':.<50} {octave_overhead:>10.4f} ms  ({octave_overhead/total_time*100:>5.1f}%)")
        
        print("="*90 + "\n")
        
        # 重置计数器
        self._timing_counter = 0
        self._timing_accumulator = {}
    
# class PerlinNoiseGPU:
#     """GPU 加速的柏林噪声生成器"""
#     def __init__(self, scale=10.0, octaves=2, device='cuda'):
#         self.scale = scale
#         self.octaves = octaves
#         self.device = device
        
#     def generate(self, shape, time_offset=0.0):
#         """生成2D柏林噪声 (全 GPU 实现)
#         Args:
#             shape: (height, width)
#             time_offset: 时间偏移,用于生成连续的噪声
#         Returns:
#             noise: shape 的噪声图 Tensor,值范围[0, 1]
#         """
#         h, w = shape
#         noise = torch.zeros((h, w), device=self.device)
        
#         # 使用多个八度叠加
#         amplitude = 1.0
#         frequency = 1.0
        
#         for octave in range(self.octaves):
#             # 生成随机梯度网格
#             grid_h = int(h / self.scale / frequency) + 2
#             grid_w = int(w / self.scale / frequency) + 2
            
#             # 🔥 使用时间偏移作为随机种子 (在 GPU 上)
#             torch.manual_seed(int((time_offset + octave) * 1000) % 2**31)
#             gradients = torch.randn(grid_h, grid_w, 2, device=self.device)
            
#             # 🔥 创建网格坐标 (GPU 张量操作,替代双重循环)
#             # 生成 (h, w) 的坐标网格
#             y_coords = torch.arange(h, device=self.device).float().unsqueeze(1) / self.scale / frequency  # [h, 1]
#             x_coords = torch.arange(w, device=self.device).float().unsqueeze(0) / self.scale / frequency  # [1, w]
            
#             # 广播到 (h, w)
#             y_grid = y_coords.expand(h, w)  # [h, w]
#             x_grid = x_coords.expand(h, w)  # [h, w]
            
#             # 四个角点的整数坐标
#             x0 = torch.floor(x_grid).long()  # [h, w]
#             y0 = torch.floor(y_grid).long()
#             x1 = x0 + 1
#             y1 = y0 + 1

#             # Clip 坐标防止越界
#             x0 = torch.clamp(x0, 0, grid_w - 1)
#             y0 = torch.clamp(y0, 0, grid_h - 1)
#             x1 = torch.clamp(x1, 0, grid_w - 1)
#             y1 = torch.clamp(y1, 0, grid_h - 1)
            
#             # 🔥 计算四个角点的梯度点积 (向量化计算)
#             # 相对坐标
#             sx = x_grid - x0.float()  # [h, w]
#             sy = y_grid - y0.float()
            
#             # 取四个角点的梯度 [h, w, 2]
#             grad_00 = gradients[y0, x0]  # [h, w, 2]
#             grad_10 = gradients[y0, x1]
#             grad_01 = gradients[y1, x0]
#             grad_11 = gradients[y1, x1]
            
#             # 计算到四个角点的向量
#             vec_00 = torch.stack([x_grid - x0.float(), y_grid - y0.float()], dim=-1)  # [h, w, 2]
#             vec_10 = torch.stack([x_grid - x1.float(), y_grid - y0.float()], dim=-1)
#             vec_01 = torch.stack([x_grid - x0.float(), y_grid - y1.float()], dim=-1)
#             vec_11 = torch.stack([x_grid - x1.float(), y_grid - y1.float()], dim=-1)
            
#             # 梯度点积 (向量化)
#             n00 = (grad_00 * vec_00).sum(dim=-1)  # [h, w]
#             n10 = (grad_10 * vec_10).sum(dim=-1)
#             n01 = (grad_01 * vec_01).sum(dim=-1)
#             n11 = (grad_11 * vec_11).sum(dim=-1)
            
#             # 🔥 双线性插值 (向量化)
#             nx0 = self._lerp(n00, n10, sx)
#             nx1 = self._lerp(n01, n11, sx)
#             value = self._lerp(nx0, nx1, sy)
            
#             noise += value * amplitude
#             amplitude *= 0.5
#             frequency *= 2.0
        
#         # 归一化到 [0, 1]
#         noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
#         return noise
    
#     def _lerp(self, a, b, t):
#         """线性插值 (GPU 张量操作)
#         Args:
#             a: 起始值 tensor
#             b: 终止值 tensor
#             t: 插值系数 tensor [0, 1]
#         Returns:
#             插值结果 tensor
#         """
#         return a + t * (b - a)

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)

        # 🔥 初始化柏林噪声生成器
        self.perlin_time_offset = 0.0
        # 🔥 初始化柏林噪声生成器 (GPU 版本)
        
        if self.cfg.depth.use_camera:
            # 🔥 预计算 Sobel 卷积核 (用于边缘检测)
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=sim_device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=sim_device)
            self._sobel_x = sobel_x.view(1, 1, 3, 3)
            self._sobel_y = sobel_y.view(1, 1, 3, 3)

            print("="*60)
            print("🔥 初始化 GPU 柏林噪声生成器...")
            
            # 🔥 使用 GPU 版本的柏林噪声生成器
            self.perlin_noise_generator = PerlinNoiseGPU(
                scale=self.cfg.depth.perlin_noise_scale,
                octaves=self.cfg.depth.perlin_noise_octaves,
                device=sim_device
            )
            
            # 每个环境的时间偏移 (在 GPU 上)
            self.perlin_time_offsets = torch.zeros(
                self.cfg.env.num_envs,
                device=sim_device,
                dtype=torch.float32
            )
            
            # 给每个环境不同的初始相位
            self.perlin_time_offsets[:] = torch.rand(
                self.cfg.env.num_envs,
                device=sim_device
            ) * 10.0  # 0-10 的随机初始相位
            
            # 🔥 环境噪声启用标志
            edge_enable_prob = getattr(self.cfg.depth, 'edge_noise_enable_prob', 1.0)
            self.env_has_edge_noise = torch.rand(self.cfg.env.num_envs, device=sim_device) < edge_enable_prob
            
            perlin_enable_prob = getattr(self.cfg.depth, 'perlin_noise_enable_prob', 1.0)
            self.env_has_perlin_noise = torch.rand(self.cfg.env.num_envs, device=sim_device) < perlin_enable_prob
            
            # 🔥 块状空洞启用标志 (新增独立概率)
            hole_enable_prob = getattr(self.cfg.depth, 'hole_noise_enable_prob', 0.2)
            self.env_has_hole_noise = torch.rand(self.cfg.env.num_envs, device=sim_device) < hole_enable_prob

            gaussian_enable_prob = getattr(self.cfg.depth, 'gaussian_noise_enable_prob', 1.0)
            self.env_has_gaussian_noise = torch.rand(self.cfg.env.num_envs, device=sim_device) < gaussian_enable_prob
            
            print(f"   ✅ GPU 柏林噪声生成器已创建")
            print(f"   噪声分配统计:")
            print(f"      - 边缘噪声: {self.env_has_edge_noise.sum().item()}/{self.cfg.env.num_envs} ({edge_enable_prob*100:.0f}%)")
            print(f"      - 块状空洞: {self.env_has_hole_noise.sum().item()}/{self.cfg.env.num_envs} ({hole_enable_prob*100:.0f}%)")
            print(f"      - 高斯噪声: {self.env_has_gaussian_noise.sum().item()}/{self.cfg.env.num_envs} ({gaussian_enable_prob*100:.0f}%)")


        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), 
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.post_physics_step()



    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = self.reindex(actions)

        actions.to(self.device)
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        if self.cfg.domain_rand.action_delay:
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)
            if self.viewer:
                self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)
            indices = -self.delay -1
            actions = self.action_history_buf[:, indices.long()] # delay for 1/50=20ms

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.extras["delta_yaw_ok"] = self.delta_yaw < 0.6
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_history_observations(self):
        return self.obs_history_buf
    
    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image
    
    def process_depth_image(self, depth_image, env_id):
        """处理深度图像 (全 GPU 优化版本)"""
        # import time
        
        # 🔥 总耗时统计
        # total_start = time.time()
        # timings = {}
        
        # 裁剪图像
        #t0 = time.time()
        depth_image = self.crop_depth_image(depth_image)
        #timings['crop'] = (time.time() - t0) * 1000  # ms
        
        # 添加均匀噪声
        #t0 = time.time()
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1, device=self.device)-0.5)[0]
        #timings['uniform_noise'] = (time.time() - t0) * 1000
        
        if getattr(self.cfg.depth, 'enable_noise', True):
            
            # ==========================================
            # 🔥 1. Clip: 近距离设为无穷 (全 GPU)
            # ==========================================
            #t0 = time.time()
            distance = torch.abs(depth_image)
            near_mask = distance < self.cfg.depth.clip_near_distance
            depth_image[near_mask] = -self.cfg.depth.far_clip
            #timings['1_clip_near'] = (time.time() - t0) * 1000
            
            # ==========================================
            # 🔥 2. Gaussian noise: 高斯噪声 (全 GPU)
            # ==========================================
            #t0 = time.time()
            if hasattr(self.cfg.depth, 'gaussian_noise_std') and self.cfg.depth.gaussian_noise_std > 0:
                if self.env_has_gaussian_noise[env_id]:
                    distance = torch.abs(depth_image)
                    valid_mask = distance <= 3.0
                    
                    base_std = self.cfg.depth.gaussian_noise_std
                    distance_factor = getattr(self.cfg.depth, 'gaussian_noise_distance_factor', 0.5)
                    adaptive_std = base_std * (1.0 + distance_factor * distance)
                    
                    gaussian_noise = torch.randn_like(depth_image) * adaptive_std
                    gaussian_noise[~valid_mask] = 0.0
                    depth_image += gaussian_noise
            #timings['2_gaussian_noise'] = (time.time() - t0) * 1000
            
            # ==========================================
            # 🔥 3. Edge noise: 边缘噪声 (全 GPU)
            # ==========================================
            #t0 = time.time()
            if hasattr(self.cfg.depth, 'edge_noise_prob') and self.cfg.depth.edge_noise_prob > 0:
                if self.env_has_edge_noise[env_id]:
                    # 梯度计算
                    grad_x = torch.zeros_like(depth_image)
                    grad_x[:, 1:-1] = (depth_image[:, 2:] - depth_image[:, :-2]) / 2
                    grad_x[:, 0] = depth_image[:, 1] - depth_image[:, 0]
                    grad_x[:, -1] = depth_image[:, -1] - depth_image[:, -2]
                    
                    grad_y = torch.zeros_like(depth_image)
                    grad_y[1:-1, :] = (depth_image[2:, :] - depth_image[:-2, :]) / 2
                    grad_y[0, :] = depth_image[1, :] - depth_image[0, :]
                    grad_y[-1, :] = depth_image[-1, :] - depth_image[-2, :]
                    
                    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
                    edge_mask = gradient_magnitude > self.cfg.depth.edge_gradient_threshold
                    
                    kernel_size = self.cfg.depth.edge_dilation_kernel_size
                    edge_4d = edge_mask.float().unsqueeze(0).unsqueeze(0)
                    edge_dilated = torch.nn.functional.max_pool2d(
                        edge_4d, 
                        kernel_size=kernel_size, 
                        stride=1, 
                        padding=kernel_size//2
                    ).squeeze().bool()
                    
                    random_mask = torch.rand_like(depth_image) < self.cfg.depth.edge_noise_prob
                    edge_noise_mask = edge_dilated & random_mask
                    depth_image[edge_noise_mask] = -self.cfg.depth.far_clip
            #timings['3_edge_noise'] = (time.time() - t0) * 1000
            
            # # ==========================================
            # # 🔥 4. Holes: 柏林噪声 (全 GPU - 实时生成)
            # # ==========================================
            # t0 = time.time()
            # if hasattr(self.cfg.depth, 'perlin_noise_threshold'):
            #     if self.env_has_perlin_noise[env_id]:
            #         # 实时生成
            #         perlin_tensor = self.perlin_noise_generator.generate(
            #             depth_image.shape,
            #             time_offset=self.perlin_time_offsets[env_id].item()
            #         )
                    
            #         # 更新时间偏移
            #         self.perlin_time_offsets[env_id] += self.cfg.depth.perlin_noise_evolution_speed
                    
            #         # 应用阈值
            #         hole_mask = perlin_tensor > self.cfg.depth.perlin_noise_threshold
            #         depth_image[hole_mask] = -self.cfg.depth.far_clip
            # timings['4_perlin_noise'] = (time.time() - t0) * 1000

            # ==========================================
            # 🔥 4. Holes: 块状随机空洞 (替代柏林噪声)
            # ==========================================
            #t0 = time.time()
            if hasattr(self.cfg.depth, 'hole_noise_prob'):
                if self.env_has_hole_noise[env_id]:  # 🔥 使用新的标志位
                    # 参数
                    hole_prob = getattr(self.cfg.depth, 'hole_noise_prob', 0.15)
                    hole_size = getattr(self.cfg.depth, 'hole_block_size', 8)
                    
                    h, w = depth_image.shape
                    
                    # 生成稀疏的随机种子点
                    sparse_h, sparse_w = h // hole_size, w // hole_size
                    sparse_mask = torch.rand(sparse_h, sparse_w, device=self.device) < hole_prob
                    
                    # 上采样到原始分辨率
                    hole_mask = torch.nn.functional.interpolate(
                        sparse_mask.float().unsqueeze(0).unsqueeze(0),
                        size=(h, w),
                        mode='nearest'
                    ).squeeze().bool()
                    
                    depth_image[hole_mask] = -self.cfg.depth.far_clip
            #timings['4_hole_noise'] = (time.time() - t0) * 1000

            # ==========================================
            # 🔥 5. Blind spot: 去除左侧列 (全 GPU)
            # ==========================================
            #t0 = time.time()
            # if hasattr(self.cfg.depth, 'blind_spot_left_columns') and self.cfg.depth.blind_spot_left_columns > 0:
            #     depth_image[:, :self.cfg.depth.blind_spot_left_columns] = -self.cfg.depth.far_clip
            #timings['5_blind_spot'] = (time.time() - t0) * 1000
            
            # ==========================================
            # 原有噪声: dropout, salt_pepper (全 GPU)
            # ==========================================
            #t0 = time.time()
            if hasattr(self.cfg.depth, 'dropout_prob') and self.cfg.depth.dropout_prob > 0:
                dropout_mask = torch.rand_like(depth_image) < self.cfg.depth.dropout_prob
                depth_image[dropout_mask] = -self.cfg.depth.far_clip
            
            if hasattr(self.cfg.depth, 'salt_pepper_prob') and self.cfg.depth.salt_pepper_prob > 0:
                salt_mask = torch.rand_like(depth_image) < (self.cfg.depth.salt_pepper_prob / 2)
                depth_image[salt_mask] = -self.cfg.depth.far_clip
                
                pepper_mask = torch.rand_like(depth_image) < (self.cfg.depth.salt_pepper_prob / 2)
                depth_image[pepper_mask] = -self.cfg.depth.near_clip
            #timings['6_dropout_salt_pepper'] = (time.time() - t0) * 1000
            
            # ==========================================
            # 🔥 7. Gaussian Blur: 平滑整个图像 (全 GPU)
            # ==========================================
            #t0 = time.time()
            if getattr(self.cfg.depth, 'apply_gaussian_blur', False):
                kernel_size = getattr(self.cfg.depth, 'gaussian_blur_kernel_size', 5)
                sigma = getattr(self.cfg.depth, 'gaussian_blur_sigma', 1.0)
                
                if not hasattr(self, '_gaussian_kernel') or self._gaussian_kernel_size != kernel_size:
                    x = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
                    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
                    gauss_1d = gauss_1d / gauss_1d.sum()
                    
                    gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
                    self._gaussian_kernel = gauss_2d.view(1, 1, kernel_size, kernel_size)
                    self._gaussian_kernel_size = kernel_size
                
                depth_4d = depth_image.unsqueeze(0).unsqueeze(0)
                depth_blurred = torch.nn.functional.conv2d(
                    depth_4d, 
                    self._gaussian_kernel, 
                    padding=kernel_size//2
                ).squeeze()
                depth_image = depth_blurred
            #timings['7_gaussian_blur'] = (time.time() - t0) * 1000
        
        # Clip 到有效范围
        #t0 = time.time()
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        #timings['clip_range'] = (time.time() - t0) * 1000
        
        # Resize
        #t0 = time.time()
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        #timings['resize'] = (time.time() - t0) * 1000
        
        # 归一化
        # t0 = time.time()
        depth_image = self.normalize_depth_image(depth_image)
        # timings['normalize'] = (time.time() - t0) * 1000
        
        # # 总耗时
        # timings['TOTAL'] = (time.time() - total_start) * 1000
        
        # # 🔥 每 100 次打印一次统计信息 (避免刷屏)
        # if not hasattr(self, '_timing_counter'):
        #     self._timing_counter = 0
        #     self._timing_accumulator = {k: 0.0 for k in timings.keys()}
        
        # # 累积时间
        # for k, v in timings.items():
        #     self._timing_accumulator[k] += v
        
        # self._timing_counter += 1
        
        # # 每 100 次打印平均值
        # if self._timing_counter % 100 == 0:
        #     print("\n" + "="*80)
        #     print(f"📊 深度图像处理性能统计 (平均值, 基于 {self._timing_counter} 次调用)")
        #     print("="*80)
            
        #     # 按顺序打印
        #     keys_order = ['crop', 'uniform_noise', '1_clip_near', '2_gaussian_noise', 
        #                 '3_edge_noise', '4_perlin_noise', '5_blind_spot', 
        #                 '6_dropout_salt_pepper', '7_gaussian_blur', 
        #                 'clip_range', 'resize', 'normalize', 'TOTAL']
            
        #     for key in keys_order:
        #         if key in self._timing_accumulator:
        #             avg_time = self._timing_accumulator[key] / self._timing_counter
        #             if key == 'TOTAL':
        #                 print("-"*80)
        #                 print(f"{'⏱️  ' + key:.<50} {avg_time:>8.3f} ms")
        #             else:
        #                 # 计算占比
        #                 total_avg = self._timing_accumulator['TOTAL'] / self._timing_counter
        #                 percentage = (avg_time / total_avg * 100) if total_avg > 0 else 0
        #                 print(f"  {key:.<50} {avg_time:>8.3f} ms  ({percentage:>5.1f}%)")
            
        #     print("="*80 + "\n")
            
        #     # 重置计数器
        #     self._timing_counter = 0
        #     self._timing_accumulator = {k: 0.0 for k in timings.keys()}
    
        return depth_image

    # def process_depth_image(self, depth_image, env_id):
    #     """处理深度图像 (全 GPU 优化版本)"""
    #     # 裁剪图像
    #     depth_image = self.crop_depth_image(depth_image)
        
    #     # 添加均匀噪声
    #     depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1, device=self.device)-0.5)[0]
        
    #     if getattr(self.cfg.depth, 'enable_noise', True):
            
    #         # ==========================================
    #         # 🔥 1. Clip: 近距离设为无穷 (全 GPU)
    #         # ==========================================
    #         distance = torch.abs(depth_image)
    #         near_mask = distance < self.cfg.depth.clip_near_distance
    #         depth_image[near_mask] = -self.cfg.depth.far_clip
            
    #         # ==========================================
    #         # 🔥 2. Gaussian noise: 高斯噪声 (全 GPU)
    #         # ==========================================
    #         if hasattr(self.cfg.depth, 'gaussian_noise_std') and self.cfg.depth.gaussian_noise_std > 0:
    #             if self.env_has_gaussian_noise[env_id]:
    #                 distance = torch.abs(depth_image)
    #                 valid_mask = distance <= 3.0
                    
    #                 base_std = self.cfg.depth.gaussian_noise_std
    #                 distance_factor = getattr(self.cfg.depth, 'gaussian_noise_distance_factor', 0.5)
    #                 adaptive_std = base_std * (1.0 + distance_factor * distance)
                    
    #                 gaussian_noise = torch.randn_like(depth_image) * adaptive_std
    #                 gaussian_noise[~valid_mask] = 0.0
    #                 depth_image += gaussian_noise
            
    #         # ==========================================
    #         # 🔥 3. Edge noise: 边缘噪声 (全 GPU - Sobel)
    #         # ==========================================
    #         if hasattr(self.cfg.depth, 'edge_noise_prob') and self.cfg.depth.edge_noise_prob > 0:
    #             if self.env_has_edge_noise[env_id]:
    #                 # 🔥 使用 PyTorch 实现 np.gradient() 的效果
    #                 # np.gradient() 计算的是相邻像素的差分
                    
    #                 # X 方向梯度 (axis=1, 列方向)
    #                 # 中间: (depth[i, j+1] - depth[i, j-1]) / 2
    #                 # 边界: 前向或后向差分
    #                 grad_x = torch.zeros_like(depth_image)
    #                 grad_x[:, 1:-1] = (depth_image[:, 2:] - depth_image[:, :-2]) / 2  # 中间列
    #                 grad_x[:, 0] = depth_image[:, 1] - depth_image[:, 0]  # 左边界
    #                 grad_x[:, -1] = depth_image[:, -1] - depth_image[:, -2]  # 右边界
                    
    #                 # Y 方向梯度 (axis=0, 行方向)
    #                 grad_y = torch.zeros_like(depth_image)
    #                 grad_y[1:-1, :] = (depth_image[2:, :] - depth_image[:-2, :]) / 2  # 中间行
    #                 grad_y[0, :] = depth_image[1, :] - depth_image[0, :]  # 上边界
    #                 grad_y[-1, :] = depth_image[-1, :] - depth_image[-2, :]  # 下边界
                    
    #                 # 计算梯度幅值
    #                 gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
                    
    #                 # 识别边缘
    #                 edge_mask = gradient_magnitude > self.cfg.depth.edge_gradient_threshold
                    
    #                 # 🔥 使用 max_pool2d 实现膨胀 (替代 cv2.dilate)
    #                 if self.cfg.depth.edge_dilation_kernel_size > 0:
    #                     kernel_size = self.cfg.depth.edge_dilation_kernel_size
    #                     edge_4d = edge_mask.float().unsqueeze(0).unsqueeze(0)
    #                     edge_dilated = torch.nn.functional.max_pool2d(
    #                         edge_4d, 
    #                         kernel_size=kernel_size, 
    #                         stride=1, 
    #                         padding=kernel_size//2
    #                     ).squeeze().bool()
    #                 else:
    #                     edge_dilated = edge_mask
                    
    #                 # 在边缘处随机设为无穷
    #                 random_mask = torch.rand_like(depth_image) < self.cfg.depth.edge_noise_prob
    #                 edge_noise_mask = edge_dilated & random_mask
    #                 depth_image[edge_noise_mask] = -self.cfg.depth.far_clip
            
    #         # ==========================================
    #         # 🔥 4. Holes: 柏林噪声 (全 GPU - 实时生成)
    #         # ==========================================
    #         if hasattr(self.cfg.depth, 'perlin_noise_threshold'):
    #             if self.env_has_perlin_noise[env_id]:
    #                 # 🔥 实时生成柏林噪声 (全 GPU,无 CPU 传输)
    #                 perlin_tensor = self.perlin_noise_generator.generate(
    #                     depth_image.shape,
    #                     time_offset=self.perlin_time_offsets[env_id].item()
    #                 )
                    
    #                 # 更新时间偏移
    #                 self.perlin_time_offsets[env_id] += self.cfg.depth.perlin_noise_evolution_speed
                    
    #                 # 应用阈值生成空洞
    #                 hole_mask = perlin_tensor > self.cfg.depth.perlin_noise_threshold
    #                 depth_image[hole_mask] = -self.cfg.depth.far_clip

    #         # ==========================================
    #         # 🔥 5. Blind spot: 去除左侧列 (全 GPU)
    #         # ==========================================
    #         if hasattr(self.cfg.depth, 'blind_spot_left_columns') and self.cfg.depth.blind_spot_left_columns > 0:
    #             depth_image[:, :self.cfg.depth.blind_spot_left_columns] = -self.cfg.depth.far_clip
            
    #         # ==========================================
    #         # 原有噪声: dropout, salt_pepper (全 GPU)
    #         # ==========================================
    #         if hasattr(self.cfg.depth, 'dropout_prob') and self.cfg.depth.dropout_prob > 0:
    #             dropout_mask = torch.rand_like(depth_image) < self.cfg.depth.dropout_prob
    #             depth_image[dropout_mask] = -self.cfg.depth.far_clip
            
    #         if hasattr(self.cfg.depth, 'salt_pepper_prob') and self.cfg.depth.salt_pepper_prob > 0:
    #             salt_mask = torch.rand_like(depth_image) < (self.cfg.depth.salt_pepper_prob / 2)
    #             depth_image[salt_mask] = -self.cfg.depth.far_clip
                
    #             pepper_mask = torch.rand_like(depth_image) < (self.cfg.depth.salt_pepper_prob / 2)
    #             depth_image[pepper_mask] = -self.cfg.depth.near_clip
            
    #         # ==========================================
    #         # 🔥 6. Gaussian Blur: 平滑整个图像 (全 GPU)
    #         # ==========================================
    #         if getattr(self.cfg.depth, 'apply_gaussian_blur', False):
    #             kernel_size = getattr(self.cfg.depth, 'gaussian_blur_kernel_size', 5)
    #             sigma = getattr(self.cfg.depth, 'gaussian_blur_sigma', 1.0)
                
    #             # 🔥 使用预计算的高斯核 (避免重复计算)
    #             if not hasattr(self, '_gaussian_kernel') or self._gaussian_kernel_size != kernel_size:
    #                 # 生成一维高斯核
    #                 x = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
    #                 gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    #                 gauss_1d = gauss_1d / gauss_1d.sum()
                    
    #                 # 二维高斯核 (外积)
    #                 gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
    #                 self._gaussian_kernel = gauss_2d.view(1, 1, kernel_size, kernel_size)
    #                 self._gaussian_kernel_size = kernel_size
                
    #             # 应用高斯模糊 (GPU 卷积)
    #             depth_4d = depth_image.unsqueeze(0).unsqueeze(0)
    #             depth_blurred = torch.nn.functional.conv2d(
    #                 depth_4d, 
    #                 self._gaussian_kernel, 
    #                 padding=kernel_size//2
    #             ).squeeze()
    #             depth_image = depth_blurred
        
    #     # Clip 到有效范围
    #     depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        
    #     # Resize (双三次插值,GPU 加速)
    #     depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        
    #     # 归一化
    #     depth_image = self.normalize_depth_image(depth_image)
    #     return depth_image
    
    # def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        # depth_image = self.crop_depth_image(depth_image)

        # depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0] 原有的均匀噪声

         
        # depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1, device=self.device)-0.5)[0]
        
        # if getattr(self.cfg.depth, 'enable_noise', True):  # 默认启用噪声
        #     # 添加随机像素点缺失(设为无穷大)
        #     if hasattr(self.cfg.depth, 'dropout_prob') and self.cfg.depth.dropout_prob > 0:
        #         # 生成随机mask,决定哪些像素点缺失
        #         dropout_mask = torch.rand_like(depth_image) < self.cfg.depth.dropout_prob
        #         depth_image[dropout_mask] = float('-inf')  # 设为负无穷(因为深度值是负的)
            
        #     # 添加椒盐噪声
        #     if hasattr(self.cfg.depth, 'salt_pepper_prob') and self.cfg.depth.salt_pepper_prob > 0:
        #         # 盐噪声(设为远端clip值)
        #         salt_mask = torch.rand_like(depth_image) < (self.cfg.depth.salt_pepper_prob / 2)
        #         depth_image[salt_mask] = -self.cfg.depth.far_clip
                
        #         # 椒噪声(设为近端clip值)
        #         pepper_mask = torch.rand_like(depth_image) < (self.cfg.depth.salt_pepper_prob / 2)
        #         depth_image[pepper_mask] = -self.cfg.depth.near_clip
            
        #     if hasattr(self.cfg.depth, 'gaussian_noise_std') and self.cfg.depth.gaussian_noise_std > 0:
        #         # 深度值是负数，取绝对值得到实际距离
        #         distance = torch.abs(depth_image)  # [H, W]

        #         # 🔥 只对 3m 以内的距离施加噪声
        #         valid_mask = distance <= 3.0  # [H, W] 布尔掩码
                
        #         # 🔥 噪声模型：σ(d) = base_std * (1 + distance_factor * d)
        #         # 基础噪声：2% 误差
        #         base_std = self.cfg.depth.gaussian_noise_std
                
        #         # 距离系数：控制噪声随距离增长的速度
        #         # distance_factor = 0.5 表示每增加 1m，噪声增加 50%
        #         distance_factor = getattr(self.cfg.depth, 'gaussian_noise_distance_factor', 0.5)
                
        #         # 计算每个像素的自适应标准差
        #         # 例如: d=0.5m -> σ=0.02*(1+0.5*0.5)=0.025 (2.5%误差)
        #         #      d=1.0m -> σ=0.02*(1+0.5*1.0)=0.03  (3%误差)
        #         #      d=2.0m -> σ=0.02*(1+0.5*2.0)=0.04  (4%误差)
        #         adaptive_std = base_std * (1.0 + distance_factor * distance)
                
        #         # 生成自适应高斯噪声
        #         gaussian_noise = torch.randn_like(depth_image) * adaptive_std
                
        #         # 🔥 只在有效区域添加噪声
        #         gaussian_noise[~valid_mask] = 0.0  # 3m 外的噪声设为 0

        #         # 添加噪声（注意深度是负值，所以直接加）
        #         depth_image += gaussian_noise


    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        self.gym.step_graphics(self.sim) # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                self.envs[i], 
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)
            rgb_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                              self.envs[i], 
                                                              self.cam_handles[i],
                                                              gymapi.IMAGE_COLOR)
            rgb_image = gymtorch.wrap_tensor(rgb_image_)
            rgb_image = self.resize_transform(rgb_image.permute(2, 0, 1)).permute(1, 2, 0)
            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
                self.rgb_buffer[i] = torch.stack([rgb_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)
                self.rgb_buffer[i] = torch.cat([self.rgb_buffer[i, 1:], rgb_image.to(self.device).unsqueeze(0)], dim=0)

        self.gym.end_access_image_tensors(self.sim)

    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0

        self.reached_goal_ids = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1) < self.cfg.env.next_goal_threshold
        self.reach_goal_timer[self.reached_goal_ids] += 1

        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        
        # self._update_jump_schedule()
        self._update_goals()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.update_depth_buffer()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            if not self.cfg.depth.use_camera:
                self._draw_height_samples()
                self._draw_goals()
                self._draw_feet()
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

                scale_factor = 10
                depth_image = self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5
                height, width = depth_image.shape[:2]
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                print(new_height, new_width)

                resized_depth_image = cv2.resize(depth_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                cv2.imshow(window_name, resized_depth_image)
                # cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)

                rgb_img = self.rgb_buffer[self.lookat_id, -1].cpu().numpy()
                # 2. 检查并转换通道 (Isaac Gym RGB -> OpenCV BGR)
                if rgb_img.shape[-1] == 4:
                    # 如果是 RGBA (4通道)，转为 BGR
                    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGBA2BGR)
                # 3. 检查数据类型 (防止全白或全黑)
                # 如果数据是 float 且最大值大于 1，说明是 0-255 的 float，需要转 uint8  (确实是0-255的float)
                print("type:", rgb_img.dtype, "max:", rgb_img.max(), "min:", rgb_img.min())
                if rgb_img.dtype != np.uint8:
                    if rgb_img.max() > 1.1:
                        rgb_img = rgb_img.astype(np.uint8)
                    else:
                        # 如果是 0-1 的 float，OpenCV 可以显示，但转为 0-255 更通用
                        rgb_img = (rgb_img * 255).astype(np.uint8)
                # (可选) 如果你想让 RGB 和 深度图一样大，可以加 resize
                rgb_img = cv2.resize(rgb_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                cv2.imshow("RGB Image", rgb_img)
                cv2.waitKey(1)

    def reindex_feet(self, vec):
        return vec[:, [1, 0, 3, 2]]

    def reindex(self, vec):
        return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        roll_cutoff = torch.abs(self.roll) > 1.5
        pitch_cutoff = torch.abs(self.pitch) > 1.5
        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals
        # height_cutoff = self.root_states[:, 2] < -0.25

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.time_out_buf |= reach_goal_cutoff

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        # self.reset_buf |= height_cutoff

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ 
        Computes observations
        """
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        if self.global_counter % 5 == 0:
            self.delta_yaw = self.target_yaw - self.yaw
            self.delta_next_yaw = self.next_target_yaw - self.yaw
        obs_buf = torch.cat((#skill_vector, 
                            self.base_ang_vel  * self.obs_scales.ang_vel,   #[1,3]
                            imu_obs,    #[1,2]
                            0*self.delta_yaw[:, None], 
                            self.delta_yaw[:, None],
                            self.delta_next_yaw[:, None],
                            0*self.commands[:, 0:2], 
                            self.commands[:, 0:1],  #[1,1]
                            (self.env_class != 17).float()[:, None], 
                            (self.env_class == 17).float()[:, None],
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            self.reindex(self.action_history_buf[:, -1]),
                            self.reindex_feet(self.contact_filt.float()-0.5),
                            ),dim=-1)
        priv_explicit = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                   0 * self.base_lin_vel,
                                   0 * self.base_lin_vel), dim=-1)
        priv_latent = torch.cat((
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0] - 1, 
            self.motor_strength[1] - 1
        ), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
            self.obs_buf = torch.cat([obs_buf, heights, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([obs_buf, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        obs_buf[:, 6:8] = 0  # mask yaw in proprioceptive history
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )
        
        
    def get_noisy_measurement(self, x, scale):
        if self.cfg.noise.add_noise:
            x = x + (2.0 * torch.rand_like(x) - 1) * scale * self.cfg.noise.noise_level
        return x

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id  # required in headless mode
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        start = time()
        print("*"*80)
        print("Start creating ground...")
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*"*80)
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # No need to use tensors as only called upon env creation
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros((1, ))
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        mass_params = np.concatenate([rand_mass, rand_com])
        return props, mass_params
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0)
        self._resample_commands(env_ids.nonzero(as_tuple=False).flatten())

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.8*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
            self.commands[:, 2] *= torch.abs(self.commands[:, 2]) > self.cfg.commands.ang_vel_clip
        
        if self.cfg.terrain.measure_heights:
            if self.global_counter % self.cfg.depth.update_interval == 0:
                self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        
    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(1, (self.cur_goal_idx[:, None, None]+future).expand(-1, -1, self.env_goals.shape[-1])).squeeze(1)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 2] *= torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_clip

        # set small commands to zero
        self.commands[env_ids, :2] *= torch.abs(self.commands[env_ids, 0:1]) > self.cfg.commands.lin_vel_clip

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            if not self.cfg.domain_rand.randomize_motor:  # TODO add strength to gain directly
                torques = self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.d_gains*self.dof_vel
            else:
                torques = self.motor_strength[0] * self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.motor_strength[1] * self.d_gains*self.dof_vel
                
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(0., 0.9, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.env.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = self.cfg.env.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                if self.cfg.env.randomize_start_pitch:
                    rand_pitch = self.cfg.env.rand_pitch_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                else:
                    rand_pitch = torch.zeros(len(env_ids), device=self.device)
                quat = quat_from_euler_xyz(0*rand_yaw, rand_pitch, rand_yaw) 
                self.root_states[env_ids, 3:7] = quat[:, :]  
            if self.cfg.env.randomize_start_y:
                self.root_states[env_ids, 1] += self.cfg.env.rand_y_range * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
            
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        
        dis_to_origin = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        threshold = self.commands[env_ids, 0] * self.cfg.env.episode_length_s
        move_up =dis_to_origin > 0.8*threshold
        move_down = dis_to_origin < 0.4*threshold

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
            
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6) # for feet only, see create_env()
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        if self.cfg.env.history_encoding:
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_dofs, device=self.device, dtype=torch.float)
        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 4, device=self.device, dtype=torch.float)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self._resample_commands(torch.arange(self.num_envs, device=self.device, requires_grad=False))
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_dof_pos_all[:] = self.default_dof_pos[0]

        self.height_update_interval = 1
        if hasattr(self.cfg.env, "height_update_dt"):
            self.height_update_interval = int(self.cfg.env.height_update_dt / (self.cfg.sim.dt * self.cfg.control.decimation))

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,  
                                            self.cfg.depth.buffer_len, 
                                            self.cfg.depth.resized[1], 
                                            self.cfg.depth.resized[0]).to(self.device)
            self.rgb_buffer = torch.zeros(self.num_envs,  
                                          self.cfg.depth.buffer_len, 
                                          self.cfg.depth.resized[1], 
                                          self.cfg.depth.resized[0],
                                          4,).to(self.device)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.cfg.terrain.horizontal_scale
        hf_params.row_scale = self.cfg.terrain.horizontal_scale
        hf_params.vertical_scale = self.cfg.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border
        hf_params.transform.p.y = -self.terrain.border
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.flatten(order='C'), hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    # def _create_trimesh(self):
    #     """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
    #         Very slow when horizontal_scale is small
    #     """
    #     tm_params = gymapi.TriangleMeshParams()
    #     tm_params.nb_vertices = self.terrain.vertices.shape[0]
    #     tm_params.nb_triangles = self.terrain.triangles.shape[0]

    #     tm_params.transform.p.x = -self.terrain.cfg.border_size 
    #     tm_params.transform.p.y = -self.terrain.cfg.border_size
    #     tm_params.transform.p.z = 0.0
    #     tm_params.static_friction = self.cfg.terrain.static_friction
    #     tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
    #     tm_params.restitution = self.cfg.terrain.restitution
    #     print("Adding trimesh to simulation...")
    #     self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
    #     print("Trimesh added")
    #     self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
    #     self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            MODIFIED: Also adds any additional trimeshes found in the terrain object (e.g., for hollow stairs).
        """
        # --- PART 1: ADD THE MAIN TERRAIN (FROM HEIGHTFIELD) ---
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        
        print("Adding main trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
        print("Main trimesh added.")

        # --- MODIFIED PART START: ADD EXTRA MESHES LIKE STAIRS ---
        # 检查 terrain 对象是否有 trimeshes 列表
        if hasattr(self.terrain, 'trimeshes') and self.terrain.trimeshes:
            print(f"Found {len(self.terrain.trimeshes)} additional meshes to add (e.g., hollow stairs)...")
            
            # 为这些额外的网格创建一个新的参数对象
            step_tm_params = gymapi.TriangleMeshParams()
            step_tm_params.static_friction = self.cfg.terrain.static_friction
            step_tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
            step_tm_params.restitution = self.cfg.terrain.restitution
            
            # 将每个台阶网格添加到仿真中
            for i, (vertices, triangles) in enumerate(self.terrain.trimeshes):
                step_tm_params.nb_vertices = vertices.shape[0]
                step_tm_params.nb_triangles = triangles.shape[0]
                
                # 重要：因为顶点坐标已经是世界坐标，所以这里的变换位置是(0,0,0)
                step_tm_params.transform.p.x = 0.0
                step_tm_params.transform.p.y = 0.0
                step_tm_params.transform.p.z = 0.0

                if i == 0:
                    # 只打印第一个台阶的信息，避免刷屏
                    print(f">>> DEBUG: Adding first step mesh. It has {vertices.shape[0]} vertices.")
                    # 打印第一个顶点的坐标，看看Z值是不是大于0
                    print(f"    First vertex coordinate: {vertices[0]}")
                    # 打印中心点坐标，看看它的大概位置
                    center_coord = np.mean(vertices, axis=0)
                    print(f"    Approximate center: {center_coord}")

                # 将一个台阶添加到仿真中
                self.gym.add_triangle_mesh(self.sim, vertices.flatten(order='C'), triangles.flatten(order='C'), step_tm_params)
            
            print("Additional meshes added.")

        # --- MODIFIED PART END ---

        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[0]
            camera_props.height = self.cfg.depth.original[1]
            camera_props.enable_tensors = True

            #camera_horizontal_fov = self.cfg.depth.horizontal_fov 
            # camera_horizontal_fov = 87.5
            #  Horizontal FOV 域随机化
            # 80% 概率使用标准值, 20% 概率从候选值中随机选择
            if hasattr(self.cfg.depth, 'horizontal_fov_range') and hasattr(self.cfg.depth, 'horizontal_fov'):
                if np.random.random() < 0.8:
                    # 80% 概率: 使用标准 FOV
                    camera_horizontal_fov = self.cfg.depth.horizontal_fov
                else:
                    # 20% 概率: 从候选值中随机选择
                    fov_candidates = self.cfg.depth.horizontal_fov_range
                    camera_horizontal_fov = np.random.choice(fov_candidates)
            else:
                # 后备方案: 使用固定值
                camera_horizontal_fov = self.cfg.depth.horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)
            print(self.cam_handles)
            local_transform = gymapi.Transform()
            
            camera_position_center = np.copy(config.position)
            camera_position = np.random.uniform(camera_position_center-config.position_rand, camera_position_center+config.position_rand)
            camera_angle = np.random.uniform(config.angle[0], config.angle[1])

            camera_z_angle = np.random.uniform(config.z_angle[0], config.z_angle[1])
            camera_x_angle = np.random.uniform(config.x_angle[0], config.x_angle[1])

            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(np.radians(camera_x_angle), np.radians(camera_angle), np.radians(camera_z_angle))
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
            
            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]


        for s in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            if self.cfg.env.randomize_start_pos:
                pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            if self.cfg.env.randomize_start_yaw:
                rand_yaw_quat = gymapi.Quat.from_euler_zyx(0., 0., self.cfg.env.rand_yaw_range*np.random.uniform(-1, 1))
                start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)
            
            self.attach_camera(i, env_handle, anymal_handle)

            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)
        thigh_names = ["FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint"]
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(thigh_names):
            self.thigh_indices[i] = self.dof_names.index(name)
        calf_names = ["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"]
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(calf_names):
            self.calf_indices[i] = self.dof_names.index(name)
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            
            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]

            self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
            self.env_goals = torch.zeros(self.num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3, device=self.device, requires_grad=False)
            self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
            temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
            last_col = temp[:, -1].unsqueeze(1)
            self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
            self.cur_goals = self._gather_cur_goals()
            self.next_goals = self._gather_cur_goals(future=1)

        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        reward_norm_factor = 1#np.sum(list(self.reward_scales.values()))
        for rew in self.reward_scales:
            self.reward_scales[rew] = self.reward_scales[rew] / reward_norm_factor
        if self.cfg.commands.curriculum:
            self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        else:
            self.command_ranges = class_to_dict(self.cfg.commands.max_ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_height_samples(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        i = self.lookat_id
        base_pos = (self.root_states[i, :3]).cpu().numpy()
        heights = self.measured_heights[i].cpu().numpy()
        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    
    def _draw_goals(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(1, 0, 0))
        sphere_geom_cur = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0, 0, 1))
        sphere_geom_reached = gymutil.WireframeSphereGeometry(self.cfg.env.next_goal_threshold, 32, 32, None, color=(0, 1, 0))
        goals = self.terrain_goals[self.terrain_levels[self.lookat_id], self.terrain_types[self.lookat_id]].cpu().numpy()
        for i, goal in enumerate(goals):
            goal_xy = goal[:2] + self.terrain.cfg.border_size
            pts = (goal_xy/self.terrain.cfg.horizontal_scale).astype(int)
            goal_z = self.height_samples[pts[0], pts[1]].cpu().item() * self.terrain.cfg.vertical_scale
            pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)
            if i == self.cur_goal_idx[self.lookat_id].cpu().item():
                gymutil.draw_lines(sphere_geom_cur, self.gym, self.viewer, self.envs[self.lookat_id], pose)
                if self.reached_goal_ids[self.lookat_id]:
                    gymutil.draw_lines(sphere_geom_reached, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            else:
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
        if not self.cfg.depth.use_camera:
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
            pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()
            for i in range(5):
                norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.1*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0.5))
            for i in range(5):
                norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.2*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
    def _draw_feet(self):
        if hasattr(self, 'feet_at_edge'):
            non_edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0))
            edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0, 0))

            feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
            for i in range(4):
                pose = gymapi.Transform(gymapi.Vec3(feet_pos[self.lookat_id, i, 0], feet_pos[self.lookat_id, i, 1], feet_pos[self.lookat_id, i, 2]), r=None)
                if self.feet_at_edge[self.lookat_id, i]:
                    gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[i], pose)
                else:
                    gymutil.draw_lines(non_edge_geom, self.gym, self.viewer, self.envs[i], pose)
    
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]
        return points

    def get_foot_contacts(self):
        foot_contacts_bool = self.contact_forces[:, self.feet_indices, 2] > 10
        if self.cfg.env.include_foot_contacts:
            return foot_contacts_bool
        else:
            return torch.zeros_like(foot_contacts_bool).to(self.device)

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_heights_points(self, coords, env_ids=None):
        if env_ids:
            points = coords[env_ids]
        else:
            points = coords

        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    ################## parkour rewards ##################

    def _reward_tracking_goal_vel(self):
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        cur_vel = self.root_states[:, 7:9]
        rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
        return rew

    def _reward_tracking_yaw(self):
        rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))
        return rew
    
    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel[:, 2])
        rew[self.env_class != 17] *= 0.5
        return rew
    
    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
     
    def _reward_orientation(self):
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        rew[self.env_class != 17] = 0.
        return rew

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)
    
    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
    
        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew
