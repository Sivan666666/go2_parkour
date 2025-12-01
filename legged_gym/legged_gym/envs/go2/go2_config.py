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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class init_state_slope( LeggedRobotCfg.init_state ):
        pos = [0.56, 0.0, 0.24] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.03,   # [rad]
            'RL_hip_joint': 0.03,   # [rad]
            'FR_hip_joint': -0.03,  # [rad]
            'RR_hip_joint': -0.03,   # [rad]

            'FL_thigh_joint': 1.0,     # [rad]
            'RL_thigh_joint': 1.9,   # [rad]1.8
            'FR_thigh_joint': 1.0,     # [rad]
            'RR_thigh_joint': 1.9,   # [rad]

            'FL_calf_joint': -2.2,   # [rad]
            'RL_calf_joint': -0.9,    # [rad]
            'FR_calf_joint': -2.2,  # [rad]
            'RR_calf_joint': -0.9,    # [rad]

            'l_finger_joint': 0.0,    # [m]
            'r_finger_joint': 0.0,    # [m]
        }
        
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'joint': 40.}  # [N*m/rad]
        # damping = {'joint': 0.5}     # [N*m*s/rad]
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_new.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]#, "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.3
        # class scales( LeggedRobotCfg.rewards.scales ):
            # torques = -0.0002
            # dof_pos_limits = -10.0

    class depth( LeggedRobotCfg.depth ):
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        # # helpful doggy
        position = [0.3, 0, 0.147]  # front camera
        position_rand = 0.02  
        angle = [29-5, 29+5]  # positive pitch down  #27-5,27+5
        z_angle = [-2, 2]
        x_angle = [-2, 2]
        # our designed position
        # position = [0.3, 0, 0.188]  # front camera
        # position_rand = 0.01  
        # angle = [30-5, 30+5]  # positive pitch down  #27-5,27+5

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (106, 60)
        resized = (87, 58)
        # 🔥 Horizontal FOV 域随机化
        horizontal_fov = 87  # 基准值
        horizontal_fov_range = [86, 87, 88]   # 随机范围 86-88 度

        buffer_len = 2
        
        # Clip 参数
        near_clip = 0.15
        far_clip = 2

        # 噪声总开关
        enable_noise = True
        dis_noise = 0.0

        # 1. Clip: 近距离设为无穷
        clip_near_distance = 0.15  # 0.15m以内设为无穷大

        # 2. Edge noise: 边缘噪声
        edge_noise_enable_prob = 0.8  #  80%概率启用边缘噪声
        edge_noise_prob = 0.5  # 边缘处30%概率设为无穷
        edge_gradient_threshold = 0.3  # 深度梯度阈值(米)
        edge_dilation_kernel_size = 3  # 边缘膨胀核大小

        # 3. Holes: 柏林噪声模拟空洞
        perlin_noise_enable_prob = 0  #  80%概率启用柏林噪声
        perlin_noise_threshold = 0.8  # 大于此阈值的区域设为空洞
        perlin_noise_scale = 10.0  # 柏林噪声频率
        perlin_noise_octaves = 2  # 柏林噪声叠加层数
        perlin_noise_evolution_speed = 0.005  # 时间演化速度

        #  启用块状空洞 代替 柏林
        hole_noise_enable_prob = 0.2      # 20% 概率启用块状空洞
        hole_noise_prob = 0.05        # 5% 区域有空洞
        hole_block_size = 8           # 8×8 像素的块

        # 4. Blind spot: 去除左侧列
        blind_spot_left_columns = 0  # 去除左侧5列

        # 5. Gaussian noise: 高斯噪声
        gaussian_noise_enable_prob = 0.6  #  80%概率启用高斯噪声
        gaussian_noise_std = 0.04
        gaussian_noise_distance_factor = 0.5

        # 6. Gaussian Blur: 最终平滑 (新增)
        apply_gaussian_blur = True  # 是否应用高斯模糊
        gaussian_blur_kernel_size = 3  # 核大小(必须是奇数: 3, 5, 7...)
        gaussian_blur_sigma = 1.0  # 标准差(越大越模糊)

        # 原有噪声
        dropout_prob = 0.001
        salt_pepper_prob = 0.0


        scale = 1
        invert = True

class Go2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'

  
