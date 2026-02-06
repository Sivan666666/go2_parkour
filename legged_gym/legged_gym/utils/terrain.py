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
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from scipy import ndimage
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation
from legged_gym.utils import trimesh

class newSubTerrain(terrain_utils.SubTerrain):
    def __init__(self, name, width, length, vertical_scale, horizontal_scale, height, downsampled_scale):
        super().__init__(name, width, length, vertical_scale, horizontal_scale)
        self.heightsamples = np.zeros((self.width, self.length), dtype=np.int16)
        self.height = height
        # print(self.height)
        self.downsampled_scale = downsampled_scale
class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        # print(self.cfg.height)
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        # 打印地形分布（名称+累计比例）、行列数与 horizontal_scale
        terrain_names = [
            "smooth slope",
            "rough slope up",
            "rough slope down",
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
        ]
        cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(cfg.terrain_proportions)
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        print("=== Terrain setup ===")
        print(f"- grid rows x cols: {cfg.num_rows} x {cfg.num_cols}")
        print(f"- horizontal_scale: {cfg.horizontal_scale}")

        # 表格打印 terrain proportions（保留1位小数）
        headers = ("Index", "Name", "Proportion")
        rows = []
        for i, p in enumerate(cfg.terrain_proportions):
            name = terrain_names[i] if i < len(terrain_names) else f"type_{i}"
            rows.append((str(i), name, f"{p:.2f}"))

        # 计算列宽
        data = [headers] + rows
        col_w = [max(len(row[c]) for row in data) for c in range(3)]
        def hline():
            return "+" + "+".join("-"*(w+2) for w in col_w) + "+"

        print("- terrain proportions:")
        print(hline())
        print("| " + headers[0].ljust(col_w[0]) + " | " +
                    headers[1].ljust(col_w[1]) + " | " +
                    headers[2].ljust(col_w[2]) + " |")
        print(hline())
        for r in rows:
            print("| " + r[0].ljust(col_w[0]) + " | " +
                        r[1].ljust(col_w[1]) + " | " +
                        r[2].rjust(col_w[2]) + " |")
        print(hline())

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
        # self.env_slope_vec = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals, 3))
        self.num_goals = cfg.num_goals

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        self.heightsamples = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        self.trimeshes = []  # 用来收集所有子地形生成的独立网格

        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            if hasattr(cfg, "max_difficulty"):
                self.curiculum(random=True, max_difficulty=cfg.max_difficulty)
            else:
                self.curiculum(random=True)
            # self.randomized_terrain()   
        
        # self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            print("Converting heightmap to trimesh...")
            if cfg.hf2mesh_method == "grid":
                self.vertices, self.triangles, auto_edge_mask = convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                                self.cfg.horizontal_scale,
                                                                                                self.cfg.vertical_scale,
                                                                                                self.cfg.slope_treshold)
                # --- [新增/修改] 合并自动与手动掩码 ---
                self.x_edge_mask = auto_edge_mask
                if hasattr(self, 'x_edge_mask_manual'):
                    # 取并集：只要是自动检测到的边缘 OR 手动标记的边缘，都计入
                    self.x_edge_mask |= self.x_edge_mask_manual
                
                # 统一进行膨胀处理，使边缘变成一个可感知的区域 edge_width_thresh = 5cm
                half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
                structure = np.ones((half_edge_width*2+1, 1))
                self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)
                if self.cfg.simplify_grid:
                    mesh_simplifier = pyfqmr.Simplify()
                    mesh_simplifier.setMesh(self.vertices, self.triangles)
                    mesh_simplifier.simplify_mesh(target_count = int(0.05*self.triangles.shape[0]), aggressiveness=7, preserve_border=True, verbose=10)

                    self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                    self.vertices = self.vertices.astype(np.float32)
                    self.triangles = self.triangles.astype(np.uint32)
            else:
                assert cfg.hf2mesh_method == "fast", "Height field to mesh method must be grid or fast"
                self.vertices, self.triangles = convert_heightfield_to_trimesh_delatin(self.height_field_raw, self.cfg.horizontal_scale, self.cfg.vertical_scale, max_error=cfg.max_error)
            print("Created {} vertices".format(self.vertices.shape[0]))
            print("Created {} triangles".format(self.triangles.shape[0]))

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            # difficulty = np.random.choice([0.5, 0.75, 0.9])
            difficulty = np.random.uniform(-0.2, 1.2)
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self, random=False, max_difficulty=False):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / (self.cfg.num_rows-1)
                choice = j / self.cfg.num_cols + 0.001
                # print(f"Terrain choice={choice:.3f}, difficulty={difficulty:.3f}")
                if random:
                    if max_difficulty:
                        terrain = self.make_terrain(choice, np.random.uniform(0.99, 1))
                    else:
                        terrain = self.make_terrain(choice, np.random.uniform(0, 1))
                else:
                    terrain = self.make_terrain(choice, difficulty)

                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = newSubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.length_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale,
                              height=self.cfg.height,
                              downsampled_scale=self.cfg.downsampled_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def add_roughness(self, terrain, difficulty=1):
        max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)

    def make_terrain(self, choice, difficulty):
        terrain = newSubTerrain(   "terrain",
                                width=self.length_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale,
                                height=self.cfg.height,
                                downsampled_scale=self.cfg.downsampled_scale)
        slope = difficulty * 0.4
        step_height = 0.02 + 0.14 * difficulty
        discrete_obstacles_height = 0.03 + difficulty * 0.15
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            idx = 0
            if choice < self.proportions[0]/ 2:
                idx = 1
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            # self.add_roughness(terrain)
        elif choice < self.proportions[2]:
            idx = 2
            height = 0.1 + 0.1 * difficulty
            if choice < self.proportions[1]:
                idx = 3
                height *= -1
            # terrain_utils.pyramid_stairs_terrain(terrain, step_width=1., step_height=height, platform_size=3.)

            num_goals = 8
            num_steps = num_goals - 1
            #step_width = 0.2  # 20cm

            # step_width: difficulty=0时为0.4m，difficulty=1时为0.2m
            step_width = 0.4 - 0.2 * difficulty
            staircase_length = 10.0  # 总楼梯区长度
            birth_area_length = 3  # 由 birth_area_length_px = 60 * 0.01 得到
            # 计算顶部平台长度
            platform_size = staircase_length - birth_area_length - step_width * num_steps
            terrain_y_flat = 1.6
            stairs_terrain(terrain, step_height=height, platform_size=platform_size, staircase_length=staircase_length, num_goals=self.num_goals, birth_area_length=birth_area_length, step_width=step_width, terrain_y_flat=terrain_y_flat)
            self.add_roughness(terrain, difficulty)
        elif choice < self.proportions[4]:
            idx = 4
            step_height_first = 0.1 + 0.12 * difficulty
            step_height_others = 0.1 + 0.13 * difficulty

            num_goals = 8
            num_steps = num_goals - 1
            #step_width = 0.2  # 20cm

            # step_width: difficulty=0时为0.4m，difficulty=1时为0.2m
            step_width = 0.4 - 0.2 * difficulty
            staircase_length = 10.0  # 总楼梯区长度
            birth_area_length = 3  # 由 birth_area_length_px = 60 * 0.01 得到

            terrain_y_flat = 1.6
            # 计算顶部平台长度
            platform_size = staircase_length - birth_area_length - step_width * num_steps
            if platform_size < 0:
                raise ValueError("楼梯太多或太宽，platform_size为负，请调整参数！")

            if choice<self.proportions[3]:
                idx = 5 
                step_height_first *= -1
                step_height_others *= -1
            hollow_stairs_terrain(
                terrain,
                step_height_first=step_height_first,
                step_height_others=step_height_others,
                slope_treshold=self.cfg.slope_treshold,
                step_thickness=0.035,
                platform_size=platform_size,
                staircase_length=staircase_length,
                num_goals=self.num_goals,
                birth_area_length=birth_area_length,
                step_width=step_width,
                terrain_y_flat=terrain_y_flat,
                difficulty = difficulty,
                is_steep=True
            )
            self.add_roughness(terrain)
        elif choice < self.proportions[5]:
            idx = 6
            num_rectangles = 20
            rectangle_min_size = 0.5
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[6]:
            idx = 7
            stones_size = 1.5 - 1.2*difficulty
            # terrain_utils.stepping_stones_terrain(terrain, stone_size=stones_size, stone_distance=0.1, stone_distance_rand=0, max_height=0.04*difficulty, platform_size=2.)
            half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            stepping_stones_terrain(terrain, stone_size=1.5-0.2*difficulty, stone_distance=0.0+0.4*difficulty, max_height=0.2*difficulty, platform_size=1.2)
            self.add_roughness(terrain)
        elif choice < self.proportions[7]:
            idx = 8
            # gap_size = random.uniform(self.cfg.gap_size[0], self.cfg.gap_size[1])
            gap_parkour_terrain(terrain, difficulty, platform_size=4)
            self.add_roughness(terrain)
        elif choice < self.proportions[8]:
            # print("flat terrain")
            idx = 9
            flat_terrain(terrain)
            self.add_roughness(terrain)
            # pass
        elif choice < self.proportions[9]:
            idx = 10
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        elif choice < self.proportions[10]:
            idx = 11
            if self.cfg.all_vertical:
                half_slope_difficulty = 1.0
            else:
                difficulty *= 1.3
                if not self.cfg.no_flat:
                    difficulty -= 0.1
                if difficulty > 1:
                    half_slope_difficulty = 1.0
                elif difficulty < 0:
                    self.add_roughness(terrain)
                    terrain.slope_vector = np.array([1, 0., 0]).astype(np.float32)
                    return terrain
                else:
                    half_slope_difficulty = difficulty
            wall_width = 4 - half_slope_difficulty * 4
            # terrain_utils.wall_terrain(terrain, height=1, start2center=0.7)
            # terrain_utils.tanh_terrain(terrain, height=1.0, start2center=0.7)
            if self.cfg.flat_wall:
                half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            else:
                half_sloped_terrain(terrain, wall_width=wall_width, start2center=0.5, max_height=1.5)
            max_height = terrain.height_field_raw.max()
            top_mask = terrain.height_field_raw > max_height - 0.05
            self.add_roughness(terrain, difficulty=1)
            terrain.height_field_raw[top_mask] = max_height
        elif choice < self.proportions[11]:
            idx = 12
            # half platform terrain
            half_platform_terrain(terrain, max_height=0.1 + 0.4 * difficulty )
            self.add_roughness(terrain, difficulty=1)
        elif choice < self.proportions[13]:
            # step_height = 0.1 + 0.3 * difficulty
            step_height_first = 0.1 + 0.05 * difficulty
            step_height_others = 0.1 + 0.05 * difficulty
            idx = 13

            num_goals = 8
            num_steps = num_goals - 1
            #step_width = 0.2  # 20cm

            # step_width: difficulty=0时为0.4m，difficulty=1时为0.2m
            step_width = 0.4 - 0.2 * difficulty
            staircase_length = 10.0  # 总楼梯区长度
            birth_area_length = 3  # 由 birth_area_length_px = 60 * 0.01 得到

            terrain_y_flat = 1.6
            # 计算顶部平台长度
            platform_size = staircase_length - birth_area_length - step_width * num_steps
            if platform_size < 0:
                raise ValueError("楼梯太多或太宽，platform_size为负，请调整参数！")

            if choice<self.proportions[12]:
                idx = 14 
                step_height_first *= -1
                step_height_others *= -1
            hollow_stairs_terrain(
                terrain,
                step_height_first=step_height_first,
                step_height_others=step_height_others,
                slope_treshold=self.cfg.slope_treshold,
                step_thickness=0.035,
                platform_size=platform_size,
                staircase_length=staircase_length,
                num_goals=self.num_goals,
                birth_area_length=birth_area_length,
                step_width=step_width,
                terrain_y_flat=terrain_y_flat,
                difficulty = difficulty
            )
            self.add_roughness(terrain)
        elif choice < self.proportions[14]:
            x_range = [-0.1, 0.1+0.3*difficulty]  # offset to stone_len
            y_range = [0.2, 0.3+0.1*difficulty]
            stone_len = [0.9 - 0.3*difficulty, 1 - 0.2*difficulty]#2 * round((0.6) / 2.0, 1)
            incline_height = 0.25*difficulty
            last_incline_height = incline_height + 0.1 - 0.1*difficulty
            parkour_terrain(terrain,
                            num_stones=self.num_goals - 2,
                            x_range=x_range, 
                            y_range=y_range,
                            incline_height=incline_height,
                            stone_len=stone_len,
                            stone_width=1.0, 
                            last_incline_height=last_incline_height,
                            pad_height=0,
                            pit_depth=[0.2, 1])
            idx = 15
            # terrain.height_field_raw[:] = 0
            self.add_roughness(terrain)
        elif choice < self.proportions[15]:
            idx = 16
            parkour_hurdle_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   stone_len=0.1+0.3*difficulty,
                                   hurdle_height_range=[0.1+0.1*difficulty, 0.15+0.25*difficulty],
                                   pad_height=0,
                                   x_range=[1.2, 2.2],
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.4, 0.8],
                                   )
            # terrain.height_field_raw[:] = 0
            self.add_roughness(terrain)
        elif choice < self.proportions[16]:
            idx = 17
            parkour_hurdle_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   stone_len=0.1+0.3*difficulty,
                                   hurdle_height_range=[0.1+0.1*difficulty, 0.15+0.15*difficulty],
                                   pad_height=0,
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.45, 1],
                                   flat=True
                                   )
            self.add_roughness(terrain)
        elif choice < self.proportions[17]:
            idx = 18
            parkour_step_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   step_height=0.1 + 0.35*difficulty,
                                   x_range=[0.3,1.5],
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.5, 1],
                                   pad_height=0,
                                   )
            self.add_roughness(terrain)
        elif choice < self.proportions[18]:
            idx = 19
            parkour_gap_terrain(terrain,
                                num_gaps=self.num_goals - 2,
                                gap_size=0.1 + 0.7 * difficulty,
                                gap_depth=[0.2, 1],
                                pad_height=0,
                                x_range=[0.8, 1.5],
                                y_range=self.cfg.y_range,
                                half_valid_width=[0.6, 1.2],
                                # flat=True
                                )
            self.add_roughness(terrain)
        elif choice < self.proportions[19]:
            idx = 20
            demo_terrain(terrain)
            self.add_roughness(terrain)
        # np.set_printoptions(precision=2)
        # print(np.array(self.proportions), choice)
        terrain.idx = idx
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw
        if(np.any(terrain.heightsamples)):
            self.heightsamples[start_x: end_x, start_y:end_y] = terrain.heightsamples
        else:
            self.heightsamples[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # --- [新增] 合并边缘掩码 ---
        # 如果全局掩码还未初始化（第一次调用时），先初始化为全 False
        if not hasattr(self, 'x_edge_mask_manual'):
            self.x_edge_mask_manual = np.zeros((self.tot_rows, self.tot_cols), dtype=bool)
        
        # 如果该子地形有手动标记的边缘，合入全局手动掩码
        if hasattr(terrain, 'local_edge_mask'):
            self.x_edge_mask_manual[start_x:end_x, start_y:end_y] |= terrain.local_edge_mask

        # [新增] 合并空心高度图
        if not hasattr(self, 'hollow_height_map'):
            # 创建一个跟大地图一样大的高度图
            self.hollow_height_map = np.zeros((self.tot_rows, self.tot_cols), dtype=np.float32)
        
        if hasattr(terrain, 'local_hollow_heights'):
            self.hollow_height_map[start_x:end_x, start_y:end_y] = terrain.local_hollow_heights

        if hasattr(terrain, 'trimeshes') and terrain.trimeshes:
            # print(f">>> DEBUG: Found {len(terrain.trimeshes)} meshes from sub-terrain ({row}, {col}). Collecting them.")
            
            origin_x_m = i * self.env_length
            origin_y_m = j * self.env_width
            
            for vertices, triangles in terrain.trimeshes:
                world_vertices = vertices.copy()
                world_vertices[:, 0] += origin_x_m
                world_vertices[:, 1] += origin_y_m
                self.trimeshes.append((world_vertices, triangles))

        env_origin_x = i * self.env_length + 1.0
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 0.5) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 0.5) / terrain.horizontal_scale)
        if self.cfg.origin_zero_z:
            env_origin_z = 0
        else:
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_type[i, j] = terrain.idx

        # --- MODIFICATION START: HANDLE BOTH 2D AND 3D GOALS ---
        if terrain.goals.shape[0] != self.goals.shape[2]:
            print(f"Warning: Goal count mismatch in terrain generation. Expected {self.goals.shape[2]}, got {terrain.goals.shape[0]}")
            return # 如果目标点数量不对，直接返回避免更严重的错误

        # 检查 goals 是 2D 还是 3D
        if terrain.goals.shape[1] == 2:
            # 这是个 2D goal, 需要我们手动添加 Z 坐标
            goals_2d_px = (terrain.goals / terrain.horizontal_scale).astype(int)
            
            # 裁剪像素坐标以确保在子地形范围内
            goals_2d_px[:, 0] = np.clip(goals_2d_px[:, 0], 0, terrain.height_field_raw.shape[0] - 1)
            goals_2d_px[:, 1] = np.clip(goals_2d_px[:, 1], 0, terrain.height_field_raw.shape[1] - 1)
            
            # 从子地形的高度场中查询 Z 值
            z_coords_px = terrain.height_field_raw[goals_2d_px[:, 0], goals_2d_px[:, 1]]
            z_coords_m = z_coords_px * terrain.vertical_scale
            
            # 将 2D goals 和 Z 坐标合并成 3D goals
            goals_3d_local = np.hstack([terrain.goals, z_coords_m[:, np.newaxis]])
        else:
            # 这已经是个 3D goal (来自 hollow_stairs_terrain)
            goals_3d_local = terrain.goals.copy()

        # 将子地形的局部3D坐标转换为世界3D坐标
        world_goals = goals_3d_local
        world_goals[:, 0] += i * self.env_length  # 平移 X
        world_goals[:, 1] += j * self.env_width  # 平移 Y
        # 注意：Z 坐标是相对高度，也需要加上 env_origin_z (如果不是0的话)
        # world_goals[:, 2] += self.env_origins[i, j, 2] # 根据你的逻辑决定是否需要这行

        self.goals[i, j, :, :] = world_goals
        # --- MODIFICATION END ---

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def gap_parkour_terrain(terrain, difficulty, platform_size=2.):
    gap_size = 0.1 + 0.3 * difficulty
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -400
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

    slope_angle = 0.1 + difficulty * 1
    offset = 1 + 9 * difficulty#10
    scale = 15
    wall_center_x = [center_x - x1, center_x, center_x + x1]
    wall_center_y = [center_y - y1, center_y, center_y + y1]

    # for i in range(center_y + y1, center_y + y2):
    #     for j in range(center_x-x1, center_x + x1):
    #         for w in wall_center_x:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[j, i] < height:
    #                 terrain.height_field_raw[j, i] = int(height)
    
    # for i in range(center_y - y2, center_y - y1):
    #     for j in range(center_x-x1, center_x + x1):
    #         for w in wall_center_x:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[j, i] < height:
    #                 terrain.height_field_raw[j, i] = int(height)

    # for i in range(center_x + x1, center_x + x2):
    #     for j in range(center_y-y1, center_y + y1):
    #         for w in wall_center_y:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[i, j] < height:
    #                 terrain.height_field_raw[i, j] = int(height)
    
    # for i in range(center_x - x2, center_x - x1):
    #     for j in range(center_y-y1, center_y + y1):
    #         for w in wall_center_y:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[i, j] < height:
    #                 terrain.height_field_raw[i, j] = int(height)

def flat_terrain(terrain, num_goals=5):
    """
    生成一个完全平坦的地形，并在其上均匀分布目标点。

    Parameters:
        terrain (terrain): 地形对象。
        num_goals (int): 必须生成的目标点数量（默认值为8）。
    Returns:
        terrain (SubTerrain): 更新后的地形对象。
    """
    # 将整个地形设为平地（高度为0）
    terrain.height_field_raw[:, :] = 0

    # 计算地形的中心位置
    terrain_mid_y_px = terrain.length // 2

    # 计算目标点的X坐标，均匀分布在地形的宽度范围内
    goal_x_coords = np.linspace(terrain.width * 0.05, terrain.width * 0.45, num_goals) # 略微偏离边缘

    # 所有目标点的Y坐标都位于地形的中心
    goal_y_coords = np.full(num_goals, terrain_mid_y_px)

    # 将X和Y坐标组合成目标点的二维数组
    goals_px = np.vstack([goal_x_coords, goal_y_coords]).T

    # 将目标点从像素坐标转换为实际坐标（米）
    terrain.goals = goals_px * terrain.horizontal_scale

    return terrain

def stairs_terrain(terrain, step_height, platform_size=1., staircase_length=5.0, num_goals=8, birth_area_length=3, step_width=0.4, terrain_y_flat = 1):
    """
    生成一个指定长度的笔直楼梯，并在平坦的出生区域和楼梯上设置目标点。

    Parameters:
        terrain (terrain): 地形对象。
        step_height (float): 每个台阶的高度（米）。
        platform_size (float): 地形末端顶部平坦区域的大小（米）。
        staircase_length (float): 楼梯在前进方向（X轴）上的总长度（米）。
        num_goals (int): 必须生成的目标点数量（固定为8）。
    Returns:
        terrain (SubTerrain): 更新后的地形对象。
    """
    # 楼梯由 (num_goals - 1) 个台阶和 1 个顶层平台组成。
    num_steps = num_goals - 1

    # --- 1. 参数转换 (米 -> 像素) ---
    platform_size_px = int(platform_size / terrain.horizontal_scale)
    step_height_px = int(step_height / terrain.vertical_scale)
    staircase_length_px = int(staircase_length / terrain.horizontal_scale)
    birth_area_length_px = int(birth_area_length / terrain.horizontal_scale)
    step_width_px = int(step_width / terrain.horizontal_scale)
    # 考虑地形的整体宽度
    terrain_width_px = terrain.width
    terrain_mid_y_px = terrain.length // 2
    # terrain_y_flat = 1.6 # 两边各留1米的平地区域
    terrain_y_flat_px = int(terrain_y_flat / terrain.horizontal_scale)

    # --- 2. 验证输入参数 ---
    # 确保楼梯总长不超过地形的总宽度
    staircase_length_px = min(staircase_length_px, terrain_width_px)
    start_y = 0 + terrain_y_flat_px
    end_y = terrain.length - terrain_y_flat_px

    # 检查楼梯总长是否足够容纳顶部平台、台阶以及出生区域
    if staircase_length_px <= platform_size_px + birth_area_length_px or num_steps <= 0:
        # 如果楼梯配置无效，则只生成一个平地，并在其上均匀分布目标点
        terrain.height_field_raw[:, :] = 0
        # 目标点从地形起点开始，均匀分布到地形终点
        goal_x_coords = np.linspace(birth_area_length_px * 0.5, terrain_width_px * 0.95, num_goals) # 略微偏离边缘
        goal_y_coords = np.full(num_goals, terrain_mid_y_px)
        goals_px = np.vstack([goal_x_coords, goal_y_coords]).T
        terrain.goals = goals_px * terrain.horizontal_scale
        return terrain

    if step_width_px < 1: step_width_px = 1 # 确保台阶至少有1像素宽

    # --- 4. 初始化地形和目标点 ---
    terrain.height_field_raw[:, :] = 0  # 先将整个地形设为平地（出生区域）
    goals_px = []
    height = 0
    current_x_position_px = birth_area_length_px # 台阶从出生区域之后开始

    # --- 5. 添加第一个导航目标点 ---
    # 第一个目标点放置在出生区域之后，第一级台阶之前
    # first_nav_goal_x = birth_area_length_px + step_width_px / 2.0
    # goals_px.append([first_nav_goal_x, terrain_mid_y_px])

    # --- 6. 循环创建台阶 ---
    for i in range(num_steps):
        # 计算当前台阶的起止位置
        start_x = current_x_position_px
        end_x = current_x_position_px + step_width_px
        
        # 增加高度并填充台阶
        height += step_height_px
        terrain.height_field_raw[start_x:end_x, start_y:end_y] = height

        # 将目标点放置在当前台阶平面的中心
        goal_x = start_x + step_width_px / 2.0
        goals_px.append([goal_x, terrain_mid_y_px])

        # 更新当前前进位置
        current_x_position_px = end_x

    # --- 7. 创建顶部平台并放置最后一个目标点 ---
    platform_start_x = current_x_position_px
    platform_end_x = staircase_length_px
    height += step_height_px
    terrain.height_field_raw[platform_start_x:platform_end_x, start_y:end_y] = height

    # 将最后一个目标点（第`num_goals`个）放置在顶部平台的中心
    top_platform_center_x = platform_start_x + (platform_end_x - platform_start_x) / 2.0
    # goals_px.append([top_platform_center_x, terrain_mid_y_px])
    
    goals_px.append([current_x_position_px + 3 * step_width_px, terrain_mid_y_px])

    terrain.goals = np.array(goals_px) * terrain.horizontal_scale

    return terrain

def hollow_stairs_terrain(terrain, step_height_first, step_height_others, slope_treshold, step_thickness=0.05, platform_size=1., staircase_length=5.0, num_goals=8, birth_area_length=3.0, step_width=0.4, terrain_y_flat = 1, difficulty = 0, is_steep=False):
    """
    生成一个镂空的楼梯，并可选择性地在其表面添加独立的Perlin噪声起伏。
    噪声生成使用TerrainPerlin，与barrier_track使用相同的逻辑。
    
    Parameters:
        terrain (terrain): 地形对象。
        step_height_first (float): 第一个台阶的高度（米）。
        step_height_others (float): 其他台阶的高度（米）。
        step_thickness (float): 台阶厚度（米）。
        platform_size (float): 地形末端顶部平坦区域的大小（米）。
        staircase_length (float): 楼梯在前进方向（X轴）上的总长度（米）。
        num_goals (int): 必须生成的目标点数量。
        add_noise (bool): 是否在台阶表面添加Perlin噪声。
        noise_kwargs (dict): 用于生成Perlin噪声的参数字典。
    """
    def add_roughness_heightfield(heightfield, width, length, difficulty=1):
        height = terrain.height 
        def random_uniform_terrain_heightfield(heightfield, min_height, max_height, step):
            downsampled_scale = terrain.downsampled_scale
            if downsampled_scale is None:
                downsampled_scale = terrain.horizontal_scale

            # switch parameters to discrete units
            min_height = int(min_height / terrain.vertical_scale)
            max_height = int(max_height / terrain.vertical_scale)
            step = int(step / terrain.vertical_scale)

            heights_range = np.arange(min_height, max_height + step, step)
            height_field_downsampled = np.random.choice(heights_range, (int(width * terrain.horizontal_scale / downsampled_scale), int(
                length * terrain.horizontal_scale / downsampled_scale)))

            x = np.linspace(0, width * terrain.horizontal_scale, height_field_downsampled.shape[0])
            y = np.linspace(0, length * terrain.horizontal_scale, height_field_downsampled.shape[1])

            f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')

            x_upsampled = np.linspace(0, width * terrain.horizontal_scale, width)
            y_upsampled = np.linspace(0, length * terrain.horizontal_scale, length)
            z_upsampled = np.rint(f(y_upsampled, x_upsampled))

            heightfield += z_upsampled.astype(np.int16)
            return heightfield
        # print(difficulty)
        # print(height[1],height[0])
        max_height = (height[1] - height[0]) * difficulty + height[0]
        height = random.uniform(height[0], max_height)
        # print(">> Adding roughness with height:", height)
        # print(max_height, height)
        return random_uniform_terrain_heightfield(heightfield, min_height=-height, max_height=height, step=0.005)

    def fill_heightfield_to_scale(heightfield):
        """ Due to the rasterization of the heightfield, the trimesh size does not match the 
        heightfield_resolution * horizontal_scale, so we need to fill enlarge heightfield to
        meet this scale.
        """
        assert len(heightfield.shape) == 2, "heightfield must be 2D"
        heightfield_x_fill = np.concatenate([
            heightfield,
            heightfield[-1:, :],
        ], axis= 0)
        heightfield_y_fill = np.concatenate([
            heightfield_x_fill,
            heightfield_x_fill[:, -2:],
        ], axis= 1)
        return heightfield_y_fill

    # --- 1. 参数转换与初始化 ---
    num_steps = num_goals - 1
    platform_size_px = int(platform_size / terrain.horizontal_scale)
    staircase_length_px = int(staircase_length / terrain.horizontal_scale)
    birth_area_length_px = int(birth_area_length / terrain.horizontal_scale)
    step_width_px = int(step_width / terrain.horizontal_scale)
    
    terrain_width_px = terrain.width   # X方向像素
    terrain_length_px = terrain.length # Y方向像素

    # print("terrain_length_px:", terrain_length_px)
    # print("terrain_width_px:", terrain_width_px)

    # terrain_y_flat = 1.6 # 两边各留1米的平地区域
    terrain_y_flat_px = int(terrain_y_flat / terrain.horizontal_scale)
    
    start_y = 0 + terrain_y_flat_px
    end_y = terrain.length - terrain_y_flat_px
    
    terrain.height_field_raw[:, :] = 0
    if not hasattr(terrain, 'trimeshes'):
        terrain.trimeshes = []

    # --- 2. 验证与计算台阶尺寸 ---
    # birth_area_length_px = int(terrain_width_px * 0.1)
    # birth_area_length_px = max(birth_area_length_px, 120)

    # 调整出生区域大小避免平地过长(确保总长度不变)
    # tmp = platform_size_px + birth_area_length_px
    # birth_area_length_px = 60
    
    # platform_size_px = tmp - birth_area_length_px

    # print("birth_area_length_px:", birth_area_length_px)
    # print("platform_size_px:", platform_size_px)
    # print("staircase_length_px:", staircase_length_px)
    if staircase_length_px <= platform_size_px + birth_area_length_px or num_steps <= 0:
        print("Warning: Staircase configuration invalid. Generating flat terrain.")
        terrain.goals = np.zeros((num_goals, 3))
        goal_x_coords_m = np.linspace(birth_area_length_px * terrain.horizontal_scale, staircase_length_px * terrain.horizontal_scale, num_goals)
        terrain.goals[:, 0] = goal_x_coords_m
        terrain.goals[:, 1] = (terrain.length / 2) * terrain.horizontal_scale
        return terrain

    # total_step_space_px = staircase_length_px - platform_size_px - birth_area_length_px
    # step_width_px = int(total_step_space_px / num_steps)
    
    
    if step_width_px < 1: step_width_px = 1

    # --- 4. 循环创建台阶网格和目标点 ---
    goals_m = []
    current_height_m = 0.0
    current_x_pos_px = birth_area_length_px
    terrain_length_m = terrain.length * terrain.horizontal_scale
    terrain_width_m = terrain.width * terrain.horizontal_scale
    # print("terrain_length_m:", terrain_length_m)
    # print("terrain_width_m:", terrain_width_m)
    step_width_m = step_width_px * terrain.horizontal_scale

    rail_thickness = 0.03
    rail_mid = rail_thickness / 2
    rail_L_y = terrain_y_flat + rail_mid
    rail_R_y = terrain_length_m - terrain_y_flat - rail_mid
    step_mid = step_thickness / 2
    width_px_for_roughness = end_y-start_y-1
    step_resolution = (
        np.ceil(step_width_px).astype(int),
        np.ceil(width_px_for_roughness).astype(int)
    )

    heightsamples = terrain.height_field_raw.copy()
    heightsamples[:, :] = 0
    step_end_x_m = birth_area_length + step_width_m * num_steps

    terrain.local_edge_mask = np.zeros_like(terrain.height_field_raw, dtype=bool)
    terrain.local_edge_mask[current_x_pos_px - 1:current_x_pos_px, start_y:end_y] = True # 前缘
    terrain.local_hollow_heights = np.zeros_like(terrain.height_field_raw, dtype=np.float32)

    for i in range(num_steps):
        if i == 0:
            current_height_m += step_height_first
        else:
            current_height_m += step_height_others

        x_s, x_e = current_x_pos_px, current_x_pos_px + step_width_px
        
        terrain.local_hollow_heights[x_s:x_e, start_y:end_y] = current_height_m
        # 只在台阶的前后两条线（X边缘）标记为边缘
        # terrain.local_edge_mask[x_s:x_s+1, start_y:end_y] = True # 前缘
        terrain.local_edge_mask[x_e-1:x_e, start_y:end_y] = True # 后缘
        # 如果需要左右边缘惩罚：
        terrain.local_edge_mask[x_s:x_e, start_y:start_y+1] = True
        terrain.local_edge_mask[x_s:x_e, end_y-1:end_y] = True
        # # --- 关键步骤：更新高度场记录 ---
        # # 我们将台阶所在的矩形区域高度写入 height_field_raw
        # # 这样在 Terrain 最终处理时，台阶的边界会产生巨大的梯度，从而被计入 x_edge_mask
        # h_val = int(current_height_m / terrain.vertical_scale)
        # x_s, x_e = current_x_pos_px, current_x_pos_px + step_width_px
        
        # # 写入高度场
        # terrain.height_field_raw[x_s:x_e, 0 + start_y: width_px_for_roughness + start_y] = h_val
        
        center_x_m = (current_x_pos_px + step_width_px / 2.0) * terrain.horizontal_scale
        center_y_m = terrain_length_m / 2.0
        center_z_m = current_height_m - step_thickness / 2.0 # 厚度补充
        current_x_pos_m = current_x_pos_px * terrain.horizontal_scale
        if i == 1:
            back_rail_z = current_height_m + rail_mid
        vertices, triangles = trimesh.box_trimesh(
            size=(step_width_m, terrain_length_m - 2*terrain_y_flat, step_thickness),
            center_position=(center_x_m, center_y_m, center_z_m)
        )
        rail_to_end_L = trimesh.box_trimesh(
            size=(step_end_x_m - current_x_pos_m - step_width_m, rail_thickness, step_thickness),
            center_position=((step_end_x_m + current_x_pos_m + step_width_m) / 2, rail_L_y, current_height_m - step_mid)
        )
        rail_to_end_R = trimesh.box_trimesh(
            size=(step_end_x_m - current_x_pos_m - step_width_m, rail_thickness, step_thickness),
            center_position=((step_end_x_m + current_x_pos_m + step_width_m) / 2, rail_R_y, current_height_m - step_mid)
        )
        heightfield_raw = np.zeros(step_resolution, dtype=np.float32)
        heightfield_raw[:, 0: width_px_for_roughness] = current_height_m / terrain.vertical_scale  # 填充当前台阶高度

        # heightfield_raw = add_roughness_heightfield(heightfield_raw, width=step_width_px, length=width_px_for_roughness, difficulty=difficulty)
        heightsamples[
            current_x_pos_px:(current_x_pos_px + step_width_px),
            0 + start_y: width_px_for_roughness + start_y
        ] = heightfield_raw.copy()  # 更新地形高度场 空心楼梯 的高度
        t_vertices, t_triangles, _ = convert_heightfield_to_trimesh(
                fill_heightfield_to_scale(heightfield_raw),
                terrain.horizontal_scale,
                terrain.vertical_scale,
            )
        t_vertices[:, 0] += (current_x_pos_px) * terrain.horizontal_scale
        t_vertices[:, 1] += (terrain_length_m / 2.0 - (width_px_for_roughness + 1 ) / 2.0 * terrain.horizontal_scale)
        # t_vertices[:, 2] += current_height_m - step_thickness / 2.0

        trimesh_template = (t_vertices, t_triangles)
        step_trimesh = (vertices, triangles)

        final_trimesh = trimesh.combine_trimeshes(
                trimesh_template,
                step_trimesh,
                rail_to_end_L,
                rail_to_end_R
            )
        # final_trimesh = step_trimesh
        terrain.trimeshes.append(final_trimesh)
        # terrain.trimeshes.append(trimesh_template)
 
        goals_m.append([center_x_m, center_y_m, current_height_m])
        current_x_pos_px += step_width_px

    # --- 5. 创建顶部平台 ---
    current_height_m += step_height_others
    # h_top = int(current_height_m / terrain.vertical_scale)
    # platform_width_px = min(staircase_length_px - current_x_pos_px, 6 * step_width_px)
    
    # terrain.height_field_raw[current_x_pos_px : current_x_pos_px + platform_width_px, 0 + start_y: width_px_for_roughness + start_y] = h_top
    
    terrain.local_edge_mask[current_x_pos_px:current_x_pos_px+1, start_y:end_y] = True # 前缘
    platform_width_px = min(staircase_length_px - current_x_pos_px, 6 * step_width_px)
    platform_width_m = platform_width_px * terrain.horizontal_scale

    platform_center_x_m = (current_x_pos_px + platform_width_px / 2.0) * terrain.horizontal_scale
    platform_center_y_m = terrain_length_m / 2.0
    platform_center_z_m = current_height_m - step_thickness / 2.0

    terrain.local_hollow_heights[current_x_pos_px:current_x_pos_px + 5 , start_y:end_y] = current_height_m

    platform_resolution = (
        np.ceil(platform_width_px).astype(int),
        np.ceil(width_px_for_roughness).astype(int)
    )
    if step_height_others < 0:
        terrain.height_field_raw[:, :] = current_height_m / terrain.vertical_scale  # 挖洞
        terrain.height_field_raw[0: birth_area_length_px, 0: terrain.length] = 0

    vertices, triangles = trimesh.box_trimesh(
        size=(platform_width_m, terrain_length_m - 2*terrain_y_flat, step_thickness),
        center_position=(platform_center_x_m, platform_center_y_m, platform_center_z_m)
    )
    
    heightfield_raw = np.zeros(platform_resolution, dtype=np.float32)
    heightfield_raw[:, 0: width_px_for_roughness] = current_height_m / terrain.vertical_scale  # 填充当前台阶高度
    # heightfield_raw += heightfield_noise[
    #     (current_x_pos_px - birth_area_length_px):(current_x_pos_px - birth_area_length_px + step_width_px),
    #     0: terrain_length_px
    # ]

    heightfield_raw = add_roughness_heightfield(heightfield_raw, width=platform_width_px, length=width_px_for_roughness, difficulty=difficulty)
    heightsamples[
        current_x_pos_px:(current_x_pos_px + platform_width_px),
        0 + start_y: width_px_for_roughness + start_y
    ] = heightfield_raw  # 更新地形高度场
    t_vertices, t_triangles, _ = convert_heightfield_to_trimesh(
            fill_heightfield_to_scale(heightfield_raw),
            terrain.horizontal_scale,
            terrain.vertical_scale,
        )
    t_vertices[:, 0] += (current_x_pos_px) * terrain.horizontal_scale
    t_vertices[:, 1] += (terrain_length_m / 2.0 - (width_px_for_roughness + 1) / 2.0 * terrain.horizontal_scale)
    # t_vertices[:, 2] += current_height_m - step_thickness / 2.0

    trimesh_template = (t_vertices, t_triangles)
    step_trimesh = (vertices, triangles)

    final_trimesh = trimesh.combine_trimeshes(
            trimesh_template,
            step_trimesh,
        )

    terrain.trimeshes.append(final_trimesh)

    # goals_m.append([platform_center_x_m, platform_center_y_m, current_height_m])
    # 设置最后一个waypoint在楼梯的前面一点点
    goals_m.append([(current_x_pos_px + 4 * step_width_px) * terrain.horizontal_scale, platform_center_y_m, current_height_m])


    # ------  6.加入栏杆和后部长方形横杆组和叉型  --------

    #栏杆
    # L实则为狗前进方向的R，R同理（因为我写代码默认L是减去，R是增加
    def bar_trimesh(center_position, delta_x, delta_z):
        vertices = np.empty((8, 3), dtype= np.float32)
        vertices[:] = center_position
        vertices[[0, 4, 2, 6], 0] -= delta_x / 2
        vertices[[1, 5, 3, 7], 0] += delta_x / 2
        vertices[[0, 1, 2, 3], 1] -= rail_thickness / 2
        vertices[[4, 5, 6, 7], 1] += rail_thickness / 2
        vertices[[3, 7], 2] += delta_z / 2
        vertices[[1, 5], 2] += delta_z / 2
        vertices[[2, 6], 2] -= delta_z / 2
        vertices[[0, 4], 2] -= delta_z / 2
        vertices[[3, 7], 2] -= rail_thickness / 2
        vertices[[1, 5], 2] += rail_thickness / 2
        vertices[[2, 6], 2] -= rail_thickness / 2
        vertices[[0, 4], 2] += rail_thickness / 2

        triangles = -np.ones((12, 3), dtype= np.uint32)
        triangles[0] = [0, 2, 1] #
        triangles[1] = [1, 2, 3]
        triangles[2] = [0, 4, 2] #
        triangles[3] = [2, 4, 6]
        triangles[4] = [4, 5, 6] #
        triangles[5] = [5, 7, 6]
        triangles[6] = [1, 3, 5] #
        triangles[7] = [3, 7, 5]
        triangles[8] = [0, 1, 4] #
        triangles[9] = [1, 5, 4]
        triangles[10]= [2, 6, 3] #
        triangles[11]= [3, 6, 7]
        
        return vertices, triangles
    
    def connect_trimesh(center_position, con_delta_z):
        vertices = np.empty((6, 3), dtype= np.float32)
        triangles = -np.ones((8, 3), dtype= np.uint32)
        vertices[[1, 2, 3, 4], 0] = center_position[0] + rail_thickness / 2
        vertices[[0, 5], 0] = center_position[0] - rail_thickness / 2
        vertices[[0, 1, 2], 1] = center_position[1] - rail_thickness / 2
        vertices[[3, 4, 5], 1] = center_position[1] + rail_thickness / 2
        vertices[[2, 3], 2] = center_position[2] + con_delta_z
        vertices[[0, 1, 4, 5], 2] = center_position[2]

        triangles[0] = [0, 1, 2]
        triangles[1] = [3, 4, 5]
        triangles[2] = [1, 3, 2]
        triangles[3] = [1, 4, 3]
        triangles[4] = [0, 4, 1]
        triangles[5] = [0, 5, 4]
        triangles[6] = [2, 3, 5]
        triangles[7] = [2, 5, 0]

        return vertices, triangles

    rail_height = 0.6
    rail_height_short = rail_height + step_height_first
    rail_height_tall = rail_height + current_height_m
    current_x_pos_m = (current_x_pos_px) * terrain.horizontal_scale

    Down_L = trimesh.box_trimesh(
        size=(rail_thickness, rail_thickness, rail_height_short),
        center_position=(birth_area_length + rail_mid, rail_L_y, rail_height_short / 2)
    )
    Down_R = trimesh.box_trimesh(
        size=(rail_thickness, rail_thickness, rail_height_short),
        center_position=(birth_area_length + rail_mid, rail_R_y, rail_height_short / 2)
    )
    delta_minus_rail_height = step_height_first - step_thickness
    rail_height_tall_adjusted = rail_height_tall - delta_minus_rail_height
    rail_height_tall_z_adjusted = rail_height_tall / 2 + delta_minus_rail_height / 2
    Up_L = trimesh.box_trimesh(
        size=(rail_thickness, rail_thickness, rail_height_tall_adjusted),
        center_position=(current_x_pos_m + rail_mid, rail_L_y, rail_height_tall_z_adjusted)
    )
    Up_R = trimesh.box_trimesh(
        size=(rail_thickness, rail_thickness, rail_height_tall_adjusted),
        center_position=(current_x_pos_m + rail_mid, rail_R_y, rail_height_tall_z_adjusted)
    )
    Up_Back_L = trimesh.box_trimesh(
        size=(rail_thickness, rail_thickness, rail_height_tall),
        center_position=(current_x_pos_m + platform_width_m - rail_mid, rail_L_y, rail_height_tall / 2)
    )
    Up_Back_R = trimesh.box_trimesh(
        size=(rail_thickness, rail_thickness, rail_height_tall),
        center_position=(current_x_pos_m + platform_width_m - rail_mid, rail_R_y, rail_height_tall / 2)
    )

    Up_Bar_length = platform_width_m - rail_thickness * 2 
    Up_Bar_3_z = rail_height_tall - rail_thickness / 2
    Up_Bar_2_z = (current_height_m + Up_Bar_3_z) / 2
    Up_Bar_L3 = trimesh.box_trimesh(
        size=(Up_Bar_length, rail_thickness, rail_thickness),
        center_position=(current_x_pos_m + platform_width_m / 2, rail_L_y, Up_Bar_3_z)
    )
    Up_Bar_R3 = trimesh.box_trimesh(
        size=(Up_Bar_length, rail_thickness, rail_thickness),
        center_position=(current_x_pos_m + platform_width_m / 2, rail_R_y, Up_Bar_3_z)
    )
    Up_Bar_L2 = trimesh.box_trimesh(
        size=(Up_Bar_length, rail_thickness, rail_thickness),
        center_position=(current_x_pos_m + platform_width_m / 2, rail_L_y, Up_Bar_2_z)
    )
    Up_Bar_R2 = trimesh.box_trimesh(
        size=(Up_Bar_length, rail_thickness, rail_thickness),
        center_position=(current_x_pos_m + platform_width_m / 2, rail_R_y, Up_Bar_2_z)
    )
    
    bar_tangent = step_height_others / step_width_m
    bar_x = step_width_m * 4 - rail_thickness
    bar_z = bar_x * bar_tangent
    Bar_1_z = (step_height_first + rail_thickness * bar_tangent + current_height_m) / 2 + rail_thickness / 2
    Bar_3_z = (rail_height_tall + rail_height_short + rail_thickness * bar_tangent) / 2 - rail_thickness / 2
    Bar_2_z = (Bar_1_z + Bar_3_z) / 2
    Bar_L1 = bar_trimesh(
        center_position=(birth_area_length + rail_thickness + bar_x / 2, 
                        rail_L_y, 
                        Bar_1_z),
        delta_x=bar_x,
        delta_z=bar_z
    )
    Bar_R1 = bar_trimesh(
        center_position=(birth_area_length + rail_thickness + bar_x / 2, 
                        rail_R_y, 
                        Bar_1_z),
        delta_x=bar_x,
        delta_z=bar_z
    )
    Bar_L3 = bar_trimesh(
        center_position=(birth_area_length + rail_thickness + bar_x / 2, 
                        rail_L_y, 
                        Bar_3_z),
        delta_x=bar_x,
        delta_z=bar_z
    )
    Bar_R3 = bar_trimesh(
        center_position=(birth_area_length + rail_thickness + bar_x / 2, 
                        rail_R_y, 
                        Bar_3_z),
        delta_x=bar_x,
        delta_z=bar_z
    )
    Bar_L2 = bar_trimesh(
        center_position=(birth_area_length + rail_thickness + bar_x / 2, 
                        rail_L_y, 
                        Bar_2_z),
        delta_x=bar_x,
        delta_z=bar_z
    )
    Bar_R2 = bar_trimesh(
        center_position=(birth_area_length + rail_thickness + bar_x / 2, 
                        rail_R_y, 
                        Bar_2_z),
        delta_x=bar_x,
        delta_z=bar_z
    )
    con_L = connect_trimesh(
        center_position=(birth_area_length + rail_mid, rail_L_y, rail_height_short),
        con_delta_z=rail_thickness * bar_tangent
    )
    con_R = connect_trimesh(
        center_position=(birth_area_length + rail_mid, rail_R_y, rail_height_short),
        con_delta_z=rail_thickness * bar_tangent
    )

    final_siderail = trimesh.combine_trimeshes(
        Down_L, 
        Down_R, 
        Up_L, 
        Up_R,
        Up_Back_L,
        Up_Back_R,
        Up_Bar_L2,
        Up_Bar_L3,
        Up_Bar_R2,
        Up_Bar_R3,
        Bar_L1,
        Bar_L2,
        Bar_L3,
        Bar_R1,
        Bar_R2,
        Bar_R3,
        con_L,
        con_R,
    )
    if(is_steep):
        Down_Mid = trimesh.box_trimesh(
            size=(rail_thickness, terrain_length_m - 2*terrain_y_flat, rail_thickness),
            center_position=(birth_area_length + rail_mid, terrain_length_m / 2, step_height_first * 0.05 / 0.22 + rail_mid)
        )
        final_siderail = trimesh.combine_trimeshes(
            final_siderail,
            Down_Mid,
        )
    terrain.trimeshes.append(final_siderail)

    # 后部长方形横杆组
    back_rail_F = trimesh.box_trimesh(
        size=(rail_thickness, terrain_length_m - 2 * terrain_y_flat, rail_thickness),
        center_position=(current_x_pos_m + rail_mid, center_y_m, back_rail_z)
    )
    back_rail_B = trimesh.box_trimesh(
        size=(rail_thickness, terrain_length_m - 2 * terrain_y_flat, rail_thickness),
        center_position=(current_x_pos_m + platform_width_m - rail_mid, center_y_m, back_rail_z)
    )
    back_rail_L_out = trimesh.box_trimesh(
        size=(platform_width_m - rail_thickness * 2, rail_thickness, rail_thickness),
        center_position=(current_x_pos_m + platform_width_m / 2, rail_L_y, back_rail_z)
    )
    back_rail_R_out = trimesh.box_trimesh(
        size=(platform_width_m - rail_thickness * 2, rail_thickness, rail_thickness),
        center_position=(current_x_pos_m + platform_width_m / 2, rail_R_y, back_rail_z)
    )
    back_rail_L_in = trimesh.box_trimesh(
        size=(platform_width_m - rail_thickness * 2, rail_thickness, rail_thickness),
        center_position=(current_x_pos_m + platform_width_m / 2, rail_L_y + rail_thickness, back_rail_z)
    )
    back_rail_R_in = trimesh.box_trimesh(
        size=(platform_width_m - rail_thickness * 2, rail_thickness, rail_thickness),
        center_position=(current_x_pos_m + platform_width_m / 2, rail_R_y - rail_thickness, back_rail_z)
    )
    back_rail_L_out_downward = trimesh.box_trimesh(
        size=(platform_width_m - rail_thickness * 2, rail_thickness, step_thickness),
        center_position=(current_x_pos_m + platform_width_m / 2, rail_L_y, step_height_first - step_mid)
    )
    back_rail_R_out_downward = trimesh.box_trimesh(
        size=(platform_width_m - rail_thickness * 2, rail_thickness, step_thickness),
        center_position=(current_x_pos_m + platform_width_m / 2, rail_R_y, step_height_first - step_mid)
    )
    final_backrail = trimesh.combine_trimeshes(
        back_rail_F,
        back_rail_B,
        back_rail_L_out,
        back_rail_R_out,
        back_rail_L_in,
        back_rail_R_in,
        back_rail_L_out_downward,
        back_rail_R_out_downward,
    )
    terrain.trimeshes.append(final_backrail)

    #叉型
    back_rail_x = current_x_pos_m + platform_width_m - rail_mid
    cross_length = np.sqrt( (current_height_m - back_rail_z - rail_thickness * 3) **2 + (terrain_length_m - 2 * terrain_y_flat - rail_thickness) **2 )
    cross_1 = trimesh.box_trimesh(
        size=(rail_thickness, cross_length, rail_thickness),
        center_position=(back_rail_x - rail_thickness, center_y_m, (back_rail_z + current_height_m - step_mid) / 2),
        rpy = (np.arctan2(current_height_m - back_rail_z - rail_thickness * 2, (terrain_length_m - 2 * terrain_y_flat)), 0, 0)
    )
    cross_2 = trimesh.box_trimesh(
        size=(rail_thickness, cross_length, rail_thickness),
        center_position=(back_rail_x + rail_thickness, center_y_m, (back_rail_z + current_height_m - step_mid) / 2),
        rpy = (-np.arctan2(current_height_m - back_rail_z - rail_thickness * 2, (terrain_length_m - 2 * terrain_y_flat)), 0, 0)
    )   
    final_cross = trimesh.combine_trimeshes(
        cross_1,
        cross_2,
    )
    terrain.trimeshes.append(final_cross)
    # ---------------------------------------

    # --- 7. 赋值 ---
    terrain.goals = np.array(goals_m)

    terrain.heightsamples[:, :] = heightsamples[:, :]
    # print("1111111111111111111111111111111111111111111111111111111111111111111")
    return terrain





def parkour_terrain(terrain, 
                    platform_len=2.5, 
                    platform_height=0., 
                    num_stones=8, 
                    x_range=[1.8, 1.9], 
                    y_range=[0., 0.1], 
                    z_range=[-0.2, 0.2],
                    stone_len=1.0,
                    stone_width=0.6,
                    pad_width=0.1,
                    pad_height=0.5,
                    incline_height=0.1,
                    last_incline_height=0.6,
                    last_stone_len=1.6,
                    pit_depth=[0.5, 1.]):
    # 1st dimension: x, 2nd dimension: y
    goals = np.zeros((num_stones+2, 2))
    terrain.height_field_raw[:] = -round(np.random.uniform(pit_depth[0], pit_depth[1]) / terrain.vertical_scale)
    
    mid_y = terrain.length // 2  # length is actually y width
    stone_len = np.random.uniform(*stone_len)
    stone_len = 2 * round(stone_len / 2.0, 1)
    stone_len = round(stone_len / terrain.horizontal_scale)
    dis_x_min = stone_len + round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = stone_len + round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    dis_z_min = round(z_range[0] / terrain.vertical_scale)
    dis_z_max = round(z_range[1] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_width = round(stone_width / terrain.horizontal_scale)
    last_stone_len = round(last_stone_len / terrain.horizontal_scale)

    incline_height = round(incline_height / terrain.vertical_scale)
    last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len - np.random.randint(dis_x_min, dis_x_max) + stone_len // 2
    goals[0] = [platform_len -  stone_len // 2, mid_y]
    left_right_flag = np.random.randint(0, 2)
    # dis_z = np.random.randint(dis_z_min, dis_z_max)
    dis_z = 0
    
    for i in range(num_stones):
        dis_x += np.random.randint(dis_x_min, dis_x_max)
        pos_neg = round(2*(left_right_flag - 0.5))
        dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
        if i == num_stones - 1:
            dis_x += last_stone_len // 4
            heights = np.tile(np.linspace(-last_incline_height, last_incline_height, stone_width), (last_stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x-last_stone_len//2:dis_x+last_stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
        else:
            heights = np.tile(np.linspace(-incline_height, incline_height, stone_width), (stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
        
        goals[i+1] = [dis_x, dis_y]

        left_right_flag = 1 - left_right_flag
    final_dis_x = dis_x + 2*np.random.randint(dis_x_min, dis_x_max)
    final_platform_start = dis_x + last_stone_len // 2 + round(0.05 // terrain.horizontal_scale)
    terrain.height_field_raw[final_platform_start:, :] = platform_height
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height
    
def parkour_gap_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_gaps=8,
                           gap_size=0.3,
                           x_range=[1.6, 2.4],
                           y_range=[-1.2, 1.2],
                           half_valid_width=[0.6, 1.2],
                           gap_depth=-200,
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False):
    goals = np.zeros((num_gaps+2, 2))
    # terrain.height_field_raw[:] = -200
    # import ipdb; ipdb.set_trace()
    mid_y = terrain.length // 2  # length is actually y width

    # dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    # dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale)
    
    # half_gap_width = round(np.random.uniform(0.6, 1.2) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    # terrain.height_field_raw[:, :mid_y-half_valid_width] = gap_depth
    # terrain.height_field_raw[:, mid_y+half_valid_width:] = gap_depth
    
    terrain.height_field_raw[0:platform_len, :] = platform_height

    gap_size = round(gap_size / terrain.horizontal_scale)
    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if not flat:
            # terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            # terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, 
            #                          gap_center-half_gap_width:gap_center+half_gap_width] = gap_depth
            terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, :] = gap_depth

        terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = gap_depth
        terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = gap_depth
        
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def parkour_hurdle_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_stones=8,
                           stone_len=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False):
    goals = np.zeros((num_stones+2, 2))
    # terrain.height_field_raw[:] = -200
    
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_len = round(stone_len / terrain.horizontal_scale)
    # stone_width = round(stone_width / terrain.horizontal_scale)
    
    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x
        if not flat:
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, :mid_y+rand_y-half_valid_width] = 0
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, mid_y+rand_y+half_valid_width:] = 0
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def parkour_step_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_stones=8,
                        #    x_range=[1.5, 2.4],
                            x_range=[0.2, 0.4],
                           y_range=[-0.15, 0.15],
                           half_valid_width=[0.45, 0.5],
                           step_height = 0.2,
                           pad_width=0.1,
                           pad_height=0.5):
    goals = np.zeros((num_stones+2, 2))
    # terrain.height_field_raw[:] = -200
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round( (x_range[0] + step_height) / terrain.horizontal_scale)
    dis_x_max = round( (x_range[1] + step_height) / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    step_height = round(step_height / terrain.vertical_scale)

    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    # stone_width = round(stone_width / terrain.horizontal_scale)
    
    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    last_dis_x = dis_x
    stair_height = 0
    goals[0] = [platform_len - round(1 / terrain.horizontal_scale), mid_y]
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if i < num_stones // 2:
            stair_height += step_height
        elif i > num_stones // 2:
            stair_height -= step_height
        terrain.height_field_raw[dis_x:dis_x+rand_x, ] = stair_height
        dis_x += rand_x
        terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = 0
        terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = 0
        
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y+rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def demo_terrain(terrain):
    goals = np.zeros((8, 2))
    mid_y = terrain.length // 2
    
    # hurdle
    platform_length = round(2 / terrain.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / terrain.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / terrain.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[0] = [platform_length + hurdle_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+hurdle_depth, round(mid_y-hurdle_width/2):round(mid_y+hurdle_width/2)] = hurdle_height
    
    # step up
    platform_length += round(np.random.uniform(1.5, 2.5) / terrain.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    first_step_height = round(np.random.uniform(0.35, 0.45) / terrain.vertical_scale)
    first_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[1] = [platform_length+first_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+first_step_depth, round(mid_y-first_step_width/2):round(mid_y+first_step_width/2)] = first_step_height
    
    platform_length += first_step_depth
    second_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    second_step_height = first_step_height
    second_step_width = first_step_width
    goals[2] = [platform_length+second_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+second_step_depth, round(mid_y-second_step_width/2):round(mid_y+second_step_width/2)] = second_step_height
    
    # gap
    platform_length += second_step_depth
    gap_size = round(np.random.uniform(0.5, 0.8) / terrain.horizontal_scale)
    
    # step down
    platform_length += gap_size
    third_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    third_step_height = first_step_height
    third_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[3] = [platform_length+third_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+third_step_depth, round(mid_y-third_step_width/2):round(mid_y+third_step_width/2)] = third_step_height
    
    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    forth_step_height = first_step_height
    forth_step_width = third_step_width
    goals[4] = [platform_length+forth_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+forth_step_depth, round(mid_y-forth_step_width/2):round(mid_y+forth_step_width/2)] = forth_step_height
    
    # parkour
    platform_length += forth_step_depth
    gap_size = round(np.random.uniform(0.1, 0.4) / terrain.horizontal_scale)
    platform_length += gap_size
    
    left_y = mid_y + round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    right_y = mid_y - round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    
    slope_height = round(np.random.uniform(0.15, 0.22) / terrain.vertical_scale)
    slope_depth = round(np.random.uniform(0.75, 0.85) / terrain.horizontal_scale)
    slope_width = round(1.0 / terrain.horizontal_scale)
    
    platform_height = slope_height + np.random.randint(0, 0.2 / terrain.vertical_scale)

    goals[5] = [platform_length+slope_depth/2, left_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * 1
    terrain.height_field_raw[platform_length:platform_length+slope_depth, left_y-slope_width//2: left_y+slope_width//2] = heights.astype(int) + platform_height
    
    platform_length += slope_depth + gap_size
    goals[6] = [platform_length+slope_depth/2, right_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * -1
    terrain.height_field_raw[platform_length:platform_length+slope_depth, right_y-slope_width//2: right_y+slope_width//2] = heights.astype(int) + platform_height
    
    platform_length += slope_depth + gap_size + round(0.4 / terrain.horizontal_scale)
    goals[-1] = [platform_length, left_y]
    terrain.goals = goals * terrain.horizontal_scale

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def half_sloped_terrain(terrain, wall_width=4, start2center=0.7, max_height=1):
    wall_width_int = max(int(wall_width / terrain.horizontal_scale), 1)
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    height2width_ratio = max_height_int / wall_width_int
    xs = np.arange(slope_start, terrain_length)
    heights = (height2width_ratio * (xs - slope_start)).clip(max=max_height_int).astype(np.int16)
    terrain.height_field_raw[slope_start:terrain_length, :] = heights[:, None]
    terrain.slope_vector = np.array([wall_width_int*terrain.horizontal_scale, 0., max_height]).astype(np.float32)
    terrain.slope_vector /= np.linalg.norm(terrain.slope_vector)
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')

def half_platform_terrain(terrain, start2center=2, max_height=1):
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    terrain.height_field_raw[:, :] = max_height_int
    terrain.height_field_raw[-slope_start:slope_start, -slope_start:slope_start] = 0
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')

def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=-1):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    def get_rand_dis_int(scale):
        return np.random.randint(int(- scale / terrain.horizontal_scale + 1), int(scale / terrain.horizontal_scale))
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance - get_rand_dis_int(0.2))
            terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance + get_rand_dis_int(0.2)
            start_y += stone_size + stone_distance + get_rand_dis_int(0.2)
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain

def convert_heightfield_to_trimesh_delatin(height_field_raw, horizontal_scale, vertical_scale, max_error=0.01):
    mesh = Delatin(np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=max_error)
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles

def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    move_x = np.zeros((num_rows, num_cols))
    move_y = np.zeros((num_rows, num_cols))
    
    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0