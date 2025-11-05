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


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", "plane"]:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(
            cfg.terrain_proportions
        )
        self.proportions = [
            np.sum(cfg.terrain_proportions[: i + 1])
            for i in range(len(cfg.terrain_proportions))
        ]
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
        # self.env_slope_vec = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals, 3))
        self.num_goals = cfg.num_goals

        # Store URDF hurdle positions for each environment: dict mapping (row, col) -> list of hurdles
        # Each hurdle is a tuple: (x, y, z, urdf_file)
        self.urdf_hurdles = {}

        # Store geometric gate obstacles for each environment: dict mapping (row, col) -> list of gates
        # Each gate is a dict with keys: x, y, z, height, gate_width, gate_depth, post_thickness
        self.gate_obstacles_dict = {}

        # Store procedural H-shaped hurdles for each environment: dict mapping (row, col) -> list of h_hurdles
        # Each h_hurdle is a dict with complete component information
        self.h_hurdles_dict = {}

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
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

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            print("Converting heightmap to trimesh...")
            if cfg.hf2mesh_method == "grid":
                self.vertices, self.triangles, self.x_edge_mask = (
                    convert_heightfield_to_trimesh(
                        self.height_field_raw,
                        self.cfg.horizontal_scale,
                        self.cfg.vertical_scale,
                        self.cfg.slope_treshold,
                    )
                )
                half_edge_width = int(
                    self.cfg.edge_width_thresh / self.cfg.horizontal_scale
                )
                structure = np.ones((half_edge_width * 2 + 1, 1))
                self.x_edge_mask = binary_dilation(
                    self.x_edge_mask, structure=structure
                )
                if self.cfg.simplify_grid:
                    mesh_simplifier = pyfqmr.Simplify()
                    mesh_simplifier.setMesh(self.vertices, self.triangles)
                    mesh_simplifier.simplify_mesh(
                        target_count=int(0.05 * self.triangles.shape[0]),
                        aggressiveness=7,
                        preserve_border=True,
                        verbose=10,
                    )

                    self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                    self.vertices = self.vertices.astype(np.float32)
                    self.triangles = self.triangles.astype(np.uint32)
            else:
                assert (
                    cfg.hf2mesh_method == "fast"
                ), "Height field to mesh method must be grid or fast"
                self.vertices, self.triangles = convert_heightfield_to_trimesh_delatin(
                    self.height_field_raw,
                    self.cfg.horizontal_scale,
                    self.cfg.vertical_scale,
                    max_error=cfg.max_error,
                )
            print("Created {} vertices".format(self.vertices.shape[0]))
            print("Created {} triangles".format(self.triangles.shape[0]))

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            # difficulty = np.random.choice([0.5, 0.75, 0.9])
            difficulty = np.random.uniform(-0.2, 1.2)
            terrain = self.make_terrain(choice, difficulty, row=i, col=j)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self, random=False, max_difficulty=False):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = (
                    i / (self.cfg.num_rows - 1) if self.cfg.num_rows > 1 else 0.0
                )
                choice = j / self.cfg.num_cols + 0.001
                if random:
                    if max_difficulty:
                        terrain = self.make_terrain(
                            choice, np.random.uniform(0.7, 1), row=i, col=j
                        )
                    else:
                        terrain = self.make_terrain(
                            choice, np.random.uniform(0, 1), row=i, col=j
                        )
                else:
                    terrain = self.make_terrain(choice, difficulty, row=i, col=j)

                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop("type")
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def add_roughness(self, terrain, difficulty=1):
        max_height = (
            self.cfg.height[1] - self.cfg.height[0]
        ) * difficulty + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        terrain_utils.random_uniform_terrain(
            terrain,
            min_height=-height,
            max_height=height,
            step=0.005,
            downsampled_scale=self.cfg.downsampled_scale,
        )

    def make_terrain(self, choice, difficulty, row=None, col=None):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.length_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        num_hurdles = 4
        height_range = [0.20, 0.50]
        x_spacing = [2.0, 2.5]
        y_spacing = [0.0, 0.0]
        post_spacing = 0.5
        use_progressive = True
        progressive_sequence_override = None

        schedule = getattr(self.cfg, "curriculum_schedule", None)
        if schedule is not None and row is not None:
            level_idx = int(np.clip(row, 0, len(schedule) - 1))
            level_cfg = schedule[level_idx]

            jump_cols = set(getattr(self.cfg, "jump_columns", [0, 1]))
            crawl_cols = set(getattr(self.cfg, "crawl_columns", [2, 3]))
            strategy_key = "default"
            if col is not None:
                if col in jump_cols:
                    strategy_key = "jump"
                elif col in crawl_cols:
                    strategy_key = "crawl"

            strategy_cfg = None
            if isinstance(level_cfg, dict):
                strategy_cfg = level_cfg.get(strategy_key) or level_cfg.get("default")

            if strategy_cfg:
                seq = strategy_cfg.get("sequence")
                if seq:
                    progressive_sequence_override = list(seq)
                    num_hurdles = len(progressive_sequence_override)
                    if num_hurdles == 0:
                        progressive_sequence_override = [height_range[0]]
                        num_hurdles = 1
                    height_range = [
                        min(progressive_sequence_override),
                        max(progressive_sequence_override),
                    ]
                    use_progressive = True
                else:
                    height_range = strategy_cfg.get("height_range", height_range)
                    use_progressive = strategy_cfg.get("progressive", use_progressive)

                x_spacing = strategy_cfg.get("x_spacing", x_spacing)
                y_spacing = strategy_cfg.get("y_range", y_spacing)
                post_spacing = strategy_cfg.get("post_spacing", post_spacing)
                num_hurdles = max(1, strategy_cfg.get("num_hurdles", num_hurdles))

        # H型栏杆地形 - 唯一的地形类型
        if choice < self.proportions[0]:
            idx = 0
            # H型栏杆：两根立柱 + 悬空横梁
            # 优化：支持play.py中的固定高度演示配置
            demo_heights = getattr(self.cfg, "demo_heights", None)
            demo_progressive_cols = getattr(self.cfg, "demo_progressive_cols", [])

            # 确定当前列的高度配置
            if demo_heights is not None and col is not None and col < len(demo_heights):
                demo_height = demo_heights[col]
                if demo_height is not None:
                    # 固定高度列
                    use_height_range = [demo_height, demo_height]
                    use_progressive = False
                elif col in demo_progressive_cols:
                    # 递进混合列：200-300-400-500mm
                    use_height_range = [0.20, 0.50]
                    use_progressive = True
                else:
                    # 回退到默认课程模式
                    use_height_range = height_range
                    use_progressive = False
            else:
                # 兼容旧版：使用原有的override逻辑
                override_height = getattr(self.cfg, "demo_passable_height", None)
                override_progressive = getattr(self.cfg, "demo_progressive", None)
                use_height_range = (
                    [override_height, override_height]
                    if override_height is not None
                    else height_range
                )
                use_progressive = (
                    bool(override_progressive)
                    if override_progressive is not None
                    else False
                )

            h_hurdle_terrain(
                terrain,
                num_hurdles=num_hurdles,
                total_goals=self.num_goals,
                x_range=x_spacing,
                y_range=y_spacing,
                height_range=use_height_range,
                pad_height=0,
                progressive_heights=use_progressive,
                post_spacing=post_spacing,
                crossbar_inset=0.05,
                progressive_sequence=progressive_sequence_override,
            )
            self.add_roughness(terrain)
        else:
            # 备用：如果出现意外，默认也使用h_hurdle（与上面逻辑一致）
            idx = 0
            demo_heights = getattr(self.cfg, "demo_heights", None)
            demo_progressive_cols = getattr(self.cfg, "demo_progressive_cols", [])

            if demo_heights is not None and col is not None and col < len(demo_heights):
                demo_height = demo_heights[col]
                if demo_height is not None:
                    use_height_range = [demo_height, demo_height]
                    use_progressive = False
                elif col in demo_progressive_cols:
                    use_height_range = [0.20, 0.50]
                    use_progressive = True
                else:
                    use_height_range = height_range
                    use_progressive = False
            else:
                override_height = getattr(self.cfg, "demo_passable_height", None)
                override_progressive = getattr(self.cfg, "demo_progressive", None)
                use_height_range = (
                    [override_height, override_height]
                    if override_height is not None
                    else height_range
                )
                use_progressive = (
                    bool(override_progressive)
                    if override_progressive is not None
                    else False
                )

            h_hurdle_terrain(
                terrain,
                num_hurdles=num_hurdles,
                total_goals=self.num_goals,
                x_range=x_spacing,
                y_range=y_spacing,
                height_range=use_height_range,
                pad_height=0,
                progressive_heights=use_progressive,
                post_spacing=post_spacing,
                crossbar_inset=0.05,
                progressive_sequence=progressive_sequence_override,
            )
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
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        # env_origin_x = (i + 0.5) * self.env_length
        env_origin_x = i * self.env_length + 1.0
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int(
            (self.env_length / 2.0 - 0.5) / terrain.horizontal_scale
        )  # within 1 meter square range
        x2 = int((self.env_length / 2.0 + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 0.5) / terrain.horizontal_scale)
        if self.cfg.origin_zero_z:
            env_origin_z = 0
        else:
            env_origin_z = (
                np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
            )
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_type[i, j] = terrain.idx
        self.goals[i, j, :, :2] = terrain.goals + [
            i * self.env_length,
            j * self.env_width,
        ]
        # self.env_slope_vec[i, j] = terrain.slope_vector

        # Store URDF hurdle positions if they exist
        if hasattr(terrain, "hurdle_positions") and terrain.hurdle_positions:
            # Transform hurdle positions to world coordinates
            hurdles_world = []
            for (
                hurdle_x,
                hurdle_y,
                hurdle_z,
                hurdle_height,
                urdf_file,
            ) in terrain.hurdle_positions:
                world_x = hurdle_x + i * self.env_length
                world_y = hurdle_y + j * self.env_width
                world_z = hurdle_z + env_origin_z
                hurdles_world.append(
                    (world_x, world_y, world_z, hurdle_height, urdf_file)
                )
            self.urdf_hurdles[(i, j)] = hurdles_world

        # Store geometric gate obstacles if they exist
        if hasattr(terrain, "gate_obstacles") and terrain.gate_obstacles:
            # Transform gate positions to world coordinates
            gates_world = []
            for gate_info in terrain.gate_obstacles:
                gate_world = gate_info.copy()
                gate_world["x"] = gate_info["x"] + i * self.env_length
                gate_world["y"] = gate_info["y"] + j * self.env_width
                gate_world["z"] = gate_info["z"] + env_origin_z
                gates_world.append(gate_world)
            self.gate_obstacles_dict[(i, j)] = gates_world

        # Store procedural H-shaped hurdles if they exist
        if hasattr(terrain, "h_hurdles") and terrain.h_hurdles:
            # Transform hurdle positions to world coordinates
            hurdles_world = []

            # 【修复】获取环境网格的世界坐标偏移
            grid_origin_x = i * self.env_length
            grid_origin_y = j * self.env_width
            # env_origin_z 是机器人起始高度，已经在上面计算过了

            # # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            # # ！！！！！！   在这里添加下面这行代码   ！！！！！！
            # print(f">>> [DEBUG] 正在为环境({i}, {j})执行【深拷贝】修复逻辑。")
            # # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！

            for hurdle_info in terrain.h_hurdles:
                # 【修复】深拷贝所有嵌套字典，避免共享引用
                hurdle_world = {}

                # 【修复】转换顶层坐标为绝对世界坐标
                # h_hurdle_terrain 中生成的坐标是相对于网格单元 (i,j) 的局部坐标
                hurdle_world["x"] = hurdle_info["x"] + grid_origin_x
                hurdle_world["y"] = hurdle_info["y"] + grid_origin_y
                hurdle_world["z"] = hurdle_info["z"] + env_origin_z

                # 复制尺寸参数（不需要坐标转换）
                hurdle_world["height"] = hurdle_info["height"]
                hurdle_world["post_spacing"] = hurdle_info["post_spacing"]

                # 【修复】深拷贝并转换 posts 子字典的坐标
                hurdle_world["posts"] = {
                    "radius": hurdle_info["posts"]["radius"],
                    "height": hurdle_info["posts"]["height"],
                    "left_y": hurdle_info["posts"]["left_y"]
                    + grid_origin_y,  # 转换Y坐标
                    "right_y": hurdle_info["posts"]["right_y"]
                    + grid_origin_y,  # 转换Y坐标
                    "color": hurdle_info["posts"]["color"],
                }

                # 【修复】深拷贝 crossbar 子字典（横梁不需要额外坐标转换）
                hurdle_world["crossbar"] = {
                    "radius": hurdle_info["crossbar"]["radius"],
                    "length": hurdle_info["crossbar"]["length"],
                    "height": hurdle_info["crossbar"]["height"],
                    "inset": hurdle_info["crossbar"]["inset"],
                    "color": hurdle_info["crossbar"]["color"],
                }

                # 【修复】深拷贝 bottom_bar 子字典（如果存在）
                if "bottom_bar" in hurdle_info:
                    hurdle_world["bottom_bar"] = {
                        "radius": hurdle_info["bottom_bar"]["radius"],
                        "length": hurdle_info["bottom_bar"]["length"],
                        "height": hurdle_info["bottom_bar"]["height"],
                        "offset_x": hurdle_info["bottom_bar"]["offset_x"],
                        "color": hurdle_info["bottom_bar"]["color"],
                    }

                hurdles_world.append(hurdle_world)
            self.h_hurdles_dict[(i, j)] = hurdles_world


def gap_terrain(terrain, gap_size, platform_size=1.0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[
        center_x - x2 : center_x + x2, center_y - y2 : center_y + y2
    ] = -1000
    terrain.height_field_raw[
        center_x - x1 : center_x + x1, center_y - y1 : center_y + y1
    ] = 0


def gap_parkour_terrain(terrain, difficulty, platform_size=2.0):
    gap_size = 0.1 + 0.3 * difficulty
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[
        center_x - x2 : center_x + x2, center_y - y2 : center_y + y2
    ] = -400
    terrain.height_field_raw[
        center_x - x1 : center_x + x1, center_y - y1 : center_y + y1
    ] = 0

    slope_angle = 0.1 + difficulty * 1
    offset = 1 + 9 * difficulty  # 10
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


def parkour_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_stones=8,
    x_range=[1.8, 1.9],
    y_range=[0.0, 0.1],
    z_range=[-0.2, 0.2],
    stone_len=1.0,
    stone_width=0.6,
    pad_width=0.1,
    pad_height=0.5,
    incline_height=0.1,
    last_incline_height=0.6,
    last_stone_len=1.6,
    pit_depth=[0.5, 1.0],
):
    # 1st dimension: x, 2nd dimension: y
    goals = np.zeros((num_stones + 2, 2))
    terrain.height_field_raw[:] = -round(
        np.random.uniform(pit_depth[0], pit_depth[1]) / terrain.vertical_scale
    )

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
    goals[0] = [platform_len - stone_len // 2, mid_y]
    left_right_flag = np.random.randint(0, 2)
    # dis_z = np.random.randint(dis_z_min, dis_z_max)
    dis_z = 0

    for i in range(num_stones):
        dis_x += np.random.randint(dis_x_min, dis_x_max)
        pos_neg = round(2 * (left_right_flag - 0.5))
        dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
        if i == num_stones - 1:
            dis_x += last_stone_len // 4
            heights = (
                np.tile(
                    np.linspace(-last_incline_height, last_incline_height, stone_width),
                    (last_stone_len, 1),
                )
                * pos_neg
            )
            terrain.height_field_raw[
                dis_x - last_stone_len // 2 : dis_x + last_stone_len // 2,
                dis_y - stone_width // 2 : dis_y + stone_width // 2,
            ] = (
                heights.astype(int) + dis_z
            )
        else:
            heights = (
                np.tile(
                    np.linspace(-incline_height, incline_height, stone_width),
                    (stone_len, 1),
                )
                * pos_neg
            )
            terrain.height_field_raw[
                dis_x - stone_len // 2 : dis_x + stone_len // 2,
                dis_y - stone_width // 2 : dis_y + stone_width // 2,
            ] = (
                heights.astype(int) + dis_z
            )

        goals[i + 1] = [dis_x, dis_y]

        left_right_flag = 1 - left_right_flag
    final_dis_x = dis_x + 2 * np.random.randint(dis_x_min, dis_x_max)
    final_platform_start = (
        dis_x + last_stone_len // 2 + round(0.05 // terrain.horizontal_scale)
    )
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


def parkour_gap_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_gaps=8,
    gap_size=0.3,
    x_range=[1.6, 2.4],
    y_range=[-1.2, 1.2],
    half_valid_width=[0.6, 1.2],
    gap_depth=-200,
    pad_width=0.1,
    pad_height=0.5,
    flat=False,
):
    goals = np.zeros((num_gaps + 2, 2))
    # terrain.height_field_raw[:] = -200
    # import ipdb; ipdb.set_trace()
    mid_y = terrain.length // 2  # length is actually y width

    # dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    # dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    gap_depth = -round(
        np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale
    )

    # half_gap_width = round(np.random.uniform(0.6, 1.2) / terrain.horizontal_scale)
    half_valid_width = round(
        np.random.uniform(half_valid_width[0], half_valid_width[1])
        / terrain.horizontal_scale
    )
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
            terrain.height_field_raw[
                dis_x - gap_size // 2 : dis_x + gap_size // 2, :
            ] = gap_depth

        terrain.height_field_raw[
            last_dis_x:dis_x, : mid_y + rand_y - half_valid_width
        ] = gap_depth
        terrain.height_field_raw[
            last_dis_x:dis_x, mid_y + rand_y + half_valid_width :
        ] = gap_depth

        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
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


def parkour_hurdle_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_stones=8,
    stone_len=0.01,
    x_range=[1.5, 2.4],
    y_range=[-0.4, 0.4],
    half_valid_width=[0.4, 0.8],
    hurdle_height_range=[0.2, 0.6],
    pad_width=0.1,
    pad_height=0.5,
    flat=False,
):
    goals = np.zeros((num_stones + 2, 2))
    # terrain.height_field_raw[:] = -200

    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
    half_valid_width = round(
        np.random.uniform(half_valid_width[0], half_valid_width[1])
        / terrain.horizontal_scale
    )
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
            terrain.height_field_raw[
                dis_x - stone_len // 2 : dis_x + stone_len // 2,
            ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            terrain.height_field_raw[
                dis_x - stone_len // 2 : dis_x + stone_len // 2,
                : mid_y + rand_y - half_valid_width,
            ] = 0
            terrain.height_field_raw[
                dis_x - stone_len // 2 : dis_x + stone_len // 2,
                mid_y + rand_y + half_valid_width :,
            ] = 0
        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
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


def parkour_step_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_stones=8,
    #    x_range=[1.5, 2.4],
    x_range=[0.2, 0.4],
    y_range=[-0.15, 0.15],
    half_valid_width=[0.45, 0.5],
    step_height=0.2,
    pad_width=0.1,
    pad_height=0.5,
):
    goals = np.zeros((num_stones + 2, 2))
    # terrain.height_field_raw[:] = -200
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round((x_range[0] + step_height) / terrain.horizontal_scale)
    dis_x_max = round((x_range[1] + step_height) / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    step_height = round(step_height / terrain.vertical_scale)

    half_valid_width = round(
        np.random.uniform(half_valid_width[0], half_valid_width[1])
        / terrain.horizontal_scale
    )

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
        terrain.height_field_raw[dis_x : dis_x + rand_x,] = stair_height
        dis_x += rand_x
        terrain.height_field_raw[
            last_dis_x:dis_x, : mid_y + rand_y - half_valid_width
        ] = 0
        terrain.height_field_raw[
            last_dis_x:dis_x, mid_y + rand_y + half_valid_width :
        ] = 0

        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
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


import numpy as np


def h_hurdle_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_hurdles=4,
    total_goals=None,
    x_range=[1.5, 2.5],
    y_range=[0.0, 0.0],
    height_range=[0.2, 0.5],  # 可通过高度范围（上横杆下端到下横杆上端）
    pad_width=0.1,
    pad_height=0.5,
    progressive_heights=True,
    progressive_sequence=None,
    post_spacing=0.5,  # 两根立柱内侧距离（宽度）
    crossbar_inset=0.0,
):

    # 初始化目标点数组（保持数组大小一致，用于Isaac Gym环境）
    if total_goals is None:
        total_goals = num_hurdles + 2
    goals = np.zeros((total_goals, 2))

    # 【改进】添加轻微地面噪声，增加真实性和鲁棒性
    # 而不是完全平坦的地面
    terrain.height_field_raw[:] = 0

    # 添加小幅度随机噪声（±5mm）提高适应性
    noise_amplitude = 0.005 / terrain.vertical_scale  # 5mm噪声
    if noise_amplitude > 0:
        noise = np.random.uniform(
            -noise_amplitude, noise_amplitude, terrain.height_field_raw.shape
        )
        terrain.height_field_raw += noise.astype(np.int16)

    mid_y = terrain.length // 2  # 地形中心线（Y轴）

    # 转换参数到像素单位
    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len_px = round(platform_len / terrain.horizontal_scale)
    platform_height_int = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len_px, :] = platform_height_int

    # 起点位置和第一个目标点
    dis_x = platform_len_px
    goals[0] = [platform_len_px - 1, mid_y]

    # 可用的栏杆高度（标准高度）
    available_heights = [0.2, 0.3, 0.4, 0.5]
    if progressive_sequence is not None:
        progressive_sequence = list(progressive_sequence)
    elif progressive_heights:
        progressive_sequence = available_heights[: max(1, num_hurdles)]
    else:
        progressive_sequence = None

    # 初始化H型栏杆位置信息列表（将在legged_robot.py中创建实际几何体）
    h_hurdles = []

    # 几何体组件尺寸定义
    post_radius = 0.008  # 立柱半径 [米]
    # 【修复】立柱间距应该使用传入的post_spacing参数，而不是hardcode
    # post_distance = 0.5  # 旧版：硬编码，导致与横梁长度不一致

    crossbar_radius = 0.005  # 横梁半径 [米]
    crossbar_length = 0.7  # 横梁长度（与立柱间距一致）

    for i in range(num_hurdles):
        # 计算下一个栏杆的X坐标位置
        rand_x = (
            np.random.randint(dis_x_min, dis_x_max)
            if dis_x_max > dis_x_min
            else dis_x_min
        )
        dis_x += rand_x

        # 计算栏杆的Y轴偏移
        rand_y = np.random.randint(dis_y_min, dis_y_max) if dis_y_max > dis_y_min else 0

        # 选择当前栏杆的可通过高度（上横杆下端到下横杆上端的距离）
        if progressive_sequence is not None and len(progressive_sequence) > 0:
            seq_idx = min(i, len(progressive_sequence) - 1)
            passable_height = progressive_sequence[seq_idx]
        else:
            # 【优化】直接在传入的 height_range [min, max] 之间随机取值
            # 这确保了无论 height_range 是什么，都能得到一个有效的高度
            # 并且高度分布更连续（不再局限于 [0.2, 0.3, 0.4, 0.5]）
            passable_height = np.random.uniform(height_range[0], height_range[1])

        # 计算栏杆在世界坐标系中的中心位置
        hurdle_x = dis_x * terrain.horizontal_scale
        hurdle_y = (mid_y + rand_y) * terrain.horizontal_scale
        hurdle_z = 0.0  # 地面高度

        # 计算立柱位置（沿Y轴分布）
        # 【修复】使用post_spacing而不是post_distance，保持与横梁长度一致
        left_post_y = hurdle_y - post_spacing / 2  # 左侧立柱
        right_post_y = hurdle_y + post_spacing / 2  # 右侧立柱

        # 底部横杆参数
        bottom_bar_height = 0.05  # 底部横杆中心高度（5cm）
        bottom_bar_offset_x = 0.0  # 底部横杆在X轴前移0cm（避免与立柱连接成墙）
        bottom_bar_radius = 0.005  # 底部横杆半径（0.5cm）
        # 【修复】底部横杆长度应该与立柱间距一致
        bottom_bar_length = post_spacing * 0.5

        # 【关键修正】根据可通过高度计算顶部横杆和立柱高度
        # 定义：可通过高度 = 上横杆下端 - 下横杆上端
        # 即：passable_height = (crossbar_height - crossbar_radius) - (bottom_bar_height + bottom_bar_radius)
        # 推导：crossbar_height = passable_height + bottom_bar_height + bottom_bar_radius + crossbar_radius
        crossbar_height = (
            passable_height + bottom_bar_height + bottom_bar_radius + crossbar_radius
        )

        # 立柱高度到横杆中心（即crossbar_height）
        post_height = crossbar_height

        # 存储H型栏杆的完整几何信息
        hurdle_info = {
            "x": hurdle_x,
            "y": hurdle_y,
            "z": hurdle_z,
            "height": post_height,  # 栏杆总高度（立柱高度）
            "passable_height": passable_height,  # 可通过高度（上横杆下端到下横杆上端）
            "post_spacing": post_spacing,  # 立柱内侧间距（宽度）
            # 立柱信息（两根垂直圆柱）
            "posts": {
                "radius": post_radius,
                "height": post_height,  # 立柱高度
                "left_y": left_post_y,  # 左侧立柱的Y坐标
                "right_y": right_post_y,  # 右侧立柱的Y坐标
                "color": [0.3, 0.3, 0.3],  # 深灰色
            },
            # 顶部横梁信息（悬空的水平圆柱，与立柱顶端平齐）
            "crossbar": {
                "radius": crossbar_radius,
                "length": crossbar_length,  # 横梁长度（等于立柱间距）
                "height": crossbar_height,  # 横梁Z坐标（中心高度）
                "inset": crossbar_inset,  # 横梁向内缩进距离
                "color": [0.8, 0.2, 0.2],  # 红色
            },
            # 底部横杆信息（用来绊倒机器人，与横梁平行）
            "bottom_bar": {
                "radius": bottom_bar_radius,
                "length": bottom_bar_length,  # 底部横杆长度（等于立柱间距）
                "height": bottom_bar_height,  # 底部横杆中心高度（5cm）
                "offset_x": bottom_bar_offset_x,  # 在X轴偏移
                "color": [0.2, 0.8, 0.2],  # 绿色
            },
        }
        h_hurdles.append(hurdle_info)

        # 设置目标点（在栏杆中心，引导机器人穿过）
        goals[i + 1] = [dis_x, mid_y + rand_y]

    # 最终目标点（放在最后一个栏杆之后）
    final_rand_x = (
        np.random.randint(dis_x_min, dis_x_max) if dis_x_max > dis_x_min else dis_x_min
    )
    final_dis_x = dis_x + final_rand_x
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[num_hurdles + 1] = [final_dis_x, mid_y]

    # 填充剩余目标点（如果total_goals > num_hurdles + 2）
    for i in range(num_hurdles + 2, total_goals):
        goals[i] = [final_dis_x, mid_y]

    # 转换目标点坐标到米制
    terrain.goals = goals * terrain.horizontal_scale

    # 将H型栏杆信息存储到terrain对象（供legged_robot.py使用）
    terrain.h_hurdles = h_hurdles

    # 地形边缘填充（防止机器人掉出边界）
    pad_width_px = int(pad_width // terrain.horizontal_scale)
    pad_height_px = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width_px] = pad_height_px
    terrain.height_field_raw[:, -pad_width_px:] = pad_height_px
    terrain.height_field_raw[:pad_width_px, :] = pad_height_px
    terrain.height_field_raw[-pad_width_px:, :] = pad_height_px


def demo_terrain(terrain):
    goals = np.zeros((8, 2))
    mid_y = terrain.length // 2

    # hurdle
    platform_length = round(2 / terrain.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / terrain.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / terrain.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[0] = [platform_length + hurdle_depth / 2, mid_y]
    terrain.height_field_raw[
        platform_length : platform_length + hurdle_depth,
        round(mid_y - hurdle_width / 2) : round(mid_y + hurdle_width / 2),
    ] = hurdle_height

    # step up
    platform_length += round(np.random.uniform(1.5, 2.5) / terrain.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    first_step_height = round(np.random.uniform(0.35, 0.45) / terrain.vertical_scale)
    first_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[1] = [platform_length + first_step_depth / 2, mid_y]
    terrain.height_field_raw[
        platform_length : platform_length + first_step_depth,
        round(mid_y - first_step_width / 2) : round(mid_y + first_step_width / 2),
    ] = first_step_height

    platform_length += first_step_depth
    second_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    second_step_height = first_step_height
    second_step_width = first_step_width
    goals[2] = [platform_length + second_step_depth / 2, mid_y]
    terrain.height_field_raw[
        platform_length : platform_length + second_step_depth,
        round(mid_y - second_step_width / 2) : round(mid_y + second_step_width / 2),
    ] = second_step_height

    # gap
    platform_length += second_step_depth
    gap_size = round(np.random.uniform(0.5, 0.8) / terrain.horizontal_scale)

    # step down
    platform_length += gap_size
    third_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    third_step_height = first_step_height
    third_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[3] = [platform_length + third_step_depth / 2, mid_y]
    terrain.height_field_raw[
        platform_length : platform_length + third_step_depth,
        round(mid_y - third_step_width / 2) : round(mid_y + third_step_width / 2),
    ] = third_step_height

    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    forth_step_height = first_step_height
    forth_step_width = third_step_width
    goals[4] = [platform_length + forth_step_depth / 2, mid_y]
    terrain.height_field_raw[
        platform_length : platform_length + forth_step_depth,
        round(mid_y - forth_step_width / 2) : round(mid_y + forth_step_width / 2),
    ] = forth_step_height

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

    goals[5] = [platform_length + slope_depth / 2, left_y]
    heights = (
        np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1))
        * 1
    )
    terrain.height_field_raw[
        platform_length : platform_length + slope_depth,
        left_y - slope_width // 2 : left_y + slope_width // 2,
    ] = (
        heights.astype(int) + platform_height
    )

    platform_length += slope_depth + gap_size
    goals[6] = [platform_length + slope_depth / 2, right_y]
    heights = (
        np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1))
        * -1
    )
    terrain.height_field_raw[
        platform_length : platform_length + slope_depth,
        right_y - slope_width // 2 : right_y + slope_width // 2,
    ] = (
        heights.astype(int) + platform_height
    )

    platform_length += slope_depth + gap_size + round(0.4 / terrain.horizontal_scale)
    goals[-1] = [platform_length, left_y]
    terrain.goals = goals * terrain.horizontal_scale


def pit_terrain(terrain, depth, platform_size=1.0):
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
    heights = (
        (height2width_ratio * (xs - slope_start))
        .clip(max=max_height_int)
        .astype(np.int16)
    )
    terrain.height_field_raw[slope_start:terrain_length, :] = heights[:, None]
    terrain.slope_vector = np.array(
        [wall_width_int * terrain.horizontal_scale, 0.0, max_height]
    ).astype(np.float32)
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


def stepping_stones_terrain(
    terrain, stone_size, stone_distance, max_height, platform_size=1.0, depth=-1
):
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
        return np.random.randint(
            int(-scale / terrain.horizontal_scale + 1),
            int(scale / terrain.horizontal_scale),
        )

    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height - 1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance - get_rand_dis_int(0.2))
            terrain.height_field_raw[0:stop_x, start_y:stop_y] = np.random.choice(
                height_range
            )
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = (
                    np.random.choice(height_range)
                )
                start_x += stone_size + stone_distance + get_rand_dis_int(0.2)
            start_y += stone_size + stone_distance + get_rand_dis_int(0.2)
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x:stop_x, 0:stop_y] = np.random.choice(
                height_range
            )
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = (
                    np.random.choice(height_range)
                )
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain


def convert_heightfield_to_trimesh_delatin(
    height_field_raw, horizontal_scale, vertical_scale, max_error=0.01
):
    mesh = Delatin(
        np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=max_error
    )
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles


def convert_heightfield_to_trimesh(
    height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None
):
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

    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += (
            hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        )
        move_x[1:num_rows, :] -= (
            hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        )
        move_y[:, : num_cols - 1] += (
            hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        )
        move_y[:, 1:num_cols] -= (
            hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        )
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1]
            > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols]
            > slope_threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3

    return vertices, triangles, move_x != 0
