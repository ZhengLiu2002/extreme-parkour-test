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
from collections import deque
import numpy as np
import os
import copy

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
    x = quat_angle[:, 0]
    y = quat_angle[:, 1]
    z = quat_angle[:, 2]
    w = quat_angle[:, 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


class LeggedRobot(BaseTask):
    def __init__(
        self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless
    ):
        """Parses the provided config file,
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
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.resize_transform = torchvision.transforms.Resize(
            (self.cfg.depth.resized[1], self.cfg.depth.resized[0]),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        )

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
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = self.reindex(actions)

        actions.to(self.device)
        self.action_history_buf = torch.cat(
            [self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1
        )
        if self.cfg.domain_rand.action_delay:
            if (
                self.global_counter % self.cfg.domain_rand.delay_update_global_steps
                == 0
            ):
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = torch.tensor(
                        self.cfg.domain_rand.action_curr_step.pop(0),
                        device=self.device,
                        dtype=torch.float,
                    )
            if self.viewer:
                self.delay = torch.tensor(
                    self.cfg.domain_rand.action_delay_view,
                    device=self.device,
                    dtype=torch.float,
                )
            indices = -self.delay - 1
            actions = self.action_history_buf[:, indices.long()]  # delay for 1/50=20ms

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = (
            self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        )
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        self.extras["delta_yaw_ok"] = self.delta_yaw < 0.6
        if (
            self.cfg.depth.use_camera
            and self.global_counter % self.cfg.depth.update_interval == 0
        ):
            self.extras["depth"] = self.depth_buffer[
                :, -2
            ]  # have already selected last one
        else:
            self.extras["depth"] = None
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def get_history_observations(self):
        return self.obs_history_buf

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (
            self.cfg.depth.far_clip - self.cfg.depth.near_clip
        ) - 0.5
        return depth_image

    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1) - 0.5)[0]
        depth_image = torch.clip(
            depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip
        )
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        self.gym.step_graphics(self.sim)  # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[i], self.cam_handles[i], gymapi.IMAGE_DEPTH
            )

            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack(
                    [depth_image] * self.cfg.depth.buffer_len, dim=0
                )
            else:
                self.depth_buffer[i] = torch.cat(
                    [
                        self.depth_buffer[i, 1:],
                        depth_image.to(self.device).unsqueeze(0),
                    ],
                    dim=0,
                )

        self.gym.end_access_image_tensors(self.sim)

    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0

        self.reached_goal_ids = (
            torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1)
            < self.cfg.env.next_goal_threshold
        )
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
        """check terminations, compute observations and rewards
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
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.base_lin_acc = (
            self.root_states[:, 7:10] - self.last_root_vel[:, :3]
        ) / self.dt

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        prev_contacts = self.last_contacts.clone()
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.0

        prev_air_time = self.feet_air_time.clone()
        self.feet_air_time += self.dt
        self.feet_air_time *= ~contact

        first_contact = (~prev_contacts) & contact
        self.feet_last_air_time.zero_()
        self.feet_last_air_time += first_contact.float() * prev_air_time

        self.contact_filt = torch.logical_or(contact, prev_contacts)
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

        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            # self._draw_height_samples()
            self._draw_goals()
            self._draw_feet()
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(
                    "Depth Image",
                    self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5,
                )
                cv2.waitKey(1)

    def reindex_feet(self, vec):
        return vec[:, [1, 0, 3, 2]]

    def reindex(self, vec):
        return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]

    def _trapezoid_shaping(self, value, start, peak_start, peak_end, end):
        """梯形奖励成形函数，start < peak_start < peak_end < end."""

        denom_up = max(peak_start - start, 1.0e-6)
        denom_down = max(end - peak_end, 1.0e-6)

        result = torch.zeros_like(value)

        ascending = (value >= start) & (value < peak_start)
        result[ascending] = (value[ascending] - start) / denom_up

        plateau = (value >= peak_start) & (value <= peak_end)
        result[plateau] = 1.0

        descending = (value > peak_end) & (value < end)
        result[descending] = 1.0 - (value[descending] - peak_end) / denom_down

        return torch.clamp(result, 0.0, 1.0)

    def _check_stuck(self):
        """
        检测机器人是否被栏杆卡住

        判断标准（更严格，减少误判）：
        1. 机器人X轴前进距离很小（只检查前进方向）
        2. 持续较长时间（避免误判正常调整姿态）
        3. 同时有较大的接触力（说明确实被卡住而不是在等待）

        Returns:
            torch.Tensor: 布尔张量，指示哪些环境的机器人被卡住
        """
        # 获取配置参数
        stuck_distance_threshold = getattr(
            self.cfg.env, "stuck_distance_threshold", 0.02
        )  # 2cm (更严格，只检查X轴前进)
        stuck_time_threshold = getattr(
            self.cfg.env, "stuck_time_threshold", 200
        )  # 200步 (约1秒 @ 200Hz，减少误判)

        # 计算当前位置
        current_position = self.root_states[:, :3]

        # 只检查X轴（前进方向）的位置变化
        forward_distance = torch.abs(current_position[:, 0] - self.last_position[:, 0])

        # 检测是否前进很少
        is_barely_moving_forward = forward_distance < stuck_distance_threshold

        # 【新增】检查是否有较大的身体接触力（被卡住的信号）
        # 如果没有接触力，说明机器人可能只是在调整姿态，不是被卡住
        if (
            hasattr(self, "penalised_contact_indices")
            and len(self.penalised_contact_indices) > 0
        ):
            body_contact_forces = self.contact_forces[
                :, self.penalised_contact_indices, :
            ]
            contact_force_magnitude = torch.norm(body_contact_forces, dim=-1).sum(dim=1)
            has_contact = contact_force_magnitude > 10.0  # 有明显接触力
        else:
            has_contact = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )

        # 只有同时满足"不动"和"有接触力"才计数
        is_stuck_candidate = is_barely_moving_forward & has_contact

        # 更新计数器
        self.stuck_detection_counter[is_stuck_candidate] += 1
        self.stuck_detection_counter[~is_stuck_candidate] = 0

        # 每隔一段时间更新参考位置（而不是每步都更新）
        # 这样可以更好地检测长期趋势
        if self.common_step_counter % 10 == 0:  # 每10步更新一次参考位置
            self.last_position[:] = current_position

        # 判断是否卡住（持续时间超过阈值）
        is_stuck = self.stuck_detection_counter > stuck_time_threshold
        self.current_stuck[:] = is_stuck

        termination_multiplier = getattr(
            self.cfg.env, "stuck_termination_multiplier", 3.0
        )
        termination_steps = int(stuck_time_threshold * termination_multiplier)
        termination_steps = max(termination_steps, stuck_time_threshold + 1)
        terminate_mask = self.stuck_detection_counter > termination_steps

        return terminate_mask

    def check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        roll_cutoff = torch.abs(self.roll) > 1.5
        pitch_cutoff = torch.abs(self.pitch) > 1.5
        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals
        height_cutoff = self.root_states[:, 2] < -0.25

        # 【新增】卡住检测
        stuck_cutoff = self._check_stuck()
        self.extras["is_stuck"] = self.current_stuck

        trench_cutoff = torch.zeros_like(self.reset_buf, dtype=torch.bool)
        walkway_width = getattr(self.cfg.terrain, "walkway_width", 0.0)
        trench_depth = getattr(self.cfg.terrain, "trench_depth", 0.0)
        if walkway_width > 0.0 and trench_depth > 0.0:
            walkway_half = walkway_width * 0.5 + 0.05  # 给机器人略微的容错区间
            lateral_offset = torch.abs(self.root_states[:, 1] - self.env_origins[:, 1])

            expected_height = getattr(self.cfg.init_state, "pos", [0.0, 0.0, 0.0])[2]
            height_threshold = expected_height - trench_depth * 0.8
            # 若课程配置导致高度阈值过低，保留一个最小值
            height_threshold = min(expected_height, max(0.05, height_threshold))
            z_relative = self.root_states[:, 2] - self.env_origins[:, 2]

            trench_cutoff = (lateral_offset > walkway_half) & (
                z_relative < height_threshold
            )

        self.extras["fell_in_trench"] = trench_cutoff

        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs
        self.time_out_buf |= reach_goal_cutoff

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= height_cutoff
        self.reset_buf |= stuck_cutoff
        self.reset_buf |= trench_cutoff

    def reset_idx(self, env_ids):
        """Reset selected environments and更新相关缓冲区。"""
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_torques[env_ids] = 0.0
        self.last_root_vel[:] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.feet_last_air_time[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        if hasattr(self, "contact_filt"):
            self.contact_filt[env_ids] = False
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.0  # reset obs history buffer TODO no 0s
        self.contact_buf[env_ids, :, :] = 0.0
        self.action_history_buf[env_ids, :, :] = 0.0
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0

        # 【新增】重置卡住检测相关的缓冲区
        self.stuck_detection_counter[env_ids] = 0
        self.last_position[env_ids] = self.root_states[env_ids, :3]
        self.current_stuck[env_ids] = False

        # 【新增】更新重置环境的静态障碍物信息
        # 当环境重置时，获取该环境对应的 (row, col) 的栏杆信息
        # 并将其存储到 static_hurdle_info 中，供 compute_observations 使用
        if len(env_ids) > 0:
            for env_id in env_ids:
                # 获取当前环境 (row, col)
                row = self.terrain_levels[env_id].item()
                col = self.terrain_types[env_id].item()

                # 获取当前环境 (row, col) 对应的栏杆列表
                hurdles_world = self.terrain.h_hurdles_dict.get((row, col), [])

                # 构建栏杆数据 [x, y, height]
                hurdle_data = []
                for k in range(4):  # 固定4个栏杆
                    if k < len(hurdles_world):
                        hurdle_data.append(
                            [
                                hurdles_world[k]["x"],
                                hurdles_world[k]["y"],
                                hurdles_world[k]["height"],
                            ]
                        )
                    else:
                        # 如果栏杆少于4个，用0填充（不应该发生，因为我们固定了num_hurdles=4）
                        hurdle_data.append([0.0, 0.0, 0.0])

                # 更新张量
                self.static_hurdle_info[env_id] = torch.tensor(
                    hurdle_data, device=self.device, dtype=torch.float
                )

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels.float()
            )
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][
                1
            ]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

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
        obs_buf = torch.cat(
            (  # skill_vector,
                self.base_ang_vel * self.obs_scales.ang_vel,  # [1,3]
                imu_obs,  # [1,2]
                0 * self.delta_yaw[:, None],
                self.delta_yaw[:, None],
                self.delta_next_yaw[:, None],
                0 * self.commands[:, 0:2],
                self.commands[:, 0:1],  # [1,1]
                (self.env_class != 17).float()[:, None],
                (self.env_class == 17).float()[:, None],
                self.reindex(
                    (self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos
                ),
                self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                self.reindex(self.action_history_buf[:, -1]),
                self.reindex_feet(self.contact_filt.float() - 0.5),
            ),
            dim=-1,
        )
        priv_explicit = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                0 * self.base_lin_vel,
                0 * self.base_lin_vel,
            ),
            dim=-1,
        )
        # 【新增】计算障碍物特权观测 (4个栏杆 * [delta_x, delta_y, height] = 12个值)
        # 这些信息将帮助 Critic 理解环境，从而更好地指导 Actor 的学习

        # 获取机器人当前位置 [N, 2]
        robot_pos_xy = self.root_states[:, :2]

        # 获取栏杆绝对位置 [N, 4, 2]
        hurdle_abs_pos_xy = self.static_hurdle_info[:, :, :2]

        # 计算世界坐标系下的相对位置 [N, 4, 2]
        hurdle_relative_pos_xy_world = hurdle_abs_pos_xy - robot_pos_xy.unsqueeze(1)

        # 【优化】将相对位置转换到机器人局部坐标系
        # 这样机器人"看到"的障碍物位置与其朝向无关，提高学习效率
        # 使用base_quat的逆旋转将世界坐标转换为局部坐标

        # 添加z=0维度，使其成为3D向量 [N, 4, 3]
        hurdle_relative_pos_3d = torch.cat(
            [
                hurdle_relative_pos_xy_world,
                torch.zeros_like(hurdle_relative_pos_xy_world[:, :, :1]),
            ],
            dim=-1,
        )

        # 对每个栏杆应用逆旋转 [N, 4, 3]
        hurdle_relative_pos_local = torch.stack(
            [
                quat_rotate_inverse(self.base_quat, hurdle_relative_pos_3d[:, i, :])
                for i in range(4)
            ],
            dim=1,
        )

        # 只取xy分量 [N, 4, 2]
        hurdle_relative_pos_xy_local = hurdle_relative_pos_local[:, :, :2]

        # 获取栏杆高度 [N, 4, 1]
        hurdle_heights = self.static_hurdle_info[:, :, 2].unsqueeze(-1)

        # 组合成 [N, 4, 3]：局部坐标系下的 [delta_x, delta_y, height]
        obstacle_priv_obs = torch.cat(
            [hurdle_relative_pos_xy_local, hurdle_heights], dim=-1
        )

        # 展平为 [N, 12]
        obstacle_priv_obs_flat = obstacle_priv_obs.view(self.num_envs, -1)

        # 构建完整的特权潜在观测
        priv_latent = torch.cat(
            (
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[0] - 1,
                self.motor_strength[1] - 1,
                obstacle_priv_obs_flat,  # 【关键】添加障碍物信息
            ),
            dim=-1,
        )
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights,
                -1,
                1.0,
            )
            self.obs_buf = torch.cat(
                [
                    obs_buf,
                    heights,
                    priv_explicit,
                    priv_latent,
                    self.obs_history_buf.view(self.num_envs, -1),
                ],
                dim=-1,
            )
        else:
            self.obs_buf = torch.cat(
                [
                    obs_buf,
                    priv_explicit,
                    priv_latent,
                    self.obs_history_buf.view(self.num_envs, -1),
                ],
                dim=-1,
            )
        obs_buf[:, 6:8] = 0  # mask yaw in proprioceptive history
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)], dim=1),
        )

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack(
                [self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1
            ),
            torch.cat(
                [self.contact_buf[:, 1:], self.contact_filt.float().unsqueeze(1)], dim=1
            ),
        )

    def get_noisy_measurement(self, x, scale):
        if self.cfg.noise.add_noise:
            x = x + (2.0 * torch.rand_like(x) - 1) * scale * self.cfg.noise.noise_level
        return x

    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id  # required in headless mode
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        mesh_type = self.cfg.terrain.mesh_type
        start = time()
        print("*" * 80)
        print("Start creating ground...")
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]"
            )
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*" * 80)
        self._create_envs()

    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0], friction_range[1], (num_buckets, 1), device="cpu"
                )
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof,
                2,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = (
                    m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                )
                self.dof_pos_limits[i, 1] = (
                    m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                )
        return props

    def _process_rigid_body_props(self, props, env_id):
        # No need to use tensors as only called upon env creation
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1,))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros((1,))
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3,))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        mass_params = np.concatenate([rand_mass, rand_com])
        return props, mass_params

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (
            self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)
            == 0
        )
        self._resample_commands(env_ids.nonzero(as_tuple=False).flatten())

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.8 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0
            )
            self.commands[:, 2] *= (
                torch.abs(self.commands[:, 2]) > self.cfg.commands.ang_vel_clip
            )

        if self.cfg.terrain.measure_heights:
            if self.global_counter % self.cfg.depth.update_interval == 0:
                self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (
            self.common_step_counter % self.cfg.domain_rand.push_interval == 0
        ):
            self._push_robots()

    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(
            1,
            (self.cur_goal_idx[:, None, None] + future).expand(
                -1, -1, self.env_goals.shape[-1]
            ),
        ).squeeze(1)

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
            self.commands[env_ids, 2] *= (
                torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_clip
            )

        # set small commands to zero
        self.commands[env_ids, :2] *= (
            torch.abs(self.commands[env_ids, 0:1]) > self.cfg.commands.lin_vel_clip
        )

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type == "P":
            if (
                not self.cfg.domain_rand.randomize_motor
            ):  # TODO add strength to gain directly
                torques = (
                    self.p_gains
                    * (actions_scaled + self.default_dof_pos_all - self.dof_pos)
                    - self.d_gains * self.dof_vel
                )
            else:
                torques = (
                    self.motor_strength[0]
                    * self.p_gains
                    * (actions_scaled + self.default_dof_pos_all - self.dof_pos)
                    - self.motor_strength[1] * self.d_gains * self.dof_vel
                )

        elif control_type == "V":
            torques = (
                self.p_gains * (actions_scaled - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def update_command_curriculum(self, env_ids):
        """渐进式扩大命令范围，提升任务难度。

        策略：以最近若干次 episode 的目标点完成度作为进度指标，
        只有在确实穿越足够多障碍后才放宽最大前向速度，避免"原地刷时间"。
        """

        if not self.cfg.commands.curriculum:
            return

        if self.command_curriculum_buffer is None:
            self.command_curriculum_buffer = deque(maxlen=20)

        if len(env_ids) == 0:
            return

        num_goals = max(getattr(self.cfg.terrain, "num_goals", 1), 1)
        goal_completion = self.cur_goal_idx[env_ids].float() / num_goals
        goal_completion = torch.clamp(goal_completion, 0.0, 1.0)

        goal_threshold = getattr(self.cfg.commands, "curriculum_goal_threshold", 0.4)

        # 记录平均目标完成度（而非纯粹靠存活时间）
        if goal_completion.numel() > 0:
            self.command_curriculum_buffer.append(goal_completion.mean().item())
        else:
            self.command_curriculum_buffer.append(0.0)

        if len(self.command_curriculum_buffer) < self.command_curriculum_buffer.maxlen:
            return

        avg_completion = sum(self.command_curriculum_buffer) / len(
            self.command_curriculum_buffer
        )

        if avg_completion < goal_threshold:
            return

        if self._increase_command_velocity_range():
            self.command_curriculum_buffer.clear()

    def _increase_command_velocity_range(self):
        """提升前向命令上限，返回是否实际发生更新。"""

        if "lin_vel_x" not in self.command_ranges:
            return False

        current_range = self.command_ranges["lin_vel_x"]
        max_range = self.command_max_ranges.get("lin_vel_x", current_range)

        if isinstance(current_range, torch.Tensor):
            current_max = current_range[1].item()
        else:
            current_max = float(current_range[1])

        max_allowed = (
            float(max_range[1])
            if isinstance(max_range, (list, tuple))
            else float(max_range)
        )

        if current_max >= max_allowed - 1e-6:
            # 已经达到上限
            return False

        increment = 0.05
        if self.command_curriculum_increments:
            incr_val = self.command_curriculum_increments.get("lin_vel_x", increment)
            if isinstance(incr_val, (list, tuple)):
                increment = (
                    float(incr_val[1]) if len(incr_val) > 1 else float(incr_val[0])
                )
            else:
                increment = float(incr_val)

        new_max = min(current_max + increment, max_allowed)

        if isinstance(current_range, torch.Tensor):
            current_range[1] = new_max
        else:
            self.command_ranges["lin_vel_x"][1] = new_max

        # 更新 clipping 阈值，确保不把新的小命令裁零
        clip_margin = getattr(self.cfg.commands, "lin_vel_clip", 0.0)
        if clip_margin >= new_max:
            new_clip = max(new_max * 0.5, 0.01)
            setattr(self.cfg.commands, "lin_vel_clip", new_clip)

        self.command_curriculum_step += 1

        self.extras.setdefault("curriculum", {})
        self.extras["curriculum"]["lin_vel_x_max"] = new_max
        self.extras["curriculum"]["lin_vel_x_step"] = self.command_curriculum_step

        return True

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(
            -0.2, 0.2, (len(env_ids), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = 0.0

        # 【关键修复】使用正确的Actor索引 (0, 17, 34, ...) 而不是 (0, 1, 2, ...)
        # 这与 _reset_root_states 中的逻辑保持一致
        # 机器人Actor的索引是 env_id * num_actors (例如: 0, 17, 34, 51...)
        env_ids_int32 = (env_ids * self.num_actors).to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
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
                self.root_states[env_ids, :2] += torch_rand_float(
                    -0.3, 0.3, (len(env_ids), 2), device=self.device
                )  # xy position within 1m of the center
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = self.cfg.env.rand_yaw_range * torch_rand_float(
                    -1, 1, (len(env_ids), 1), device=self.device
                ).squeeze(1)
                if self.cfg.env.randomize_start_pitch:
                    rand_pitch = self.cfg.env.rand_pitch_range * torch_rand_float(
                        -1, 1, (len(env_ids), 1), device=self.device
                    ).squeeze(1)
                else:
                    rand_pitch = torch.zeros(len(env_ids), device=self.device)
                quat = quat_from_euler_xyz(0 * rand_yaw, rand_pitch, rand_yaw)
                self.root_states[env_ids, 3:7] = quat[:, :]
            if self.cfg.env.randomize_start_y:
                self.root_states[
                    env_ids, 1
                ] += self.cfg.env.rand_y_range * torch_rand_float(
                    -1, 1, (len(env_ids), 1), device=self.device
                ).squeeze(
                    1
                )

        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # 【关键修复】只更新机器人的状态到all_root_states
        if self.num_actors > 1:
            # 更新机器人actor的状态
            self.all_root_states[env_ids, 0, :] = self.root_states[env_ids]

            # 机器人Actor的索引是 [0, num_actors, 2*num_actors, ...]
            robot_actor_indices = (env_ids * self.num_actors).to(dtype=torch.int32)

            # 使用扁平化的all_root_states作为源张量
            # Isaac Gym会从中提取robot_actor_indices指定的行来更新模拟器
            all_states_flat = self.all_root_states.view(-1, 13).contiguous()

            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(all_states_flat),  # 源数据：所有actors的状态
                gymtorch.unwrap_tensor(robot_actor_indices),  # 目标索引：只更新机器人
                len(robot_actor_indices),
            )
        else:
            # 只有机器人actor的情况（无障碍物）
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids.to(dtype=torch.int32)),
                len(env_ids),
            )

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )  # lin vel x/y

        # 【关键修复】只更新机器人Actor的状态
        if self.num_actors > 1:
            # 更新 all_root_states 中机器人的状态
            self.all_root_states[:, 0, :] = self.root_states

            # 获取所有机器人Actor的索引
            all_robot_indices = torch.arange(
                0,
                self.num_envs * self.num_actors,
                self.num_actors,
                device=self.device,
                dtype=torch.int32,
            )

            # 使用扁平化的all_root_states作为源张量
            all_states_flat = self.all_root_states.view(-1, 13).contiguous()

            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(all_states_flat),  # 源数据：所有actors的状态
                gymtorch.unwrap_tensor(all_robot_indices),  # 目标索引：只更新机器人
                len(all_robot_indices),
            )
        else:
            # 保持原样 (num_actors == 1)
            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self.root_states)
            )

    def _update_terrain_curriculum(self, env_ids):
        """Implements the game-inspired curriculum with task switching.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return

        # 对于栏杆任务，使用目标点完成率而不是前进距离
        # 原因：
        # 1. 栏杆任务总共约8-10米，但threshold计算需要6-20米
        # 2. 前进距离判断导致terrain_level永远不变
        # 3. 目标点完成率更直接反映任务完成度
        goal_completion_rate = (
            self.cur_goal_idx[env_ids].float() / self.cfg.terrain.num_goals
        )

        # 收紧课程升级条件，防止"绕过栏杆"就升级
        # 原来的问题：level 0-2只需完成1/8 (12.5%)就升级，机器人可以绕过
        # 新策略：所有级别都需要完成至少50%的目标点，确保真正学会穿越
        current_level = self.terrain_levels[env_ids]

        # 统一升级门槛：必须完成至少50%的目标点（4/8个栏杆）
        move_up_threshold = 0.5  # 从动态阈值改为固定50%

        # 降级门槛：完成少于25%的目标点（2/8个栏杆）
        move_down_threshold = 0.25  # 从动态阈值改为固定25%

        # 通过 (阈值) 目标点 -> 提升难度
        move_up = goal_completion_rate > move_up_threshold
        # 通过不到 (阈值) 目标点 -> 降低难度
        move_down = goal_completion_rate < move_down_threshold

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

        # ========== 【核心优化】任务切换逻辑 ==========
        # 当机器人完成最高难度 (level 7) 后，在跳跃和钻爬课程之间切换
        # 这样可以在单一策略网络中训练两种通过方式
        #
        # 切换规则：
        # - 完成跳跃课程 (cols 0,1) 的 level 7 -> 切换到钻爬课程 (col 2)
        # - 完成钻爬课程 (cols 2,3) 的 level 7 -> 切换到跳跃课程 (col 0)
        # ================================================

        # 找到达到最高难度的环境
        reached_max = self.terrain_levels[env_ids] >= self.max_terrain_level

        if reached_max.any():
            # 获取达到最高难度的环境索引
            max_level_env_ids = env_ids[reached_max]

            # 获取这些环境当前的地形类型 (col)
            current_types = self.terrain_types[max_level_env_ids]

            # 判断是跳跃课程还是钻爬课程
            is_jump_curriculum = (current_types == 0) | (
                current_types == 1
            )  # cols 0 or 1
            is_crawl_curriculum = (current_types == 2) | (
                current_types == 3
            )  # cols 2 or 3

            # 跳跃课程 -> 钻爬课程 (切换到 col 2)
            self.terrain_types[max_level_env_ids] = torch.where(
                is_jump_curriculum,
                torch.tensor(
                    2, dtype=torch.long, device=self.device
                ),  # 切换到钻爬课程的第一列 (col 2)
                self.terrain_types[max_level_env_ids],
            )

            # 钻爬课程 -> 跳跃课程 (切换到 col 0)
            self.terrain_types[max_level_env_ids] = torch.where(
                is_crawl_curriculum,
                torch.tensor(
                    0, dtype=torch.long, device=self.device
                ),  # 切换到跳跃课程的第一列 (col 0)
                self.terrain_types[max_level_env_ids],
            )

            # 重置难度等级为 0（从新课程的最简单难度开始）
            self.terrain_levels[max_level_env_ids] = 0

            # 打印切换信息（用于调试，可选）
            if (
                hasattr(self, "print_curriculum_switch")
                and self.print_curriculum_switch
            ):
                num_jump_to_crawl = is_jump_curriculum.sum().item()
                num_crawl_to_jump = is_crawl_curriculum.sum().item()
                if num_jump_to_crawl > 0:
                    print(
                        f"[Curriculum] {num_jump_to_crawl} 个环境从跳跃课程切换到钻爬课程"
                    )
                if num_crawl_to_jump > 0:
                    print(
                        f"[Curriculum] {num_crawl_to_jump} 个环境从钻爬课程切换到跳跃课程"
                    )

        # 确保所有难度等级都在有效范围内
        self.terrain_levels[env_ids] = torch.clip(
            self.terrain_levels[env_ids], 0, self.max_terrain_level - 1
        )

        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]
        self.env_class[env_ids] = self.terrain_class[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat(
            (temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1
        )[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
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
        all_root_states = gymtorch.wrap_tensor(actor_root_state)

        # 计算每个环境的actors数量（如果有H型栏杆，会有额外的actors）
        self.num_actors = all_root_states.shape[0] // self.num_envs
        print(f"每个环境有 {self.num_actors} 个actors")

        # 如果每个环境有多个actors，重塑为(num_envs, num_actors, 13)
        # 然后只使用第一个actor（机器人）的状态
        if self.num_actors > 1:
            # 保存所有actors的状态
            self.all_root_states = all_root_states.view(
                self.num_envs, self.num_actors, 13
            )
            # root_states只包含机器人的状态，保持向后兼容
            self.root_states = self.all_root_states[:, 0, :]
        else:
            self.all_root_states = all_root_states
            self.root_states = all_root_states

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(
            self.num_envs, -1, 13
        )
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(
            self.num_envs, 4, 6
        )  # for feet only, see create_env()
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.d_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.reach_goal_timer = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(
            2,
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        ) + str_rng[0]
        if self.cfg.env.history_encoding:
            self.obs_history_buf = torch.zeros(
                self.num_envs,
                self.cfg.env.history_len,
                self.cfg.env.n_proprio,
                device=self.device,
                dtype=torch.float,
            )
        self.action_history_buf = torch.zeros(
            self.num_envs,
            self.cfg.domain_rand.action_buf_len,
            self.num_dofs,
            device=self.device,
            dtype=torch.float,
        )
        self.contact_buf = torch.zeros(
            self.num_envs,
            self.cfg.env.contact_buf_len,
            4,
            device=self.device,
            dtype=torch.float,
        )

        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading
        self._resample_commands(
            torch.arange(self.num_envs, device=self.device, requires_grad=False)
        )
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )  # TODO change this
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.feet_last_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.contact_filt = torch.zeros_like(self.last_contacts)

        # 【新增】卡住检测相关的缓冲区
        self.stuck_detection_counter = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.current_stuck = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.last_position = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.default_dof_pos_all = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
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
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero"
                    )
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_dof_pos_all[:] = self.default_dof_pos[0]

        self.height_update_interval = 1
        if hasattr(self.cfg.env, "height_update_dt"):
            self.height_update_interval = int(
                self.cfg.env.height_update_dt
                / (self.cfg.sim.dt * self.cfg.control.decimation)
            )

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(
                self.num_envs,
                self.cfg.depth.buffer_len,
                self.cfg.depth.resized[1],
                self.cfg.depth.resized[0],
            ).to(self.device)

        # 用于存储每个环境的4个栏杆的绝对世界坐标和高度 [abs_x, abs_y, height]
        # shape: (num_envs, 4, 3)
        # 这个信息将在 reset_idx 中更新，并在 compute_observations 中用于计算相对位置
        self.static_hurdle_info = torch.zeros(
            self.num_envs,
            4,
            3,
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in self.reward_scales.keys()
        }

    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
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

        self.gym.add_heightfield(
            self.sim, self.terrain.heightsamples.flatten(order="C"), hf_params
        )
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    def _create_trimesh(self):
        """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order="C"),
            self.terrain.triangles.flatten(order="C"),
            tm_params,
        )
        print("Trimesh added")
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )
        self.x_edge_mask = (
            torch.tensor(self.terrain.x_edge_mask)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[0]
            camera_props.height = self.cfg.depth.original[1]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)

            local_transform = gymapi.Transform()

            camera_position = np.copy(config.position)
            camera_angle = np.random.uniform(config.angle[0], config.angle[1])

            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(
                0, np.radians(camera_angle), 0
            )
            root_handle = self.gym.get_actor_root_rigid_body_handle(
                env_handle, actor_handle
            )

            self.gym.attach_camera_to_body(
                camera_handle,
                env_handle,
                root_handle,
                local_transform,
                gymapi.FOLLOW_TRANSFORM,
            )

    def _create_envs(self):
        """Creates environments:
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
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
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

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
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

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )

        # 初始化障碍物资产缓存字典，避免重复创建资产导致显存浪费和"幽灵障碍物"问题
        self.hurdle_asset_cache = {}
        # 存储所有静态障碍物的actor句柄，便于后续管理和查询
        self.static_obstacle_handles = []

        # 【关键修复】先创建virtual_crossbars供奖励函数使用（不创建Actor）
        if hasattr(self.terrain, "h_hurdles_dict") and self.terrain.h_hurdles_dict:
            print("Creating virtual crossbars for rewards...")
            self._create_virtual_crossbars_from_h_hurdles()

        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            if self.cfg.env.randomize_start_pos:
                pos[:2] += torch_rand_float(
                    -1.0, 1.0, (2, 1), device=self.device
                ).squeeze(1)
            if self.cfg.env.randomize_start_yaw:
                rand_yaw_quat = gymapi.Quat.from_euler_zyx(
                    0.0, 0.0, self.cfg.env.rand_yaw_range * np.random.uniform(-1, 1)
                )
                start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i
            )
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            # 使用更大的filter值确保碰撞生效
            robot_collision_filter = -1  # 使用-1作为全1位掩码（32位有符号整数）

            # Isaac Gym API: create_actor(env, asset, pose, name, group, filter, segmentationId)
            # group: 碰撞组，相同组的actor会碰撞
            # filter: 碰撞过滤器（位掩码），按位与结果非零时才能碰撞
            # segmentationId: 分割ID，用于可视化，不影响碰撞
            anymal_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                "anymal",
                i,  # group: 每个环境使用独立的碰撞组，隔离不同环境的Actor
                robot_collision_filter,  # filter: 使用全1位掩码确保碰撞
                0,  # segmentationId: 分割ID（用于可视化）
            )

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, anymal_handle
            )
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, anymal_handle, body_props, recomputeInertia=True
            )
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)

            self.attach_camera(i, env_handle, anymal_handle)

            # 【关键修复】在创建机器人Actor后立即创建障碍物Actors
            # 确保Actor顺序为: [R0, H0_actors, R1, H1_actors, ...]
            if hasattr(self.terrain, "h_hurdles_dict") and self.terrain.h_hurdles_dict:
                self._add_h_hurdle_static_geometry(env_handle, i)

            self.mass_params_tensor[i, :] = (
                torch.from_numpy(mass_params).to(self.device).to(torch.float)
            )
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = (
                self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)
            )

        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.zeros(
            len(hip_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)
        thigh_names = [
            "FR_thigh_joint",
            "FL_thigh_joint",
            "RR_thigh_joint",
            "RL_thigh_joint",
        ]
        self.thigh_indices = torch.zeros(
            len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i, name in enumerate(thigh_names):
            self.thigh_indices[i] = self.dof_names.index(name)
        calf_names = [
            "FR_calf_joint",
            "FL_calf_joint",
            "RR_calf_joint",
            "RL_calf_joint",
        ]
        self.calf_indices = torch.zeros(
            len(calf_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i, name in enumerate(calf_names):
            self.calf_indices[i] = self.dof_names.index(name)

        # 【修复完成】障碍物Actor已在主循环内创建，此处无需额外代码

    def _create_virtual_crossbars_from_h_hurdles(self):
        """
        【关键修复】从h_hurdles_dict创建virtual_crossbars列表供奖励函数使用
        """
        # 使用第一个地形块（row=0, col=0）的栏杆作为模板
        template_hurdles = self.terrain.h_hurdles_dict.get((0, 0), [])

        if not template_hurdles:
            print("[警告] 未找到H型栏杆信息，virtual_crossbars将为空")
            self.terrain.virtual_crossbars = []
            return

        # 【新增】获取 (0, 0) 环境的地形原点
        template_origin = self.terrain_origins[0, 0]  # [x, y, z]
        template_origin_x = template_origin[0].item()
        template_origin_y = template_origin[1].item()

        virtual_crossbars = []
        for hurdle_info in template_hurdles:
            # h_hurdles_dict中的坐标是世界坐标
            world_x = hurdle_info["x"]
            world_y = hurdle_info["y"]

            # 【修改】使用减法计算相对坐标，而不是模运算
            relative_x = world_x - template_origin_x
            relative_y = world_y - template_origin_y

            crossbar_info = {
                "x": relative_x,
                "y": relative_y,
                "z": hurdle_info["z"],
                "height": hurdle_info["crossbar"]["height"],
                "width": hurdle_info["post_spacing"],
                "depth": 0.1,  # 栏杆前后范围
            }
            virtual_crossbars.append(crossbar_info)

        self.terrain.virtual_crossbars = virtual_crossbars
        # print(f"[虚拟横杆] 已创建 {len(virtual_crossbars)} 个虚拟横杆用于奖励计算")
        # for i, cb in enumerate(virtual_crossbars):
        #     print(
        #         f"  横杆{i+1}: x={cb['x']:.2f}m, y={cb['y']:.2f}m, 高度={cb['height']:.2f}m, 宽度={cb['width']:.2f}m"
        #     )

    def _create_gate_assets(self):
        """创建门框几何体assets（立柱和横梁）"""
        print("Creating gate assets...")

        # 扫描所有环境，找出需要的所有尺寸组合
        gate_configs = set()
        for gates in self.terrain.gate_obstacles_dict.values():
            for gate in gates:
                config = (
                    round(gate["height"], 2),
                    round(gate["gate_width"], 2),
                    round(gate["gate_depth"], 2),
                    round(gate["post_thickness"], 2),
                )
                gate_configs.add(config)

        # 为每种配置创建assets
        for height, gate_width, gate_depth, post_thickness in gate_configs:
            # 创建立柱asset（box）
            asset_options = gymapi.AssetOptions()
            asset_options.density = 1000.0  # 密度
            asset_options.fix_base_link = True  # 固定不动
            asset_options.disable_gravity = True  # 禁用重力

            # 立柱尺寸：宽×深×高
            post_asset = self.gym.create_box(
                self.sim,
                post_thickness,  # x: 厚度
                gate_depth,  # y: 深度
                height,  # z: 高度
                asset_options,
            )

            # 上横梁asset（box）
            # 横梁厚度约为4cm
            beam_thickness = 0.04
            beam_asset = self.gym.create_box(
                self.sim,
                gate_width,  # x: 横跨整个门宽
                gate_depth,  # y: 深度（与立柱一致）
                beam_thickness,  # z: 横梁厚度
                asset_options,
            )

            # 存储assets
            config_key = (height, gate_width, gate_depth, post_thickness)
            self.gate_assets[config_key] = {
                "post": post_asset,
                "beam": beam_asset,
                "beam_thickness": beam_thickness,
            }

        print(f"Created {len(self.gate_assets)} gate asset configurations")

    def _add_gate_static_geometry(self, env_handle, env_id):
        """在指定环境中添加门框静态几何体（不作为actors，避免影响tensor尺寸）"""
        # 获取当前环境的地形坐标
        row = self.terrain_levels[env_id].item()
        col = self.terrain_types[env_id].item()

        # 获取当前环境的门框列表
        gates = self.terrain.gate_obstacles_dict.get((row, col), [])

        if not gates:
            return

        for gate_idx, gate_info in enumerate(gates):
            height = gate_info["height"]
            gate_width = gate_info["gate_width"]
            gate_depth = gate_info["gate_depth"]
            post_thickness = gate_info["post_thickness"]
            x = gate_info["x"]
            y = gate_info["y"]
            z = gate_info["z"]

            # 计算立柱位置（门框左右两侧）
            half_width = gate_width / 2.0
            beam_thickness = 0.04

            # 使用 gym.add_ground_geometry 添加静态几何体
            # 这些不会被计入actors，不影响root_states

            # 左立柱
            left_pos = gymapi.Vec3(
                x,
                y - half_width + post_thickness / 2.0,
                z + height / 2.0,
            )
            self._add_box_geometry(
                env_handle,
                left_pos,
                post_thickness,
                gate_depth,
                height,
                gymapi.Vec3(0.2, 0.2, 0.8),  # 蓝色
            )

            # 右立柱
            right_pos = gymapi.Vec3(
                x,
                y + half_width - post_thickness / 2.0,
                z + height / 2.0,
            )
            self._add_box_geometry(
                env_handle,
                right_pos,
                post_thickness,
                gate_depth,
                height,
                gymapi.Vec3(0.2, 0.2, 0.8),  # 蓝色
            )

            # 上横梁
            beam_pos = gymapi.Vec3(x, y, z + height + beam_thickness / 2.0)
            self._add_box_geometry(
                env_handle,
                beam_pos,
                gate_width,
                gate_depth,
                beam_thickness,
                gymapi.Vec3(0.9, 0.9, 0.9),  # 白色
            )

    def _add_box_geometry(self, env_handle, pos, width, depth, height, color):
        """添加一个box几何体到环境中（作为静态障碍，不是actor）"""
        # 创建box asset作为静态物体
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.density = 1000.0

        box_asset = self.gym.create_box(self.sim, width, depth, height, asset_options)

        # 创建actor但标记为静态环境的一部分
        # 使用特殊的collision group使其不计入机器人的actor索引
        pose = gymapi.Transform(pos, gymapi.Quat(0, 0, 0, 1))
        actor_handle = self.gym.create_actor(
            env_handle,
            box_asset,
            pose,
            "gate_static",
            self.num_envs,  # 使用num_envs作为collision group，隔离开机器人
            0,
            0,
        )

        # 设置颜色
        self.gym.set_rigid_body_color(
            env_handle, actor_handle, 0, gymapi.MESH_VISUAL, color
        )

    def _add_h_hurdle_static_geometry(self, env_handle, env_id):
        """
        在指定环境中添加H型栏杆静态几何体
        """
        # 步骤1：获取当前env_id对应的 (row, col)
        row = self.terrain_levels[env_id].item()
        col = self.terrain_types[env_id].item()

        # 步骤2：获取当前 (row, col) 对应的障碍物列表
        # terrain.py 的 add_terrain_to_map 已将所有坐标转换为世界坐标
        hurdles_world = self.terrain.h_hurdles_dict.get((row, col), [])

        if not hurdles_world:
            return

        # 步骤3：遍历障碍物并使用它们存储的绝对世界坐标创建actor
        for hurdle_info in hurdles_world:
            # 直接从hurdle_info中提取已计算好的世界坐标
            x = hurdle_info["x"]
            y = hurdle_info["y"]
            z = hurdle_info["z"]

            # 提取几何参数
            height = hurdle_info["height"]
            post_spacing = hurdle_info["post_spacing"]

            # 立柱信息
            posts = hurdle_info["posts"]
            post_radius = posts["radius"]
            post_height = posts["height"]
            # 立柱的Y坐标也必须使用绝对坐标（已在terrain.py中转换）
            left_post_y = posts["left_y"]
            right_post_y = posts["right_y"]
            post_color = gymapi.Vec3(*posts["color"])

            # 横梁信息
            crossbar = hurdle_info["crossbar"]
            crossbar_radius = crossbar["radius"]
            crossbar_length = crossbar["length"]
            crossbar_height = crossbar["height"]
            crossbar_color = gymapi.Vec3(*crossbar["color"])

            # 底部横杆信息
            bottom_bar = hurdle_info["bottom_bar"]
            bottom_bar_length = bottom_bar["length"]
            bottom_bar_height = bottom_bar["height"]
            bottom_bar_offset_x = bottom_bar["offset_x"]
            bottom_bar_half_height = bottom_bar.get(
                "half_height", bottom_bar.get("radius", 0.01)
            )
            bottom_bar_half_width = bottom_bar.get(
                "half_width", bottom_bar.get("radius", 0.01)
            )
            bottom_bar_color = gymapi.Vec3(*bottom_bar["color"])

            # 步骤5：使用正确的绝对世界坐标创建几何体

            # 创建左立柱（圆柱体）
            left_post_pos = gymapi.Vec3(x, left_post_y, z + post_height / 2.0)
            self._add_cylinder_geometry(
                env_handle,
                env_id,  # 传入env_id
                left_post_pos,
                post_radius,
                post_height,
                post_color,
                vertical=True,
            )

            # 创建右立柱（圆柱体）
            right_post_pos = gymapi.Vec3(x, right_post_y, z + post_height / 2.0)
            self._add_cylinder_geometry(
                env_handle,
                env_id,  # 传入env_id
                right_post_pos,
                post_radius,
                post_height,
                post_color,
                vertical=True,
            )

            # 创建顶部横梁（长方体）
            crossbar_center_z = z + crossbar_height + crossbar_radius
            crossbar_pos = gymapi.Vec3(x, y, crossbar_center_z)
            self._add_box_geometry(
                env_handle,
                env_id,  # 传入env_id
                crossbar_pos,
                crossbar_length,  # Y轴方向（长度）
                2 * crossbar_radius,  # Z轴方向（高度）
                2 * crossbar_radius,  # X轴方向（宽度）
                crossbar_color,
            )

            # 创建底部横杆
            bottom_bar_center_z = z + bottom_bar_height + bottom_bar_half_height
            bottom_bar_pos = gymapi.Vec3(
                x + bottom_bar_offset_x,
                y,
                bottom_bar_center_z,
            )
            self._add_box_geometry(
                env_handle,
                env_id,  # 传入env_id
                bottom_bar_pos,
                bottom_bar_length,
                2 * bottom_bar_half_height,
                2 * bottom_bar_half_width,
                bottom_bar_color,
            )

    def _add_cylinder_geometry(
        self, env_handle, env_id, pos, radius, length, color, vertical=True
    ):
        """添加一个圆柱体几何到环境中（作为静态障碍，不创建actor）

        Args:
            env_handle: 环境句柄
            env_id: 环境ID（用于collision_group）
            pos: 圆柱体中心位置
            radius: 圆柱体半径
            length: 圆柱体长度
            color: 颜色 (Vec3)
            vertical: 是否垂直放置（True=垂直Z轴，False=水平Y轴）
        """
        # Isaac Gym的capsule默认是沿X轴（机器人运动方向）的
        if vertical:
            # 垂直圆柱体（沿Z轴）- 从X轴旋转到Z轴，需要绕Y轴旋转90度
            # 使用四元数：绕Y轴旋转90度
            angle = np.pi / 2
            axis = gymapi.Vec3(0, 1, 0)  # Y轴
            quat = gymapi.Quat.from_axis_angle(axis, angle)
        else:
            # 水平圆柱体（沿Y轴，垂直于运动方向）- 从X轴旋转到Y轴，需要绕Z轴旋转90度
            # 使用四元数：绕Z轴旋转90度
            angle = np.pi / 2
            axis = gymapi.Vec3(0, 0, 1)  # Z轴
            quat = gymapi.Quat.from_axis_angle(axis, angle)

        pose = gymapi.Transform(pos, quat)

        # 使用资产缓存机制
        # 1. 创建基于几何属性的唯一键
        key = ("cylinder", float(radius), float(length))

        # 2. 检查缓存
        if key not in self.hurdle_asset_cache:
            # 3. 如果不在缓存中，创建新资产并存储
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            asset_options.density = 1000.0

            cylinder_asset = self.gym.create_capsule(
                self.sim, radius, length, asset_options
            )
            self.hurdle_asset_cache[key] = cylinder_asset
        else:
            # 4. 如果在缓存中，直接复用
            cylinder_asset = self.hurdle_asset_cache[key]

        # 5. 使用缓存的资产创建演员 (Actor)
        # 关键：障碍物使用与机器人相同的collision_group，确保它们在同一个环境中能碰撞
        # 使用全1的filter值（0xFFFFFFFF）确保碰撞检测生效
        obstacle_collision_filter = -1  # 使用-1作为全1位掩码（32位有符号整数）

        # 调试信息：检查资产的碰撞形状
        # num_shapes = self.gym.get_asset_rigid_shape_count(cylinder_asset)
        # if env_id == 0:  # 只在第一个环境打印，避免过多输出
        #     print(f"[DEBUG] 创建圆柱障碍物 (env={env_id}):")
        #     print(f"  - collision_group = {env_id}")
        #     print(
        #         f"  - collision_filter = {obstacle_collision_filter} (二进制全1，32位整数)"
        #     )
        #     print(f"  - self_collisions = 0")
        #     print(f"  - 圆柱资产碰撞形状数量: {num_shapes}")
        #     print(f"  - 位置: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
        #     print(f"  - 半径: {radius:.3f}, 长度: {length:.3f}")

        actor_handle = self.gym.create_actor(
            env_handle,
            cylinder_asset,  # <-- 使用缓存的 asset
            pose,
            "h_hurdle_static",
            env_id,  # collision_group=env_id: 与机器人使用相同的碰撞组，确保能碰撞
            obstacle_collision_filter,  # collision filter: 使用全1位掩码确保碰撞
            0,  # self_collisions=0: 障碍物无自碰撞
        )

        # 验证创建后的actor属性（仅第一个环境）
        # if env_id == 0:
        #     num_bodies = self.gym.get_actor_rigid_body_count(env_handle, actor_handle)
        #     body_shape_props = self.gym.get_actor_rigid_shape_properties(
        #         env_handle, actor_handle
        #     )
        #     print(f"  - 圆柱障碍物Actor刚体数量: {num_bodies}")
        #     print(f"  - 圆柱障碍物Actor碰撞形状数量: {len(body_shape_props)}")
        #     if len(body_shape_props) > 0:
        #         print(f"  - 第一个形状类型: {type(body_shape_props[0])}")

        # 存储actor句柄，便于后续管理
        self.static_obstacle_handles.append(actor_handle)

        # 设置颜色
        self.gym.set_rigid_body_color(
            env_handle, actor_handle, 0, gymapi.MESH_VISUAL, color
        )

    def _add_box_geometry(
        self, env_handle, env_id, pos, length_y, length_z, length_x, color
    ):
        """添加一个长方体几何到环境中（作为静态障碍）

        Args:
            env_handle: 环境句柄
            env_id: 环境ID（用于collision_group）
            pos: 长方体中心位置
            length_y: Y轴方向长度（横梁的长度方向）
            length_z: Z轴方向长度（高度）
            length_x: X轴方向长度（宽度）
            color: 颜色 (Vec3)
        """
        pose = gymapi.Transform(pos, gymapi.Quat(0, 0, 0, 1))

        # 使用资产缓存机制
        # 1. 创建基于几何属性的唯一键
        key = ("box", float(length_x), float(length_y), float(length_z))

        # 2. 检查缓存
        if key not in self.hurdle_asset_cache:
            # 3. 如果不在缓存中，创建新资产并存储
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            asset_options.density = 1000.0

            box_asset = self.gym.create_box(
                self.sim, length_x, length_y, length_z, asset_options
            )
            self.hurdle_asset_cache[key] = box_asset
        else:
            # 4. 如果在缓存中，直接复用
            box_asset = self.hurdle_asset_cache[key]

        # 5. 使用缓存的资产创建演员 (Actor)
        # 关键：障碍物使用与机器人相同的collision_group，确保它们在同一个环境中能碰撞
        # 使用全1的filter值（0xFFFFFFFF）确保碰撞检测生效
        obstacle_collision_filter = -1  # 使用-1作为全1位掩码（32位有符号整数）

        # 调试信息：检查资产的碰撞形状
        # num_shapes = self.gym.get_asset_rigid_shape_count(box_asset)
        # if env_id == 0:  # 只在第一个环境打印，避免过多输出
        #     print(f"[DEBUG] 创建长方体障碍物 (env={env_id}):")
        #     print(f"  - collision_group = {env_id}")
        #     print(
        #         f"  - collision_filter = {obstacle_collision_filter} (二进制全1，32位整数)"
        #     )
        #     print(f"  - self_collisions = 0")
        #     print(f"  - 长方体资产碰撞形状数量: {num_shapes}")
        #     print(f"  - 位置: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
        #     print(f"  - 尺寸: X={length_x:.3f}, Y={length_y:.3f}, Z={length_z:.3f}")

        actor_handle = self.gym.create_actor(
            env_handle,
            box_asset,  # <-- 使用缓存的 asset
            pose,
            "h_hurdle_static",
            env_id,  # collision_group=env_id: 与机器人使用相同的碰撞组，确保能碰撞
            obstacle_collision_filter,  # collision filter: 使用全1位掩码确保碰撞
            0,  # self_collisions=0: 障碍物无自碰撞
        )

        # 验证创建后的actor属性（仅第一个环境）
        # if env_id == 0:
        #     num_bodies = self.gym.get_actor_rigid_body_count(env_handle, actor_handle)
        #     body_shape_props = self.gym.get_actor_rigid_shape_properties(
        #         env_handle, actor_handle
        #     )
        #     print(f"  - 长方体障碍物Actor刚体数量: {num_bodies}")
        #     print(f"  - 长方体障碍物Actor碰撞形状数量: {len(body_shape_props)}")
        #     if len(body_shape_props) > 0:
        #         print(f"  - 第一个形状类型: {type(body_shape_props[0])}")

        # 存储actor句柄，便于后续管理
        self.static_obstacle_handles.append(actor_handle)

        # 设置颜色
        self.gym.set_rigid_body_color(
            env_handle, actor_handle, 0, gymapi.MESH_VISUAL, color
        )

    def _get_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            self.env_class = torch.zeros(
                self.num_envs, device=self.device, requires_grad=False
            )
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(
                0, max_init_level + 1, (self.num_envs,), device=self.device
            )
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.terrain.num_cols),
                rounding_mode="floor",
            ).to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = (
                torch.from_numpy(self.terrain.env_origins)
                .to(self.device)
                .to(torch.float)
            )
            self.env_origins[:] = self.terrain_origins[
                self.terrain_levels, self.terrain_types
            ]

            self.terrain_class = (
                torch.from_numpy(self.terrain.terrain_type)
                .to(self.device)
                .to(torch.float)
            )
            self.env_class[:] = self.terrain_class[
                self.terrain_levels, self.terrain_types
            ]

            self.terrain_goals = (
                torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
            )
            self.env_goals = torch.zeros(
                self.num_envs,
                self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs,
                3,
                device=self.device,
                requires_grad=False,
            )
            self.cur_goal_idx = torch.zeros(
                self.num_envs, device=self.device, requires_grad=False, dtype=torch.long
            )
            temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
            last_col = temp[:, -1].unsqueeze(1)
            self.env_goals[:] = torch.cat(
                (temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1
            )[:]
            self.cur_goals = self._gather_cur_goals()
            self.next_goals = self._gather_cur_goals(future=1)

        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        reward_norm_factor = 1  # np.sum(list(self.reward_scales.values()))
        for rew in self.reward_scales:
            self.reward_scales[rew] = self.reward_scales[rew] / reward_norm_factor
        max_command_ranges = class_to_dict(self.cfg.commands.max_ranges)
        self.command_max_ranges = copy.deepcopy(max_command_ranges)
        if self.cfg.commands.curriculum:
            self.command_ranges = class_to_dict(self.cfg.commands.ranges)
            self.command_curriculum_increments = (
                class_to_dict(self.cfg.commands.crclm_incremnt)
                if hasattr(self.cfg.commands, "crclm_incremnt")
                else {}
            )
            self.command_curriculum_buffer = deque(maxlen=20)
            self.command_curriculum_step = 0
        else:
            self.command_ranges = copy.deepcopy(max_command_ranges)
            self.command_curriculum_increments = {}
            self.command_curriculum_buffer = None
            self.command_curriculum_step = 0
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(
            self.cfg.domain_rand.push_interval_s / self.dt
        )

    def _draw_height_samples(self):
        """Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        # Check if envs list is properly initialized
        if len(self.envs) == 0 or self.lookat_id >= len(self.envs):
            return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        i = self.lookat_id
        base_pos = (self.root_states[i, :3]).cpu().numpy()
        heights = self.measured_heights[i].cpu().numpy()
        height_points = (
            quat_apply_yaw(
                self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]
            )
            .cpu()
            .numpy()
        )
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(
                sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose
            )

    def _draw_goals(self):
        # Check if envs list is properly initialized
        if len(self.envs) == 0 or self.lookat_id >= len(self.envs):
            return

        sphere_geom = gymutil.WireframeSphereGeometry(
            0.1, 32, 32, None, color=(1, 0, 0)
        )
        sphere_geom_cur = gymutil.WireframeSphereGeometry(
            0.1, 32, 32, None, color=(0, 0, 1)
        )
        sphere_geom_reached = gymutil.WireframeSphereGeometry(
            self.cfg.env.next_goal_threshold, 32, 32, None, color=(0, 1, 0)
        )
        goals = (
            self.terrain_goals[
                self.terrain_levels[self.lookat_id], self.terrain_types[self.lookat_id]
            ]
            .cpu()
            .numpy()
        )
        for i, goal in enumerate(goals):
            goal_xy = goal[:2] + self.terrain.cfg.border_size
            pts = (goal_xy / self.terrain.cfg.horizontal_scale).astype(int)
            goal_z = (
                self.height_samples[pts[0], pts[1]].cpu().item()
                * self.terrain.cfg.vertical_scale
            )
            pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)
            if i == self.cur_goal_idx[self.lookat_id].cpu().item():
                gymutil.draw_lines(
                    sphere_geom_cur,
                    self.gym,
                    self.viewer,
                    self.envs[self.lookat_id],
                    pose,
                )
                if self.reached_goal_ids[self.lookat_id]:
                    gymutil.draw_lines(
                        sphere_geom_reached,
                        self.gym,
                        self.viewer,
                        self.envs[self.lookat_id],
                        pose,
                    )
            else:
                gymutil.draw_lines(
                    sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose
                )

        if not self.cfg.depth.use_camera:
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(
                0.02, 16, 16, None, color=(1, 0.35, 0.25)
            )
            pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()
            for i in range(5):
                norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.target_pos_rel / (norm + 1e-5)
                pose_arrow = (
                    pose_robot[:2]
                    + 0.1 * (i + 3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                )
                pose = gymapi.Transform(
                    gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None
                )
                gymutil.draw_lines(
                    sphere_geom_arrow,
                    self.gym,
                    self.viewer,
                    self.envs[self.lookat_id],
                    pose,
                )

            sphere_geom_arrow = gymutil.WireframeSphereGeometry(
                0.02, 16, 16, None, color=(0, 1, 0.5)
            )
            for i in range(5):
                norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
                pose_arrow = (
                    pose_robot[:2]
                    + 0.2 * (i + 3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                )
                pose = gymapi.Transform(
                    gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None
                )
                gymutil.draw_lines(
                    sphere_geom_arrow,
                    self.gym,
                    self.viewer,
                    self.envs[self.lookat_id],
                    pose,
                )

    def _draw_feet(self):
        if (
            hasattr(self, "feet_at_edge")
            and len(self.envs) > 0
            and self.lookat_id < len(self.envs)
        ):
            non_edge_geom = gymutil.WireframeSphereGeometry(
                0.02, 16, 16, None, color=(0, 1, 0)
            )
            edge_geom = gymutil.WireframeSphereGeometry(
                0.02, 16, 16, None, color=(1, 0, 0)
            )

            feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
            for i in range(4):
                pose = gymapi.Transform(
                    gymapi.Vec3(
                        feet_pos[self.lookat_id, i, 0],
                        feet_pos[self.lookat_id, i, 1],
                        feet_pos[self.lookat_id, i, 2],
                    ),
                    r=None,
                )
                if self.feet_at_edge[self.lookat_id, i]:
                    gymutil.draw_lines(
                        edge_geom,
                        self.gym,
                        self.viewer,
                        self.envs[self.lookat_id],
                        pose,
                    )
                else:
                    gymutil.draw_lines(
                        non_edge_geom,
                        self.gym,
                        self.viewer,
                        self.envs[self.lookat_id],
                        pose,
                    )

    def _init_height_points(self):
        """Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(
            self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False
        )
        x = torch.tensor(
            self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False
        )
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs,
            self.num_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_envs):
            offset = torch_rand_float(
                -self.cfg.terrain.measure_horizontal_noise,
                self.cfg.terrain.measure_horizontal_noise,
                (self.num_height_points, 2),
                device=self.device,
            ).squeeze()
            xy_noise = (
                torch_rand_float(
                    -self.cfg.terrain.measure_horizontal_noise,
                    self.cfg.terrain.measure_horizontal_noise,
                    (self.num_height_points, 2),
                    device=self.device,
                ).squeeze()
                + offset
            )
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
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs,
                self.num_height_points,
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points),
                self.height_points[env_ids],
            ) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(
                self.base_quat.repeat(1, self.num_height_points), self.height_points
            ) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_heights_points(self, coords, env_ids=None):
        if env_ids:
            points = coords[env_ids]
        else:
            points = coords

        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    ################## parkour rewards ##################

    def _reward_forward_progress_trapezoid(self):
        """鼓励在可控速度区间内持续向前移动。"""
        forward_speed = torch.clamp(self.base_lin_vel[:, 0], min=0.0)
        speed_params = getattr(
            self.cfg.rewards,
            "progress_speed_trapezoid",
            [0.18, 0.4, 0.75, 1.05],
        )
        p0, p1, p2, p3 = speed_params
        speed_score = self._trapezoid_shaping(forward_speed, p0, p1, p2, p3)

        heading_tol = getattr(self.cfg.rewards, "progress_heading_tolerance", 0.35)
        heading_error = torch.abs(
            getattr(self, "delta_yaw", torch.zeros_like(forward_speed))
        )
        heading_gate = torch.clamp(1.0 - heading_error / heading_tol, 0.0, 1.0)

        alive_gate = 1.0 - self.reset_buf.float()
        return speed_score * heading_gate * alive_gate

    def _reward_lin_vel_z(self):
        """惩罚垂直方向的剧烈运动。"""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        """惩罚横滚、俯仰方向的角速度。"""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """惩罚姿态偏离水平面，避免翻滚。"""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_collision(self):
        """惩罚非脚部的障碍物接触。"""
        contacts = torch.norm(
            self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
        )
        return torch.sum(contacts > 0.1, dim=1).float()

    def _reward_action_rate(self):
        """惩罚动作突变，鼓励平滑控制。"""
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_torques(self):
        """惩罚能耗（扭矩平方）。"""
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_feet_stumble(self):
        """惩罚脚在非垂直方向上的碰撞，避免被绊住。"""
        lateral = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        vertical = torch.abs(self.contact_forces[:, self.feet_indices, 2]) + 1e-6
        stumble = lateral > 4 * vertical
        return torch.any(stumble, dim=1).float()

    def _reward_centerline_alignment(self):
        """鼓励机器人沿赛道中心线前进，防止绕行。"""
        tolerance = getattr(self.cfg.rewards, "centerline_tolerance", 0.25)
        tolerance = max(tolerance, 1.0e-3)
        rel_y = self.root_states[:, 1] - self.env_origins[:, 1]
        return torch.exp(-torch.abs(rel_y) / tolerance)

    def _reward_base_height_strategy(self):
        """根据前方障碍高度调整躯干高度，引导跳跃或钻爬策略。"""
        base_normal = getattr(self.cfg.rewards, "base_height_normal", 0.36)
        crawl_target = getattr(self.cfg.rewards, "base_height_target", 0.25)
        jump_target = getattr(self.cfg.rewards, "jump_height_target", 0.42)
        detection_range = getattr(self.cfg.rewards, "obstacle_detection_range", 1.5)
        high_threshold = getattr(self.cfg.rewards, "base_height_high_threshold", 0.35)
        shaping_gain = getattr(self.cfg.rewards, "base_height_strategy_gain", 6.0)

        target_heights = torch.full(
            (self.num_envs,),
            base_normal,
            dtype=torch.float,
            device=self.device,
        )

        if hasattr(self, "static_hurdle_info"):
            robot_xy = self.root_states[:, :2]
            relative = self.static_hurdle_info[:, :, :2] - robot_xy.unsqueeze(1)
            forward_dist = relative[:, :, 0]
            hurdle_heights = self.static_hurdle_info[:, :, 2]

            valid = (
                (hurdle_heights > 1.0e-4)
                & (forward_dist > 0.0)
                & (forward_dist < detection_range)
            )

            if torch.any(valid):
                large = torch.full_like(forward_dist, detection_range + 1.0)
                masked_dist = torch.where(valid, forward_dist, large)
                min_dist, min_idx = torch.min(masked_dist, dim=1)
                has_target = min_dist < detection_range

                nearest_height = hurdle_heights[
                    torch.arange(self.num_envs, device=self.device), min_idx
                ]
                nearest_height = torch.where(
                    has_target, nearest_height, torch.zeros_like(nearest_height)
                )

                high_mask = has_target & (nearest_height >= high_threshold)
                low_mask = (
                    has_target
                    & (nearest_height > 0.0)
                    & (nearest_height < high_threshold)
                )

                low_target = torch.full_like(target_heights, jump_target)
                high_target = torch.full_like(target_heights, crawl_target)

                target_heights = torch.where(high_mask, high_target, target_heights)
                target_heights = torch.where(low_mask, low_target, target_heights)

        robot_height = self.root_states[:, 2] - self.env_origins[:, 2]
        height_error = torch.abs(robot_height - target_heights)
        return torch.exp(-shaping_gain * height_error)

    def _reward_feet_clearance_simple(self):
        clearance_height = getattr(self.cfg.rewards, "feet_clearance_height", 0.10)
        detection_window = getattr(self.cfg.rewards, "feet_clearance_window", 0.30)
        walkway_half = getattr(self.cfg.terrain, "walkway_width", 0.8) * 0.5

        penalty = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if not hasattr(self, "static_hurdle_info"):
            return penalty

        hurdle_pos = self.static_hurdle_info[:, :, :2]
        hurdle_heights = self.static_hurdle_info[:, :, 2]
        has_hurdle = hurdle_heights > 1.0e-4

        robot_xy = self.root_states[:, :2]
        relative = hurdle_pos - robot_xy.unsqueeze(1)

        near_x = torch.abs(relative[:, :, 0]) < detection_window
        near_y = torch.abs(relative[:, :, 1]) < walkway_half
        near_hurdle = has_hurdle & near_x & near_y
        near_any = near_hurdle.any(dim=1)
        if not torch.any(near_any):
            return penalty

        foot_heights = self.rigid_body_states[:, self.feet_indices, 2]
        foot_heights = foot_heights - self.env_origins[:, 2].unsqueeze(1)
        low_feet = (foot_heights < clearance_height).float()
        return torch.sum(low_feet, dim=1) * near_any.float()

    def _reward_feet_air_time(self):
        """奖励合适的腾空时间，避免碎步或过度腾空。"""
        if not hasattr(self, "feet_last_air_time"):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        air_time = self.feet_last_air_time
        if air_time.numel() == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        target = getattr(self.cfg.rewards, "feet_air_time_target", 0.22)
        tolerance = getattr(self.cfg.rewards, "feet_air_time_tolerance", 0.10)
        min_time = max(0.0, target - tolerance)
        max_time = target + tolerance

        within_left = (air_time >= min_time) & (air_time <= target)
        within_right = (air_time > target) & (air_time <= max_time)

        left_span = max(target - min_time, 1.0e-6)
        right_span = max(max_time - target, 1.0e-6)

        reward_left = torch.where(
            within_left, (air_time - min_time) / left_span, torch.zeros_like(air_time)
        )
        reward_right = torch.where(
            within_right,
            (max_time - air_time) / right_span,
            torch.zeros_like(air_time),
        )

        reward = reward_left + reward_right
        reward = reward.sum(dim=1)
        return reward

    def _reward_stand_still(self):
        """惩罚前向速度过低的原地不动行为。"""
        forward_vel = torch.abs(self.base_lin_vel[:, 0])
        return (forward_vel < 0.1).float()

    def _reward_alive_bonus(self):
        """鼓励策略保持存活。"""
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_termination(self):
        """在触发环境重置时给予惩罚。"""
        termination = self.reset_buf.clone().float()
        if hasattr(self, "cur_goal_idx") and hasattr(self.cfg.terrain, "num_goals"):
            reached_all_goals = self.cur_goal_idx >= self.cfg.terrain.num_goals
            termination[reached_all_goals] = 0.0
        return termination

    def get_post_contact_forces_info(self, env_idx=0):
        """
        获取指定环境中机器人与柱子的接触力详细信息

        Args:
            env_idx: 环境索引，默认为0

        Returns:
            dict: 包含接触力详细信息的字典
        """
        info = {
            "has_contact": False,
            "max_contact_force": 0.0,
            "contact_body_parts": [],
            "robot_position": None,
            "nearest_post": None,
            "distance_to_post": None,
        }

        if (
            not hasattr(self, "penalised_contact_indices")
            or len(self.penalised_contact_indices) == 0
        ):
            return info

        # 获取身体部件接触力
        body_contact_forces = self.contact_forces[
            env_idx, self.penalised_contact_indices, :
        ]
        contact_force_magnitude = torch.norm(body_contact_forces, dim=-1)

        threshold = getattr(self.cfg.rewards, "obstacle_contact_force_threshold", 5.0)

        # 获取机器人位置
        robot_pos = self.root_states[env_idx, :3].cpu().numpy()
        info["robot_position"] = robot_pos

        # 检查接触
        has_contact_mask = contact_force_magnitude > threshold
        if torch.any(has_contact_mask):
            info["has_contact"] = True
            info["max_contact_force"] = torch.max(contact_force_magnitude).item()

            # 找出哪些身体部件有接触
            for i, has_contact in enumerate(has_contact_mask):
                if has_contact:
                    force = contact_force_magnitude[i].item()
                    body_idx = self.penalised_contact_indices[i]
                    body_name = (
                        self.body_names[body_idx]
                        if hasattr(self, "body_names")
                        else f"body_{body_idx}"
                    )
                    info["contact_body_parts"].append(
                        {"name": body_name, "force": force, "index": body_idx}
                    )

        # 查找最近的柱子
        if hasattr(self, "terrain") and hasattr(self.terrain, "virtual_crossbars"):
            env_origin = self.env_origins[env_idx].cpu().numpy()
            robot_x = robot_pos[0] - env_origin[0]
            robot_y = robot_pos[1] - env_origin[1]

            min_distance = float("inf")
            nearest_post = None

            for crossbar_info in self.terrain.virtual_crossbars:
                post_x = crossbar_info["x"]
                post_y = crossbar_info["y"]

                # 计算到柱子中心的距离
                distance = np.sqrt((robot_x - post_x) ** 2 + (robot_y - post_y) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    nearest_post = {
                        "position": (post_x, post_y),
                        "height": crossbar_info["height"],
                        "width": crossbar_info["width"],
                        "depth": crossbar_info.get("depth", 0.12),
                    }

            info["nearest_post"] = nearest_post
            info["distance_to_post"] = min_distance

        return info

    def print_contact_forces_summary(self, num_envs=1):
        """
        打印多个环境的接触力摘要信息

        Args:
            num_envs: 要打印的环境数量
        """
        print("\n" + "=" * 80)
        print("机器人与柱子接触力摘要")
        print("=" * 80)

        for env_idx in range(min(num_envs, self.num_envs)):
            info = self.get_post_contact_forces_info(env_idx)

            print(f"\n环境 {env_idx}:")
            print(
                f"  机器人位置: ({info['robot_position'][0]:.2f}, "
                f"{info['robot_position'][1]:.2f}, {info['robot_position'][2]:.2f})"
            )

            if info["nearest_post"]:
                post = info["nearest_post"]
                print(
                    f"  最近柱子: 位置=({post['position'][0]:.2f}, {post['position'][1]:.2f}), "
                    f"高度={post['height']:.2f}m, 距离={info['distance_to_post']:.2f}m"
                )

            if info["has_contact"]:
                print(f"  ⚠️ 检测到接触! 最大接触力: {info['max_contact_force']:.2f} N")
                print(f"  接触部件:")
                for part in info["contact_body_parts"]:
                    print(f"    - {part['name']}: {part['force']:.2f} N")
            else:
                print(f"  ✓ 无接触")

        print("=" * 80 + "\n")

    # def _reward_feet_air_time(self):
    #     """
    #     奖励适当的腾空时间，鼓励良好的步态
    #     - 腾空时间过短：步态僵硬，效率低
    #     - 腾空时间适中：步态自然，效率高
    #     - 腾空时间过长：可能是跳跃或失控
    #     """
    #     # 更新腾空时间
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
    #     contact_filt = torch.logical_or(contact, self.last_contacts)
    #     self.last_contacts = contact

    #     # 当脚接触地面时，记录腾空时间并重置
    #     first_contact = (self.feet_air_time > 0.0) * contact_filt
    #     self.feet_air_time += self.dt

    #     # 理想的腾空时间范围：0.1-0.3秒
    #     # 低于0.1秒：步态太快或拖地
    #     # 高于0.3秒：可能在跳跃
    #     rew_air_time = torch.sum(
    #         (self.feet_air_time - 0.1) * first_contact, dim=1
    #     )  # 奖励接近0.1秒的腾空时间

    #     # 限制奖励范围，避免过长的腾空时间获得高奖励
    #     rew_air_time = torch.clamp(rew_air_time, -0.5, 0.5)

    #     # 重置已接触的脚的腾空时间
    #     self.feet_air_time *= ~contact_filt

    #     return rew_air_time
