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
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
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
    if checkpoint == -1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint


def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = env_cfg.env.episode_length_s

    terrain_mode = getattr(args, "terrain_mode", "stage")
    terrain_mode = terrain_mode.lower() if isinstance(terrain_mode, str) else "stage"
    if terrain_mode not in {"stage", "demo"}:
        print(f"[警告] 未识别的 terrain_mode='{terrain_mode}'，自动切换为stage模式。")
        terrain_mode = "stage"

    selected_stage_index = None
    selected_terrain_type = None  # 用于保存 terrain_type (col)
    if terrain_mode == "demo":
        env_cfg.env.num_envs = 5 if not args.save else 64
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.max_difficulty = False
        env_cfg.terrain.height = [0.02, 0.02]
        env_cfg.terrain.terrain_dict = {"h_hurdle": 1.0}
        env_cfg.terrain.terrain_proportions = [1.0]
        env_cfg.terrain.demo_heights = [0.20, 0.30, 0.40, 0.50, None]
        env_cfg.terrain.demo_progressive_cols = [4]

        # env_cfg.terrain.horizontal_scale = 0.10 # (原为 0.05)
        # env_cfg.terrain.border_size = 1.0 # (原为 5.0)
    else:
        # 【stage模式】完全匹配训练时的地形配置，确保能正确反映训练中的地形
        env_cfg.env.num_envs = 4 if not args.save else 64

        # 使用与训练完全相同的配置
        # 训练配置：num_rows=6, num_cols=4, curriculum=True, max_init_terrain_level=0
        env_cfg.terrain.num_rows = 6  # 6个难度级别 (terrain_level: 0-5)
        env_cfg.terrain.num_cols = 4  # 4种地形类型 (col 0,1=跳跃课程, col 2,3=钻爬课程)
        env_cfg.terrain.curriculum = True  # 启用课程模式
        env_cfg.terrain.max_difficulty = False
        env_cfg.terrain.height = [0.02, 0.02]
        env_cfg.terrain.terrain_dict = {"h_hurdle": 1.0}
        env_cfg.terrain.terrain_proportions = [1.0]
        env_cfg.terrain.demo_heights = [None]
        env_cfg.terrain.demo_progressive_cols = []

        # curriculum_stage 对应训练中的 terrain_level (0-5)
        # difficulty = terrain_level / (num_rows - 1) = terrain_level / 5
        stage_arg = getattr(args, "curriculum_stage", -1)
        if stage_arg < 0:
            # 默认使用最高难度 (terrain_level = 5)
            selected_terrain_level = 5
        else:
            # 限制在有效范围内 (0-5)
            selected_terrain_level = max(0, min(stage_arg, 5))

        # 设置 max_init_terrain_level，但实际会在环境创建后固定为 selected_terrain_level
        env_cfg.terrain.max_init_terrain_level = selected_terrain_level

        # 计算对应的 difficulty
        difficulty = (
            selected_terrain_level / (env_cfg.terrain.num_rows - 1)
            if env_cfg.terrain.num_rows > 1
            else 0.0
        )

        # 可选：通过参数指定 terrain_type (col)，默认使用 col=0 (跳跃课程)
        terrain_type = getattr(args, "terrain_type", 0)
        terrain_type = max(0, min(terrain_type, env_cfg.terrain.num_cols - 1))

        print(
            f"[地形] stage模式 - terrain_level={selected_terrain_level}, "
            f"difficulty={difficulty:.2f}, terrain_type(col)={terrain_type}"
        )

        # 保存这些值，以便在环境创建后使用
        selected_stage_index = selected_terrain_level  # 为了兼容后续代码
        selected_terrain_type = terrain_type

    env_cfg.depth.angle = [0, 1]
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

    # 【关键修复】在 stage 模式下，固定 terrain_levels 和 terrain_types
    # 确保使用与训练时完全相同的 terrain_level 和 terrain_type
    obs = env.get_observations()

    if terrain_mode == "stage" and selected_stage_index is not None:
        print(
            f"  [play.py] 强行锁定所有环境到 Level={selected_stage_index}, "
            f"Type={selected_terrain_type}"
        )
        env.terrain_levels.fill_(selected_stage_index)
        if selected_terrain_type is not None:
            env.terrain_types.fill_(selected_terrain_type)
        all_envs = torch.arange(env.num_envs, device=env.device)
        env.reset_idx(all_envs)
        env.cfg.terrain.curriculum = False
        print("  [play.py] env.cfg.terrain.curriculum 已禁用。")

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(
        log_root=log_pth,
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        return_log_dir=True,
    )

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
    infos["depth"] = (
        env.depth_buffer.clone().to(ppo_runner.device)[:, -1]
        if ppo_runner.if_depth
        else None
    )

    for i in range(10 * int(env.max_episode_length)):
        if args.use_jit:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    depth_latent = torch.ones(
                        (env_cfg.env.num_envs, 32), device=env.device
                    )
                    actions, depth_latent = policy_jit(
                        obs.detach(), True, infos["depth"], depth_latent
                    )
                else:
                    depth_buffer = torch.ones(
                        (env_cfg.env.num_envs, 58, 87), device=env.device
                    )
                    actions, depth_latent = policy_jit(
                        obs.detach(), False, depth_buffer, depth_latent
                    )
            else:
                obs_jit = torch.cat(
                    (
                        obs.detach()[:, : env_cfg.env.n_proprio + env_cfg.env.n_priv],
                        obs.detach()[
                            :, -env_cfg.env.history_len * env_cfg.env.n_proprio :
                        ],
                    ),
                    dim=1,
                )
                actions = policy(obs_jit)
        else:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    obs_student = obs[:, : env.cfg.env.n_proprio].clone()
                    obs_student[:, 6:8] = 0
                    depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = depth_latent_and_yaw[:, -2:]
                obs[:, 6:8] = 1.5 * yaw

            else:
                depth_latent = None

            if hasattr(ppo_runner.alg, "depth_actor"):
                actions = ppo_runner.alg.depth_actor(
                    obs.detach(), hist_encoding=True, scandots_latent=depth_latent
                )
            else:
                actions = policy(
                    obs.detach(), hist_encoding=True, scandots_latent=depth_latent
                )

        obs, _, rews, dones, infos = env.step(actions.detach())
        if args.web:
            web_viewer.render(
                fetch_results=True,
                step_graphics=True,
                render_all_camera_sensors=True,
                wait_for_page_load=True,
            )
        print(
            "time:",
            env.episode_length_buf[env.lookat_id].item() / 50,
            "cmd vx",
            env.commands[env.lookat_id, 0].item(),
            "actual vx",
            env.base_lin_vel[env.lookat_id, 0].item(),
        )

        id = env.lookat_id


if __name__ == "__main__":
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
