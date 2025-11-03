# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GalileoParkourCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 512
        # 【新增】卡住检测配置（优化后，减少误判）
        stuck_distance_threshold = 0.02  # X轴前进距离小于2cm视为不动
        stuck_time_threshold = 200  # 持续200步（约1秒）不动且有接触力视为卡住

    # 【修复】合并terrain配置，避免重复定义
    class terrain(LeggedRobotCfg.terrain):
        num_rows = 8  # 8个难度级别（0-7）
        num_cols = 4  # 4种地形类型
        terrain_dict = {"h_hurdle": 1.0}
        terrain_proportions = list(terrain_dict.values())
        num_goals = 8
        curriculum = True
        max_init_terrain_level = 0  # 【关键】从最简单的level 0开始，避免一开始太难

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # 抬高初始高度，避免触地
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,  # 增加后大腿弯曲
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,  # 增加后大腿弯曲
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        # 【可选优化】稍微提高刚度，增强可控性（如果当前动作过软可尝试）
        stiffness = {"joint": 50.0}  # 从50.0提高到60.0，增强关节响应
        damping = {"joint": 1.5}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/galileo_e1_v1d6_e1r/e1_v1d6_e1r.urdf"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base_link"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        push_robots = True

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False

        class scales:
            termination = -2.0
            tracking_goal_vel = 1.5
            tracking_yaw = 0.5
            orientation = -0.5
            ang_vel_xy = -0.05
            action_rate = -0.05
            dof_acc = -1.5e-7
            dof_error = -0.05
            collision = -0.0
            body_obstacle_contact = -0.01
            feet_stumble = -0.5
            feet_contact_forces = -0.01
            stand_still = -1.0
            alive_bonus = 1.0
            excessive_leg_width = -1.0
            height_based_guidance = 10.0
            hurdle_alignment = 0.5

        soft_dof_pos_limit = 0.9
        base_height_target = 0.25  # 钻爬时的目标高度
        base_height_normal = 0.35  # 正常站立高度
        obstacle_contact_force_threshold = 5.0
        low_hurdle_threshold = 0.35  # 低栏杆阈值（跳跃/爬）：小于35cm
        high_hurdle_threshold = 0.35  # 高栏杆阈值（钻）：大于等于35cm
        obstacle_detection_range = 1.0
        post_contact_proximity_threshold = 0.5
        contact_force_penalty_scaling = 50.0
        max_contact_force_penalty = 2.0
        enable_contact_force_logging = True
        obstacle_safe_distance = 1.0
        max_leg_width = 0.5  # 修改：左右腿最大Y轴距离500mm（对应栏杆宽度）
        guidance_detection_range = 2.0  # 扩大：引导奖励检测范围（更易触发）
        jump_height_target = 0.40  # 新增：跳跃时的目标身体高度
        # 对准检测参数
        alignment_detection_range = 1.5  # 对准检测范围（1.5m内）
        y_alignment_tolerance = 0.15  # Y轴对准容忍度（15cm）
        yaw_alignment_tolerance = 0.3  # 航向对准容忍度（约17度）
        # 引导奖励的高度容忍度（放宽以降低稀疏性）
        low_guidance_tolerance = 0.20  # 低栏：±20cm
        high_guidance_tolerance = 0.15  # 高栏：±15cm

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        num_commands = 4
        resampling_time = 6.0
        heading_command = True

        class ranges:
            lin_vel_x = [0.3, 1.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0, 0]
            heading = [0, 0]

    class depth(LeggedRobotCfg.depth):
        use_camera = False
        camera_num_envs = 32
        position = [0.28, 0, 0.08]
        angle = [-15, 5]
        update_interval = 4
        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 90
        buffer_len = 2
        near_clip = 0.0
        far_clip = 3.5
        dis_noise = 0.01
        scale = 1.0
        invert = True


class GalileoParkourCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        num_mini_batches = 4
        num_learning_epochs = 5

    class depth_encoder(LeggedRobotCfgPPO.depth_encoder):
        if_depth = GalileoParkourCfg.depth.use_camera
        depth_shape = GalileoParkourCfg.depth.resized
        buffer_len = GalileoParkourCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.0e-3
        num_steps_per_env = GalileoParkourCfg.depth.update_interval * 24

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "galileo"
        # 【优化】增加采样长度，提升轨迹稳定性和梯度质量
        num_steps_per_env = 48  # ↑ 从24提高到48，更长的轨迹更有利于学习步态
        max_iterations = 100000
        save_interval = 100
