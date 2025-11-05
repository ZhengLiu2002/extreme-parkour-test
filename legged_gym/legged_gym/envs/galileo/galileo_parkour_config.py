# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GalileoParkourCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 32
        stuck_distance_threshold = (
            0.05  # 【放宽】X轴前进距离小于5cm视为不动（从2cm放宽）
        )
        stuck_time_threshold = (
            400  # 【放宽】持续400步（约2秒）不动且有接触力视为卡住（从200步放宽）
        )

    class terrain(LeggedRobotCfg.terrain):
        num_rows = 8  # 对应8个渐进式课程等级
        num_cols = 4  # 两列跳跃课程 + 两列钻爬课程
        terrain_dict = {"h_hurdle": 1.0}
        terrain_proportions = list(terrain_dict.values())
        num_goals = 10
        curriculum = True
        max_init_terrain_level = 0
        jump_columns = [0, 1]
        crawl_columns = [2, 3]
        demo_heights = [None, None, None, None]
        demo_progressive_cols = []
        curriculum_schedule = [
            {
                "jump": {
                    "sequence": [0.12, 0.15, 0.18, 0.2],
                    "x_spacing": [2.8, 3.1],
                    "y_range": [-0.05, 0.05],
                    "post_spacing": 0.50,
                },
                "crawl": {
                    "sequence": [0.30, 0.32, 0.34, 0.36],
                    "x_spacing": [2.8, 3.1],
                    "y_range": [-0.05, 0.05],
                    "post_spacing": 0.46,
                },
            },
            {
                "jump": {
                    "sequence": [0.18, 0.2, 0.22, 0.25],
                    "x_spacing": [2.6, 3.0],
                    "y_range": [-0.05, 0.05],
                    "post_spacing": 0.50,
                },
                "crawl": {
                    "sequence": [0.32, 0.35, 0.37, 0.4],
                    "x_spacing": [2.6, 3.0],
                    "y_range": [-0.05, 0.05],
                    "post_spacing": 0.45,
                },
            },
            {
                "jump": {
                    "sequence": [0.2, 0.25, 0.3, 0.32],
                    "x_spacing": [2.5, 2.9],
                    "y_range": [-0.06, 0.06],
                    "post_spacing": 0.48,
                },
                "crawl": {
                    "sequence": [0.35, 0.38, 0.42, 0.45],
                    "x_spacing": [2.5, 2.9],
                    "y_range": [-0.06, 0.06],
                    "post_spacing": 0.44,
                },
            },
            {
                "jump": {
                    "sequence": [0.22, 0.28, 0.34, 0.36],
                    "x_spacing": [2.4, 2.8],
                    "y_range": [-0.06, 0.06],
                    "post_spacing": 0.46,
                },
                "crawl": {
                    "sequence": [0.38, 0.42, 0.45, 0.48],
                    "x_spacing": [2.4, 2.8],
                    "y_range": [-0.06, 0.06],
                    "post_spacing": 0.43,
                },
            },
            {
                "jump": {
                    "sequence": [0.24, 0.3, 0.36, 0.4],
                    "x_spacing": [2.3, 2.7],
                    "y_range": [-0.07, 0.07],
                    "post_spacing": 0.45,
                },
                "crawl": {
                    "sequence": [0.4, 0.44, 0.46, 0.5],
                    "x_spacing": [2.3, 2.7],
                    "y_range": [-0.07, 0.07],
                    "post_spacing": 0.42,
                },
            },
            {
                "jump": {
                    "sequence": [0.26, 0.32, 0.38, 0.44],
                    "x_spacing": [2.2, 2.6],
                    "y_range": [-0.07, 0.07],
                    "post_spacing": 0.45,
                },
                "crawl": {
                    "sequence": [0.38, 0.42, 0.46, 0.5],
                    "x_spacing": [2.2, 2.6],
                    "y_range": [-0.07, 0.07],
                    "post_spacing": 0.42,
                },
            },
            {
                "jump": {
                    "sequence": [0.2, 0.3, 0.4, 0.45],
                    "x_spacing": [2.1, 2.4],
                    "y_range": [-0.08, 0.08],
                    "post_spacing": 0.44,
                },
                "crawl": {
                    "sequence": [0.35, 0.4, 0.45, 0.5],
                    "x_spacing": [2.1, 2.4],
                    "y_range": [-0.08, 0.08],
                    "post_spacing": 0.42,
                },
            },
            {
                "jump": {
                    "sequence": [0.2, 0.3, 0.4, 0.5],
                    "x_spacing": [2.0, 2.3],
                    "y_range": [-0.08, 0.08],
                    "post_spacing": 0.42,
                },
                "crawl": {
                    "sequence": [0.2, 0.3, 0.4, 0.5],
                    "x_spacing": [2.0, 2.3],
                    "y_range": [-0.08, 0.08],
                    "post_spacing": 0.42,
                },
            },
        ]

    class curriculum:
        enabled = True
        evaluation_window = 80

        stage1_name = "基础引导"
        stage1_terrain_types = [0, 1]
        stage1_obstacle_heights = [0.12, 0.20]
        stage1_num_obstacles = 4
        stage1_vel_range = [0.35, 0.5]
        stage1_success_threshold = 0.65
        stage1_min_iterations = 150
        stage1_obstacle_spacing = [2.7, 3.1]

        stage2_name = "低姿构建"
        stage2_terrain_types = [2, 3]
        stage2_obstacle_heights = [0.32, 0.42]
        stage2_num_obstacles = 4
        stage2_vel_range = [0.35, 0.6]
        stage2_success_threshold = 0.68
        stage2_min_iterations = 180
        stage2_obstacle_spacing = [2.5, 2.9]

        stage3_name = "混合策略"
        stage3_terrain_types = [0, 1, 2, 3]
        stage3_obstacle_heights = [0.24, 0.48]
        stage3_num_obstacles = 4
        stage3_vel_range = [0.4, 0.7]
        stage3_success_threshold = 0.72
        stage3_min_iterations = 220
        stage3_obstacle_spacing = [2.2, 2.7]

        stage4_name = "竞赛节奏"
        stage4_terrain_types = [0, 1, 2, 3]
        stage4_obstacle_heights = [0.2, 0.5]
        stage4_num_obstacles = 4
        stage4_vel_range = [0.45, 0.85]
        stage4_success_threshold = 0.78
        stage4_min_iterations = 260
        stage4_obstacle_spacing = [2.0, 2.4]

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
        stiffness = {"joint": 70.0}
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
        push_robots = False

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False

        class scales:
            tracking_goal_vel = 1.7
            tracking_yaw = 0.45
            lin_vel_z = -0.03
            ang_vel_xy = -0.04
            orientation = -0.2
            dof_acc = -2.5e-7
            collision = -0.18
            action_rate = -0.05
            delta_torques = -1.0e-7
            torques = -1.2e-5
            dof_error = -0.02
            feet_stumble = -1.4
            feet_edge = -0.5
            feet_contact_forces = -0.015
            stand_still = -0.15
            termination = -1.0
            body_obstacle_contact = -1.0
            base_height_stability = 0.85
            height_based_guidance = 0.85
            hurdle_alignment = 0.5
            strategy_efficiency = 0.6
            obstacle_approach_speed = 0.25
            hurdle_progress_trapezoid = 1.4
            feet_air_time = 0.12
            excessive_leg_width = 0.25
            feet_clearance = 0.9
            rear_leg_follow = 1.0
            feet_drag_penalty = -0.9
            successful_traversal = 2.5
            alive_bonus = 0.3

        soft_dof_pos_limit = 0.9
        base_height_target = 0.24
        base_height_normal = 0.36
        height_guidance_detection_range = 2.2
        base_height_stability_gain = 6.5
        obstacle_contact_force_threshold = 5.0
        low_hurdle_threshold = 0.32
        high_hurdle_threshold = 0.45
        obstacle_detection_range = 1.8
        post_contact_proximity_threshold = 0.5
        contact_force_penalty_scaling = 50.0
        max_contact_force_penalty = 2.0
        enable_contact_force_logging = True
        obstacle_safe_distance = 1.0
        max_leg_width = 0.5
        target_leg_width_near_hurdle = 0.35
        leg_width_detection_range = 3.0
        guidance_detection_range = 3.2
        jump_height_target = 0.44
        foot_clearance_margin = 0.06
        foot_clearance_front_window = 0.4
        foot_clearance_rear_window = 0.45
        foot_clearance_sigmoid_k = 35.0
        foot_clearance_tolerance = 0.06
        foot_drag_window = 0.32
        rear_follow_window = 0.38
        rear_follow_front_clearance = 0.08
        # 对准检测参数
        alignment_detection_range = 2.4
        y_alignment_tolerance = 0.14
        yaw_alignment_tolerance = 0.25
        # 引导奖励的高度容忍度（放宽以降低稀疏性）
        low_guidance_tolerance = 0.18
        high_guidance_tolerance = 0.14
        # 优化：跳跃阶段相关阈值（增强350mm+高栏的跳跃引导）
        preload_height_drop = 0.085
        preload_window = [
            1.3,
            0.35,
        ]
        clearance_window = 0.32
        post_landing_window = [
            0.85,
            0.12,
        ]
        clearance_margin = 0.06
        trapezoid_approach = [1.6, 0.5]
        trapezoid_plateau_half = 0.16
        trapezoid_exit = [0.35, 1.15]
        trapezoid_alignment_sigma = 0.22
        trapezoid_min_speed = 0.38
        trapezoid_stall_penalty = 0.65

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        num_commands = 4
        resampling_time = 6.0
        heading_command = True
        lin_vel_clip = 0.12
        curriculum_success_fraction = 0.4
        curriculum_success_threshold = 0.5
        curriculum_goal_threshold = 0.4

        class ranges:
            lin_vel_x = [0.35, 0.55]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0, 0]
            heading = [0, 0]

        class max_ranges(LeggedRobotCfg.commands.max_ranges):
            lin_vel_x = [0.3, 1.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [-1.6, 1.6]

        class crclm_incremnt(LeggedRobotCfg.commands.crclm_incremnt):
            lin_vel_x = 0.04

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
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.5
        continue_from_last_std = True

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.012
        num_mini_batches = 5
        num_learning_epochs = 6
        learning_rate = 2.5e-4
        desired_kl = 0.009

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
        num_steps_per_env = 64
        max_iterations = 120000
        save_interval = 100
