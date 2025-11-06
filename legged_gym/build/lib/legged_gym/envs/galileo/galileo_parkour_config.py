# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GalileoParkourCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 512
        stuck_distance_threshold = (
            0.05  # 【放宽】X轴前进距离小于5cm视为不动（从2cm放宽）
        )
        stuck_time_threshold = (
            400  # 【放宽】持续400步（约2秒）不动且有接触力视为卡住（从200步放宽）
        )

    class terrain(LeggedRobotCfg.terrain):
        num_rows = 6  # 6个难度级别，对应渐进课程阶段
        num_cols = 4  # 4种地形类型
        terrain_dict = {"h_hurdle": 1.0}
        terrain_proportions = list(terrain_dict.values())
        num_goals = 8
        curriculum = True
        max_init_terrain_level = 0  # 【关键】从最简单的level 0开始，避免一开始太难
        demo_heights = [None, None, None, None]
        demo_progressive_cols = []
        walkway_width = 0.8  # 中央跑道宽度（米）
        trench_depth = 0.4  # 两侧壕沟深度（米）
        hurdle_profiles = [
            {
                "num_hurdles": 0,
                "height_range": [0.0, 0.0],
                "x_range": [2.4, 2.8],
                "add_roughness": False,
            },
            {
                "num_hurdles": 2,
                "custom_heights": [0.20, 0.25],
                "x_range": [2.3, 2.6],
                "add_roughness": False,
            },
            {
                "num_hurdles": 3,
                "custom_heights": [0.20, 0.30, 0.35],
                "x_range": [2.2, 2.5],
            },
            {
                "num_hurdles": 4,
                "custom_heights": [0.20, 0.30, 0.40, 0.50],
                "x_range": [2.0, 2.4],
            },
            {
                "num_hurdles": 4,
                "custom_heights": [0.20, 0.30, 0.40, 0.50],
                "x_range": [1.9, 2.3],
                "post_spacing": 0.48,
            },
            {
                "num_hurdles": 4,
                "custom_heights": [0.20, 0.30, 0.40, 0.50],
                "x_range": [1.8, 2.1],
                "post_spacing": 0.45,
            },
        ]

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
            tracking_goal_vel = 1.2
            forward_progress_trapezoid = 1.0
            tracking_yaw = 0.5
            lin_vel_z = -0.02
            ang_vel_xy = -0.04
            orientation = -0.2
            dof_acc = -2.5e-7
            collision = -0.15
            action_rate = -0.05
            delta_torques = -1.0e-7
            torques = -1.0e-5
            dof_error = -0.02
            feet_stumble = -1.5
            feet_edge = -0.5
            feet_contact_forces = -0.01
            stand_still = -0.1
            termination = -1.0
            body_obstacle_contact = -1.0
            base_height_stability = 0.8
            height_based_guidance = 0.85
            hurdle_alignment = 0.45
            strategy_efficiency = 0.6
            obstacle_approach_speed = 0.25
            feet_air_time = 0.12
            excessive_leg_width = 0.3
            feet_clearance = 0.95
            rear_leg_follow = 0.85
            feet_drag_penalty = -0.8
            successful_traversal = 2.2
            alive_bonus = 0.25

        soft_dof_pos_limit = 0.9
        base_height_target = 0.25  # 钻爬时的目标高度
        base_height_normal = 0.36  # 正常站立高度
        height_guidance_detection_range = 2.0  # 【扩大】从1.2m到2.0m，更早引导高度调整
        base_height_stability_gain = 6.0
        obstacle_contact_force_threshold = 5.0
        low_hurdle_threshold = 0.32
        high_hurdle_threshold = 0.45
        obstacle_detection_range = 1.5
        post_contact_proximity_threshold = 0.5
        contact_force_penalty_scaling = 50.0
        max_contact_force_penalty = 2.0
        enable_contact_force_logging = True
        obstacle_safe_distance = 1.0
        max_leg_width = 0.5
        target_leg_width_near_hurdle = 0.35
        leg_width_detection_range = 3.0
        guidance_detection_range = 3.0
        jump_height_target = 0.42
        foot_clearance_margin = 0.05
        foot_clearance_front_window = 0.35
        foot_clearance_rear_window = 0.4
        foot_clearance_sigmoid_k = 35.0
        foot_clearance_tolerance = 0.05
        foot_drag_window = 0.3
        rear_follow_window = 0.35
        rear_follow_front_clearance = 0.05
        progress_speed_trapezoid = [0.18, 0.4, 0.75, 1.05]
        progress_heading_tolerance = 0.35
        approach_speed_trapezoid = [0.12, 0.35, 0.7, 1.05]
        approach_distance_trapezoid = [1.6, 1.0, 0.45, 0.15]
        # 对准检测参数
        alignment_detection_range = 2.0
        y_alignment_tolerance = 0.15
        yaw_alignment_tolerance = 0.3
        # 引导奖励的高度容忍度（放宽以降低稀疏性）
        low_guidance_tolerance = 0.20
        high_guidance_tolerance = 0.15
        # 优化：跳跃阶段相关阈值（增强350mm+高栏的跳跃引导）
        preload_height_drop = (
            0.08  # 优化：蓄力时下蹲更多（从0.06提高到0.08），增强起跳储能
        )
        preload_window = [
            1.2,
            0.4,
        ]  # 优化：蓄力窗口扩大（从[0.8,0.3]到[1.2,0.4]），更早开始引导
        clearance_window = 0.30  # 优化：越杆窗口扩大（从0.25到0.30），增加引导时间
        post_landing_window = [
            0.8,
            0.1,
        ]  # 优化：落地窗口扩大（从[0.6,0.1]到[0.8,0.1]），更长的稳定评估
        clearance_margin = 0.05  # 优化：安全余量增大（从0.03到0.05），避免蹭杆

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        num_commands = 4
        resampling_time = 6.0
        heading_command = True
        lin_vel_clip = 0.1  # 避免小命令被裁剪为0
        curriculum_success_fraction = 0.35  # 【降低】进一步放宽，让课程更容易推进
        curriculum_success_threshold = 0.45  # 【降低】从0.5降到0.45
        curriculum_goal_threshold = 0.4

        class ranges:
            lin_vel_x = [0.25, 0.85]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0, 0]
            heading = [0, 0]

        class max_ranges(LeggedRobotCfg.commands.max_ranges):
            lin_vel_x = [0.2, 1.1]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [-1.6, 1.6]

        class crclm_incremnt(LeggedRobotCfg.commands.crclm_incremnt):
            lin_vel_x = 0.03  # 【降低】从0.05改为0.03，更平缓的课程推进

    class curriculum:
        enabled = True
        evaluation_window = 80

        stage1_name = "平地推进"
        stage1_terrain_types = [0, 1, 2, 3]
        stage1_obstacle_heights = []
        stage1_num_obstacles = 0
        stage1_vel_range = [0.25, 0.45]
        stage1_success_threshold = 0.7
        stage1_min_iterations = 60
        stage1_obstacle_spacing = [2.6, 2.8]

        stage2_name = "双杆切入"
        stage2_terrain_types = [0, 1]
        stage2_obstacle_heights = [0.20, 0.25]
        stage2_num_obstacles = 2
        stage2_vel_range = [0.3, 0.55]
        stage2_success_threshold = 0.72
        stage2_min_iterations = 80
        stage2_obstacle_spacing = [2.3, 2.6]

        stage3_name = "三杆节奏"
        stage3_terrain_types = [1, 2]
        stage3_obstacle_heights = [0.20, 0.30, 0.35]
        stage3_num_obstacles = 3
        stage3_vel_range = [0.32, 0.60]
        stage3_success_threshold = 0.73
        stage3_min_iterations = 100
        stage3_obstacle_spacing = [2.1, 2.4]

        stage4_name = "竞赛四杆"
        stage4_terrain_types = [2, 3]
        stage4_obstacle_heights = [0.20, 0.30, 0.40, 0.50]
        stage4_num_obstacles = 4
        stage4_vel_range = [0.35, 0.80]
        stage4_success_threshold = 0.75
        stage4_min_iterations = 140
        stage4_obstacle_spacing = [1.8, 2.2]

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
        init_noise_std = 0.6
        continue_from_last_std = True

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
        num_steps_per_env = 48  # ↑ 从24提高到48，更长的轨迹更有利于学习步态
        max_iterations = 100000
        save_interval = 100
