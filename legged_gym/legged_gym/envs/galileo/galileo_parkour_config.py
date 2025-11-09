# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GalileoParkourCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 32
        stuck_distance_threshold = 0.05  # X轴前进距离小于5cm视为不动
        stuck_time_threshold = 200
        stuck_termination_multiplier = 4.0  # 卡住超过原阈值4倍时才强制终止

    class terrain(LeggedRobotCfg.terrain):
        num_rows = 6  # 6个难度级别，对应渐进课程阶段
        num_cols = 4  # 4种地形类型
        terrain_dict = {"h_hurdle": 1.0}
        terrain_proportions = list(terrain_dict.values())
        num_goals = 8
        curriculum = True
        max_init_terrain_level = 0  # 从最简单的level 0开始，避免一开始太难
        demo_heights = [None, None, None, None]
        demo_progressive_cols = []
        walkway_width = 0.8  # 中央跑道宽度（米）
        trench_depth = 0.8  # 两侧壕沟深度（米）

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # 抬高初始高度，避免触地
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 70.0}
        damping = {"joint": 1.5}
        action_scale = 0.4  # target angle = actionScale * action + defaultAngle
        decimation = 5  # 40Hz

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
        only_positive_rewards = False  # 是否将所有负奖励截断为0

        class scales:
            forward_progress_trapezoid = 2.0  # 基于梯形积分的前进速度奖励
            lin_vel_z = -0.02  # 垂直速度惩罚
            ang_vel_xy = -0.04  # 横滚/俯仰角速度惩罚
            orientation = -0.2  # 姿态偏差惩罚
            action_rate = -0.01  # 连续动作差分惩罚（平滑控制）
            torques = -1.0e-6  # 扭矩幅值惩罚，抑制过大驱动
            collision = -0.15  # 非脚部碰撞惩罚
            feet_stumble = -1.5  # 脚部被底杆绊住惩罚
            stand_still = -0.1  # 原地不动惩罚
            termination = -2.0  # 终止惩罚
            alive_bonus = 0.25  # 存活奖励
            centerline_alignment = 0.5  # 保持赛道中心
            base_height_strategy = 0.8  # 根据障碍调整基座高度
            feet_clearance_simple = -1.2  # 底杆附近抬脚不足惩罚
            feet_air_time = 0.3  # 鼓励合适的腾空时间，避免碎步

        progress_speed_trapezoid = [0.18, 0.4, 0.75, 1.05]  # 前进速度梯形参数
        progress_heading_tolerance = 0.35  # 前进方向容差（弧度）
        base_height_target = 0.15  # 遇到高障碍时的低姿态目标高度（米）
        base_height_normal = 0.45  # 平地或正常移动的基座高度（米）
        jump_height_target = 0.7  # 面对低障碍时的抬升目标高度（米）
        obstacle_detection_range = 1.0  # 前方障碍检测范围（米）
        base_height_high_threshold = 0.35  # 判断高障碍的高度阈值（米）
        base_height_strategy_gain = 6.0  # 高度策略的指数塑形增益

        high_hurdle_threshold = 0.35  # 兼容旧逻辑使用的高障碍阈值（米）
        centerline_tolerance = 0.25  # 赛道中心线容忍度（米）
        feet_clearance_height = 0.15  # 障碍附近脚部最低安全高度（米）
        feet_clearance_window = 0.80  # 判定靠近障碍的窗口（米）
        feet_air_time_target = 0.24  # 理想腾空时间（秒）
        feet_air_time_tolerance = 0.12  # 腾空时间容忍区间（秒）

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        num_commands = 4
        resampling_time = 6.0
        heading_command = True
        lin_vel_clip = 0.1

        class ranges:
            lin_vel_x = [0.2, 1.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0, 0]
            heading = [0, 0]

        class max_ranges(LeggedRobotCfg.commands.max_ranges):
            lin_vel_x = [0.2, 1.1]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [-1.6, 1.6]

        class crclm_incremnt(LeggedRobotCfg.commands.crclm_incremnt):
            lin_vel_x = 0.03

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
        num_steps_per_env = 48
        max_iterations = 100000
        save_interval = 100
