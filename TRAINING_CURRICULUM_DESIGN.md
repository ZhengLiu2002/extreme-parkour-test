# 训练课程设计详解

## 一、训练配置参数

### 基础配置（galileo_parkour_config.py）
- `num_rows = 6`: 6个难度级别（terrain_level: 0-5）
- `num_cols = 4`: 4种地形类型（col: 0-3）
- `curriculum = True`: 启用课程学习
- `max_init_terrain_level = 0`: 训练开始时所有环境从 level 0 开始

## 二、训练中的地形生成逻辑

### 1. 地形网格生成（terrain.py - curiculum()方法）

地形被组织成一个 6×4 的网格：
- **行（Row）**: 对应 `terrain_level` (0-5)，控制难度
- **列（Col）**: 对应 `terrain_type` (0-3)，控制地形类型

```
地形网格 (6 rows × 4 cols):
        col 0    col 1    col 2    col 3
row 0  [0,0]    [0,1]    [0,2]    [0,3]
row 1  [1,0]    [1,1]    [1,2]    [1,3]
row 2  [2,0]    [2,1]    [2,2]    [2,3]
row 3  [3,0]    [3,1]    [3,2]    [3,3]
row 4  [4,0]    [4,1]    [4,2]    [4,3]
row 5  [5,0]    [5,1]    [5,2]    [5,3]
```

### 2. 难度计算（difficulty）

```python
difficulty = terrain_level / (num_rows - 1) = terrain_level / 5
```

| terrain_level | difficulty | 说明 |
|--------------|-----------|------|
| 0 | 0.0 | 最简单 |
| 1 | 0.2 | |
| 2 | 0.4 | |
| 3 | 0.6 | |
| 4 | 0.8 | |
| 5 | 1.0 | 最困难 |

### 3. 地形类型（col）决定障碍物高度范围

#### 跳跃课程（col 0, 1）
```python
min_height = 0.02m (20mm)
max_height = 0.35m (350mm)
height = 0.02 + difficulty * (0.35 - 0.02) = 0.02 + difficulty * 0.33
```

#### 钻爬课程（col 2, 3）
```python
max_height = 0.60m (600mm)
min_height = 0.35m (350mm)
height = 0.60 - difficulty * (0.60 - 0.35) = 0.60 - difficulty * 0.25
```

### 4. 每个难度下的具体地形设计

#### terrain_level = 0 (difficulty = 0.0)

**跳跃课程 (col 0, 1)**:
- 障碍物高度: 0.02m = **20mm**
- 说明: 极低障碍物，几乎可以走过

**钻爬课程 (col 2, 3)**:
- 障碍物高度: 0.60m = **600mm**
- 说明: 极高障碍物，很容易钻过（机器人正常高度约420mm）

---

#### terrain_level = 1 (difficulty = 0.2)

**跳跃课程 (col 0, 1)**:
- 障碍物高度: 0.02 + 0.2 × 0.33 = **86mm**

**钻爬课程 (col 2, 3)**:
- 障碍物高度: 0.60 - 0.2 × 0.25 = **550mm**

---

#### terrain_level = 2 (difficulty = 0.4)

**跳跃课程 (col 0, 1)**:
- 障碍物高度: 0.02 + 0.4 × 0.33 = **152mm**

**钻爬课程 (col 2, 3)**:
- 障碍物高度: 0.60 - 0.4 × 0.25 = **500mm**

---

#### terrain_level = 3 (difficulty = 0.6)

**跳跃课程 (col 0, 1)**:
- 障碍物高度: 0.02 + 0.6 × 0.33 = **218mm**

**钻爬课程 (col 2, 3)**:
- 障碍物高度: 0.60 - 0.6 × 0.25 = **450mm**

---

#### terrain_level = 4 (difficulty = 0.8)

**跳跃课程 (col 0, 1)**:
- 障碍物高度: 0.02 + 0.8 × 0.33 = **284mm**

**钻爬课程 (col 2, 3)**:
- 障碍物高度: 0.60 - 0.8 × 0.25 = **400mm**

---

#### terrain_level = 5 (difficulty = 1.0)

**跳跃课程 (col 0, 1)**:
- 障碍物高度: 0.02 + 1.0 × 0.33 = **350mm**
- 说明: 最高跳跃难度

**钻爬课程 (col 2, 3)**:
- 障碍物高度: 0.60 - 1.0 × 0.25 = **350mm**
- 说明: 最低钻爬高度，需要极低姿态

**注意**: 两种课程在最高难度时高度相同（350mm），但策略不同：
- 跳跃课程：需要跳过去
- 钻爬课程：需要钻过去

## 三、训练中的环境分配

### 初始化时（legged_robot.py - _get_env_origins()）

```python
# terrain_levels: 从 [0, max_init_terrain_level] = [0, 0] 随机选择
# 所以训练开始时所有环境都是 terrain_level = 0

# terrain_types: 根据环境索引均匀分配到4个col
env 0-127:   col 0 (跳跃课程)
env 128-255: col 1 (跳跃课程)
env 256-383: col 2 (钻爬课程)
env 384-511: col 3 (钻爬课程)
```

### 课程升级（legged_robot.py - _update_terrain_curriculum()）

机器人根据表现自动升级：
- **升级条件**: 完成 ≥50% 的目标点（4/8个栏杆）
- **降级条件**: 完成 <25% 的目标点（2/8个栏杆）
- **最高难度**: 完成 level 7 后，在跳跃和钻爬课程之间切换

## 四、play.py 中 stage 模式是否完整复现训练？

### ✅ 是的，完全对应！

**play.py 的 stage 模式配置**:
```python
env_cfg.terrain.num_rows = 6      # 与训练一致
env_cfg.terrain.num_cols = 4      # 与训练一致
env_cfg.terrain.curriculum = True # 与训练一致
```

**对应关系**:
- `--curriculum_stage N` → `terrain_level = N` (0-5)
- `--terrain_type M` → `terrain_type = M` (0-3)，默认 0

**固定地形**:
- 环境创建后，固定 `terrain_levels = selected_terrain_level`
- 固定 `terrain_types = selected_terrain_type`

### 使用示例

```bash
# 测试训练中最简单的难度（terrain_level=0, 跳跃课程）
python play.py --task=galileo --exptid galileo_teacher \
  --terrain_mode stage --curriculum_stage 0 --terrain_type 0

# 测试训练中最高难度（terrain_level=5, 跳跃课程）
python play.py --task=galileo --exptid galileo_teacher \
  --terrain_mode stage --curriculum_stage 5 --terrain_type 0

# 测试钻爬课程（terrain_level=3, 钻爬课程）
python play.py --task=galileo --exptid galileo_teacher \
  --terrain_mode stage --curriculum_stage 3 --terrain_type 2

# 默认使用最高难度（terrain_level=5）
python play.py --task=galileo --exptid galileo_teacher \
  --terrain_mode stage --curriculum_stage -1
```

## 五、完整地形对照表

| terrain_level | difficulty | col 0,1 (跳跃) | col 2,3 (钻爬) |
|--------------|-----------|----------------|----------------|
| 0 | 0.0 | 20mm | 600mm |
| 1 | 0.2 | 86mm | 550mm |
| 2 | 0.4 | 152mm | 500mm |
| 3 | 0.6 | 218mm | 450mm |
| 4 | 0.8 | 284mm | 400mm |
| 5 | 1.0 | 350mm | 350mm |

## 六、总结

1. **训练使用 6×4 地形网格**，通过 `terrain_level` 和 `terrain_type` 控制
2. **play.py 的 stage 模式完全匹配训练配置**，可以精确复现任意难度
3. **通过 `--curriculum_stage` 和 `--terrain_type` 可以完整描述训练中的地形**

