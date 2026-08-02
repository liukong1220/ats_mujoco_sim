# ats_mujoco_sim

## 简介

ATS 四驱四转舵轮的 MuJoCo 动力学、LiDAR/ToF、定位输入、场景和 ROS 2 bridge。正式闭环
使用自研 ATS action、ROGMap adapter、MINCO 和 SE2 MPC，不启动 Nav2 map/lifecycle 或 `/plan`。

## 模块

- MuJoCo 车体、四轮驱动/转向、接触与 telemetry。
- 仿真 LiDAR/ToF、`/registered_scan`、`/odometry` 和定位输入。
- `static_map_publisher` 与 RMUC2026 场景。
- `twist_to_motion_ctrl`：`/cmd_vel_mpc -> /motion_control` 唯一 bridge。
- ROGMap/MINCO/MPC 自研闭环 launch 与 ROGMap/MINCO/MPC 诊断 RViz。

## 依赖

ROS 2 Humble、Python 3、MuJoCo、NumPy、SciPy、Pillow、PyYAML，以及工作区活动导航包。

## 构建

从工作区根目录构建：

```bash
source /opt/ros/humble/setup.bash
MAKEFLAGS=-j1 colcon build --base-paths src --packages-select ats_mujoco_sim \
  --symlink-install --parallel-workers 1
source install/setup.bash
```

`colcon build --base-paths src` 是正式 ROS 2 构建方式；单独 CMake 不构成包级验证。

## 启动

无界面 RMUC2026 闭环：

```bash
ROS_DOMAIN_ID=231 PLANNING_GRID_OWNER=rog_map \
ros2 launch ats_mujoco_sim rmuc_2026_mujoco.launch.py \
  use_viewer:=false show_viewer:=false use_rviz:=false
```

回归脚本会为每个场景启动新的 domain 和 MuJoCo 进程：

```bash
PLANNING_GRID_OWNER=rog_map TEST_PROFILE=red_box GOAL_TIMEOUT=180 \
scripts/test_mujoco_minco_mpc_chain.sh
```

## 接口

| Topic | producer | consumer |
| :--- | :--- | :--- |
| `/localization` | localization fusion | ROGMap/Goal Manager/MPC |
| `/registered_scan` | MuJoCo sensor bridge | ROGMap |
| `/cmd_vel_mpc` | `ats_swerve_mpc` | `twist_to_motion_ctrl` |
| `/motion_control` | `twist_to_motion_ctrl` | MuJoCo vehicle |
| `/swerve/telemetry` | MuJoCo vehicle | test/evaluator |

每条规划、速度和底盘输入链在运行图中必须各有一个 owner。

## 配置

`launch/rmuc_2026_mujoco.launch.py` 连接自研地图/规划/控制，并用根
`node_params.yaml` 统一 ROS 参数。地图由 `static_map_publisher` 保留 frame、origin/yaw、
resolution、占据语义和 transient-local QoS。RViz 默认展示 ROGMap occupy/inflated/unknown、
bounds、MINCO 与 MPC 真实 topic。

## 架构

```text
MuJoCo sensors -> localization/registered_scan -> ROGMap -> adapter -> MINCO
               -> Goal Manager -> SE2 MPC -> /cmd_vel_mpc -> bridge -> /motion_control
```

ROS graph 由独立 localization fusion 发布 `map -> odom`；仿真不再提供竞争 TF。

## 验证与限制

本轮 rectangle、red_box 和九个 P2/P3 fault 场景均在独立 domain 通过。red_box 终点误差为
`0.003696 m`，raw/reference 点数为 `37/708`，离散 footprint 冲突和
`contact_violation_count` 均为 `0`。每个 stale/unknown/unreachable/action failure 场景都验证
`emergency_stop=true -> /cmd_vel_mpc=0 -> /motion_control=0`。

接触 evaluator 仅覆盖现有 MuJoCo 计数器，不能替代连续 swept footprint 或实车物理验证。
