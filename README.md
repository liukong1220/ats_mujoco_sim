# ats_mujoco_sim

ATS 四驱四转舵轮底盘的 MuJoCo 动力学、场景、LiDAR/ToF 和 ROS 2 桥接仓库。

> 当前 `mujoco_navigation.launch.py` 默认启动 Nav2 对照链。它也能启动
> MINCO/MPC 旁路，但默认值和现有脚本不能自动证明 P3 Nav2-free。

## 目录

- [功能模块](#功能模块)
- [依赖](#依赖)
- [Quick Start](#quick-start)
- [启动入口](#启动入口)
- [接口](#接口)
- [配置与场景](#配置与场景)
- [数据流](#数据流)
- [测试与验证边界](#测试与验证边界)
- [参考与致谢](#参考与致谢)

## 功能模块

| 模块 | 说明 |
| :--- | :--- |
| 四舵轮模型 | 车体、四轮转向/驱动、接触和执行器反馈 |
| 传感器桥 | MID-360 风格点云、可选 ToF、真值与里程计输出 |
| 随机地图 | 从 YAML/seed 生成可复现地图资产 |
| RMUC2026 场景 | 加载比赛场地 mesh 和导航测试初始状态 |
| 速度桥 | `Twist -> /motion_control`，保持唯一底盘输入 |
| 导航编排 | map server、Nav2 对照、自研 MINCO/MPC 和 RViz |

## 依赖

- ROS 2 Humble、Python 3
- MuJoCo、NumPy、SciPy、Pillow、PyYAML
- `ats_nav_bringup`、`ats_rog_map*`、`minco_planner`、`ats_swerve_mpc`
- `nav2_map_server`/`nav2_lifecycle_manager` 当前用于默认对照 profile

## Quick Start

```bash
cd /home/ats/ATS_2026_snetry_test
source /opt/ros/humble/setup.bash
MAKEFLAGS=-j1 colcon build --base-paths src \
  --packages-up-to ats_mujoco_sim \
  --parallel-workers 1 --symlink-install
source install/setup.bash
```

无 viewer 的 CI/回归模式：

```bash
ros2 launch ats_mujoco_sim mujoco_navigation.launch.py \
  use_viewer:=false show_viewer:=false use_rviz:=false
```

## 启动入口

| Launch | 用途 |
| :--- | :--- |
| `launch/ats_mujoco_sim.launch.py` | 仅仿真本体与传感器 |
| `launch/planner_mujoco.launch.py` | 随机地图生成后启动仿真 |
| `launch/mujoco_navigation.launch.py` | 地图、导航、控制桥和 RViz 总入口 |
| `launch/rmuc_2026_mujoco.launch.py` | RMUC2026 mesh 场景 |

重要默认值：

| 参数 | 默认值 | 边界 |
| :--- | :--- | :--- |
| `launch_nav2` | `true` | 当前默认仍是 Nav2 基线 |
| `launch_swerve_mpc` | `false` | 开启后桥接 MPC 命令 |
| `launch_twist_bridge` | `true` | `/motion_control` 唯一 bridge |
| `use_viewer`/`show_viewer` | `true` | 自动回归需显式关闭 |
| `use_rviz` | `true` | headless 回归需显式关闭 |

## 接口

| Topic | 方向 | 说明 |
| :--- | :--- | :--- |
| `/localization` | 输出 | 仿真主里程计/定位输入 |
| `/registered_scan` | 输出 | ROGMap/terrain 配准点云 |
| `/local_pointcloud` | 输出 | LiDAR 原始仿真点云 |
| `/perception/tof/points_merged` | 输出 | 可选 ToF 合并点云 |
| `/cmd_vel_mpc` 或兼容 cmd topic | 输入到 bridge | profile 决定，必须只有一个 producer |
| `/motion_control` | bridge 输出 | MuJoCo 底盘唯一输入 |

## 配置与场景

- `config/random_map.yaml`：随机地图几何与 seed。
- `models/`：MuJoCo XML、mesh 和材质。
- `rviz/mujoco_navigation.rviz`：导航对照视图。
- `/tmp/ats_mujoco_sim_maps`：默认生成地图目录，不提交到 Git。

ROGMap 的正式 P2 参数不在本仓重复维护，位于导航仓：

- `ats_rog_map/config/rog_map_ground_planning_mujoco.yaml`
- `ats_rog_map_adapter/config/rog_map_ground_planning.yaml`

## 数据流

```text
MuJoCo state/contact
  -> localization/registered_scan/TF/sensor topics
  -> ROGMap + terrain + planning adapter
  -> Nav2 对照或 MINCO/MPC
  -> Twist bridge
  -> /motion_control
  -> MuJoCo actuators
```

`/cmd_vel_mpc` 有值、`/motion_control` 有值或机器人局部移动都不等价于闭环通过。
最终证据必须包含 owner 数量、终点误差、安全故障注入和物理接触 evaluator。

## 测试与验证边界

Nav2 基线：

```bash
scripts/test_mujoco_nav_chain.sh
```

P2 红框：

```bash
PLANNING_GRID_OWNER=rog_map P2_FAULT_CASE=none \
  TEST_PROFILE=red_box GOAL_TIMEOUT=180 \
  scripts/test_mujoco_minco_mpc_chain.sh
```

stale、service timeout、heartbeat 中断、unknown 和 unreachable 必须分别使用新的
`ROS_DOMAIN_ID` 与新的 MuJoCo launch 注入。

- **已验证**：README 中入口、默认参数和 topic 由 launch/package 静态核对。
- **未验证**：本轮未启动 MuJoCo，未测红框终点、footprint 冲突或物理接触。
- **未完成**：MuJoCo 默认 Nav2-free、自研 action 下扩大矩形/红框和独立 contact evaluator 门禁。

工作区级说明见
[仿真域说明](../../../docs/仿真域说明.md)。

## 参考与致谢

物理仿真基于 MuJoCo，导航与地图链复用 ATS 导航仓及其上游开源组件。模型、mesh
和第三方代码的许可证与来源以仓内对应文件为准。
