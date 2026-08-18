# TASK-005 十一动作动态控制器执行计划

> 执行说明：Linux VS Code Codex 直接执行本文档。不要调用或假设安装任何 superpowers
> 技能、插件、子代理或编排命令。目录名只保留仓库历史约定。

## 目标与起点

目标是在 TASK-004 已验收动态力和力矩环境上，实现固定一秒的十一动作控制器、平面定量验收、
胃部无适配迁移以及一次按键触发一次动作的连续可视化。

Linux 必须先获取 workflow/TASK-005-eleven-action-dynamic-controller 的远端精确 head，确认其
历史包含 TASK-004 基线 87a80adcc367a3210fc1f8cfadea410f340e3918，然后从规划 head 创建
feature/TASK-005-eleven-action-dynamic-controller。不得从 main、旧 TASK-004 报告头或本地
脏工作树开始，不得直接合并到 main。

开始编辑前完整阅读 AGENTS.md、活动合同、设计文档、本文计划和 TASK-004 最终报告。记录
base、planning head、分支和初始 status。TASK-004 的四动作实现、冻结 profile、报告和证据
都必须保留。

## 全局实施规则

所有新动作运行期只允许使用世界系 COM force/torque 和 PhysX 积分。正常 reset 以外严禁
write_root_pose_to_sim、write_root_velocity_to_sim、set_transforms、set_velocities、
kinematic、teleport、projection 或任何等价状态写入。

本任务明确授权新 controller 和 validator 使用胶囊真值、接触和表面网格，但不得把这些字段
加入 policy observation。不得修改 reward、coverage、VLM、Actor-Critic 或训练配置。

物理固定 240 Hz；控制器 apply 在每个物理子步更新。默认 render interval 为 2，即目标
120 FPS，CLI 可选 60、120、240。环境动作处理 cadence 可以保持 TASK-004 的 decimation 4，
但每次动作的物理子步计数必须独立且精确为 240。

tests 和 docs/superpowers 在当前 .gitignore 中被忽略。新增这些文件时必须使用
git add -f；不得因为普通 git add 没有包含文件而漏交测试或计划。

## Task 1：建立独立十一动作数据合同和冻结 profile

**新建文件**

tests/eleven_action/conftest.py

tests/eleven_action/test_types_and_profile.py

configs/eleven_action/dynamic_profile.json

source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action/__init__.py

source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action/types.py

source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action/config.py

**第一步：先写失败测试**

测试固定 ID 0 至 10、公开动作没有 -1、ActionResult 只有 COMPLETED、REJECTED、FAULT，
内部 lifecycle 可表达 READY_HOLD、EXECUTING、FAULTED。测试 profile 严格拒绝缺键、额外键、
非有限数值和越界时序，并产生 64 字符规范化 JSON SHA-256。

profile 初值必须复制 TASK-004 的已验收权限并加入下列固定字段：

~~~json
{
  "schema_version": "task005_eleven_action_dynamic_v1",
  "capsule_mass_kg": 0.0057349997,
  "capsule_radius_m": 0.0065,
  "capsule_cylinder_half_length_m": 0.006,
  "physics_hz": 240,
  "action_duration_s": 1.0,
  "view_motion_duration_s": 0.8,
  "view_hold_duration_s": 0.2,
  "view_cone_half_angle_deg": 15.0,
  "move_min_tilt_deg": 60.0,
  "contact_history_s": 0.05,
  "axis_kp_nm_per_rad": 0.02,
  "axis_kd_nms_per_rad": 0.0016,
  "support_kp_n_per_m": 10.0,
  "support_kd_ns_per_m": 0.4,
  "support_normal_preload_n": 0.1,
  "total_force_limit_n": 1.25,
  "total_torque_limit_nm": 0.02,
  "force_slew_limit_n_per_s": 50.0,
  "torque_slew_limit_nm_per_s": 0.2,
  "support_drift_limit_m": 0.002,
  "move_force_k": 0.9,
  "move_force_k_max": 3.0,
  "move_force_k_step": 0.1
}
~~~

允许添加实现必需的有限容差字段，但不得改变设计文档的动作语义。动作时序验证必须证明
1.0 秒乘 240 Hz 精确等于 240 子步，0.05 秒精确等于 12 子步。

**第二步：运行并确认失败**

~~~bash
./run_isaaclab.sh -p -m pytest tests/eleven_action/test_types_and_profile.py -q
~~~

预期因模块和 profile 尚不存在而失败。

**第三步：最小实现并通过测试**

ElevenActionId 使用 IntEnum，ActionResult 和 Lifecycle 使用 Enum。遥测至少包含 action_id、
request_id、substep_index、result、constrained、direction_degenerate、start/end axis、surface
normal、support anchor/drift、move direction/displacement、contact flags、force、torque、slew
flags 和 profile digest。

不得复用 TASK-004 的 PrimitiveId 或四浮点 pulse decoder 作为公开协议。可以复用其不可变
向量校验模式和 wrench 数据结构。

**第四步：提交**

~~~bash
./run_isaaclab.sh -p -m pytest tests/eleven_action/test_types_and_profile.py -q
git add configs/eleven_action/dynamic_profile.json source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action
git add -f tests/eleven_action/conftest.py tests/eleven_action/test_types_and_profile.py
git commit -m "feat: define eleven-action dynamic contract"
~~~

## Task 2：实现统一表面查询、相机 frame 和接触分类

**新建文件**

tests/eleven_action/test_surface_query.py

tests/eleven_action/test_geometry_and_contacts.py

source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action/geometry.py

source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action/surface_query.py

source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action/contact_history.py

**第一步：写纯函数失败测试**

构造非均匀面积三角网格，验证最近三角形的一环包含所有共享顶点面，法向先统一朝内再按面积
加权。验证平面 adapter 和胃部 adapter 输入相同几何后返回相同结果。验证 camera frame 正交，
八个画面方向与数字九宫格一致，外围目标与起始光轴夹角恰好 15 度。

构造 spherocylinder 的 upright、60 度、side 和随机旋转，验证动作开始得到的底部材料点局部
offset 在整个动作中保持不变，而世界点从实时姿态重建。验证切向支撑误差没有法向分量。

用接触点轴向坐标验证：大于 6 mm 为相机半球，小于负 6 mm 为非相机半球，绝对值不超过
6 mm 为圆柱侧壁。接触历史保持最近 12 个子步，不使用 impulse、接触点数量或稳定比例门限。

**第二步：实现纯 SurfaceQuery**

优先复用 controllers/ideal_surface 中经过测试的 Spherocylinder、几何归一化和
SurfaceNavigationMesh 数据读取能力，但不要修改 ideal_surface 的动作、mask、kinematic 或
状态写入代码。新增 LocalSurfaceHit，使其返回 nearest point、nearest triangle、one-ring
IDs、area-weighted inward normal 和 geometry digest。

flat provider 在代码中生成覆盖测试区域的规则三角化平面，normal 为世界正 Z。stomach
provider 后续通过 reference_from_stage 读取既有胃壁 prim 和已验证 inward sign。不要增加
新 USD。

**第三步：实现接触历史纯缓冲**

ContactSample 保存 physics_substep、世界点、法向和可用 impulse，仅分类时忽略 impulse。
SideContactHistory 提供 had_sidewall_contact(last_n_substeps=12) 和
camera_constraints(last_n_substeps=12)。历史清理必须按物理子步而不是环境 step。

**第四步：运行、提交**

~~~bash
./run_isaaclab.sh -p -m pytest tests/eleven_action/test_surface_query.py tests/eleven_action/test_geometry_and_contacts.py -q
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action
git add -f tests/eleven_action/test_surface_query.py tests/eleven_action/test_geometry_and_contacts.py
git commit -m "feat: add local surface and contact geometry"
~~~

## Task 3：用纯 NumPy 实现 240 子步状态机

**新建文件**

tests/eleven_action/test_trajectory.py

tests/eleven_action/test_controller.py

source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action/trajectory.py

source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action/controller.py

**第一步：测试一秒边界与九宫格轨迹**

对 8 个 VIEW 验证子步 0 从真实光轴开始，前 192 个子步使用 quintic swing，后 48 个子步保持
末端 15 度目标，第 240 子步产生结果并回到 READY_HOLD。HOLD_VIEW 同样必须 240 子步。
连续同方向 VIEW 必须以第二次真实起点重新生成相对 15 度目标，不能复用世界绝对目标。

测试在动作 100 个子步发生相机接触时，下一 controller update 冻结真实光轴并移除朝墙内
驱动力矩，动作仍在 240 子步 COMPLETED。测试持续 camera constraint 会把继续推墙的 VIEW
变成完整一秒 constrained HOLD，反向 VIEW 可离开；12 个无接触子步后约束清除。

**第二步：测试 MOVE 门限和三段时序**

在 59.999 度、无侧壁接触、有过期接触时请求 MOVE，均执行 240 子步 HOLD 并返回 REJECTED。
在 60、75、90 度和最近侧壁接触时均锁存为 accepted。accepted MOVE 的子步 0 至 59 为零
wrench，子步 60 重新查询并冻结方向，子步 60 至 179 只施加 COM force，子步 180 至 239 为
零 wrench。整个 MOVE 的 torque 必须为零。

验证 POS 与 NEG 严格相反且同时垂直于局部法向和长轴切向投影。验证 0.25 秒方向退化时返回
COMPLETED 与 direction_degenerate，不返回 FAULT 或 REJECTED。

**第三步：实现控制器**

Controller.submit 只在 READY_HOLD 接受一个 ID。Controller.step 每个 physics dt 调用一次，
只依据 integer substep 选择阶段，避免浮点时间累计少一帧或多一帧。

VIEW/HOLD 开始时冻结 surface normal、camera frame、底部材料点和切向 anchor。姿态 torque
只控制 directed optical axis 的 swing，不生成平行光轴的计划 twist。支撑力在 frozen tangent
plane 内闭环，转换为等效 COM wrench，并和姿态 torque 一起做幅值及 vector slew。

动作结束时从真实状态建立新的 READY_HOLD target 和支撑点。不得先输出一帧零 wrench；从上个
wrench 经过同一个 slew history 平滑转入 hold。MOVE 明确的零段仍按合同执行。

**第四步：严格 FAULT 测试**

只有 nonfinite state、程序异常、状态机子步越界或不可恢复内部状态损坏进入 FAULTED。有限大
角速度、接触、滑动、低位移、方向退化和轻微支撑漂移都必须保持普通结果。

**第五步：运行、提交**

~~~bash
./run_isaaclab.sh -p -m pytest tests/eleven_action/test_trajectory.py tests/eleven_action/test_controller.py -q
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action
git add -f tests/eleven_action/test_trajectory.py tests/eleven_action/test_controller.py
git commit -m "feat: implement one-second eleven-action controller"
~~~

## Task 4：接入 Isaac Lab 动态刚体和实时接触

**新建文件**

tests/eleven_action/test_action_term.py

tests/eleven_action/test_runtime_contract.py

source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/eleven_action.py

**修改文件**

source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.py

source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.pyi

**第一步：写 decoder 和禁止项测试**

公开 request ID 只能是 0 至 10。ActionTerm 内部允许 -1 代表本环境 step 没有新请求。重复的
相同 ID 在两次独立 READY 边界都必须可执行；EXECUTING 时的新 ID 必须丢弃且不排队。

静态扫描新 runtime controller、action term 和 launcher，禁止所有 pose/velocity setter、
kinematic 和 robot/magnet API。测试 action_dim 为 1，policy observation 未增加真值字段。

**第二步：实现 ActionTerm**

从 root_com_pose_w 读取 COM 位置，从 root_link_pose_w 读取几何姿态，从 root_com_vel_w 读取
速度，沿用 TASK-004 已验证的 quaternion 坐标转换。检查物理 dt 为 1/240、num_envs 为 1、
胶囊非运动学、重力和 CCD 开启。

订阅 PhysX contact report，按 collider path 只收集胶囊接触，在每个 physics substep 把实时
contact samples 送入 controller。胃部 mesh runtime 使用既有 validated prim path 和 geometry
hash；flat task 使用代码生成平面 provider。

每个子步只把 controller 输出复制进独立 torch tensor，然后调用
permanent_wrench_composer.set_forces_and_torques_index，positions=None、is_global=True。
不得复用可能被 PhysX 后续修改的 NumPy view。

ActionTerm 对外提供 ready、last_result、telemetry、substep telemetry、profile digest 和
discarded_request_count。FAULT 只在严格异常时设置 environment termination signal；不得把
普通状态直接转换成 hard failure。

**第三步：实时前置检查**

新增 scripts/eleven_action/inspect_eleven_action_prerequisites.py，输出质量、惯量、几何 axis、
接触字段、mesh digest、physics/render cadence、dynamic/gravity/CCD、COM wrench API 和禁止项
扫描。平面和胃部都必须运行。

**第四步：运行、提交**

~~~bash
./run_isaaclab.sh -p -m pytest tests/eleven_action/test_action_term.py tests/eleven_action/test_runtime_contract.py -q
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp scripts/eleven_action/inspect_eleven_action_prerequisites.py
git add -f tests/eleven_action/test_action_term.py tests/eleven_action/test_runtime_contract.py
git commit -m "feat: apply eleven-action COM wrench in Isaac Lab"
~~~

## Task 5：创建平面任务、键盘协议和连续启动器

**新建文件**

tests/eleven_action/test_keyboard.py

tests/eleven_action/test_task_cfg.py

tests/eleven_action/test_launcher_protocol.py

source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/eleven_action_keyboard.py

source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_eleven_action_flat_env_cfg.py

scripts/eleven_action/teleop_eleven_action.py

**修改文件**

source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/__init__.py

source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py

**第一步：键盘测试**

验证数字小键盘 7/8/9、4/5/6、1/2/3 正确映射 VIEW，5 映射 HOLD，Q/E 映射 NEG/POS。
同一物理按下只发一次，release 后才可再次发；EXECUTING 时 launcher 丢弃而非缓存。测试
Backspace、F12、Esc，并验证无 overlay 依赖。

**第二步：平面任务**

注册 Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0。继承 TASK-004 平面动态场景，
只替换为 eleven_action term。保持 240 Hz physics、decimation 4、CCD 和相机。render_fps
参数严格映射 60/120/240 到 interval 4/2/1，默认 120。不要强制 CPU；接受 CLI device，
交互默认使用可用的 cuda:0，并在 session 中记录实际 device。

**第三步：连续启动器**

启动后持续 env.step 内部 no-request 值 -1，使 READY_HOLD、物理和渲染一直更新。只在 READY
收到一次按键时发一个 ID 一次，下一 env step 恢复 -1。执行期间收到的动作按键立即丢弃并
增加 discarded count。

终端只在 READY、REQUEST、RESULT、RESET、SNAPSHOT、FAULT 和 SESSION 事件打印，不按每帧
刷屏。RESULT 打印 240 子步、结果、constrained、初末角、支撑漂移、MOVE signed displacement、
接触和 profile digest。默认不显示 HUD。

**第四步：实时 smoke**

~~~bash
./run_isaaclab.sh -p scripts/eleven_action/inspect_eleven_action_prerequisites.py --task Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0 --device cuda:0 --headless
./run_isaaclab.sh -p scripts/eleven_action/teleop_eleven_action.py --task Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0 --device cuda:0 --render_fps 120 --capsule_camera_view
~~~

人工 smoke 至少依次按 8、6、5、Q、E，确认流程是按键、完整动作、READY_HOLD 观察、下一按键，
动作中按键不会补执行。

**第五步：提交**

~~~bash
./run_isaaclab.sh -p -m pytest tests/eleven_action/test_keyboard.py tests/eleven_action/test_task_cfg.py tests/eleven_action/test_launcher_protocol.py -q
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab scripts/eleven_action/teleop_eleven_action.py
git add -f tests/eleven_action/test_keyboard.py tests/eleven_action/test_task_cfg.py tests/eleven_action/test_launcher_protocol.py
git commit -m "feat: add eleven-action flat keyboard task"
~~~

## Task 6：平面姿态校准、MOVE 最小共享力和随机验收

**新建文件**

tests/eleven_action/test_calibration.py

tests/eleven_action/test_acceptance_summary.py

scripts/eleven_action/calibrate_eleven_action.py

scripts/eleven_action/validate_eleven_action_flat.py

**第一步：分离 calibration 和 validation seeds**

calibration 使用固定 seed 42，validation 使用固定 seed 20260818。reset 可以写随机起始状态，
动作期间不能写。每个 trial 从几何支持高度初始化，短暂 settle 后记录真实初始倾角、方位、
roll、接触历史和 surface normal；按 post-settle 状态分层。

**第二步：VIEW 确定性校准**

先使用 profile 初值在独立 canonical states 验证全部八个方向。若失败，只允许在下列网格内
选择一组所有 VIEW 和 HOLD 共享参数：

~~~text
axis_kp_nm_per_rad = [0.005, 0.01, 0.02]
axis_kd_nms_per_rad = [0.0008, 0.0016, 0.0032]
support_kp_n_per_m = [5.0, 10.0, 20.0]
support_kd_ns_per_m = [0.2, 0.4, 0.8]
~~~

总力、总力矩和 slew 上限保持 TASK-004 值。按最大角误差、最大支撑漂移、总 wrench 积分的
字典序选择通过者，并保存全部尝试。不得按方向使用不同增益。无候选通过则停止并返回
needs_decision。

**第三步：MOVE \(k\) 标定**

对 \(k=0.9,1.0,\ldots,3.0\) 按升序运行 POS 和 NEG 各至少 10 个相同配对合法随机状态。每个
方向至少 90% 的 signed displacement 达到 5 mm 才算该 \(k\) 通过。选择第一个两个方向都
通过的共享 \(k\)，规范化写入 tracked profile。3.0 仍失败则停止，不得继续增力。

**第四步：正式单动作验收**

validate_eleven_action_flat.py 必须自动生成机器可读 summary.json 和 samples.jsonl。八个 VIEW
各持续采样到至少 10 个 unblocked valid trials；碰壁样本单独分类。HOLD 至少 10 个随机 trial。
每个 MOVE 至少 10 个 valid 和 10 个 invalid trial，invalid 同时覆盖 angle 与 contact 两类。

summary evaluator 必须按设计文档精确判断，并保证低效果 MOVE 是 COMPLETED 但 batch FAIL，
正常碰壁是 COMPLETED 且不进入角度精度分母，有限不稳定不是 FAULT。

**第五步：执行**

~~~bash
./run_isaaclab.sh -p scripts/eleven_action/calibrate_eleven_action.py --task Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0 --device cuda:0 --seed 42 --write_profile configs/eleven_action/dynamic_profile.json --headless
./run_isaaclab.sh -p scripts/eleven_action/validate_eleven_action_flat.py --task Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0 --device cuda:0 --seed 20260818 --render_fps 120 --headless
~~~

只有输出 ELEVEN_ACTION_FLAT_ACCEPTANCE_PASS 后才能继续。

**第六步：测试、冻结、提交**

~~~bash
./run_isaaclab.sh -p -m pytest tests/eleven_action/test_calibration.py tests/eleven_action/test_acceptance_summary.py -q
git add configs/eleven_action/dynamic_profile.json scripts/eleven_action/calibrate_eleven_action.py scripts/eleven_action/validate_eleven_action_flat.py
git add -f tests/eleven_action/test_calibration.py tests/eleven_action/test_acceptance_summary.py
git commit -m "test: pass randomized flat eleven-action gate"
~~~

记录最终 canonical profile digest。从此提交后禁止修改 profile。

## Task 7：执行 100 动作连续压力测试

**新建文件**

tests/eleven_action/test_stress_sequence.py

scripts/eleven_action/stress_eleven_action.py

**第一步：生成并锁定序列**

使用 seed 20260818 生成 100 个 ID 并保存 sequence JSON 到外部 session 证据目录，同时在
summary 记录其 SHA-256。每个 0 至 10 至少出现 5 次。序列构造器必须包含重复、反向 VIEW、
过早 MOVE、合法 POS/NEG、相机接触和离开接触的片段；只能通过动作 ID 组织，不得在动作间
写状态或 reset。

**第二步：平面执行**

~~~bash
./run_isaaclab.sh -p scripts/eleven_action/stress_eleven_action.py --task Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0 --device cuda:0 --seed 20260818 --actions 100 --render_fps 120 --headless
~~~

检查 100 个非 FAULT 结果各 240 子步、每个 ID 计数、REJECTED 来源、合法双向 MOVE、碰壁
constraint 清除、状态机 READY 回归、连续 support recapture、无主动 reset 和无运行期状态写入。
输出 ELEVEN_ACTION_STRESS_PASS。

**第三步：测试、提交**

~~~bash
./run_isaaclab.sh -p -m pytest tests/eleven_action/test_stress_sequence.py -q
git add scripts/eleven_action/stress_eleven_action.py
git add -f tests/eleven_action/test_stress_sequence.py
git commit -m "test: pass continuous eleven-action stress"
~~~

## Task 8：无适配迁移到胃部并提供可视化

**新建文件**

tests/eleven_action/test_no_stomach_adaptation.py

source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_eleven_action_stomach_env_cfg.py

**修改文件**

source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py

**第一步：写隔离测试**

注册 Template-Robotarm-Magnetic-Eleven-Action-Stomach-Lab-v0。测试 flat 和 stomach 的
controller_cfg_values 及 profile_sha256 完全相同；胃部只替换 SurfaceQuery provider，并继承
TASK-004 胃部 scene、reset、dynamic/CCD/contact/camera/timing。扫描胃部 wrapper，不允许
gain、threshold、move_force、recovery、state setter 或 task-specific controller branch。

**第二步：实现场景 wrapper**

胃部 provider 使用既有 prim path、geometry digest 和 inward sign。不得修改胃部 USD 或
TASK-004 的相机眩光修复。profile 文件保持冻结。

**第三步：运行胃部前置检查和相同 100 ID**

将平面 session 中保存的 sequence JSON 作为输入，不重新生成、不根据胃部状态改序列。

~~~bash
./run_isaaclab.sh -p scripts/eleven_action/inspect_eleven_action_prerequisites.py --task Template-Robotarm-Magnetic-Eleven-Action-Stomach-Lab-v0 --device cuda:0 --headless
./run_isaaclab.sh -p scripts/eleven_action/stress_eleven_action.py --task Template-Robotarm-Magnetic-Eleven-Action-Stomach-Lab-v0 --device cuda:0 --sequence_file <flat-session>/sequence.json --render_fps 120 --headless
./run_isaaclab.sh -p scripts/eleven_action/teleop_eleven_action.py --task Template-Robotarm-Magnetic-Eleven-Action-Stomach-Lab-v0 --device cuda:0 --render_fps 120 --capsule_camera_view
~~~

交互会话逐个触发九宫格方向、Q、E，并在动作间观察。保存至少一组外部视图、胶囊相机快照和
终端 session。胃部不要求重新达到 5 mm，不因碰壁或低效果改 profile。记录实际视觉结果，不
代替用户作“好用”的主观结论。

**第四步：提交**

~~~bash
./run_isaaclab.sh -p -m pytest tests/eleven_action/test_no_stomach_adaptation.py tests/eleven_action/test_task_cfg.py -q
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab
git add -f tests/eleven_action/test_no_stomach_adaptation.py
git commit -m "feat: migrate eleven actions unchanged to stomach"
~~~

## Task 9：完整回归、文档、报告与推送

**新建文件**

docs/ELEVEN_ACTION_DYNAMIC_CONTROLLER.md

handoffs/reports/TASK-005-eleven-action-dynamic-controller-report.md

**修改文件**

docs/PROJECT_RUN_LOG.md

**第一步：回归**

~~~bash
./run_isaaclab.sh -p -m pytest tests/eleven_action tests/local_primitives tests/dynamic_force -q --disable-warnings
./run_isaaclab.sh -p -m pytest tests/ideal_surface tests/coverage tests/action_layer/test_atomic_protocol.py tests/action_layer/test_executor.py tests/action_layer/test_safety.py tests/action_layer/test_atomic_stomach_teleop_cfg.py tests/action_layer/test_atomic_keyboard_protocol.py -q --disable-warnings
./run_isaaclab.sh -p scripts/dynamic_force/inspect_dynamic_force_prerequisites.py --task Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0 --num_envs 1 --headless
./run_isaaclab.sh -p scripts/action_layer/validate_coverage_geometry.py --check all
./run_isaaclab.sh -p scripts/action_layer/validate_atomic_stomach_teleop.py --num_envs 1 --coverage_samples 5
./run_isaaclab.sh -p -m compileall -q scripts/eleven_action source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab
git diff --check
~~~

若 CUDA 接触回调存在环境限制，可以用 CPU 复核原因，但不得静默改变最终 flat 与 stomach 的
设备、profile 或数值。所有 fallback 都写入偏差。

**第二步：禁止项扫描**

对新增 runtime controller、action term、task cfg 和 launcher 扫描状态 setter、kinematic、
robot、magnet、reward、coverage 和 policy observation 真值。reset fixture 中合法状态写入必须
限定在 reset 路径并在报告说明。

**第三步：报告**

报告必须包含 base、planning head、feature head、分支、全部提交、实际命令、测试数量、每个
动作随机样本数、VIEW 角误差和支撑漂移、碰壁取消延迟、HOLD 指标、MOVE \(k\)、双向成功率、
REJECTED/COMPLETED/FAULT 计数、100 动作结果、flat/stomach digest、render schedule 与实测
wall FPS、胃部观察、偏差和未验证声明。

大日志、视频、数据和快照留在 Git 外，报告路径、字节数和 SHA-256。不得把主观胃部可用性
写成已验证事实。

Disposition 使用规则如下：平面门禁未通过或 \(k=3.0\) 失败为 needs_decision；平面通过但
胃部无法初始化、无法渲染或出现系统 FAULT 为 partial；平面通过、相同 digest 胃部完成压力
和交互证据、回归通过为 complete，即使个别胃部动作因正常接触低效果。

**第四步：最终提交和推送**

~~~bash
git add docs/ELEVEN_ACTION_DYNAMIC_CONTROLLER.md docs/PROJECT_RUN_LOG.md handoffs/reports/TASK-005-eleven-action-dynamic-controller-report.md
git commit -m "docs: report eleven-action dynamic controller"
git diff --check
git status --short
git push -u origin feature/TASK-005-eleven-action-dynamic-controller
~~~

不得合并 main。报告中给出远端分支和 head，等待 Windows 端验收。

## 最终人工核对

最终核对必须确认公开动作只有 0 至 10；每个正常请求恰好 240 子步；VIEW 是相机坐标相对
15 度；底部支撑点按动作重算并冻结；相机碰壁停止仍为 COMPLETED；MOVE 只按 60 度加最近
侧壁接触判定；REJECTED 只有 MOVE 前置失败；FAULT 没有吞掉有限物理不稳定；MOVE 使用最小
共享 \(k\)；动作中按键不排队；平面和胃部同 digest；所有运行期运动只来自 force/torque；
胃部没有适配；没有改动资产、相机标定、物理参数、奖励、覆盖或训练代码。
