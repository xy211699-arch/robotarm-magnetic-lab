# 胶囊相机实时窗口

## 功能

启动参数 `--capsule_camera_view` 会在 Isaac Lab 的交互式主视口之外创建
`Capsule Camera | Policy RGB | <sensor Hz>` 窗口。

- 数据源：模型输入接口 `observations["vision"]["rgb"]` 对应的圆形广角处理链；
- 图像规格：1280×720、RGB、120°圆形视场；
- 刷新频率：跟随当前任务的相机更新频率；胃部任务为1 Hz，台架任务保持30 Hz；
- 策略频率仍为20 Hz，不因显示窗口而改变；
- 未添加参数时不创建窗口，也不执行额外的广角重映射和图像上传。

## 启动

零动作/场景观察：

```bash
./run_isaaclab.sh -p scripts/zero_agent.py \
  --task Template-Robotarm-Magnetic-Stomach-Lab-v0 \
  --num_envs 1 \
  --capsule_camera_view
```

磁驱动测试：

```bash
./run_isaaclab.sh -p scripts/test_magnetic_tilt.py \
  --task Template-Robotarm-Magnetic-Stomach-Lab-v0 \
  --capsule_camera_view
```

九轴接口测试和数据采集也接受同一参数：

```bash
./run_isaaclab.sh -p scripts/validate_interfaces.py --capsule_camera_view
./run_isaaclab.sh -p scripts/collect_finetune_dataset.py --capsule_camera_view
```

关闭该可视化只需省略 `--capsule_camera_view`。该参数不能与 `--headless` 或
`--viz none` 同时使用；开启后会自动选择 Kit 可视化器。

## 正常启动标志

控制台应出现：

```text
[CAPSULE_CAMERA_VIEW] enabled ... source=policy.vision.rgb
resolution=1280x720 sensor_hz=1.0 render_hz=30.0
```

主视口用于观察全局场景，第二窗口显示模型实际接收的圆形胶囊相机RGB。胃部任务中
图像每1秒更新一次；主视口仍可按30 Hz渲染，因此相机帧之间场景动画不会被冻结。
