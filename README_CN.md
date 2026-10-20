# 软体熊双臂穿孔包裹仿真

新场景完成：双臂抓布料前侧两角 → 提起并环绕熊腹部 → 魔术贴实际接触后粘合 → 机械臂换抓熊的手臂 → 分别从两个真实布孔中拉出。

旧仿真的全部既有输出保存在 [bear-velcro-before-arms-20261018](/home/chloe/Documents/Codex/2026-10-17/na-m/outputs/archives/bear-velcro-before-arms-20261018)，含逐文件 SHA-256 清单。旧场景与原启动脚本保留。

## 本次完整录像

- [完整过程（总览与近景）](/home/chloe/Desktop/genesis-local/bear_arms_scene/runs/stabilized-dressing-complete-48/wrapping-process.mp4)
- [桌子左侧](/home/chloe/Desktop/genesis-local/bear_arms_scene/runs/stabilized-dressing-complete-48/wrapping-left-side.mp4)
- [玩偶背后](/home/chloe/Desktop/genesis-local/bear_arms_scene/runs/stabilized-dressing-complete-48/wrapping-rear.mp4)

三个视频来自同一次运行的相同步骤，帧率为 15 FPS，模拟时长 65.67 秒。成功结果位于 `stabilized-dressing-complete-48`；其他运行是保留的开发记录，状态见 `development-runs.json`。

## 打开完整环境并重新运行

```bash
bash /home/chloe/Desktop/genesis-local/run-bear-arms.sh
```

无窗口运行并输出新录像：

```bash
bash /home/chloe/Desktop/genesis-local/run-bear-arms.sh --headless --backend cuda --width 1280 --height 720
```

默认每次创建新的结果目录。使用原有 `/home/chloe/Desktop/genesis-local/.venv` 环境。

## 场景与结果

熊的双臂与躯干属于同一连续软体四面体网格，熊的底部固定。棉布保持原来的 86 × 20 厘米尺寸，两侧带魔术贴；两个孔由删除实际网格面形成。机械臂通过关节控制运动，抓取后采用局部无滑移约束。局部有向体积约束用于抑制夹持处的单元塌陷、翻转，残余数值误差记录在独立检查报告中。

本次运行目录保存三段视频、阶段截图、粒子和关节状态、轨迹日志、报告、源代码及素材快照。`replay-frames.npz` 保存每个视频采样时刻的熊、布料和机械臂状态。`checkpoint-history` 保留同名保存点的早期版本。

`independent-dressing-review.json` 检查固定底部、真实孔口、手臂穿孔、熊掌整体位于孔外、布料与身体的离散穿透采样及粘合保持；`video-validation.json` 记录三个视频的完整解码和帧同步检查。

## 模型范围

这是动作流程仿真。夹持与魔术贴使用接触触发的理想几何约束；一次局部粘接不代表整条魔术贴完全覆盖粘合，也没有标定真实拉力或剥离强度。额外的布料表面接触将当前形变后的熊作为障碍，原生粒子和机械臂接触仍参与计算。检查基于保存状态中的离散采样，不构成连续碰撞检测保证。最终机械臂保持夹住已拉出的熊手臂。
