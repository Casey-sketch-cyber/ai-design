# 任务 02：DieLineDiff 代码骨架与最小验证实验（S0 + S1）

- 执行者：Codex（在作者装有 GPU 的电脑上运行）
- 下达：Claude，2026-11-24
- 分支：`codex/dielinediff-s0-s1`
- 方案依据：**先完整读 `docs/METHOD_PLAN.md`**，本任务只做其中的 S0 和 S1。
- 沟通语言：中文（代码、变量名、注释用英文）

## 0. 开工前

1. 读 `AGENTS.md`、`docs/PROGRESS.md`、`docs/DECISIONS.md`、`docs/METHOD_PLAN.md`。
2. 本任务**只做 S0 和 S1**。S1 跑完后停下，提交 PR，等 Claude 审查。不要自行开始 S2。

## 1. 硬件与环境约束

- **必须能在 RTX 3090（24 GB 显存）上跑通**，这是验收条件。作者另有一张 RTX 5090（32 GB）可用于加速批量实验；在 5090 上运行需要支持 Blackwell 的 PyTorch（CUDA 12.8 及以上）。两张卡上的结果用同一套配置和种子。
- 主干模型：SDXL base 1.0（fp16）+ VAE `madebyollin/sdxl-vae-fp16-fix`。不训练任何参数。
- 省显存手段：fp16、注意力切片或 SDPA、VAE tiling、窗口逐个计算。在报告中写明峰值显存和单盒耗时。
- 每个第三方基线（SyncMVD、Paint3D）单独建 conda 环境，放在 `third_party/` 下，以 git submodule 或脚本克隆的方式引入，并**固定 commit**。不要把第三方代码复制进主代码。
- 不提交模型权重、大文件和 `outputs/` 下的批量结果（加进 `.gitignore`）。报告中引用的示例图压缩后放 `docs/experiments/E01/`，总量不超过 15 MB。
- 只用免费公开资源，不购买数据和服务。

## 2. 代码结构（建议，可调整，但要在 PR 中说明）

```
code/
  dielinediff/
    dieline/        # 参数化刀版：SVG、region_map、topology.json、box.obj
    color/          # ICC → 3D LUT，往返映射，OOG 计算
    methods/
      ours_tf.py    # DieLineDiff-TF 原型
      b2_controlnet.py
      b3_syncmvd.py # 调用 third_party/SyncMVD 的封装
      b4_paint3d.py # 调用 third_party/Paint3D 的封装（S1 尽量跑通，跑不通在报告中说明）
    metrics/        # 第 5 节的指标
    render/         # 折叠检查渲染（trimesh/pyrender 或 Blender 无头，二选一）
  configs/          # YAML：盒型、方法、指标阈值、种子
  scripts/          # 一键脚本：生成刀版、跑方法、算指标、汇总表格
third_party/
data/briefs/E01_briefs.jsonl
docs/experiments/E01/
```

## 3. S0：骨架与工具（2—3 天）

### 3.1 参数化刀版（`dieline/`）

- 实现反向插舌管式折叠盒（RTE，ECMA A20 类），参数：长 L、宽 W、高 H（mm）、纸厚、出血 3 mm、安全距 3 mm、粘口宽、插舌和防尘翼尺寸。默认尺寸 70×45×160 mm。
- 输出（按 `METHOD_PLAN.md` 第 3.1 节）：
  - `dieline.svg`（裁切 / 压痕 / 出血 / 安全线分图层）；
  - `region_map.png` + 标签对照表（F, B, L, R, T, Btm, Glue, Tuck, Dust, Bleed, Outside）；
  - `topology.json`（每个面的 UV 矩形、折叠后正立旋转角、3D 邻接边及其方向）；
  - `box.obj`（只含 6 个外观面，UV 等于刀版坐标）。
- 单元测试：对任意一张测试纹理（例如带方向箭头和面名的棋盘格），贴到 `box.obj` 上做折叠渲染，6 个视角下所有箭头朝上，邻接边处棋盘格对得上。**把这张自检图放进报告。**

### 3.2 色域工具（`color/`）

- ICC 特性文件：优先 ECI 的 ISO Coated v2 (FOGRA39)；若许可不清楚或无法下载，用 Ghostscript 自带的 `default_cmyk.icc`。报告中写明所用文件、来源和许可。
- 生成 33³ 的 sRGB → CMYK → sRGB 往返 LUT（PIL `ImageCms`，相对比色，黑场补偿），并实现 PyTorch 版三线性插值（S1 用于投影，S2 用于梯度引导）。

### 3.3 指标（`metrics/`）

实现 `METHOD_PLAN.md` 第 5 节中标 ★ 的全部指标：接缝不连续比 SDR、接缝 LPIPS、正立率、禁墨区油墨覆盖率、出血填充率、光照残留、OOG% 与平均往返 ΔE00、CLIPScore、HPS v2、单盒耗时、峰值显存。

- 每个指标写清楚定义、阈值和依据，放在 `code/dielinediff/metrics/README.md`。
- 先用 3.1 的自检纹理和几张人为破坏的纹理（接缝错位、倒置、粘口涂色、出血留白、强阴影、高饱和色）做健全性检查：**每个指标都必须能区分“好”和“坏”**。结果写进报告。

### 3.4 测试简报

- `data/briefs/E01_briefs.jsonl`：50 条乡村农产品简报，字段为 `id, category, region_style, main_color, product_name_zh, front_prompt, side_prompt, top_prompt`。
- 品类覆盖茶叶、小米、大米、蜂蜜、香菇、果干、杂粮、花椒等。风格词用公开、通用的地域文化描述，不使用具体商标，不抓取商业包装图片。
- 其中至少 20 条要求“主视觉跨面连续”（例如山水长卷环绕盒身），用于接缝指标的单独统计。

## 4. S1：最小验证实验 E01（约 1 周）

### 4.1 要比较的方法

| 方法 | 要求 |
| --- | --- |
| B2：SDXL + ControlNet（刀版线稿或分区图） | 一次生成整张展开图 |
| B3：SyncMVD 跑在 `box.obj` 上 | 使用官方代码和默认设置；纹理分辨率与原型对齐 |
| B4：Paint3D 跑在 `box.obj` 上 | 尽量跑通；24 GB 上跑不通就在报告中写明原因，不硬改 |
| Ours：DieLineDiff-TF 原型 | 实现 `METHOD_PLAN.md` 的 3.2（多面同步）、3.3（区域掩码与出血）、3.4 的投影版（色域）。**不做** 3.5 双轨文字 |

- 不要替基线做后处理补救（例如不要帮基线把粘口涂白）；如实评测。
- 所有方法统一采样步数（例如 30 步）、统一工作分辨率（展开图短边约 1024 px），使用同一组简报和种子。

### 4.2 运行规模

1 种盒型 × 50 条简报 × 3 个种子 × 每个方法。先用 5 条简报 × 1 个种子跑通全流程，确认无误后再跑全量。

### 4.3 原型的小规模调参（只允许在前 10 条简报上调）

- 可调：窗口边带宽度 `m`、色域投影的作用步数和 `λ`、底色估计方式。
- 调定后冻结，在其余 40 条上报告结果，并在报告中写明调参过程。**不得用全部 50 条反复调参。**

### 4.4 消融（S1 只做 2 项）

- 去掉多面同步（各面独立生成后拼接）；
- 去掉色域投影。

## 5. 交付物

1. 代码与配置，一条命令能复现 E01：`bash code/scripts/run_E01.sh`。
2. `docs/experiments/E01/REPORT.md`：
   - 第一段直接回答：**3D 纹理同步方法直接用于纸盒刀版，是否已满足印刷要求？是否建议进入 S2？**
   - 结果总表：各方法 × 各指标，给出均值、95% 置信区间和配对 Wilcoxon 检验结果；“跨面连续”子集单独列表；
   - 每个方法 3 组示例：刀版展开图 + 折叠渲染图（压缩后放入仓库）；
   - 自检图和指标健全性检查结果；
   - 调参记录、失败与踩坑记录（例如基线跑不通的原因）；
   - 3090 上的峰值显存和单盒耗时；如果也用了 5090，一并列出。
3. 结果原始数据：`docs/experiments/E01/results.csv`（每行 = 方法 × 简报 × 种子）。
4. 更新 `docs/PROGRESS.md`；重要取舍写进 `docs/DECISIONS.md`。

**图表**：报告里的图只需要能看清数据（matplotlib 默认样式即可）。论文正式插图以后由 Claude 用 PowerPoint 绘制，Codex 不需要做论文排版级的图。

## 6. 结束

在分支 `codex/dielinediff-s0-s1` 上提交，推送并创建 PR。PR 说明中列出：完成了什么、没完成什么、与 `METHOD_PLAN.md` 不一致的地方及原因。**不要自己合并到 `main`**，由 Claude 审查。

## 验收标准（Claude 审查时对照）

- [ ] 刀版与折叠渲染自检图正确：箭头全部朝上，接缝对齐。
- [ ] 每个 ★ 指标都通过健全性检查（能区分好坏样本）。
- [ ] B2、B3、Ours 全部跑完 50 × 3；B4 跑完，或写明跑不通的原因。
- [ ] 原型只在前 10 条简报上调参，其余 40 条报告结果。
- [ ] 报告第一段给出明确结论和是否进入 S2 的建议，负结果如实报告。
- [ ] 在 RTX 3090 24 GB 上跑通，并记录显存和耗时。
- [ ] `bash code/scripts/run_E01.sh` 可复现；第三方代码 commit 已固定。
