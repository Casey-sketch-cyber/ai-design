# 任务 21：C3 联合环盖画布（约 2 MP）的去重复与稳态设置，及 C3 失败时的后备方案（Gemini）

- 执行者：Gemini
- 下达：Claude，2026-11-26
- 分支：`gemini/c3-highres`
- 前置：PR #18、#19 已合并。先读 PR #18 中 Claude 的留言、`tasks/20-codex-joint-cap.md`、`docs/experiments/E02/CAP-ENERGY.md`。
- 并行：Codex 在做任务 20。
- 沟通语言：中文
- 选题与创新点判断遵循 findInnovation 仓库的 research-innovation-points skill。

## 0. 背景

任务 18 证明，初图侧的补救（包括你的 C1 拼贴初图）无效：成图的 9—12 mm 能量总会被去噪冲回约 1.0。

下一轮的主方案是你的 C3：把环带画布纵向延展，让顶盖、底盖从 F、B 原生长出来。主要风险是面积约 2 MP（例如 1280 × 1600 左右，具体取决于盒子的盖深），超出 SDXL 的舒适面积，可能出现：
- 物体重复；
- 构图崩坏；
- 任务 17 中 M3b 那样的跨种子波动。

## 1. 要回答的问题

1. **去重复设置**：比较 ScaleCrafter、HiDiffusion（RAU-Net 与 MSW-MSA）、DemoFusion、FouriScale、MultiDiffusion / SyncDiffusion 分块，以及“先在低分辨率下联合生成、再放大细化”的两段式做法。
   - 哪些与环向循环卷积、区域提示词兼容？
   - 哪些免训练？
   - 哪些在 RTX 3090 上能把 Stage 1 控制在 25 s/盒以内？
   - 给出**一个首选设置和一个备选设置**，写到能直接填进配置的程度：哪些层、哪些步、什么参数。
2. **纵向非循环边界**：上下两端不循环。延展区边缘会不会出现边框伪影？缓冲区要多宽？
3. **区域提示词**：盖面区与侧面区在纵向上是相邻的两块区域。如何避免顶盖区直接复制 F 面的主体物？例如，主体物的注意力掩码只给侧面中部。
4. **后备方案**：如果 C3 在闸门 1 止损，还有什么方法能让 F-T、B-Btm 两条原生折线的盖面侧在 0—12 mm 内纹理均匀？
   - 不能改初图（已证伪），也不能在折线处压低贴边能量（违规）；
   - 给出至少 1 个新方法，附机制、依据文献、3090 成本和一天内的最小验证。

## 2. 要求

- 文献必须核对官方摘要与 DOI 或 arXiv 号；新增文献录入 `literature/papers.csv` 与 `literature/search-log.md`。不得编造。
- 结合本仓库代码给出具体实现位置：`v3.py`、`v3_attention.py`、`sdxl.py`。
- 写成 `docs/C3_HIGHRES_GEMINI.md`。开头三行写结论：首选设置、备选设置、后备方案。

## 3. 结束

更新 `docs/PROGRESS.md`。在分支 `gemini/c3-highres` 上提交，推送并创建 PR。**不要自己合并**。
