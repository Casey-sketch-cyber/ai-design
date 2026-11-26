# 任务 23：盖面“直边贴片”与“原生联合生成下盖面仍稀疏”的机制分析（Gemini）

- 执行者：Gemini
- 下达：Claude，2026-11-26
- 分支：`gemini/panel-sparsity`
- 前置：PR #21 已合并。先读 PR #21 中 Claude 的留言、`docs/experiments/E02/JOINT-CAP.md`、`tasks/22-codex-panel-j4b-fix.md`。
- 并行：Codex 在做任务 22。
- 沟通语言：中文
- 判断创新点时遵循 findInnovation 仓库的 research-innovation-points skill。

## 0. 背景

任务 20 的进展：
- J3 γ=0.30 的条带 v1 只比上限高 0.0012；
- J4b 的条带 v1 已达标。

剩下两个问题需要解释清楚，论文中也需要写这部分机制。

## 1. 要回答的问题

1. **直边贴片**：J0、J1、J3 的盖面内部出现竖直或水平的硬边，把画面切成小块（逐盒判读见 `docs/experiments/E02/joint-cap-content-review.json`）。
   - 结合原像素和代码分析，这是语义原因（`STYLE` 中的 “packaging print artwork, clean graphic design” 诱导卡片和面板），还是几何原因（已知掩码角点、初图分区边界、潜格网格），或者两者兼有？
   - 给出你的判断和依据，并评估任务 22 中 K1 的处理是否足够。
2. **原生联合生成下盖面仍然稀疏**：J4 的 Stage 1 中，F-T、B-Btm 两侧在同一次生成里连续，没有 inpainting 边界，但盖面侧的条带 v1 仍有 2.7—3.6。
   - 这说明“贴边密、内部疏”不只是 inpainting 造成的。可能的原因包括：
     - SDXL 构图先验（画面上下边缘偏空，如天空、留白）；
     - F、B 中部 60% 放主体、上下段只放延续词，造成的密度梯度；
     - 两段式放大的影响。
   - 请用 `docs/experiments/E02/joint-cap-gate1/` 下的数据支持你的解释。
   - J4b 的底盖向下外扩，9—12 mm 能量反而很高（3.66）。这与上面的现象是否矛盾？
3. **评估任务 22 的 K1—K4**：预测各自最可能失败的方式。

   重点是 K3 的混合初图：靠近原生折用外扩内容，靠近三条非原生边用 U1-σ28 初图，强度在三条非原生边处为 0.85。这两种初图的过渡区会不会产生新的拼贴？如何避免？
4. **提出至少 1 个新方法**：如果 K 系列仍差一点，用于补救。要求：
   - 不改初图来补纹理（已证伪）；
   - 不压低贴边；
   - 附机制、依据文献、3090 成本和一天内的最小验证。

## 2. 要求

- 文献必须核对官方摘要与 DOI 或 arXiv 号；新增文献录入 `literature/papers.csv` 与 `literature/search-log.md`。不得编造。
- 写成 `docs/PANEL_SPARSITY_GEMINI.md`，开头三行写结论。

## 3. 结束

更新 `docs/PROGRESS.md`。在分支 `gemini/panel-sparsity` 上提交，推送并创建 PR。**不要自己合并**。
