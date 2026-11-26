# ai-design

设计 + AI 方向的论文选题与研究仓库，多个 AI 协作维护（规则见 `AGENTS.md`）。

当前课题：**DieLineDiff：面向折叠包装设计的印刷约束多面扩散生成方法**（DieLineDiff: Print-Constrained Multi-Face Diffusion for Foldable Packaging Design），以乡村农产品包装为测试场景。方案见 `docs/METHOD_PLAN.md`。

## 发给 Gemini 的指令（直接复制）

```
请先 clone 或 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git ，
阅读根目录的 AGENTS.md 和 GEMINI.md，然后执行 tasks/01-gemini-literature-tree-opc-packaging.md 里的任务。
方法和模板来自 https://github.com/Casey-sketch-cyber/findInnovation.git（skills/research-innovation-points/SKILL.md 和 templates/）。
在分支 gemini/literature-tree-opc-packaging 上完成，推送并创建 PR，不要自己合并。全程用中文。
```

网页版 Gemini 不能执行命令时：把 `tasks/01-gemini-literature-tree-opc-packaging.md` 和 findInnovation 的 `SKILL.md`、两个模板的内容一起粘贴给它，产出再由你或其他 AI 提交到本仓库。

## 发给 Codex 的指令（任务 22，直接复制）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
先读 AGENTS.md 和 PR #21 里 Claude 的审查留言，
然后执行 tasks/22-codex-panel-j4b-fix.md（先用 CPU 诊断盖面直边贴片是几何还是语义原因，再查清并修复 J4b 顶盖暗带；然后测 K1 去版式化提示词、K2 γ=0.40、K3 修复后的 J4b-v2（混合初图 + 非原生边强度 0.85）、K4 K3 加环内上托；环带固定 R0，冻结指标和关口不动，只能补足内部纹理，不得压低贴边细节；两个种子都报告关口，做完就停）。
必须能在 RTX 3090 24GB 上跑通。在分支 codex/panel-j4b-fix 上完成，推送并创建 PR，不要自己合并。全程用中文沟通。
```

## 发给 Gemini 的指令（任务 23，直接复制，与任务 22 并行）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
阅读 AGENTS.md、tasks/23-gemini-panel-sparsity.md、PR #21 里 Claude 的留言、docs/experiments/E02/JOINT-CAP.md 和 tasks/22-codex-panel-j4b-fix.md，
分析盖面直边贴片的成因、原生联合生成下盖面仍然稀疏的原因，评估任务 22 的 K1—K4，并提出至少 1 个补救新方法，写入 docs/PANEL_SPARSITY_GEMINI.md。
在分支 gemini/panel-sparsity 上完成，推送并创建 PR，不要自己合并。全程用中文。
```

## 发给 Codex 的指令（任务 20，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
先读 AGENTS.md、PR #19、#18、#20 里 Claude 的留言、docs/CAP_INTERIOR_GEMINI.md 第 1、3 节和 docs/C3_HIGHRES_GEMINI.md，
然后执行 tasks/20-codex-joint-cap.md（按顺序测 J1 提示词净化、J2 SDXL base 混合潜变量替代 inpainting、J3 环内单向统计上托；再并行做 J4 原生联合环盖生成（两段式设置，先过闸门 1 再做 Stage 2）和 J4b 冻结 R0 的环向带状外扩；冻结指标和关口不动，只能补足内部纹理，不得压低贴边细节；两个种子都报告关口，做完就停）。
必须能在 RTX 3090 24GB 上跑通。在分支 codex/joint-cap 上完成，推送并创建 PR，不要自己合并。全程用中文沟通。
```

## 发给 Gemini 的指令（任务 21，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
阅读 AGENTS.md、tasks/21-gemini-c3-highres.md、PR #18 里 Claude 的留言、tasks/20-codex-joint-cap.md 和 docs/experiments/E02/CAP-ENERGY.md，
为 C3 联合环盖画布（约 2 MP）给出可直接填进配置的去重复首选与备选设置，并给出 C3 失败时的后备方法，写入 docs/C3_HIGHRES_GEMINI.md。
在分支 gemini/c3-highres 上完成，推送并创建 PR，不要自己合并。全程用中文。
```

## 发给 Codex 的指令（任务 18，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
先读 AGENTS.md 和 PR #17 里 Claude 的审查留言，
然后执行 tasks/18-codex-cap-energy-match.md（先做盖面逐毫米能量诊断，再测 E1—E4：能量匹配初图与 SDEdit 强度；环带固定 R0，冻结指标和关口不动，只能补足内部纹理，不得压低贴边细节；两个种子都报告关口，做完就停）。
必须能在 RTX 3090 24GB 上跑通。在分支 codex/cap-energy-match 上完成，推送并创建 PR，不要自己合并。全程用中文沟通。
```

## 发给 Gemini 的指令（任务 19，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
阅读 AGENTS.md、tasks/19-gemini-cap-interior-ideas.md、docs/experiments/E02/RING-M2M3.md、PR #17 里 Claude 的审查留言和 tasks/18-codex-cap-energy-match.md，
解释底盖“贴边密、内部疏”的机制，评估任务 18 的方案，并提出至少 2 个新方法，写入 docs/CAP_INTERIOR_GEMINI.md。
在分支 gemini/cap-interior-ideas 上完成，推送并创建 PR，不要自己合并。全程用中文。
```

## 发给 Codex 的指令（任务 17，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
先读 AGENTS.md、PR #16 里 Claude 的审查留言、docs/BEAT_BASELINES_GEMINI.md 第 2 节和 PR #15 里 Claude 的回应，
然后执行 tasks/17-codex-v3-ring-m2-m3.md（环带 R0/M3b/M3/M2/M2+M3 按登记规则选择，盖面比较 U1 与 U1-σ28，B2-regional-M2 作对照；冻结指标和关口不动；两个种子都报告关口，做完就停）。
必须能在 RTX 3090 24GB 上跑通。在分支 codex/v3-ring-m2-m3 上完成，推送并创建 PR，不要自己合并。全程用中文沟通。
```

## 发给 Gemini 的指令（任务 16，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
阅读 AGENTS.md、GEMINI.md 和 tasks/16-gemini-beat-all-baselines.md，以及 docs/experiments/E02/CAPTEXTURE.md 和 captexture-stripe-sources.csv，
独立研究如何让 v3 在全部五条关口上超过对手（尤其是条带 v1），检验 Claude 的假设 H1—H4，并提出至少 2 个新方法。
在分支 gemini/beat-all-baselines 上完成，推送并创建 PR，不要自己合并。全程用中文。
```

## 发给 Codex 的指令（任务 15，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
先读 AGENTS.md、PR #14 里 Claude 的审查留言和 PR #12 里 Gemini 的任务 14 留言，
然后执行 tasks/15-codex-v3-texture-uniform-ring.md（盖面纹理均匀化 U1/U2 + 环带负向提示词 R1，B2-regional 同步加 R1；冻结指标和关口不动；种子 42 和 1234 都报告关口，做完就停）。
必须能在 RTX 3090 24GB 上跑通。在分支 codex/v3-texture-uniform-ring 上完成，推送并创建 PR，不要自己合并。全程用中文沟通。
```

## 发给 Codex 的指令（任务 13，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
先读 AGENTS.md、PR #12 里 Claude 的审查留言（立场修正）和 PR #10 里 Gemini 的讨论留言，
然后执行 tasks/13-codex-v3-cap-texture.md（盖面颜色对齐 + 纹理延续：ILVR 低频引导、纹理注入初始化、高强度 G2 对照；环带、冻结指标和关口不动，做完就停）。
必须能在 RTX 3090 24GB 上跑通。在分支 codex/v3-cap-texture 上完成，推送并创建 PR，不要自己合并。全程用中文沟通。
```

## 发给 Gemini 的指令（任务 14，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
阅读 tasks/14-gemini-respond-revised-stance.md、docs/experiments/E02/STRIPE-DIAG.md、PR #12 里 Claude 的审查留言和 tasks/13-codex-v3-cap-texture.md，
然后在 PR #12 下用一条留言回答任务 14 的 4 个问题。全程用中文。
```

## 发给 Gemini 的指令（任务 11，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
阅读 tasks/11-gemini-v3-g2-stripe-discussion.md、docs/experiments/E02/CAPGEOM.md 和 PR #10 里 Claude 的审查留言，
然后在 PR #10 下对任务 11 的 5 点逐条表态（同意 / 部分同意 / 反对，附理由和证据），并给出下一步实验建议。
Codex 的诊断结果（docs/experiments/E02/STRIPE-DIAG.md）出来后再补充一条留言。全程用中文。
```

## 发给 Codex 的指令（任务 12，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
先读 AGENTS.md 和 PR #10 里 Claude 的审查留言，
然后执行 tasks/12-codex-stripe-diagnostic.md（只用 CPU，不生成任何新图，不改冻结指标；做条带指数的合成对照、真实图裁图、三种候选 v2 的试算和七缝 LPIPS 拆分）。
在分支 codex/stripe-diagnostic 上完成，推送并创建 PR，不要自己合并。全程用中文沟通。
```

## 发给 Codex 的指令（任务 10，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
先读 AGENTS.md、PR #9 里 Claude 的审查留言和 PR #7 里 Gemini 的审阅留言，
然后执行 tasks/10-codex-v3-cap-geometry.md（去掉盖面 halos、加调和初始化 C5、修正共享注意力时间步，并诊断条带来源；只改盖面阶段，环带和已冻结的指标不动，做完就停）。
必须能在 RTX 3090 24GB 上跑通。在分支 codex/v3-cap-geometry 上完成，推送并创建 PR，不要自己合并。全程用中文沟通。
```

## 发给 Codex 的指令（任务 08，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
先读 AGENTS.md 和 PR #7 里 Claude 的审查留言（如有 Gemini 的留言也一并看），
然后执行 tasks/08-codex-v3-cap-fix.md（只修盖面阶段，环带和已冻结的指标不动；在前 10 条上逐项选择后重过 Step 1 关口，做完就停）。
必须能在 RTX 3090 24GB 上跑通。在分支 codex/v3-cap-fix 上完成，推送并创建 PR，不要自己合并。全程用中文沟通。
```

## 发给 Gemini 的指令（任务 09，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
阅读 tasks/09-gemini-v3-cap-review.md，以及 docs/experiments/E02/REPORT.md、PR #7 里 Claude 的审查留言和 tasks/08-codex-v3-cap-fix.md，
然后在 PR #7 下用一条留言回答任务 09 的 4 个问题。全程用中文。
```

## 发给 Codex 的指令（任务 07，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
先读 AGENTS.md，再完整阅读 docs/METHOD_V3_CONSENSUS.md（规格书），
然后执行 tasks/07-codex-v3-step0-step1.md（先做 Step 0 诊断并冻结指标，再做 Step 1 最小验证，做完就停）。
必须能在 RTX 3090 24GB 上跑通。在分支 codex/v3-step0-step1 上完成，推送并创建 PR，不要自己合并。全程用中文沟通。
```

## 发给 Gemini 的指令（任务 06 第二轮，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git，切到你的分支 gemini/method-v3-research，并合并最新 main。
阅读 PR #5 中 Claude 的第一轮审查意见，按意见修改 docs/METHOD_V3_PROPOSALS.md，
然后在 PR #5 里逐条回复（同意的写明怎么改的，不同意的给出理由和证据）。在同一分支追加提交，不要强制推送，不要合并。全程用中文。
```

## 发给 Gemini 的指令（任务 06 第一轮，已完成）

```
请先 clone 或 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
阅读 AGENTS.md 和 GEMINI.md，然后执行 tasks/06-gemini-method-v3-research.md：
读完仓库里的方案设计、全部实验报告和 PR #2、#3、#4 中 Claude 的审查留言，
深入研究如何让 DieLineDiff 在接缝、出血和折边上真正优于基线，提出 3—5 个方案。
在分支 gemini/method-v3-research 上完成，推送并创建 PR，不要自己合并；之后在 PR 里与 Claude 逐条讨论。全程用中文。
```

## 发给 Codex 的指令（任务 05，直接复制）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
先读 AGENTS.md 和 PR #4 里 Claude 的审查留言，
然后执行 tasks/05-codex-e01b-bottom-frame.md（先做第 1 节的三项诊断，再把底面改为刀版方向跨折线生成，并加入 B2-post 基线；关口通过后完成 E01b 全量，做完就停）。
必须能在 RTX 3090 24GB 上跑通。在分支 codex/e01b-bottom-frame 上完成，推送并创建 PR，不要自己合并。全程用中文沟通。
```

## 发给 Codex 的指令（任务 04，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
先读 AGENTS.md 和 PR #3 里 Claude 的审查留言，
然后执行 tasks/04-codex-e01b-crease-fix.md（先用坐标编码可视化定位折边伪影根因再修复；关口通过后完成 E01b 全量，做完就停）。
必须能在 RTX 3090 24GB 上跑通。在分支 codex/e01b-crease-fix 上完成，推送并创建 PR，不要自己合并。全程用中文沟通。
```

## 发给 Codex 的指令（任务 03，已完成）

```
请 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git 的最新 main，
先读 AGENTS.md、PR #2 里 Claude 的审查留言和 docs/experiments/E01/REPORT.md 开头的审查注记，
然后执行 tasks/03-codex-e01b-prototype-fix.md（E01b：修正原型并重跑，基线复用 E01 输出，做完就停）。
必须能在 RTX 3090 24GB 上跑通。在分支 codex/e01b-prototype-fix 上完成，推送并创建 PR，不要自己合并。全程用中文沟通。
```

## 发给 Codex 的指令（任务 02，已完成）

```
请先 clone 或 pull 仓库 https://github.com/Casey-sketch-cyber/ai-design.git ，
阅读根目录的 AGENTS.md，然后完整阅读 docs/METHOD_PLAN.md，
再执行 tasks/02-codex-dielinediff-s0-s1.md 里的任务（只做 S0 和 S1，做完就停）。
必须能在 RTX 3090 24GB 上跑通。在分支 codex/dielinediff-s0-s1 上完成，推送并创建 PR，不要自己合并。全程用中文沟通。
```

## 目录

| 路径 | 内容 |
| --- | --- |
| `tasks/` | 给各个 AI 的任务书 |
| `code/`、`third_party/` | 实验代码与第三方基线 |
| `docs/` | 文献树、创新点评估、期刊调研、进度与决策 |
| `literature/` | 检索记录与文献清单 |
