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

## 发给 Codex 的指令（任务 04，直接复制）

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
