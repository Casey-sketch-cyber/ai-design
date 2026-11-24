# 决策记录

| 日期 | 决策 | 理由 | 决定人 |
| --- | --- | --- | --- |
| 2026-11-24 | 选题调研统一使用 findInnovation 的文献树方法和模板 | 与作者其他课题保持一致，便于审查 | 作者 |
| 2026-11-24 | 文献调研由 Gemini 执行，Claude 审查合并 | 作者的多 AI 协作约定 | 作者 |
| 2026-11-24 | 方向偏向“用深度学习等 AI 方法做包装设计”，OPC / 方案交付降为可选框架 | 作者偏好 | 作者 |
| 2026-11-24 | 不招募专家或消费者，只用客观指标评价 | 找不到评价人 | 作者 |
| 2026-11-24 | 若论文讨论/应用部分提及 OPC，英文术语采用 `AI-driven One-Person Company (OPC)`（`1 + N + AI` 人机社会技术系统） | 已核实 2026 年 *Systems* (SSCI Q1)、IEEE *IT Professional* (SCIE) 与 *SEJ* (FT50) 均正式使用 OPC 术语 | Gemini（提议） |
| 2026-11-24 | 数据策略采用“高许可公开底座（Open Food Facts / RPC / ABO / Crello / AnyWord-3M）+ Blender 程序化合成 2D 刀版↔3D 几何评测基准 `PackDieline-Syn`”，不直接公开发布爬取的商业包装原图 | 规避《著作权法》第二十四条三步检验法、商标权与反不正当竞争风险，同时保证评测基准可开源复现 | Gemini（提议） |
| 2026-11-24 | `DieLineDiff` 按情况 2 定位，主方法优先采用免训练的推理期同步与印刷硬约束引导，`PackDieline-Syn` 仅作评测基准而不作贴图训练监督 | 补全 TexFusion/SyncMVD/Paint3D/MVPaint 3D 纹理脉络后，折叠纸盒 UV 同步已有成熟远域解法，且合成贴图无法提供有效排版监督（回应 PR #1 审查意见） | Gemini（按 Claude 审查意见调整） |
| 2026-11-24 | 暂定题目：DieLineDiff: Print-Constrained Multi-Face Diffusion for Foldable Packaging Design | 方法类 SCIE 期刊惯例；乡村农产品放副标题或案例 | 作者 |
| 2026-11-24 | 实验必须能在 RTX 3090 24GB 上跑通，主干用 SDXL，不用 Flux / SD3.5 Large | 作者可用显卡为 3090 和 5090 | 作者、Claude |
| 2026-11-24 | 实验由 Codex 执行，分 S0—S3 阶段，每阶段结束由 Claude 审查后再继续 | 先用最小实验验证核心假设，降低返工 | 作者 |
| 2026-11-24 | E01 在 AutoDL RTX 3090 上执行，仓库、模型、环境和所有实验数据均放服务器数据盘 | 依照作者指定的存储位置与 24 GB 验收环境 | 作者 |
