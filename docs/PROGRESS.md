# 项目进度（交接记录）

> 每个 AI 结束工作前在最上方追加一条记录，新的在上面。

## 当前状态

任务 05 在 AutoDL RTX 3090 完成三组诊断和五盒试跑。底面改按刀版方向跨折线生成，B2-post 已复用全部 150 张 B2 原图。背底条带通过，仍见横带，全盒折边条带和 CLIP 等效关口未通过。停在试跑，建议暂缓 S2。待 Claude 审查任务 05，Gemini 继续任务 06 的方法深研。

## 待办

- [x] Codex：任务 02，见 `tasks/02-codex-dielinediff-s0-s1.md`（S0 + S1，做完停下等审查）
- [x] Claude：审查任务 02 的 PR #2，通过并合并（2026-11-25），审查意见见 PR 留言
- [x] Codex：任务 03，见 `tasks/03-codex-e01b-prototype-fix.md`（E01b，做完停下等审查）
- [x] Claude：审查任务 03 的 PR #3，通过并合并（2026-11-25），审查意见见 PR 留言
- [x] Codex：任务 04，见 `tasks/04-codex-e01b-crease-fix.md`（做完停下等审查）
- [x] Claude：审查任务 04 的 PR #4，通过并合并（2026-11-25），审查意见见 PR 留言
- [x] Codex：任务 05，见 `tasks/05-codex-e01b-bottom-frame.md`（做完停下等审查）
- [ ] Claude：审查任务 05 的 PR
- [ ] Gemini：任务 06，见 `tasks/06-gemini-method-v3-research.md`（方法深研，提出 3—5 个方案，与 Claude 在 PR 中讨论）
- [ ] Claude：审查任务 06 的 PR 并与 Gemini 讨论；任务 05 结果出来后汇总成 `docs/METHOD_V3_CONSENSUS.md`，交作者确认
- [x] Claude：审查任务 01 的 PR #1（2026-11-24，退回）
- [x] Gemini：按 PR #1 审查意见完成 4 项必改与 2 项建议修改，并已在 PR #1 逐条回复
- [x] Claude：复审 PR #1，通过并合并（2026-11-24；顺手更正 Heliyon 解除 On Hold 日期为 2026-05-14）
- [x] 作者：已决定不招募专家或消费者，只用评价方案 A（2026-11-24）
- [x] 作者：指示在 AutoDL RTX 3090 上启动任务 02（S0 和 S1）
- [ ] 作者：确认单位对 MDPI / IEEE Access 的认定口径

## 已知问题

- `literature/papers.csv` 中共有 5 篇标为 `verified=no`（2 篇未定位独立 DOI 的种子文献 + 3 篇 Elsevier 付费墙阻挡摘要的文献），均已单独标注且未作为核心论证依据；其余 102 篇均已核对官方摘要（`summary_source=abstract`）。
- 公开数据集的国内网络拉取速度与 PKU PosterLayout 协议审批周期需在实验启动第一周实测。
- `docs/JOURNALS.md` 中的 JIF 与中科院 2025 分区来自公开汇总数据，正式投稿前须在 Clarivate MJL 官网与本单位图书馆目录复核。

---

## 工作记录

### 2026-11-24 · codex · 底面刀版方向与 B2-post

- **完成**：以任务 04 原版完成潜变量留存、关闭底窗局部写入和恢复真实上下文的三组对照。实现背底原生同向窗口与 ±2 mm 共享写入，并核对五条实折边。
- **基线**：复用 150 张 B2 原图，执行同源 LUT 与统一印前复位，五盒试跑发布包含 B2-post 的 60 行双版本评分。
- **关口**：背底指数 1.0446 通过，仍有可见横带。全盒最大均值 1.8047 高于 1.3020，CLIPScore 高于 B2 0.0551，超出 ±0.05 等效容差。
- **交接**：未启动调参、消融或留出集检验。建议暂缓 S2，提交 Claude 审查。见 [试跑报告](experiments/E01b/REPORT.md)。

### 2026-11-25 · claude · main
- **完成**：按作者决定（不换方向，继续在方法上做到优于基线），下达任务 06 给 Gemini 做方法深研，与任务 05 并行。

### 2026-11-25 · claude · main
- **完成**：审查并合并 PR #4；修订 METHOD_PLAN 的方向策略（底面图案按刀版方向生成）；下达任务 05。

### 2026-11-24 · codex · codex/e01b-crease-fix
- **定位**：旧映射会在两个原生盖面折边重复读取相邻侧面，底窗同时以不同正立方向写入侧面。坐标编码、OBJ 邻接与写权测试已入库。
- **完成**：修复映射，实现统一 4 px/mm、折边条带指数和顶底专项，回算历史 905 张独有原图，提交双口径评分与新旧尺度对照。
- **关口**：修订后五盒的覆盖、CLIPScore 达标，条带指数及原像素检查失败。按任务停止调参与全量，建议暂不进入 S2。见 `docs/experiments/E01b/REPORT.md`。

### 2026-11-25 · claude · main
- **完成**：审查并合并 PR #3，下达任务 04。
- **注意**：B2（SDXL + ControlNet）是最强的基线，v2 的优势要靠跨面连续子集和顶面、底面朝向来证明。

### 2026-11-24 · codex · codex/e01b-prototype-fix
- **完成**：实现 6.4 px/mm 与环带、顶底和外缘出血窗口。覆盖检查、统一印前复位、ImageReward、50 行双口径试跑指标及图像证据均已交付。
- **关口**：五盒均通过覆盖和 CLIPScore 校验，五盒均有背面与底面交界的彩色条带。调参和全量已验证阻断。
- **结论**：试跑中出血和色域较 v1 改善，折边伪影仍可见，建议暂不进入 S2。见 `docs/experiments/E01b/REPORT.md`。

### 2026-11-25 · claude · main
- **完成**：审查并合并 PR #2；在 E01 报告开头加审查注记；下达任务 03（E01b）。
- **注意**：粘口 0% 来自解码后的像素复位，不算方法优势，E01b 对所有方法统一做“印前复位”。

### 2026-11-24 · codex · codex/dielinediff-s0-s1
- **完成**：在服务器数据盘克隆最新主分支，建立任务分支。默认 RTE 几何在实际 OBJ 渲染中通过六面正立和 12 条接缝检查。
- **完成**：16 项指标健全性检查通过，四个主方法各完成 5 盒正式试跑。仅在前 10 条简报比较三组参数并冻结配置。
- **完成**：四个主方法与两项消融共 900 条正式记录，均已评分。交付完整汇总、95% 置信区间、配对检验、三份共同比较示例和 3090 实测成本。
- **结论**：原型实现粘口留白，多面同步改善接缝，色域投影有小幅收益，出血与综合质量不符合 S2 关口。结论与复现入口见 `docs/experiments/E01/REPORT.md`。

### 2026-11-24 · claude · main
- **完成**：写方案设计 `docs/METHOD_PLAN.md`，下达任务 02 给 Codex。
- **注意**：E01 的关键是如实比较 SyncMVD、Paint3D 直接贴纸盒的效果，负结果同样有价值。

### 2026-11-24 · claude · main
- **完成**：复审并合并 PR #1。独立核实 HSSC 2026 两篇、IEEE Access 饼干包装论文均真实存在（首次审查时怀疑有误，已更正）。
- **下一步**：等作者决定是否启动第一个实验。

### 2026-11-24 · gemini · gemini/literature-tree-opc-packaging（PR #1 审查返修）
- **完成**：
  - 合并 `origin/main`（`8eb149a`），按 Claude 的 4 项必改与 2 项建议意见逐项整改：
    1. **文献核对审计（消灭全部 `verified=yes` + `snippet`）**：逐条核查原 44 条 `snippet` 文献，其中 41 条已通过 OpenAlex / arXiv 官方页获取并核对完整摘要（升级为 `summary_source=abstract`）；3 篇 Elsevier 付费墙未能读取摘要原文的条目严格降为 `verified=no`。针对 Claude 点名标疑的 5 篇 2026 论文（*HSSC* `10.1057/s41599-026-08755-y`、`10.1057/s41599-026-06948-z`；*IEEE Access* `10.1109/access.2026.3723820`、`10.1109/access.2026.3730226`；*Mathematics* `10.3390/math14203759`），均已通过直接 DOI 跳转及 OpenAlex 官方摘要核实为真实已发表论文。
    2. **补全 3D 网格纹理生成脉络并下调创新判断**：新增并核对 7 篇 3D 纹理生成里程碑官方摘要与 DOI（Text2Tex ICCV 2023、TexFusion ICCV 2023、Point-UV Diffusion ICCV 2023、SyncMVD SIGGRAPH Asia 2024、Meta 3D TextureGen 2024、RoCoTex 2024、MVPaint CVPR 2025），`papers.csv` 增至 107 篇（102 篇 `verified=yes, abstract`，5 篇 `verified=no`）。在 `literature-tree-v1.md` 与 `INNOVATION.md` 中将 `DieLineDiff` 从情况 3 回调为**情况 2**，将“直接把 TexFusion / SyncMVD / Paint3D / MVPaint 跑在以刀版为 UV 的纸盒 3D 网格上”列为必选基线，并将贡献边界精准收束至 6 项印刷硬约束（粘口/插舌不可见面、3 mm 出血与安全区、每面文字物理朝上、无光照平稿、CMYK 色域、密集中文小字法定标签）。
    3. **厘清 `PackDieline-Syn` 监督信号并确立免训练主方法**：明确指出 CC0 贴图合成图不能作为包装排版训练监督目标，将 `PackDieline-Syn` 明确定位为**开源客观评测基准**；在 `INNOVATION.md` 中将**免训练的推理期同步与印刷硬约束引导（`DieLineDiff-TF`）**设为保底主方法，仅将基于真实商品数据（Open Food Facts / RPC / ABO / Crello / AnyWord-3M）监督的轻量训练适配器列为 2 周限时加分项。
    4. **补齐期刊表风险出处链接**：在 `docs/JOURNALS.md` 中为中科院 2025 预警名单（5 本）、*IEEE Access* 预警史（修正为 2020 年试行版中档、2021 年起移出）与 *Heliyon* On Hold 状态（2024-09-30 列入、2026-08-07 Retraction Watch 报道解除）全部附上已实测 HTTP 200 的可点击来源链接。
    5. **中性化措辞与场景定位校准**：通篇清理“极高”“100%”“根治”“铁证”等夸大表述，统一将乡村农产品明确定位为深度学习包装设计方法的主要测试场景。
- **未完成**：待 Claude 复审 PR #1 后合并，随后进入最小验证实验阶段。

### 2026-11-24 · claude · main
- **完成**：审查 PR #1（Gemini 任务 01），退回修改，意见见 PR 留言。
- **注意 / 坑**：Gemini 最初从旧提交开分支，后来已自行合并修订版任务书。

### 2026-11-24 · gemini · gemini/literature-tree-opc-packaging（修订版：深度学习做包装设计 + 纯客观指标 A 套 + 公开数据集）
- **完成**：
  - 合并 `origin/main`（`3bbc3d8`），按修订版任务书升级文献树、创新点、数据集调研（`docs/DATASETS.md`）、期刊表与 100 篇文献清单。

### 2026-11-24 · gemini · gemini/literature-tree-opc-packaging（初版，按 OPC 交付方向完成）
- **完成**：
  - 按 `findInnovation` 的 `SKILL.md` 及模板完成首版文献树、创新点评估、期刊调研与 64 篇文献清单，提交 PR #1。

### 2026-11-24 · claude · main
- **完成**：建立仓库结构（AGENTS.md、GEMINI.md、docs/、tasks/），下达任务 01。
- **未完成**：文献调研本身由 Gemini 完成。
- **注意 / 坑**：OPC 是 2025—2026 年中文语境的流行词，学术文献里未必有统一英文术语，需要调研确认后再定论文用词。
