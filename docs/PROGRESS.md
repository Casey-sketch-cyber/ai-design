# 项目进度（交接记录）

> 每个 AI 结束工作前在最上方追加一条记录，新的在上面。

## 当前状态

任务 01 已按作者 2026-11-24 修订方向（**以深度学习等 AI 方法做包装设计为核心、只用纯客观指标方案 A、新增公开数据集与自采版权调研、OPC 降为可选讨论框架**）由 Gemini 在分支 `gemini/literature-tree-opc-packaging` 全面重构完成并更新 PR #1，等待 Claude 审查合并。

## 待办

- [x] Gemini：按 2026-11-24 修订版任务 01 全面更新文献树、创新点、数据集调研、期刊表与 100 篇文献清单（已更新 PR #1）
- [ ] Claude：审查 PR #1（按修订版任务 01 验收）并决定是否合并
- [x] 作者：已决定不招募专家或消费者，只用评价方案 A（2026-11-24）
- [ ] 作者：确认首选方法主线（推荐 **DL-1 `DieLineDiff` 刀版拓扑感知多面一致扩散**，或稳健版 **DL-2 `PackLayText` 双轨中文图文与印前约束**）以及本单位对 MDPI / IEEE Access 的认定口径（影响首选 *IEEE Access*/*Applied Sciences*/*Electronics* 还是 *Scientific Reports*/*PLOS ONE*）
- [ ] 下一步实验：拉取免费公开子集（Open Food Facts + RPC + Crello + AnyWord-3M）并运行 1 周最小验证实验（ECMA A20 折叠盒 × 50 组 Prompt，对比 `SDXL+ControlNet` vs. 刀版共边潜特征同步推理的 Seam PSNR/LPIPS 与 3D 重投影 SSIM）

## 已知问题

- 7 篇种子文献中有 2 篇（Tao Chen 的 Semarak Ilmu 标题、SAUNIC 2026 寨子乡论文）在 OpenAlex / Crossref / 公开网页中未检索到独立 DOI 或正式出版页，已在 `literature/papers.csv` 与 `literature/search-log.md` 中单独标注 `verified=no`。
- 公开数据集（`docs/DATASETS.md`）的规模与许可已根据官方页/论文核实，但国内网络实际下载 HuggingFace / AWS S3 / Kaggle 速度及 PKU PosterLayout 协议审批耗时需在实验启动第一周实测。
- `docs/JOURNALS.md` 的 JIF 与中科院 2025 分区来自公开汇总数据，正式投稿前须在 Clarivate MJL 官网与本单位图书馆最新目录做实时终审。

---

## 工作记录

### 2026-11-24 · gemini · gemini/literature-tree-opc-packaging（修订版：深度学习做包装设计 + 纯客观指标 A 套 + 公开数据集）
- **完成**：
  - 合并 `origin/main`（`3bbc3d8`），保留作者全部最新决策，按修订版任务书全面升级全部产出并新增 [`docs/DATASETS.md`](DATASETS.md)：
    - [`docs/literature-tree/literature-tree-v1.md`](literature-tree/literature-tree-v1.md)：以**脉络一（深度学习做包装/平面/版式/中文文字/3D 贴图，45 篇顶会顶刊）**为绝对主线，系统拆解深度学习包装设计的 **5 大具体技术难点（T1 中文文字渲染、T2 品牌与跨 SKU 一致性、T3 2D 刀版与 3D 多面对应、T4 品类与风格解耦、T5 可印刷性与多图层可编辑）**，将 H1—H4 结论归结到深度学习方法可行性，OPC 降为可选应用/讨论框架，评价全面锁定纯客观 A 套。
    - [`docs/DATASETS.md`](DATASETS.md)：调研 **14 个免费公开数据集**（Open Food Facts、ABO、RPC、Grocery Store、SKU-110K、M5Product、Product1M、Crello、PKU PosterLayout、CGL、CreatiPoster 100K、AnyWord-3M、MARIO-10M、SVG-Stack），逐一列明规模、标注、License、下载方式、是否需申请及单卡 RTX 5090 适配度；并按《著作权法》第二十四条三步检验法、商标权与反不正当竞争法深度剖析自采电商包装图的版权风险，给出“公开合规底座 + Blender 程序化合成 2D 刀版↔3D 渲染基准 `PackDieline-Syn`”零版权雷区蓝图。
    - [`docs/INNOVATION.md`](INNOVATION.md)：全面转向深度学习方法创新（彻底移除方案 B），提出核心推荐 **DL-1 `DieLineDiff`（刀版拓扑感知跨面注意力扩散 + 中文双轨图文解耦 + 印前引导，属情况 3 空白）**与 5 档发散方案，给出 **6 维零人类被试纯客观指标矩阵（FID/LPIPS/HPS v2/ImageReward/PickScore/CLIPScore/VQAScore/PaddleOCR Acc & 1-NED/版式 mIoU & Overlap/Seam PSNR/3D 重投影 SSIM/CMYK OOG%/GB 7718 规则通过率）**、5 组开源 SOTA 基线对照、4 项模块消融实验及单卡 5090 1 周最小验证实验。
    - [`docs/JOURNALS.md`](JOURNALS.md)：转向面向深度学习方法 + 零被试纯客观指标的 **10 本纯 OA SCIE 期刊**（新增并核实 ***IEEE Access***、***Electronics***、***Symmetry***、***Mathematics***、***Sensors*** 2024–2026 相似论文真实 DOI，含 *IEEE Access* 2026 饼干包装 AI 设计与 *DesignCritic* 零人工设计审计论文），将纯 SSCI 期刊（*Systems*、*HSSC*）降为文末补充参考。
    - [`literature/search-log.md`](../literature/search-log.md) & [`literature/papers.csv`](../literature/papers.csv)：扩展至 **34 组中英文检索、100 篇文献（98 篇 `verified=yes` 均经 OpenAlex/Crossref 核对带可点开 DOI，2 篇未核实种子透明标注 `verified=no`）**。
- **未完成**：
  - 未启动数据下载与代码实验（属后续阶段）；未自行合并 PR。
- **注意 / 坑**：
  - *IEEE Access*（2020/2021）及部分 MDPI 刊（*Applied Sciences*/*Sustainability*/*Sensors*/*Symmetry* 2020 试行版）虽均**不在 2025 中科院预警名单**，但部分国内高校仍有内部黑/提醒名单；若单位限制，非 MDPI/非 IEEE Access 的首选纯 OA SCIE 为 ***Scientific Reports*** 与 ***PLOS ONE***。
  - 自采国内电商平台包装图存在著作权+商标权+反不正当竞争三重叠加风险，切勿直接将爬取的原图随论文开源发布；应以 Open Food Facts/RPC/ABO/Crello/AnyWord-3M + 自建 Blender 程序化合成基准 `PackDieline-Syn` 作为可公开复现主数据。

### 2026-11-24 · gemini · gemini/literature-tree-opc-packaging（初版，按 OPC 交付方向完成）
- **完成**：
  - 按 `findInnovation` 的 `SKILL.md` 及模板完成首版文献树、创新点评估、期刊调研与 64 篇文献清单（62 篇已核对），提交 PR #1。
  - 核实 2026 年 *Systems*、IEEE *IT Professional*、*SEJ* 均正式使用 `One-Person Company (OPC)` 术语。
- **后续**：因 `main` 上新增作者方向调整（转向深度学习方法 + 纯客观评价 + 公开数据集调研），已在本分支合并 `origin/main` 并按新方向全面修订。

### 2026-11-24 · claude · main
- **完成**：建立仓库结构（AGENTS.md、GEMINI.md、docs/、tasks/），下达任务 01；随后根据作者偏好更新任务 01（偏向深度学习做包装设计、只用客观评价指标 A 套、新增公开数据集调研）。
- **未完成**：文献调研本身由 Gemini 完成。
- **注意 / 坑**：OPC 是 2025—2026 年中文语境的流行词，学术文献里未必有统一英文术语，需要调研确认后再定论文用词。
