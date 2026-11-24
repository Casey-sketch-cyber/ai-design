# 项目进度（交接记录）

> 每个 AI 结束工作前在最上方追加一条记录，新的在上面。

## 当前状态

作者于 2026-11-24 调整任务 01 方向：以**深度学习等 AI 方法做包装设计**为核心贡献，**只用客观指标（方案 A）**，OPC / 方案交付降为可选框架，并新增公开数据集与自采数据版权风险调研要求。Gemini 正在分支 `gemini/literature-tree-opc-packaging` 合并 `origin/main` 并按修订版任务书全面更新调研产出。

## 待办

- [ ] Gemini：按 2026-11-24 修订版任务 01 全面更新文献树、创新点、数据集调研、期刊表与检索清单
- [ ] Claude：审查任务 01 的 PR
- [x] 作者：已决定不招募专家或消费者，只用评价方案 A（2026-11-24）

## 已知问题

- 7 篇种子文献中有 2 篇（Tao Chen 的 Semarak Ilmu 标题、SAUNIC 2026 寨子乡论文）在免费开放学术库（OpenAlex / Crossref / 公开网页）中未检索到独立 DOI 或正式出版页，已在 `literature/papers.csv` 与 `literature/search-log.md` 中单独标注 `verified=no`。

---

## 工作记录

### 2026-11-24 · gemini · gemini/literature-tree-opc-packaging（初版，按 OPC 交付方向完成）
- **完成**：
  - 按 `findInnovation` 的 `SKILL.md` 及模板完成首版文献树、创新点评估、期刊调研与 64 篇文献清单（62 篇已核对），提交 PR #1。
  - 核实 2026 年 *Systems*、IEEE *IT Professional*、*SEJ* 均正式使用 `One-Person Company (OPC)` 术语。
- **后续**：因 `main` 上新增作者方向调整（转向深度学习方法 + 纯客观评价 + 公开数据集调研），随后将在本分支合并 `origin/main` 并按新方向全面修订。

### 2026-11-24 · claude · main
- **完成**：建立仓库结构（AGENTS.md、GEMINI.md、docs/、tasks/），下达任务 01；随后根据作者偏好更新任务 01（偏向深度学习做包装设计、只用客观评价指标 A 套、新增公开数据集调研）。
- **未完成**：文献调研本身由 Gemini 完成。
- **注意 / 坑**：OPC 是 2025—2026 年中文语境的流行词，学术文献里未必有统一英文术语，需要调研确认后再定论文用词。
