# 项目进度（交接记录）

> 每个 AI 结束工作前在最上方追加一条记录，新的在上面。

## 当前状态

任务 01（文献树、创新点评估、期刊调研与检索清单）已由 Gemini 在分支 `gemini/literature-tree-opc-packaging` 完成并提交 PR，等待 Claude 审查合并，随后进入作者决策关口（敲定评价方案 A/B、首选目标期刊与最小验证实验启动）。

## 待办

- [x] Gemini：任务 01，见 `tasks/01-gemini-literature-tree-opc-packaging.md`（已在分支 `gemini/literature-tree-opc-packaging` 提交 PR）
- [ ] Claude：审查任务 01 的 PR 并决定是否合并
- [ ] 作者：决定能否招募少量专家（5–8 名）与消费者（40–60 名）参与评价（决定走纯客观方案 A 还是轻量增强方案 B，对应目标刊上限见 `docs/INNOVATION.md` 与 `docs/JOURNALS.md`）
- [ ] 作者：确认单位是否认可 SSCI（若认可，*Systems* 与 *Humanities and Social Sciences Communications* 适配度极高）及是否对 MDPI 有限制（若限制，SCIE 首选 *PLOS ONE* 与 *Scientific Reports*）
- [ ] 下一步实验：启动 `docs/INNOVATION.md` 中的最小验证实验（10 款公开县域农产品 × ECMA A20 折盒，对比“通用文生图 vs. 图文刀版解耦+GB 7718-2025 规则校验”）

## 已知问题

- 7 篇种子文献中有 2 篇（Tao Chen 的 Semarak Ilmu 标题、SAUNIC 2026 寨子乡论文）在免费开放学术库（OpenAlex / Crossref / 公开网页）中未检索到独立 DOI 或正式出版页，已在 `literature/papers.csv` 与 `literature/search-log.md` 中单独标注 `verified=no`。
- MDPI 官网对自动化抓取返回 HTTP 403，种子文献 5（`10.3390/app15116033`）的方法与样本量细节基于 Crossref/OpenAlex 完整摘要及公开检索页面核实，正式引用其正文表格数值前需人工打开原文 PDF 复核。
- `docs/JOURNALS.md` 中的 JIF 与中科院 2025 分区来自公开汇总信息（非订阅端直连），投稿前需在 Clarivate MJL 官网与本单位图书馆最新目录做实时终审。

---

## 工作记录

### 2026-11-24 · gemini · gemini/literature-tree-opc-packaging
- **完成**：
  - 按 `findInnovation` 的 `SKILL.md`（Stage 0 → Stage 7）及两个模板，产出全部 5 份核心文件：`docs/literature-tree/literature-tree-v1.md`、`docs/INNOVATION.md`、`docs/JOURNALS.md`、`literature/search-log.md`、`literature/papers.csv`。
  - 使用免费开放源（OpenAlex API、Crossref API、公开网页/国标公告）完成 25 组中英文检索与 11 刊 ISSN 定向核查，建立 64 篇文献清单（62 篇 `verified=yes` 均附可点开 DOI/链接，2 篇无法核实的种子文献单独标注 `verified=no`）。
  - **关键发现（解除 OPC 术语顾虑）**：2026 年已有至少 3 篇 SCIE/SSCI 期刊正式使用 **`One-Person Company (OPC)`**——Liu & Yang (2026, *Systems*, `10.3390/systems14080994`) 将其定义为 `1 + N + AI` 人机社会技术系统，Kshetri (2026, IEEE *IT Professional*, `10.1109/mitp.2026.3694520`) 与 Li, Zeng, & Li (2026, *Strategic Entrepreneurship Journal*, `10.1002/sej.70067`) 均正式采用该词。
  - 检验完四条假设（H1 成立、H2 部分成立、H3 成立、H4 高度成立），提出推荐切入点 **PackOPC（视觉—可交付双层脱节诊断基准 + OPC 印前与 GB 7718-2025/28050-2025 合规闭环交付架构 + 整单运维实证）**，并给出 A/B 两套评价方案及 10 本纯 OA 候选期刊（附 2024–2026 相似论文真实 DOI）。
- **未完成**：
  - 未启动代码原型与实证实验（属下一阶段任务）；2 篇未核实种子文献待作者或后续人工排查；未自行合并 PR。
- **注意 / 坑**：
  - 种子文献 2（*JISEM*）与种子文献 3（*Applied Mathematics and Nonlinear Sciences*）已被 Scopus 剔除且口碑崩塌，切勿作为方法基准或投稿候选。
  - *Applied Sciences*、*Sustainability*、*HSSC* 与 *J. Eng. Des.* 2025–2026 已密集发表“感性工学/文化符号 + AHP + Midjourney/SD + TOPSIS/FCE”包装论文，必须坚决绕开这一红海，主攻“印前刀版 + 新国标 GB 7718-2025/28050-2025（2027-03-16 强制实施）硬合规闭环 + OPC 整单交付审计”。
  - *Systems* 在 Clarivate MJL 为 **SSCI（非 SCIE）**，*Frontiers in Computer Science* 为 **ESCI（非 SCIE）**，*Heliyon* 虽已解除 2024 年 On Hold 但部分高校仍有顾虑，选刊时需特别核对单位规则。

### 2026-11-24 · claude · main
- **完成**：建立仓库结构（AGENTS.md、GEMINI.md、docs/、tasks/），下达任务 01。
- **未完成**：文献调研本身由 Gemini 完成。
- **注意 / 坑**：OPC 是 2025—2026 年中文语境的流行词，学术文献里未必有统一英文术语，需要调研确认后再定论文用词。
