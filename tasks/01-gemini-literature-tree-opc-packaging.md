# 任务 01：OPC 驱动的乡村农产品包装设计——文献树、创新点与投稿可行性调研

- 执行者：Gemini（其他 AI 接手时同样适用）
- 下达：Claude，2026-11-24
- 分支：`gemini/literature-tree-opc-packaging`
- 产出语言：中文（论文题目、关键词、期刊名保留英文原文）

## 0. 开工前必读

1. 本仓库的 `AGENTS.md`（工作规则、分支与合并约定）。
2. 选题与创新点方法：
   https://github.com/Casey-sketch-cyber/findInnovation/blob/main/skills/research-innovation-points/SKILL.md
   模板：
   https://github.com/Casey-sketch-cyber/findInnovation/tree/main/templates
   （`LITERATURE_TREE_TEMPLATE.md`、`INNOVATION_ASSESSMENT_TEMPLATE.md`）
   严格按 SKILL.md 的 Stage 0 → Stage 7 走，文献树和创新点评估都按模板的结构写。

## 1. 要回答的核心问题

> **“从图像生成到方案交付：OPC 驱动的乡村农产品包装设计实践”这个方向，能不能发一篇开放获取（OA）的 SCI/SSCI 期刊论文？如果能，最值得做的切入点是什么、投哪几本、主要风险是什么？**

所有产出的第一段都要直接回答这个问题，再展开依据。

## 2. 概念界定（先查清楚再用）

- **OPC**：One-Person Company，“一人公司”，2025—2026 年在中国流行的说法，指“1 个创始人 + N 个 AI 智能体”独立完成从设计到交付的全链路业务。
  - 需要查清：学术文献里对应的英文说法是什么（候选：one-person company、solopreneur / solo entrepreneur、AI-augmented micro-enterprise、super-individual、human–AI team、agentic workflow 等），有没有同行评审论文用过 OPC 这个词。如果学术界没有统一术语，建议论文里怎么定义、怎么引用。
- **从图像生成到方案交付**：AI 出图只是“效果图”，乡村客户（农户、合作社、县域品牌）真正需要的是能直接交给印厂、能上架电商的方案。

## 3. 待检验的初步假设（Claude 提出，请用文献证实或推翻，不要默认它对）

- H1（交付缺口）：现有“AIGC + 包装设计”研究大多停在出图和审美评价，很少处理从图像到可生产交付物之间的环节：刀版 / 结构展开图、印刷规范（CMYK、出血、分辨率、专色）、中文文字正确性、食品标签强制信息（如 GB 7718 预包装食品标签通则、GB 28050 营养标签）、条码与追溯码、材料与成本、电商主图套系、品牌一致性。
- H2（组织形态）：现有研究多把 AI 当作设计师的辅助工具，很少以“一个人 + 多个 AI 智能体完成整单交付”为研究对象，衡量交付周期、成本、返工率、合规率。
- H3（场景价值）：乡村农产品客户预算低、需求碎片化、缺乏设计资源，正好是 OPC 模式最有价值的场景。
- H4（风险）：“AI + 农产品包装 + 乡村振兴”这个大方向已有不少论文，但其中相当一部分发表在会议论文集或影响力较低的期刊；如果只是“用 AI 做了几套包装 + 问卷评价”，贡献不足，难进 SCI。

每条假设给出结论：成立 / 部分成立 / 不成立，附依据。也欢迎发散，提出比上面更好的切入点。

## 4. 文献树：建议的脉络（可调整，写明理由）

1. **生成式 AI 用于包装、平面与品牌设计**：文生图 / 图生图、版式生成（layout generation）、图像中的文字渲染、矢量与刀版生成、LLM / 多智能体设计工作流（agentic design workflow）。
2. **农产品包装设计与乡村振兴**：区域公用品牌、地域文化元素转译、消费者感知与购买意愿、可持续包装。
3. **OPC / 一人公司 / AI 增强的小微创意组织**：设计服务的组织形态、人机协作的设计实践、创意劳动与 AI。
4. **评测**：这类研究怎么评价 AI 设计结果（专家评分、消费者问卷、CLIP 等自动指标、可生产性 / 合规性检查表、设计科学研究 DSR 的评价框架、时间与成本记录）。

每条脉络按模板给时间线表（年份、工作带链接、里程碑或后续、解决了什么、留下什么），再做“难点—解法树”和“四种情况”判断。

### 种子文献（Claude 用网页检索初步找到，**全部需要打开原文核对**标题、作者、年份、发表处，并评估发表处质量）

- Tao Chen, The Application of Artificial Intelligence-Assisted Technology in Packaging Design of Agricultural Products（Semarak Ilmu 旗下期刊）
- A Framework for AI-driven Rural Revitalization Strategies: Balancing Brand Image, Cultural Compliance and Consumer Behavior Focusing on Agri Products Packaging Designs，doi:10.55267/iadt.07.15214
- Songyong Hu (2024), Exploration of Agricultural Product Packaging Design Innovation in the Context of Deep Learning, Applied Mathematics and Nonlinear Sciences 9(1)
- Generative AI-Driven Optimization for Cultural Packaging Design: Translating Chinese Poetic Imagery into Tea Packaging Design, AHFE 2025, doi:10.54941/ahfe1006930
- Sustainable Application and Evaluation of the Novel Stingray Model in Non-Heritage Packaging: The Case of Clay Sculptures in Joon County, Applied Sciences 2025, 15(11), 6033, doi:10.3390/app15116033（MDPI 期刊里已有“生成工具 + 县域包装”的先例，重点看它的研究设计和评价方法）
- Humanity in the Loop: Ethical AI-Enabled Packaging Design for Rural Agricultural Revitalization in Zhaizixiang, China（SAUNIC 2026 会议论文）
- An AIGC-Driven Teaching Model for Packaging Design Courses: Design and Experimental Study（ADDT 2026，doi:10.2991/978-94-6239-737-8_27）

种子只是起点：必须向前（谁引用了它）、向后（它引用了谁）追，并补上 CVPR/SIGGRAPH/CHI/UIST/DIS 等计算机与人机交互顶会里关于版式生成、图形设计生成、设计智能体的里程碑工作。

## 5. 检索约束

- **只用免费来源**：Google Scholar、Semantic Scholar、OpenAlex、Crossref、arXiv、Connected Papers、DOAJ、出版社官网（MDPI、Elsevier、Springer、Frontiers 等）、知网 / 万方的免费摘要页。作者没有 Scopus 和 Web of Science 权限，不要依赖它们。
- 每条检索记到 `literature/search-log.md`：日期、数据库、查询词、结果数、筛选后保留数。中英文关键词都要查。
- **不许编造文献**。只把打开过原文或官方页面的条目写成“已核对”；只在别人参考文献里见过的标“未核对”。每条写明概括来自全文、摘要还是检索摘录。
- 区分发表处质量：顶会 / SCI·SSCI 期刊 / ESCI / 普通会议论文集 / 预警或口碑差的期刊。弱来源只简单引用，不能让它决定研究设计。

## 6. 研究设计的可行性约束（判断“当下可解、与条件匹配”时用）

- 数据：只能用公开数据或自己从网上采集的数据，不能购买数据集。
- 算力：一张 RTX 5090（32GB 显存），内存 700GB 以上；可以本地跑开源文生图 / 多模态模型，也可以用商用 AI 工具。
- 是否能招募被试或专家评分：**未定**。请给出两套评价方案：
  - A 套：不需要招募人（例如：可生产性 / 标签合规检查表的客观审计、文字识别正确率、交付时间与成本记录、公开电商数据、自动指标）；
  - B 套：需要少量专家或消费者评价。
  并说明各自能支撑到什么档次的期刊。
- 作者背景：设计学院教师，研究方向是环境、室内、家具、产品设计与 AI 的交叉；希望论文偏产品 / 包装设计，而不是纯算法。

## 7. 期刊调研

列出 6—10 本候选 OA 期刊，做成 `docs/JOURNALS.md` 表格：

| 期刊 | 出版社 | 收录（SCIE / SSCI / ESCI，以 Clarivate Master Journal List 免费查询为准） | 影响因子与中科院分区（写明来源与年份） | 是否在中科院预警名单 | APC | 平均审稿周期 | 近两年发过的相似论文（带 DOI） | 适配度判断 |

候选（全部需要核实，可增删）：Applied Sciences、Sustainability、Systems、Agriculture、Heliyon、Scientific Reports、PLOS ONE、Humanities and Social Sciences Communications、Frontiers 系列相关期刊，以及包装、设计类的 OA 期刊。注意区分 SCI 与 SSCI、ESCI：作者要的是 SCI（SSCI 也请标出来，供作者决定）。

## 8. 产出（全部放在本仓库）

| 文件 | 内容 |
| --- | --- |
| `docs/literature-tree/literature-tree-v1.md` | 按 findInnovation 文献树模板 |
| `literature/search-log.md` | 检索记录 |
| `literature/papers.csv` | 文献清单：title, authors, year, venue, venue_tier, doi_or_url, strand, verified(yes/no), summary_source(full/abstract/snippet), one_line_takeaway |
| `docs/INNOVATION.md` | 按 findInnovation 创新点评估模板；包括 3—5 个候选切入点（从稳到激进）、四种情况判断、推荐的一个核心贡献 + 一个大胆尝试、第一个实验 |
| `docs/JOURNALS.md` | 第 7 节的期刊表 |
| `docs/PROGRESS.md` | 按 AGENTS.md 追加交接记录 |

## 9. 结束

按 `AGENTS.md` 的工作规则：在分支 `gemini/literature-tree-opc-packaging` 上提交，推送并创建 PR，PR 说明写清楚做了什么、哪些没核实。**不要自己合并到 `main`**，由 Claude 审查后合并。

## 验收标准（Claude 审查时对照）

- [ ] 第一段直接回答第 1 节的核心问题。
- [ ] 四条假设各有结论和依据。
- [ ] 文献树至少 3 条脉络，已核对文献不少于 30 篇，每篇有可点开的 DOI 或链接。
- [ ] 检索记录完整，中英文关键词都有。
- [ ] 给出 A、B 两套评价方案。
- [ ] 期刊表的收录、分区、预警信息注明来源和日期。
- [ ] 没有编造的文献；未核对的条目单独标出。
