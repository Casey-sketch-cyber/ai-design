# OPC 驱动的乡村农产品包装设计（从图像生成到方案交付）文献树

- 版本：v1（Gemini 起草，2026-11-24）
- 状态：草稿（待 Claude 核查）
- 检索截至：2026-11-24。检索记录见 [`literature/search-log.md`](../../literature/search-log.md)（共 25 组中英文检索与期刊定向核查），完整文献清单见 [`literature/papers.csv`](../../literature/papers.csv)（共 64 篇，其中已核对 62 篇，未核对 2 篇单独标出）。

**核心问题结论**：**“从图像生成到方案交付：OPC（一人公司）驱动的乡村农产品包装设计实践”这个方向能够发表开放获取（OA）的 SCI/SSCI 期刊论文**，但前提是彻底跳出“AIGC 出概念图 + 问卷/AHP 审美评价”的红海套路。文献调研表明：别人已经较好解决了单张包装概念图生成、文化符号提取与感性工学多准则打分（如 *Journal of Engineering Design* 2025、*Applied Sciences* 2025、*Humanities and Social Sciences Communications* 2026），而真正空着且高价值的是两个环节——**(1) 从概念图到“印前可生产 + 符合 GB 7718-2025/GB 28050-2025 法规 + 多 SKU 一致”的确定性闭环交付流水线**（填补“视觉好看但不可印刷、不合规”的断层），以及 **(2) 以 OPC（`1 + N + AI` 人机社会技术系统）为组织分析单元，在真实乡村农产品订单中对交付周期、单款成本、人工返工率与首次合规率进行系统级实证审计**。最推荐投稿的 OA SCIE 期刊为 ***Applied Sciences***、***Sustainability***、***PLOS ONE*** 与 ***Scientific Reports***（若侧重 OPC 组织工作流架构，SSCI 期刊 ***Systems*** 与 ***Humanities and Social Sciences Communications*** 适配度极高）；最大风险是若只停留在几套效果图与小样本主观打分，将因“贡献不足/明显答案”在主流 SCI 期刊被直接拒稿，或误投已被 Scopus 剔除的低质期刊。

---

## 怎么读

- **方法来源**：遵循彭思达的科研选题与领域视野方法（[learning_research](https://github.com/pengsida/learning_research)、[选题课件](https://pengsida.net/files/research_topic_selection.pdf)）与本项目采纳的 `findInnovation` 流程：先按时间线梳理里程碑工作与技术演变，再沉淀“难点—解法树”（武器库），最后基于“四种情况”与价值优先原则定位候选课题。
- **脉络选择（四条）**：本课题属“生成式 AI 工程 × 乡村农产品包装设计 × 小微创意组织形态”三方交叉的应用型研究，仅看单一学科易产生盲区，故划定 4 条紧扣问题的脉络：(1) 生成式 AI 用于包装、平面与品牌设计；(2) 农产品包装设计与乡村振兴；(3) OPC / 一人公司与 AI 增强的小微创意组织及人机协作设计；(4) AI 设计产出与包装方案的评测框架。
- **核对程度与来源说明**：
  - 凡带可点开 DOI/URL 链接的条目（共 62 篇），均已通过 OpenAlex API、Crossref API 或公开页面核对标题、作者、年份、发表处，并在 `literature/papers.csv` 中标明 `verified=yes` 与 `summary_source`（`abstract` 为基于完整摘要概括，`snippet` 为基于官方元数据与检索摘录概括；注：因无付费权限且 MDPI 官网直抓返回 403，均未以未核实正文臆测细节）。
  - 7 篇种子文献中，有 2 篇（Tao Chen 的 Semarak Ilmu 论文标题、SAUNIC 2026 寨子乡论文）在免费开放学术库中未定位到独立原文或 DOI，已在文末及 `papers.csv` 中标注 `verified=no`，不作为核心论证依据。
  - 发表处已按“顶会/顶刊 → SCIE/SSCI 期刊 → 普通会议论文集 → 弱来源/被剔除期刊”分级；被 Scopus 剔除的弱来源（如 *AMNS*、*JISEM*）仅用于证伪或揭示行业风险，不指导本研究的方法设计。

---

## 概念界定与初步假设检验

### 1. 核心概念界定（OPC 学术术语核实与论文用词建议）

- **检索结果（推翻“学术界尚无 OPC 术语”的担忧）**：2026 年已有至少 **3 篇 SCIE/SSCI 同行评审期刊论文**正式采用 **`One-Person Company (OPC)`** 这一术语：
  1. Liu & Yang (2026) 在 MDPI ***Systems***（SSCI Q1，[`10.3390/systems14080994`](https://doi.org/10.3390/systems14080994)）中，正式将 OPC 定义为 **“human–AI socio-technical system with a ‘1 + N + AI’ architecture”**（单人创始人协同 N 个 AI 智能体作为“数字员工”覆盖整条价值链）；
  2. Kshetri (2026) 在 IEEE ***IT Professional***（SCIE，[`10.1109/mitp.2026.3694520`](https://doi.org/10.1109/mitp.2026.3694520)）中以 *Artificial Intelligence-Driven One-Person Companies* 剖析 OPC 的组织扩张机制与可靠性瓶颈；
  3. Li, Zeng, & Li (2026) 在 ***Strategic Entrepreneurship Journal***（SSCI/FT50 顶刊，[`10.1002/sej.70067`](https://doi.org/10.1002/sej.70067)）中提出最小可行智能（MVI）与“智能闭环（Intelligence Loop）”理论，明确将 **one-person company** 作为智能体超越 MVI 后的极限组织形态。
  4. 同源预印本亦使用 `solopreneurship via Agentic AI`（Rezazadeh & Bonehgazy, 2025, `arXiv:2511.09533`）与 `One-Person Agentic Company (OPAC)`（Preprints.org, 2026）。
- **论文用词与引用建议**：英文论文可直接使用 **`AI-driven One-Person Company (OPC)`**，首次出现时锚定为 **`an AI-augmented solo design micro-enterprise operating as a “1 + N + AI” human–AI socio-technical system`**，并引用 Liu & Yang (2026, *Systems*)、Kshetri (2026, *IT Professional*) 与 Li, Zeng, & Li (2026, *SEJ*)。如此既保留了 2025–2026 年中国创新语境的鲜明辨识度，又与国际组织管理/系统工程学界术语完全无缝衔接。
- **“从图像生成到方案交付”的学术界定**：区分**概念效果图（concept rendering / visual mockup）**与**可生产合规交付包（print-ready, regulatory-compliant deliverable package）**。后者在乡村农产品场景中特指一套可直接发往印厂与电商平台的完整工程与营销资产包：精确刀版展开图（含压痕/模切/糊口/安全区）、CMYK 印刷规范（300 DPI、3 mm 出血、套印与黑版文字规则）、零错字中文矢量排版、符合 **GB 7718-2025《预包装食品标签通则》与 GB 28050-2025《预包装食品营养标签通则》**（2025-03 发布、2027-03-16 正式强制实施）的强制性标示信息、EAN-13 条码与数字标签/追溯二维码，以及保持品牌一致性的多 SKU 衍生款与电商主图套系。

### 2. 四条初步假设（H1—H4）检验结论

| 假设 | 结论 | 核心依据（均来自已核对文献与官方标准） |
| --- | --- | --- |
| **H1（交付缺口）**：现有“AIGC + 包装设计”多停在出图与审美评价，很少处理刀版、印刷、中文文字、食品标签法规、条码、电商套系与品牌一致性 | **成立** | 现有 SCIE/SSCI 包装新作——Gao & Li (2025, *J. Eng. Des.*)、Song & Bai (2025, *Applied Sciences*)、Zhao et al. (2026, *HSSC*) 及 Chen & Fu (2025, *AHFE*)——均止步于“需求/文化词提取 → Midjourney/SD 出概念图 → AHP/TOPSIS/FCE 主观评价 + 人工 CAD 修图”。尽管视觉技术线已推出多图层可编辑海报（COLE 2023、PosterLLaVA 2026、CreatiPoster 2025）与中文文字渲染（AnyText 2024、Glyph-ByT5-v2 2024），但**未见将刀版展开几何、CMYK/出血印前规则、GB 7718-2025/GB 28050-2025 食品法规硬约束及多 SKU 一致性纳入端到端交付评测的同行评审实证研究**。 |
| **H2（组织形态）**：现有研究多把 AI 当作设计师辅助工具，很少以“一人 + 多智能体整单交付”为研究对象衡量周期、成本、返工率、合规率 | **部分成立** | **需修正前半句判断**：2026 年宏观/管理学期刊已正式将 OPC 作为独立组织形态研究（Liu & Yang 2026 *Systems*; Kshetri 2026 *IT Professional*; Li et al. 2026 *SEJ*），工程设计也出现了多智能体协同原型系统（Chen et al. 2026 *J. Eng. Des.*）。**但后半句高度成立**：上述研究要么停留在城市面板/概念推演，要么聚焦早期概念/机械仿真；**在包装与平面设计服务业，尚未见以 OPC 整单交付（order-level delivery）为分析单元、量化追踪周期/单款成本/人工返工率/首次合规率的实证论文**。 |
| **H3（场景价值）**：乡村农产品客户预算低、需求碎片化、缺设计资源，正是 OPC 模式最有价值的场景 | **成立** | Räisänen & Tuovinen (2019, *J. Rural Studies*) 揭示农村小微企业长期受限于资金与专业人才；Liu & Yang (2026, *Systems*) 实证表明 AI 生态对 OPC 创业的促进效应在原本资源稀疏的中西部与普通城市最显著；Tang et al. (2025, *JTAER*) 证明直播电商中农产品包装视觉显著影响信任与复购。更关键的是，**GB 7718-2025 将“预包装散卖”纳入监管、禁用“零添加”并新增“1+6”营养标签**，令县域农户与合作社面临“低设计预算 × 碎 SKU × 高合规处罚/下架风险”的三重挤压，构成 OPC 标准化合规交付流水线的刚性切入场景。 |
| **H4（风险）**：“AI + 农产品包装 + 乡村振兴”已有不少低档论文，若只是“AI 做几套包装 + 问卷评价”，贡献不足难进 SCI | **高度成立** | 7 篇种子文献中，2 篇发表于已被 Scopus 剔除、口碑崩塌的期刊（*AMNS* 2024、*JISEM* 2024），2 篇为普通会议论文集（*AHFE* 2025、*ACSR* 2026），2 篇未能核实原文，仅 1 篇为 SCIE（*Applied Sciences* 2025）。同时，高区 SCI/SSCI 已在 2025–2026 年被“感性工学/评论挖掘 + AHP + AIGC + TOPSIS/FCE”范式占满（*J. Eng. Des.* 2025、*HSSC* 2026）。继续沿用“做几套效果图 + 问卷”将直接掉入彭思达警示的“明显答案 / 贡献不足”陷阱。 |

---

## 长远目标与路线图

**长远目标**：构建面向县域农产品合作社、小微农企与独立设计服务者（OPC）的**端到端可审计智能包装交付系统与评测标准**，让 1 名设计师借助多智能体协作在低成本预算下稳定交付“印厂直接可印、电商直接可上架、法规零违规”的系列化农产品包装，缩小城乡品牌服务鸿沟。

| 里程碑任务 | 内容 | 现状 | 判断 |
| --- | --- | --- | --- |
| **M0：概念图生成与文化转译** | 根据农产品地域文化与消费偏好生成包装视觉概念图 | **已做到**（Midjourney/SD + ControlNet + 感性工学/AHP/TOPSIS 已成 2024–2026 标准套路） | 已做到（红海，不宜作为主贡献） |
| **M1：可编辑多图层平面与准确文字渲染** | 输出分层可编辑海报/标签（SVG/JSON），解决中文字形错误与排版遮挡 | **部分解决**（COLE/OpenCOLE/CreatiPoster/PosterLLaVA/AnyText/StarVector 覆盖了 2D 平面海报，但未覆盖 3D 折叠刀版展开面与印刷级小字法规文本） | 别处已有解法，可迁移至包装展开面 |
| **M2：印前可生产性与食品法规合规闭环交付** | 将生成资产自动映射至参数化刀版（出血/折线/安全区/CMYK/300DPI），并引入 GB 7718-2025 & GB 28050-2025 规则校验与多轮自动修正，输出印厂 PDF 与电商套系 | **未见**（仅有孤立的 LLM 食品标签文本审校构想，或机械工程领域的“多智能体+FEA 仿真闭环”） | **空白，最值得做（核心切入点）** |
| **M3：OPC（`1+N+AI`）整单交付实证与运维度量** | 以“单人设计师 + N 个专业智能体”真实执行多品类乡村农产品包装订单，量化周期、成本、返工率、首次通过率及“视觉—可交付双层脱节” | **未见**（仅见宏观城市面板 OPC 研究与建筑/城市设计工作流比较，缺包装交付订单级实证） | **空白，值得做（与 M2 耦合形成高辨识度论文）** |
| **M4：端到端市场转化与县域商业闭环验证** | 追踪落地包装在线下商超/直播电商的真实点击率、复购率与农户增收归因 | **部分解决（传统包装已有 SEM/联合分析，AIGC 落地追踪极少）** | 较远（周期长，适合作为后续追踪论文或与合作社长期联合实验） |

---

## 领域视野六问

| 问题 | 回答 | 依据 |
| --- | --- | --- |
| **里程碑论文有哪些** | ① 图像与控制底座：LDM (CVPR 2022)、ControlNet (ICCV 2023)；② 可编辑版式/矢量/文字：Content-aware Layout (TOG 2019)、CanvasVAE (ICCV 2021)、LayoutDM (CVPR 2023)、LayoutGPT (NeurIPS 2023)、COLE (2023)、AnyText (ICLR 2024)、Glyph-ByT5 (ECCV 2024)、StarVector (CVPR 2025)、PosterLLaVA (IEEE TMM 2026)；③ AIGC 包装与多智能体工程设计：Gao & Li (*J. Eng. Des.* 2025)、Song & Bai (*Applied Sciences* 2025)、Zhao et al. (*HSSC* 2026)、Chen et al. (*J. Eng. Des.* 2026)；④ OPC 与人机协同：Noy & Zhang (*Science* 2023)、Dell'Acqua et al. (*Org. Sci.* 2026)、OptiMuse (CHI 2024)、Li et al. (*SEJ* 2026)、Liu & Yang (*Systems* 2026)；⑤ 评测框架：Hevner (*MISQ* 2004)、Venable FEDS (*EJIS* 2016)、Jung (*Sustainability* 2026 双层脱节评测)。 | 详见下文四条时间线表及 `literature/papers.csv` |
| **技术如何演变** | **第一代（2019–2021）**：判别/生成式版式模型（GAN/VAE）与传统图像特征拼接；**第二代（2022–2024）**：扩散模型（LDM/Midjourney）+ 空间控制（ControlNet）+ 感性工学/AHP 提示工程，聚焦“概念效果图”；**第三代（2024–2026）**：MLLM 驱动的多图层可编辑生成（JSON/SVG）、中文字符精准渲染（AnyText/Glyph-ByT5-v2/FonTS）、多智能体协同工作流（含工程仿真闭环）；**正在发生的第四代（2026→）**：从“Prompt 优化”转向“受硬约束校验驱动的 Workflow Architecture & Human–AI Socio-Technical System (OPC)”。 | Jung (2026, *Systems*), Chen et al. (2026, *J. Eng. Des.*), Yang et al. (2026, *IEEE TMM*), Liu & Yang (2026, *Systems*) |
| **领域的终极目标** | 实现高审美、强文化辨识度、零合规缺陷、即刻可制造且成本可负担的自动化/半自动化包装全链路交付，让资源匮乏的小微农业主体也能享有专业级品牌与供应链包装能力。 | Rundh (2009), Boz et al. (2020), Räisänen & Tuovinen (2019), Liu & Yang (2026) |
| **目前达到什么水平** | 在**视觉概念发散、地域文化风格转译、单张平面海报布局**上已高度成熟；在**机械早期 3D 原型 + 仿真闭环**与**网页截图转可执行代码**上已跑通原型；但在**包装实体生产交付（3D 展开刀版 + 印前色彩/出血 + 中国食品标签强制国标合规 + 多 SKU 电商衍生套系）**上仍严重依赖设计师手工返工，且极易出现“视觉拟真但结构/法规失效”的双层脱节。 | Song & Bai (2025), Gao & Li (2025), Jung (2026, *Sustainability*), Dell'Acqua et al. (2026) |
| **哪些重要问题未解决** | ① **视觉表征与工程/法规逻辑的系统性脱节**（AIGC 包装在刀版安全区、CMYK 色彩、中文小字配料表与 GB 7718-2025/28050-2025 合规性上的高频失效）；② **从单图到套系（跨 SKU / 跨媒材：主盒+内袋+电商主图）的品牌视觉与信息一致性**；③ **OPC 一人公司模式下的工作流治理与人机监督边界**（单人面对多智能体高频产出时的审计成本、返工瓶颈与 MVI 门槛）；④ **无高成本大规模用户实验条件下的客观可生产性评测基准缺失**。 | 见“难点—解法树”与 `docs/INNOVATION.md` |
| **当前热点** | ① 多图层可编辑平面生成与 SVG 代码化（CreatiPoster 2025、StarVector 2025、PosterLLaVA 2026）；② 从提示词工程转向多智能体工作流架构与确定性校验闭环（*J. Eng. Des.* 2026、*Systems* 2026）；③ OPC（一人公司）作为 `1+N+AI` 社会技术系统的组织效率与治理机制（*SEJ* 2026、*Systems* 2026、*IT Professional* 2026）；④ 双层脱节评测（视觉层 vs. 可操作逻辑层，*Sustainability* 2026）与大模型食品标签合规审核（GB 7718-2025 过渡期热点）。 | OpenAlex & Crossref 2025–2026 检索聚类 |

---

## 时间线一：生成式 AI 用于包装、平面与品牌设计

近七年，该脉络从“矩形框版式生成”演进到“高保真文生图”，再演进到“多图层可编辑 SVG/JSON 生成与多智能体协同设计”，但**仍缺面向三维包装展开刀版、印前 CMYK/出血规范与食品法规强制排版的可生产交付闭环**。

| 年份 | 工作（带链接） | 里程碑 / 后续 | 解决了什么 | 留下什么 |
| --- | --- | --- | --- | --- |
| 2019 | Zheng et al., *Content-aware generative modeling of graphic design layouts* ([ACM TOG / SIGGRAPH](https://doi.org/10.1145/3306346.3322971)) | 里程碑 | 首次将图像与文本内容语义纳入深度平面版式生成 | 仅输出二维杂志矩形框，无矢量资产与印前约束 |
| 2021 | Li et al., *LayoutGAN* ([IEEE TPAMI](https://doi.org/10.1109/tpami.2019.2963663)) | 里程碑 | 通过可微线框渲染在几何与像素域联合训练矢量版式 | 对复杂层级嵌套与文字长度自适应能力有限 |
| 2021 | Yamaguchi, *CanvasVAE* ([ICCV](https://doi.org/10.1109/iccv48922.2021.00543)) | 里程碑 | 将多图层矢量平面文档建模为画布+属性序列，奠定可编辑文档生成基准 | 未耦合现代高保真图像生成模型 |
| 2022 | Rombach et al., *Latent Diffusion Models (LDM)* ([CVPR](https://doi.org/10.1109/cvpr52688.2022.01042)) | 里程碑 | 潜在空间扩散大幅降低高分辨率文生图算力门槛（Stable Diffusion 底座） | 纯像素光栅输出、文字乱码、空间结构难以精确控制 |
| 2023 | Zhang et al., *ControlNet* ([ICCV](https://doi.org/10.1109/iccv51070.2023.00355)) | 里程碑 | 零卷积注入线稿/深度/分割等空间条件，实现结构可控扩散生成 | 仅控制像素轮廓对齐，不保证矢量可编辑性与印刷精度 |
| 2023 | Inoue et al., *LayoutDM* ([CVPR](https://doi.org/10.1109/cvpr52729.2023.00980)) | 里程碑 | 离散扩散统一求解多类条件约束版式生成 | 仍局限于平面海报矩形元素，不支持包装折叠多面拓扑 |
| 2023 | Feng et al., *LayoutGPT* ([NeurIPS](https://doi.org/10.52202/075280-0802)) | 里程碑 | 证明 LLM 可用 CSS 声明式语言进行 2D/3D 空间布局规划 | 需结合下游生图器，缺乏硬规则校验反馈回路 |
| 2023 | Jain et al., *VectorFusion* ([CVPR](https://doi.org/10.1109/cvpr52729.2023.00190)) | 里程碑 | 通过 SDS 蒸馏像素扩散生成文本到 SVG 矢量图 | 优化耗时长、路径冗余，难以直接用于印刷刀版与精细排版 |
| 2023 | Liu et al., *3DALL-E* ([ACM DIS](https://doi.org/10.1145/3563657.3596098)) | 后续 | 将文生图嵌入 CAD 软件用于三维造型灵感启发 | 仅服务于概念发散，未生成可制造工程图纸 |
| 2023 | Jia et al., *COLE* ([arXiv](https://doi.org/10.48550/arxiv.2311.16974)) | 里程碑 | 提出 LLM+LMM+DM 分层协同生成多图层可编辑平面设计（类 Canva） | 依赖私有数据与闭源管线；聚焦广告海报，无印前/法规模块 |
| 2024 | Tuo et al., *AnyText* ([ICLR](https://doi.org/10.48550/arxiv.2311.03054)) | 里程碑 | 引入字形潜在模块与 OCR 笔画编码，首次显著提升中文等多语种图像文字渲染 | 仍属光栅像素级文字，小字号密集配料表与营养成分表无法满足 300 DPI 零差错印刷 |
| 2024 | Liu et al., *Glyph-ByT5* ([ECCV](https://doi.org/10.1007/978-3-031-73226-3_21)) & *v2* ([arXiv](https://doi.org/10.48550/arxiv.2406.10208)) | 后续 | 字符级 ByT5 编码器提升 10 语种（含中文）视觉文字拼写与审美 | 同为像素渲染，无法替代矢量字体与印刷黑版套印 |
| 2024 | Inoue et al., *OpenCOLE* ([arXiv](https://doi.org/10.48550/arxiv.2406.08232)) | 后续 | 基于公开数据完全复现开源版层次化多图层设计框架 | 仅覆盖通用平面海报，无包装结构/合规领域适配 |
| 2025 | Rodriguez et al., *StarVector* ([CVPR](https://doi.org/10.1109/cvpr52734.2025.01508)) | 里程碑 | MLLM 直接在 SVG 代码空间生成紧凑语义矢量图元（含文本与几何图形） | 偏图标/图表级矢量，尚未扩展至印刷级多面包装工程文件 |
| 2025 | Shi et al., *FonTS* ([ICCV](https://doi.org/10.1109/iccv51701.2025.01716)) | 后续 | 在 DiT 上实现词级字体与风格一致性可控渲染 | 仍为位图渲染，不具备矢量排版可编辑性 |
| 2025 | Si et al., *Design2Code* ([NAACL](https://doi.org/10.18653/v1/2025.naacl-long.199)) | 里程碑（跨领域） | 构建视觉设计到可执行前端代码生成与自动渲染核验基准 | 面向网页 DOM，但“视觉→结构化代码→确定性渲染校验”内核可迁移至包装 SVG/PDF 交付 |
| 2025 | Hong et al., *CreatiPoster* ([arXiv](https://doi.org/10.48550/arxiv.2506.10890)) | 后续 | RGBA 协议模型先出 JSON 前景规范再条件生成背景，支持多尺寸适配 | 仍止于二维数字海报，不含印刷出血、CMYK 与食品法规逻辑 |
| 2025 | Gao & Li, *A tea packaging design method based on AIGC* ([J. Eng. Des.](https://doi.org/10.1080/09544828.2025.2489647)) | 里程碑（包装应用） | 电商评论爬取+BTM+AHP+ChatGPT+Midjourney+TOPSIS 完整包装概念流 | 止步于单款概念图与手工 CAD 修图，无自动化印前合规交付 |
| 2026 | Yang et al., *PosterLLaVA* ([IEEE TMM](https://doi.org/10.1109/tmm.2026.3753095)) | 里程碑 | MLLM 结合 JSON 指令微调处理复杂约束并导出可编辑 SVG 海报 | 未处理三维包装多面展开关系、压痕/出血及法定标签模块 |
| 2026 | Chen et al., *An LLM-based multi-agent system to assist early-stage product design and evaluation* ([J. Eng. Des.](https://doi.org/10.1080/09544828.2026.2616583)) | 里程碑（跨领域） | 多智能体协同需求解析、3D 原型生成与有限元（FEA）确定性仿真闭环 | 聚焦机械结构早期概念，未下沉至平面/包装生产交付与合规审校 |

**本线技术范式小结**：视觉生成界已形成清晰的**“解耦共识”**——光栅扩散模型（LDM/ControlNet/DiT）擅长主视觉插画与材质氛围，但**文字、版式与结构必须交由结构化表示（JSON/SVG/参数化代码）分层合成**（COLE → PosterLLaVA → CreatiPoster → StarVector）。然而，现有包装设计应用论文（如 Gao & Li 2025）仍停留在“端到端直接用 Midjourney 生整张包装图”的上一代范式，未吸收视觉界的多图层解耦与代码化校验进展，更未引入印前工程与食品法规确定性验证器。

---

## 时间线二：农产品包装设计与乡村振兴

近二十年，该脉络从“包装营销功能与消费者感知”发展到“可持续农废材料与直播电商视觉信任”，2024–2026 年爆发“AIGC + 地域/非遗文化包装”研究，但**高质量期刊被文化符号转译与感性工学占满，低质会议/水刊扎堆泛滥，几乎无人触及 2025 年新版 GB 7718/28050 法规倒逼下的乡村小微合规交付痛点**。

| 年份 | 工作（带链接） | 里程碑 / 后续 | 解决了什么 | 留下什么 |
| --- | --- | --- | --- | --- |
| 2007 | Silayoi & Speece, *The importance of packaging attributes* ([Eur. J. Mark.](https://doi.org/10.1108/03090560710821279)) | 里程碑 | 联合分析揭示食品包装“视觉美学”与“标签信息”对不同客群的差异化权重 | 传统线下零售场景，未涉及 AI 与电商套系 |
| 2009 | Rundh, *Packaging design: creating competitive advantage* ([Br. Food J.](https://doi.org/10.1108/00070700910992880)) | 里程碑 | 确立食品包装“营销传播 + 供应链物流”双职能竞争模型 | 概念性多案例研究，面向传统品牌商而非乡村小农户 |
| 2020 | Boz et al., *Consumer Considerations for Sustainable Packaging* ([Sustainability](https://doi.org/10.3390/su12062192)) | 里程碑 | 系统梳理可持续包装价值链阻力、生物基材料与环保标签声称对消费者采纳的影响 | 聚焦欧美宏观市场，未解决小微主体的包装设计成本瓶颈 |
| 2023 | Tanaiutchawoot et al., *Packaging design factors on customer purchasing decisions* ([PTS](https://doi.org/10.1002/pts.2755)) | 后续 | 实证对比新旧农业包装，证实传统农业包装重盛装轻信息与品牌识别 | 人工传统改版设计，未引入 AI 提效 |
| 2024 | Duan et al., *Corn Stover as the Biomaterial in Packaging Designs* ([Sustainability](https://doi.org/10.3390/su16041381)) | 后续 | 利用山东玉米秸秆开发低成本乡村农业包装生物材料 | 偏材料配方实验，未关联数字化视觉与法规交付 |
| 2024 | Giri & Chaulagai, *Enhancing sustainable agriculture through innovative packaging* ([Cogent Food Agric.](https://doi.org/10.1080/23311932.2024.2399793)) | 后续 | 量化发展中地区消费者对创新农产品包装的支付意愿 | 问卷调研，无设计生成与交付方法 |
| 2024 | Rivers et al., *Labeling compliance of prepackaged frozen shrimp* ([Food Control](https://doi.org/10.1016/j.foodcont.2023.110101)) | 后续（合规基准） | 对市售预包装农水产品开展客观标签合规与净含量核查，暴露高比例不合规现状 | 事后抽检，未将合规规则前置嵌入包装设计环节 |
| 2024 | Chen Tao et al., *AI-driven Rural Revitalization Strategies...* ([JISEM](https://doi.org/10.55267/iadt.07.15214)) | 种子2（弱来源） | 访谈农户/分销商提出 AI 乡村包装概念框架 | **期刊已被 Scopus 剔除**；纯定性框图，无实证交付系统 |
| 2024 | Songyong Hu, *Agricultural Product Packaging Design Innovation...* ([AMNS](https://doi.org/10.2478/amns-2024-1882)) | 种子3（弱来源） | 用传统 HSV/灰度共生矩阵分析农产品包装视觉特征 | **期刊已被 Scopus 剔除**；非现代生成式 AI，方法陈旧 |
| 2025 | Song & Bai, *Novel Stingray Model in Non-Heritage Packaging* ([Applied Sciences](https://doi.org/10.3390/app15116033)) | 种子5（SCIE 直接先例） | 感性工学（N=100）+ AHP + Midjourney + 秸秆材料 + TOPSIS 评价浚县泥咕咕包装 | 停留在单款工艺品包装概念与主观排序，不涉及食品法规、印前刀版与多 SKU 商业交付 |
| 2025 | Chen & Fu, *Translating Chinese Poetic Imagery into Tea Packaging* ([AHFE](https://doi.org/10.54941/ahfe1006930)) | 种子4（会议） | ChatGPT + LDA 提取诗词意象驱动 Midjourney 生成茶包装 + AHP/FCE | 普通会议论文，同质化“文化出图+主观评价”套路 |
| 2025 | Tang et al., *Agricultural Product Packaging Images in Livestreaming* ([JTAER](https://doi.org/10.3390/jtaer20030248)) | 里程碑（电商场景） | 392 名直播电商用户 SEM+ANN 证实包装视觉（色彩/字体/插画）通过信任驱动复购 | 仅做消费端因果检验，未解决前端如何低成本生成一致的电商+实体套系 |
| 2026 | Zhao et al., *Mongolian saddle culture in kumis packaging based on AIGC* ([HSSC](https://doi.org/10.1057/s41599-026-06948-z)) | 后续（SSCI 1区竞品） | 专家访谈+AHP+文化符号库+AIGC+FCE 完成马奶酒文化包装设计 | 验证特色食品文化包装可发高区 OA 刊，但同样停留在概念出图与主观评价 |
| 2026 | Liu et al., *Generative AI and design as a bridge for rural craft revitalization* ([HSSC](https://doi.org/10.1057/s41599-026-08755-y)) | 里程碑（乡村工作流） | 提出面向乡村手工艺振兴的“以工作流为中心”的生成式 AI 设计框架 | 聚焦手工艺文创，未覆盖预包装食品的强制法规与工业印刷交付 |
| 2026 | Wu et al., *AIGC-Driven Teaching Model for Packaging Design Courses* ([ACSR](https://doi.org/10.2991/978-94-6239-737-8_27)) | 种子7（会议） | 60 名学生实验验证四阶段 AIGC 包装教学法提升效率与创意 | 属教学改革而非乡村商业交付实践 |
| 2026 | 李雪娜 & 孙雨琪，*大语言模型在预包装食品标签合规审核中的应用与展望——基于 GB 7718-2025 与 GB28050-2025* ([现代工程技术与管理](https://doi.org/10.64216/3104-9664.26.04.014)) | 后续（中文线索） | 针对 2027-03-16 即将强制实施的新国标探讨 LLM 合规审校框架 | 中文普通刊综述，停留在事后文本审校构想，未与包装视觉生成与排版闭环耦合 |

**本线范式小结**：国际 SCIE/SSCI 对“乡村/地域特色包装 + AIGC”已形成高度趋同的固定模板——**“文化/评论提取 → AHP/Kano 定权 → Midjourney/SD 出图 → TOPSIS/FCE/眼动评分”**。与此同时，国家卫健委与市场监管总局于 2025 年 3 月发布的 **GB 7718-2025 与 GB 28050-2025**（2027 年 3 月 16 日强制实施，扩大至预包装散卖、禁用“零添加”、推行数字标签与到期日直标、营养表升级“1+6”）在现实中制造了巨大的合规换版窗口期，而现有设计学术界对此几乎完全失语。

---

## 时间线三：OPC / 一人公司与 AI 增强的小微创意组织及人机协作设计

近七年，该脉络从“农村小微企业数字化”与“个体 AI 生产率实验”跃升至 **2026 年同行评审期刊正式确立 `One-Person Company (OPC)` 作为 `1+N+AI` 人机社会技术系统**，并从单次提示词转向非线性协同与工作流架构，但**缺乏设计服务业真实订单级的交付经济与治理实证**。

| 年份 | 工作（带链接） | 里程碑 / 后续 | 解决了什么 | 留下什么 |
| --- | --- | --- | --- | --- |
| 2019 | Räisänen & Tuovinen, *Digital innovations in rural micro-enterprises* ([J. Rural Studies](https://doi.org/10.1016/j.jrurstud.2019.09.010)) | 里程碑 | 揭示农村小微企业因资本、技能与外部生态匮乏导致数字化采纳滞后 | 前生成式 AI 时代，未探讨 AI 智能体对外部轻资产服务供给的重构 |
| 2023 | Noy & Zhang, *Productivity effects of generative AI* ([Science](https://doi.org/10.1126/science.adh2586)) | 里程碑 | 453 人实验证明生成式 AI 令知识任务耗时降 40%、质量升 18% | 单任务文本写作，未涉及多环节协同交付与硬合规约束 |
| 2024 | Choi et al., *CreativeConnect* ([ACM CHI](https://doi.org/10.1145/3613904.3642794)) | 里程碑 | 面向平面设计师构建参考图关键词重组生成管道，显著增强发散构思 | 仅覆盖前期构思，不涉及工程落地与交付 |
| 2024 | Zhou et al., *Understanding Nonlinear Collaboration between Human and AI Agents (OptiMuse)* ([ACM CHI](https://doi.org/10.1145/3613904.3642812)) | 里程碑 | 指出线性提示违背平面设计实践，构建非线性人机共创框架并将 AI 视为协同同事 | 仍聚焦数字海报创意探索，未考量商业订单交付的返工与成本 |
| 2025 | Chen et al., *CoExploreDS* ([ACM CHI](https://doi.org/10.1145/3706598.3713869)) | 后续 | 基于“问题—方案共演化”节点化管理产品设计中的复杂多约束人机探索 | 偏概念阶段设计空间探索，未接入印前/法规验证工具链 |
| 2025 | Rezazadeh & Bonehgazy, *Digital Co-Founders: Solo Business via Agentic AI* ([arXiv](https://doi.org/10.48550/arxiv.2511.09533)) | 里程碑（预印本） | 构建智能体作为“数字联合创始人”的 solopreneurship 三阶段框架，警示认知过载 | 概念框架，缺乏垂直行业实证与量化指标 |
| 2025 | Shao et al., *Future of Work with AI Agents (WORKBank)* ([arXiv](https://doi.org/10.48550/arxiv.2506.06576)) | 里程碑（预印本） | 提出人类主体性量表（HAS）审计 104 职业，证实创意设计偏好人机协同而非全自动 | 宏观职业调查，未深入单人设计公司工作流微机制 |
| 2026 | Dell'Acqua et al., *Navigating the Jagged Technological Frontier* ([Organization Science](https://doi.org/10.1287/orsc.2025.21838)) | 里程碑 | 758 人田野实验证明：AI 在前沿内提效 25.1%，但前沿外任务正确率反降 19% | 强烈启示包装交付必须清晰界定“前沿内（视觉发散）vs. 前沿外（刀版/法规硬校验）”并设置护栏 |
| 2026 | Li, Zeng, & Li, *The intelligence loop and value creation in the age of AI agents* ([Strategic Entrepreneurship Journal](https://doi.org/10.1002/sej.70067)) | 里程碑 | 提出 MVI 门槛与“智能闭环”理论，将 OPC 作为极限形态，并论证**削减监督与评测将破坏闭环** | 理论推演，亟需垂直行业的 MVI 评测与治理实证案例 |
| 2026 | Kshetri, *AI-Driven One-Person Companies* ([IEEE IT Professional](https://doi.org/10.1109/mitp.2026.3694520)) | 里程碑 | SCIE 期刊正式确立 AI 驱动 OPC 的概念，剖析其可靠性与规模化约束 | 行业综述视角，缺乏设计/制造交付实证数据 |
| 2026 | Liu & Yang, *The Emergence of One-Person Companies as Human–AI Socio-Technical Systems* ([Systems](https://doi.org/10.3390/systems14080994)) | 里程碑 | 正式提出 OPC 的 **`1 + N + AI`** 社会技术系统定义，并用中国 35 城面板证明其在中西部/普通城市拉动最强 | 宏观城市计量研究，未深入微观业务交付流程 |
| 2026 | Jung, *From Prompt Optimisation to Workflow Architecture* ([Systems](https://doi.org/10.3390/systems14070771)) | 里程碑 | 通过 151 次提示片段证实强约束设计中单靠改 Prompt 必败，必须重构多阶段人机工作流架构 | 面向城市设计教学，可直接迁移至强约束食品包装 OPC 交付工作流 |
| 2026 | He et al., *Cognitive determinants in Chinese artisanal design practices mediated by AIGC* ([Scientific Reports](https://doi.org/10.1038/s41598-026-65179-z)) | 后续 | 用 SEM 揭示中国传统手工艺人面对 AIGC 的文化本真性顾虑与采纳机制 | 偏态度调查，未构建实际协同交付工具 |
| 2026 | Mirzaei et al., *AI across design thinking: a qualitative review* ([Design Science](https://doi.org/10.1017/dsj.2026.10054)) | 后续 | 综述 205 篇文献，主张“AI 管发散、人类管收敛与伦理”，强调可审计治理（auditability） | 综述性框架，呼吁面向具体边界条件的实证检验 |

**本线范式小结**：2026 年的顶刊与主流期刊已为本研究搭建了极具说服力的理论护城河——**Liu & Yang (2026) 确立了 `1 + N + AI` 的 OPC 社会技术系统定义**，**Dell'Acqua et al. (2026) 的“锯齿状前沿”解释了为何纯 AIGC 在合规/印前环节反而容易出错**，**Li et al. (2026) 的“MVI 与智能闭环”论证了评测与治理节点是 OPC 能否跑通的生死线**，**Jung (2026) 证明强约束设计必须从提示词优化跃升至工作流架构**。缺的正是包装设计领域一份扎实、可复现的微观实证与工程交付基准。

---

## 时间线四：AI 设计产出与包装方案的评测框架

该脉络分化为三条相对割裂的分支：信息系统/工程的 **DSR 人工制品评测**、传统包装的 **感性工学/MCDM/眼动主观评价**、AI 领域的 **CLIPScore/LLM-as-a-Judge 自动指标**；直到 2026 年才出现“视觉表征层 vs. 可操作逻辑层”的双层脱节评测思路，**包装领域的客观可生产性与法规合规评测基准仍属空白**。

| 年份 | 工作（带链接） | 里程碑 / 后续 | 解决了什么 | 留下什么 |
| --- | --- | --- | --- | --- |
| 2004 | Hevner et al., *Design science in information systems research* ([MIS Quarterly](https://doi.org/10.2307/25148625)) | 里程碑 | 确立设计科学研究（DSR）人工制品构建与多模态严谨评测范式 | 通用元方法论，需针对包装领域实例化指标 |
| 2007 | Peffers et al., *A Design Science Research Methodology (DSRM)* ([JMIS](https://doi.org/10.2753/mis0742-1222240302)) | 里程碑 | 提供六步 DSR 论文标准研发与呈现流程 | 需与具体设计学科的工程测试和专家/用户验证结合 |
| 2016 | Venable et al., *FEDS: Framework for Evaluation in DSR* ([EJIS](https://doi.org/10.1057/ejis.2014.36)) | 里程碑 | 构建“形成性/总结性 × 人工实验/自然情境”二维评测策略矩阵 | 直接支撑本课题的方案 A（纯客观人工审计）与方案 B（自然情境人因验证）分层论证 |
| 2021 | Hessel et al., *CLIPScore* ([EMNLP](https://doi.org/10.18653/v1/2021.emnlp-main.595)) | 里程碑 | 提供无参考图文语义对齐自动指标 | 仅测语义相关性，对中文字形错漏、印刷出血、法规缺项完全盲视 |
| 2023 | Zheng et al., *Judging LLM-as-a-Judge* ([NeurIPS](https://doi.org/10.52202/075280-2020)) | 里程碑 | 揭示大模型裁判与人类偏好一致性及位置/长度/自我增强偏差 | 单独使用 VLM 打分易产生主观幻觉，必须与确定性规则审计互补 |
| 2024 | Wang et al., *Kansei Engineering and Eye Movement to Tea Packaging* ([PTS](https://doi.org/10.1002/pts.2798)) | 里程碑 | 结合感性工学语义差异与眼动指标客观验证包装视觉焦点 | 需昂贵眼动仪与线下被试，且仅测视觉吸引不测生产合规 |
| 2024 | Yang et al., *Packaging design evaluation based on image emotion perception* ([Heliyon](https://doi.org/10.1016/j.heliyon.2024.e31408)) | 后续 | 用图像情感计算替代部分人工美学打分 | 仍停留在表层情绪维度，缺乏生产与法规可交付指标 |
| 2025 | Zhang & Jiang, *Visual appeal and local cultural identity in wooden packaging* ([BioResources](https://doi.org/10.15376/biores.21.1.188-207)) | 后续 | 10 名专家 FAHP-CRITIC-TOPSIS 主客观赋权 + 44 人眼动交叉验证云南咖啡包装 | 是高规范人因评价标杆（方案 B 参照），但未测印前/合规/OPC 效率 |
| 2025 | Sun & Liu, *Evaluating generative AI tools for visual communication design* ([Scientific Reports](https://doi.org/10.1038/s41598-025-18506-9)) | 后续 | 用球面模糊 CoCoSo 多准则方法比较视觉生成工具 | 偏工具选型主观矩阵，缺乏端到端交付物审计 |
| 2026 | Jung, *Visually Sustainable but Spatially Broken? A Two-Level Assessment* ([Sustainability](https://doi.org/10.3390/su18062943)) | 里程碑（跨领域） | 提出**“视觉表征层 vs. 可操作逻辑层”双层评测框架**，量化生成式 AI 的系统性脱节 | 面向可持续城市设计，**其双层脱节评测范式可直接迁移为包装设计的“视觉审美层 vs. 印前与法规可交付层”双层审计体系** |

**本线范式小结**：现有包装论文过度依赖“主观打分（AHP/TOPSIS/FCE）”或“语义对齐（CLIPScore）”，恰恰缺失了印厂与市场监管局真正关心的**确定性硬指标**（刀版几何/出血/CMYK/300DPI/OCR 零错误/GB 7718-2025 逐条合规/条码可扫率/跨 SKU 品牌一致性/OPC 整单耗时与返工率）。借鉴 Venable (2016) FEDS 与 Jung (2026) 双层评测框架，可以在**不强制依赖大规模人类被试的情况下**建立极具说服力的客观审计基线（方案 A），并在条件允许时叠加小规模印厂/设计专家与消费者验证（方案 B）。

---

## 按里程碑任务归类的代表性 pipeline

| 里程碑任务 | 代表性 pipeline | 代表论文 |
| --- | --- | --- |
| **M0：包装概念视觉生成** | `电商评论/文化文本 → LDA/BTM/感性工学 → AHP 定权 → LLM 转 Prompt → Midjourney/SD 出图 → TOPSIS/FCE 排序` | Gao & Li (*J. Eng. Des.* 2025); Song & Bai (*Applied Sciences* 2025); Chen & Fu (*AHFE* 2025); Zhao et al. (*HSSC* 2026) |
| **M1：可编辑多图层与准确文字生成** | `用户意图 → MLLM/协议模型输出 JSON 图层与布局规范 → 条件化背景生成 + 矢量/字形渲染器（AnyText/Glyph-ByT5/SVG 代码） → 可编辑画布` | COLE (2023); OpenCOLE (2024); AnyText (ICLR 2024); StarVector (CVPR 2025); CreatiPoster (2025); PosterLLaVA (IEEE TMM 2026) |
| **M2（跨领域参照）：多智能体 + 确定性工程验证闭环** | `需求解析 Agent → 结构化原型/代码生成 Agent → 确定性校验器（FEA 仿真 / DOM 渲染 / 空间逻辑规则） → 失败诊断与自动迭代修正` | Chen et al. (*J. Eng. Des.* 2026); Design2Code (NAACL 2025); Jung (*Systems* 2026 & *Sustainability* 2026) |
| **M2+M3（本课题目标空白）：OPC 驱动的农产品包装合规交付闭环** | `乡村客户简报/产品参数 → 1(设计师决策) + N(文化转译/主视觉/参数化刀版/GB 7718-2025 法规排版/多SKU电商衍生/确定性印前与合规审计 Agent) → 自动返工闭环 → 印厂 CMYK-PDF + 电商套系 + 运维审计日志` | **当前文献空白（本课题拟填补）** |

---

## 难点—解法树（武器库）

在“视觉概念出图”与“2D 海报多图层编辑”上已有成熟解法，但在**三维折叠刀版硬约束、中国食品新国标（GB 7718-2025/28050-2025）零差错排版、印前 CMYK/出血自动校验、多 SKU 品牌一致衍生及 OPC 整单运维治理**五个环节仍存在显著空白。

| 难点 | 现有解法 | 来自 | 在本方向上的空白（待实验验证判断） | 别的数据域里有没有同内核的解法 |
| --- | --- | --- | --- | --- |
| **D1：光栅文生图中文小字乱码与版式不可编辑** | 字符级编码/OCR 笔画约束（AnyText、Glyph-ByT5-v2、FonTS）或图层解耦（COLE、PosterLLaVA、CreatiPoster 输出 JSON/SVG） | ICLR 2024; ECCV 2024; ICCV 2025; IEEE TMM 2026 | 包装背面密集的中文配料表、1+6 营养成分表与生产商信息**不能用像素生成**，必须由“扩散生主视觉底图 + 程序化矢量排版引擎（SVG/ReportLab/Scribus/LaTeX-PDF）”严格解耦 | 有：**网页前端生成**（Design2Code, NAACL 2025）通过结构化 HTML/CSS 代码渲染彻底避免像素文字畸变 |
| **D2：二维画面与三维包装刀版（出血/模切/压痕/折面方向）脱节** | ControlNet 线稿控制（ICCV 2023）、传统参数化折叠纸盒 CAD（ECMA/FEFCO 标准模板） | ICCV 2023; 传统包装 CAD 文献 | 现有 AIGC 包装把整张概念图当“平面画”生成，导致折面倒置、关键文字落入模切/出血/糊口区；缺乏将 ECMA/FEFCO 参数化刀版 SVG 与区域掩码绑定到生成+排版闭环的公开框架 | 有：**机械/建筑参数化生成 + 几何约束检查**（Chen et al., *J. Eng. Des.* 2026; Jung, *Sustainability* 2026） |
| **D3：食品标签强制法规（GB 7718-2025 & GB 28050-2025）高违规风险** | 人工法务核对；刚出现的 LLM 事后文本合规审校构想（李雪娜等, 2026） | 2025 国标公告; 李雪娜等 (2026) | 尚未将新版国标硬规则（禁用“零添加”、字符高度 ≥1.8 mm、净含量与品名同展示面、1+6 营养表逻辑校验、致敏原加粗、数字标签二维码）编码为**机器可执行的规则校验器 + 冲突自修复节点**嵌入设计生成流 | 有：**软件工程/金融合规中的 Rule-Engine + LLM Verifier** 与 *SEJ* (2026) 提出的 MVI 治理闭环 |
| **D4：低预算乡村品牌的多 SKU 跨媒材一致性（主盒+内袋+手提袋+电商主图）** | IP-Adapter/LoRA 风格微调；CreatiPoster 多尺寸适配 | 扩散微调社区; CreatiPoster (2025) | 乡村合作社常需同一县域品牌下 3–8 款不同规格/口味/电商主图，单次提示极易导致风格漂移、色彩色差（RGB→CMYK 色域溢出）与标识变形 | 有：**品牌视觉系统（Design Token）与多端响应式组件库**（前端 UI 设计系统） |
| **D5：OPC 一人公司的认知过载、返工黑箱与质量不可审计** | 宏观“1+N+AI”模型与工作流分层架构概念（Liu & Yang 2026; Jung 2026; Rezazadeh 2025） | *Systems* 2026; *SEJ* 2026; arXiv 2025 | 缺少面向单人包装设计师的**过程可追踪、失败可定位、人类只在关键决策点（审美定调+最终签发）介入**的轻量化多智能体编排与审计日志体系 | 有：**AI 编程智能体工作队列与独立审校者模式**（2026 OPC 编码治理报告）及 **FEDS 形成性/总结性评测**（Venable 2016） |

---

## 候选课题

| 候选 | 设定（输入 → 输出） | 要解决的问题或 failure case | 来源（视野六问 / failure case / 新锤子 / 工程问题 / 用户需求） |
| --- | --- | --- | --- |
| **C1：乡村农产品包装“视觉—可交付”双层脱节实证审计与基准构建** | 输入：主流文生图/多模态工具 + 电商/乡村包装任务集（30–50 个真实县域农产品简报） → 输出：量化“视觉审美 vs. 印前/GB 7718-2025 法规/OCR/刀版可交付率”的双层诊断基准（Benchmark）与失效分类学 | 揭示并量化当前 AIGC 包装“视觉高分、交付归零”的系统性 failure case，填补无客观基准空白 | failure case + 视野六问（参照 Jung 2026 *Sustainability*） |
| **C2：OPC（1+N+AI）合规闭环农产品包装交付架构（PackOPC）与整单效率实证** | 输入：农产品结构化简报（品名/配料/营养/规格/地域文化/目标渠道）+ 参数化刀版 → 输出：印厂直接可印 CMYK-PDF（含刀版/出血/矢量文字/GB 7718-2025 合规标签/条码与数字标签二维码）+ 电商套系主图 + 交付审计报告 | 打通从图像生成到合规可生产交付的断层，并量化对比“单人传统流程 vs. 单人通用 AIGC vs. OPC 多智能体闭环架构”的周期、成本、返工率与首次合规率 | 用户需求 + 工程问题 + 新锤子（迁移 *J. Eng. Des.* 2026 确定性校验闭环） |
| **C3：基于 GB 7718-2025 数字标签与多 SKU 品牌一致性的县域农产品“一码多端”智能生成系统** | 输入：县域区域公用品牌视觉规范（Design Tokens）+ 多规格农产品元数据 → 输出：系列化实体包装 + GB 7718-2025 规范化 H5 数字标签页 + 直播电商主图套系 | 解决县域合作社小包装字小看不清、多 SKU 风格割裂、线上线下视觉不统一问题 | 用户需求 + 政策新窗口（GB 7718-2025 数字标签条款） |
| **C4：感性工学 + AIGC + AHP/TOPSIS 乡村特色农产品文化包装设计** | 输入：某县域特色农产品文化元素与用户问卷 → 输出：AI 概念包装方案 + AHP/TOPSIS 排序 | 提升文化表达与视觉满意度（传统套路） | 现有文献延续（对照组/反面基准） |
| **C5：端到端 3D 结构刀版神经网络生成 + 实时物理抗压仿真** | 输入：农产品物理尺寸/重量/物流跌落条件 → 输出：端到端神经生成的非标异形纸盒结构与力学仿真 | 从零发明新的包装力学结构生成模型 | 视野六问（纯算法/力学硬核方向） |

---

## 按四种情况判断

四种情况的定义（引自彭思达《科研选题》课件，2025-08-12，[PDF](https://pengsida.net/files/research_topic_selection.pdf)）：

1. “同样输入输出的任务，已经有了不错的解决方案，只是有些地方做得还不够好。”
2. “输入或输出有所变化的任务，已经有了不错的解决方案。或者在多个完全不同 data domain、技术问题内核相同的任务中，都有了不错的类似解决方案。”
3. “只有少数一两个完全不同 data domain、技术问题内核相同的任务，有不错的解决方案。”
4. “在各个不同领域的任务中，虽然有类似的技术问题，但都没有较好的解决方案。”

> 注：彭思达课件只指出“不同问题的创新空间区别很大”，并未对四种情况排序，也未将其固定分配给新手或老手。下表判断与优先级由本研究结合设计学院教师背景、算力与数据约束独立作出。

| 候选 | 属于哪种情况 | 理由 | 与已有工作的区别类型 | 初步结论 |
| --- | --- | --- | --- | --- |
| **C2：OPC（1+N+AI）合规闭环农产品包装交付架构与整单实证** | **情况 3**（兼含情况 2 特征） | 与本任务内核相同（“LLM 多智能体 + 确定性工程/规范校验闭环”）的解法目前仅在极少数远域出现（如机械产品设计的 LLM+FEA 闭环 *J. Eng. Des.* 2026、前端网页的 Design2Code NAACL 2025）；迁移至“农产品包装印前+GB 7718-2025 法规+OPC 整单交付”面临全新约束（3D 展开面拓扑、中文强制法规规则、低预算碎 SKU 运维经济） | 克服现有方案的局限 + 满足新需求（乡村低预算合规场景）+ 换一个角度看问题（从“生图工具”转向“OPC 可审计交付系统”） | **优先（核心推荐）** |
| **C1：乡村农产品包装“视觉—可交付”双层脱节审计与基准** | **情况 2** | 视觉评估基准与双层脱节评估在城市设计（Jung 2026 *Sustainability*）与前端代码（Design2Code 2025）已有清晰先例，迁移至预包装农产品领域能首次暴露全新失效模式（法规禁用词、出血截断、中文字形错误、CMYK 色域溢损） | 换一个角度看问题 + 在新环境里检验已有结论（且不需招募被试即可高质量完成，极适合作为 C2 的前半段诊断基线） | **优先（建议与 C2 整合为“诊断基准 + 闭环系统”一体论文）** |
| **C3：GB 7718-2025 数字标签与多 SKU 品牌一致生成系统** | **情况 2** | 跨尺寸海报适配（CreatiPoster 2025）与品牌设计系统（Design Tokens）在数字广告/UI 领域较成熟，结合 2025 新国标数字标签有鲜明时效与场景亮点，但单做这一项偏工程功能集成 | 满足新需求（新国标数字标签）+ 改善使用体验 | **备选（可作为 C2 的“大胆尝试”扩展模块）** |
| **C4：感性工学 + AIGC + AHP/TOPSIS 文化包装设计** | **情况 1** | 同样输入输出已有 *J. Eng. Des.* (2025)、*Applied Sciences* (2025)、*HSSC* (2026) 等多篇高区 SCI/SSCI 成熟示范，简单 A+B 组合已解得很好 | 仅微调案例对象，缺乏实质区别 | **换题（原因：A+B 已有成熟解法，且极易落入“明显答案/贡献不足”陷阱）** |
| **C5：端到端 3D 非标刀版神经生成 + 物理力学仿真** | **情况 4** | 跨领域均无成熟端到端神经网络直接生成可制造异形折叠包装力学结构方案；但高度依赖纯算法研发、大规模 3D 结构训练集与力学实验室验证，与设计学院“偏产品/包装实践、单卡 5090、无付费数据”约束不匹配 | 潜在新范式，但超出当前团队与周期条件 | **换题（原因：与作者背景、数据与时间约束不匹配）** |

---

## 未核实事项与下一步

### 未核实事项

1. **未核对的 2 篇种子文献**：
   - Tao Chen, *The Application of Artificial Intelligence-Assisted Technology in Packaging Design of Agricultural Products*（Semarak Ilmu 旗下期刊）：在 OpenAlex 与 Crossref 未检出独立 DOI，仅核实同一作者组（Tao Chen & Mohamed Razeef Abdul Razak）的另外两篇论文（`10.55267/iadt.07.15214`、`10.28978/nesciences.1567827`）。
   - *Humanity in the Loop: Ethical AI-Enabled Packaging Design for Rural Agricultural Revitalization in Zhaizixiang, China*（SAUNIC 2026）：在 OpenAlex、Crossref 与公开检索均未找到正式出版页或 DOI。
2. **检索覆盖边界说明**：
   - 本文对“未见将 AIGC 与中国 GB 7718-2025 食品标签硬约束及参数化刀版印前校验打通的 SCI 实证研究”这一判断，基于 2026-11-24 在 OpenAlex、Crossref、arXiv 与公开中文检索（见 `literature/search-log.md` Q01–Q25）的有限结果，未涵盖尚未分配 DOI 的在审稿件或需付费权限的知网全文库全部硕士学位论文。
   - MDPI 官网直抓触发 403，故 `10.3390/app15116033` 的方法细节基于 Crossref/OpenAlex 完整摘要及公开检索页面核实，正式引用其具体表格数据前建议由作者在浏览器直接打开 PDF 复核。

### 下一步

- [ ] 阅读 [`docs/INNOVATION.md`](../INNOVATION.md)，确定核心切入点（推荐 **C1+C2 整合方案**）与 A/B 两套评价方案的选择。
- [ ] 阅读 [`docs/JOURNALS.md`](../JOURNALS.md)，由作者结合单位认定规则（是否认可 MDPI、偏 SCIE 还是 SSCI）敲定 1–2 本首选目标期刊。
- [ ] 启动最小验证实验（First Experiment）：抽取 15–20 款真实县域预包装农产品（含不同盒型与品类），先用“通用文生图 vs. 结构化矢量解耦+GB 7718-2025 规则校验”跑通第一个客观审计对照实验。
