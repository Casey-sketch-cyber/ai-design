# 候选开放获取（OA）SCIE 期刊调研表（面向深度学习包装设计 + 纯客观指标）

- 课题：**用人工智能（深度学习等）做包装设计（乡村农产品场景，纯客观指标方案 A）**
- 调研执行者：Gemini（2026-11-24 按作者修订方向重构：优先面向**纯 OA SCIE** 且天然接受**深度学习方法 + 客观指标 + 基线对比 + 消融实验（零人工被试）**的期刊）
- 核查日期：2026-11-24
- 数据来源说明：
  - **收录类别（SCIE / SSCI / ESCI）**：以 Clarivate Master Journal List (MJL) 公开收录信息及出版社官方索引声明交叉核对（2026-11-24 核查）；**正式投稿前务必在 [Clarivate MJL 官网](https://mjl.clarivate.com/) 免费输入 ISSN 做最终实时复核**。
  - **影响因子（JIF）与中科院分区**：综合 2025 年发布的 JCR 数据及 2025 年中国科学院文献情报中心期刊分区表升级版公开汇总（不同第三方镜像对最新一版微调略有滞后，投前请以本单位图书馆官方分区表为准）。
  - **中科院预警名单**：依据中国科学院文献情报中心 **2025 年 3 月发布的《国际期刊预警名单》**（2025 版全名单仅 5 本期刊，**本表全部 10 本候选 OA SCIE 期刊均不在 2025 预警名单内**；往年曾入榜但已移出或曾被 Clarivate 标记 On Hold 的历史风险均在表中如实披露）。
  - **近两年相似论文（2024–2026）**：全部通过 Crossref API 按期刊 ISSN 实时检出并核实真实 DOI，可直接点击查阅。

**核心问题结论（选刊维度）**：转向**深度学习方法做包装设计 + 纯客观评价指标（方案 A，不招募专家与消费者）**后，**纯 OA SCIE 期刊的录用路径反而比传统设计问卷类论文更清晰、更稳妥**。最推荐的 5 本核心纯 OA SCIE 期刊为：
1. ***IEEE Access***（2026 年刚发饼干包装 AI 方法文与 *DesignCritic* 零人工自动设计审计文，审稿 1–1.5 个月，但需核实本单位是否受 2020–2021 历史预警印象影响）；
2. ***Applied Sciences***（MDPI，SCIE，对“深度学习生成 + 工程应用 + 客观消融”极友好，审稿 1.5–3 个月）；
3. ***Scientific Reports***（Springer Nature，SCIE 综合 3 区，国内高校认可度稳，接收完整开源数据+纯客观定量消融的深度学习应用论文）；
4. ***PLOS ONE***（SCIE 综合 3 区，非 MDPI，极看重“方法严谨 + 开源基准 + 纯客观统计可复现”，几乎不挑剔缺少主观问卷）；
5. ***Electronics*** / ***Symmetry*** / ***Mathematics***（MDPI 旗下偏计算机/算法/数学交叉的 SCIE 刊，2025–2026 持续刊发扩散模型、设计奖励对齐与包装设计方法论文，对纯客观 DL 实验天然适配）。

> **重要避坑**：① 按新方向（深度学习方法 SCIE 论文），纯 SSCI 期刊（*Systems*、*Humanities and Social Sciences Communications*）降为文末可选参考，不再作为首选；*Frontiers in Computer Science* 为 **ESCI（非 SCIE）**，已排除；② 种子文献中出现过的 *Applied Mathematics and Nonlinear Sciences* 与 *Journal of Information Systems Engineering & Management* 均非 SCI 且**已被 Scopus 剔除**，切勿投稿。

---

## 一、10 本核心候选纯 OA SCIE 期刊对比表（均友好接收零被试 DL 方法论文）

| 期刊 | 出版社 | 收录（SCIE / SSCI / ESCI，以 Clarivate MJL 免费查询为准） | 影响因子与中科院分区（写明来源与年份） | 是否在中科院预警名单 | APC | 平均审稿周期 | 近两年发过的相似论文（带 DOI） | 适配度判断（面向 DL 包装方法 + 方案 A 纯客观指标） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **IEEE Access** (ISSN: `2169-3536`) | IEEE | **SCIE** | JIF ≈ 3.4–4.2（JCR 2024/2025）；中科院 2025 工程技术/计算机科学 3区–4区 | **否**（2025 不在列；⚠️ 曾入 2020/2021 早期预警，后已移出；部分高校仍有历史顾虑，投前务必核查单位名单） | USD 2,075–2,500（约 ¥15,000–18,000） | 初审 3–4 周；整体 1–1.5 个月（Binary 审稿机制） | 1. Liu et al. (2026), *AI-Based Human–Computer Collaborative Packaging Design (Biscuit)*, [`10.1109/access.2026.3723820`](https://doi.org/10.1109/access.2026.3723820)<br>2. Wu & Song (2026), *DesignCritic: Principle-Aware Auditing for AI Graphic Design*, [`10.1109/access.2026.3730226`](https://doi.org/10.1109/access.2026.3730226)<br>3. Wu (2025), *Fuzzy Logic & Deep Learning for Product Packaging Design*, [`10.1109/access.2025.3596858`](https://doi.org/10.1109/access.2025.3596858) | **极高（纯 DL 方法 + 纯客观指标最对口）**：2025–2026 连发包装深度学习与零人工平面审计论文，最偏爱“清晰网络模块 + FID/OCR/几何指标 + 消融实验” |
| **Applied Sciences** (ISSN: `2076-3417`) | MDPI | **SCIE** | JIF ≈ 2.5–2.9（JCR 2024/2025）；中科院 2025 工程技术 4区（部分平台标新锐 3区） | **否**（2025 不在列；曾入 2020 试行版，2023–2025 已移出；需核对单位对 MDPI 态度） | CHF 2,400（约 ¥19,500） | 初审 ~15 天；整体 1.5–3 个月 | 1. Song & Bai (2025), *Stingray Model in Non-Heritage Packaging*, [`10.3390/app15116033`](https://doi.org/10.3390/app15116033)<br>2. Wu et al. (2026), *GenAI for Cultural Heritage Based on SD-LoRA*, [`10.3390/app16094427`](https://doi.org/10.3390/app16094427)<br>3. Behara & Naidoo (2026), *GenAI Framework in Engineering Design Systems*, [`10.3390/app16105159`](https://doi.org/10.3390/app16105159) | **极高（SCIE 保底首选）**：覆盖“计算机与工程设计应用”专题，接收 SD-LoRA/扩散架构改进 + 客观消融，审稿快且对设计学院应用导向极友好 |
| **Scientific Reports** (ISSN: `2045-2322`) | Springer Nature | **SCIE** | JIF ≈ 3.8–4.9（JCR 2024/2025）；中科院 2025 综合性期刊 3区 | **否**（中科院历年从未上榜；注：曾受 NLM/MEDLINE 生物医学库审视，Clarivate SCIE 保持收录） | USD 1,990–2,590（约 ¥15,000–18,500） | 初审 3–4 周；整体 2–4 个月 | 1. Sun & Liu (2025), *Evaluating GenAI tools for visual communication design*, [`10.1038/s41598-025-18506-9`](https://doi.org/10.1038/s41598-025-18506-9)<br>2. He et al. (2026), *Chinese artisanal design practices mediated by AIGC*, [`10.1038/s41598-026-65179-z`](https://doi.org/10.1038/s41598-026-65179-z)<br>3. Liu et al. (2026), *GenAI-assisted design problem exploration*, [`10.1038/s41598-026-78577-0`](https://doi.org/10.1038/s41598-026-78577-0) | **极高（非 MDPI 的 SCIE 旗舰首选）**：品牌与国内高校认可度高；只要开源基准与代码完整、统计显著性检验（如配对 t 检验/置信区间）规范，纯客观方案 A 即可扎实过审 |
| **PLOS ONE** (ISSN: `1932-6203`) | PLOS | **SCIE** | JIF ≈ 2.8–2.9（JCR 2024/2025）；中科院 2025 综合性期刊 3区 | **否**（历年从未上榜；非 MDPI，高校普遍认可） | USD 2,290–2,382（约 ¥16,500–17,200） | 初审 1–2 周；整体 2–4 个月 | 1. Wiese (2026), *Comparison of GenAI with bias-calibrated LLM-as-judge*, [`10.1371/journal.pone.0339920`](https://doi.org/10.1371/journal.pone.0339920)<br>2. Lin (2026), *GenAI as a cognitive support tool in early-stage design*, [`10.1371/journal.pone.0358542`](https://doi.org/10.1371/journal.pone.0358542) | **高（纯客观基准 + 可复现方法稳妥首选）**：明确以“科学严谨与数据代码完全公开”为唯一录用标准，极契合开源 `PackDieline-Syn` 基准 + 纯客观指标消融 |
| **Electronics** (ISSN: `2079-9292`) | MDPI | **SCIE** | JIF ≈ 2.6–2.9（JCR 2024/2025）；中科院 2025 计算机科学/工程技术 3区–4区 | **否**（2025 不在列，历年未上中科院预警） | CHF 2,400（约 ¥19,500） | 初审 ~15 天；录用上线极快（~1.5–2.5 个月） | 1. Singh et al. (2026), *Taxonomy of Generative Models & Diffusion Denoising*, [`10.3390/electronics15061293`](https://doi.org/10.3390/electronics15061293)<br>2. Lei et al. (2026), *Automatic Layout Generation Strategies*, [`10.3390/electronics15091807`](https://doi.org/10.3390/electronics15091807) | **高（AI/CV 子栏目高度适配）**：设有 *Artificial Intelligence* 与 *Computer Vision* 专栏，极熟悉扩散模型、注意力适配器、FID/CLIP/OCR 客观消融范式，审稿快 |
| **Symmetry** (ISSN: `2073-8994`) | MDPI | **SCIE** | JIF ≈ 2.2（JCR 2024/2025）；中科院 2025 综合性期刊 3区 | **否**（2025 不在列；曾入 2020 试行版，后续多年已移出） | CHF 2,400（约 ¥19,500） | 初审 ~16 天；整体 1.5–2.5 个月 | 1. Fang & Qu (2025), *Symmetric Alignment in Sustainable Packaging Design*, [`10.3390/sym18010019`](https://doi.org/10.3390/sym18010019)<br>2. Sun et al. (2026), *Symbol Recognition of Layout Drawings*, [`10.3390/sym18050874`](https://doi.org/10.3390/sym18050874) | **中高（几何对称/刀版展开天然契合）**：折叠纸盒刀版的面片对称性、跨折线共边连续性（对称/镜像拓扑映射）与 *Symmetry* 主题高度呼应 |
| **Mathematics** (ISSN: `2227-7390`) | MDPI | **SCIE** | JIF ≈ 2.2–2.3（JCR 2024/2025，数学 **Q1**）；中科院 2025 数学 3区 | **否**（历年从未上榜） | CHF 2,600（约 ¥21,000） | 初审 ~17–18 天；整体 1.5–2.5 个月 | 1. Cao & Chen (2026), *Fast Generative Interior Design: Reward Alignment with ARIA-Diff*, [`10.3390/math14203759`](https://doi.org/10.3390/math14203759)<br>2. Huang et al. (2025), *ConceptVoid in Generative Diffusion*, [`10.3390/math13162652`](https://doi.org/10.3390/math13162652) | **中高（适合突出数学形式化）**：2026 已发扩散设计奖励对齐论文；若论文突出刀版拓扑矩阵、CMYK 色彩空间投影损失或优化公式形式化，适配度极佳 |
| **Sustainability** (ISSN: `2071-1050`) | MDPI | **SCIE + SSCI 双收录** | JIF ≈ 3.3–4.1（JCR 2024/2025）；中科院 2025 环境科学与生态学 3区 | **否**（2025 不在列；曾入 2020 试行版，2023–2025 已移出） | CHF 2,400（约 ¥19,500） | 初审 2–4 周；整体 2–3 个月 | 1. Jung (2026), *Visually Sustainable but Spatially Broken? Two-Level Assessment*, [`10.3390/su18062943`](https://doi.org/10.3390/su18062943)<br>2. Chen et al. (2025), *GDT Framework: Generative Design*, [`10.3390/su17010372`](https://doi.org/10.3390/su17010372) | **中高（需补足可持续视角）**：若在 DieLineDiff 中强调“减少打样试错浪费、面向乡村可持续包装/农废纸材色域适配”，非常匹配；纯算法叙事易被退回 |
| **Heliyon** (ISSN: `2405-8440`) | Cell Press / Elsevier | **SCIE** | JIF ≈ 3.4–4.1（JCR 2024/2025）；中科院 2025 综合性期刊 4区 | **否**（中科院未上榜；⚠️ 2024-09 曾被 Clarivate 标记 **On Hold**，2026 年中已解除恢复，部分高校仍谨慎） | USD 2,100–2,270（约 ¥15,000–16,500） | 整体 3–5 个月 | 1. Yang et al. (2024), *Packaging design evaluation based on image emotion computing*, [`10.1016/j.heliyon.2024.e31408`](https://doi.org/10.1016/j.heliyon.2024.e31408) | **中（备选）**：学科包容但因 On Hold 历史与周期偏慢，优先级低于 *Scientific Reports* 与 *PLOS ONE* |
| **Sensors** (ISSN: `1424-8220`) | MDPI | **SCIE** | JIF ≈ 3.4–4.0（JCR 2024/2025）；中科院 2025 综合性期刊 3区 | **否**（2025 不在列；曾入 2020 试行版，后续多年已移出） | CHF 2,600（约 ¥21,000） | 初审 ~17–21 天；整体 1.5–2.5 个月 | 1. Liu et al. (2025), *Inverse Design of Sustainable Food Packaging Materials via GANs*, [`10.3390/s25113320`](https://doi.org/10.3390/s25113320)<br>2. Gu et al. (2025), *Quantifiable Evaluation of Cabin Layout Design*, [`10.3390/s25165053`](https://doi.org/10.3390/s25165053) | **中偏低（需挂靠机器视觉/光学感知特刊）**：若侧重 OCR/条码光学检测与计算机视觉子栏可投，否则不如 *Electronics* / *Applied Sciences* 对口 |

---

## 二、补充参考：两本专业 SCIE 混合 OA 刊 & 两本可选 SSCI 纯 OA 刊

| 期刊 | 出版社 | 收录与 OA 模式 | 近两年代表同类文（DOI） | 何时考虑 |
| --- | --- | --- | --- | --- |
| **Packaging Technology and Science** (`0894-3214`) | Wiley | **SCIE**（混合 OA，非 OA 免费 / Gold OA ~$3,500–4,000） | [`10.1002/pts.2798`](https://doi.org/10.1002/pts.2798); [`10.1002/pts.2755`](https://doi.org/10.1002/pts.2755) | 包装学界顶刊；若后续补充真实印厂刀版打样物理测量数据，可冲刺（审稿偏慢） |
| **Journal of Engineering Design** (`0954-4828`) | Taylor & Francis | **SCIE**（混合 OA，非 OA 免费） | Gao & Li (2025) [`10.1080/09544828.2025.2489647`](https://doi.org/10.1080/09544828.2025.2489647); Chen et al. (2026) [`10.1080/09544828.2026.2616583`](https://doi.org/10.1080/09544828.2026.2616583) | 工程设计主流 SCIE，已发 AIGC 包装与多智能体产品设计文 |
| **Systems** (`2079-8954`) | MDPI | **SSCI**（纯 OA，⚠️ **非 SCIE**） | Liu & Yang (2026) OPC [`10.3390/systems14080994`](https://doi.org/10.3390/systems14080994); Jung (2026) [`10.3390/systems14070771`](https://doi.org/10.3390/systems14070771) | 仅当未来把 OPC 系统工作流重新升为主线、且单位认可 SSCI 时考虑 |
| **Humanities and Social Sciences Communications** (`2662-9992`) | Springer Nature | **SSCI + AHCI**（纯 OA，⚠️ **非 SCIE**） | Zhao et al. (2026) [`10.1057/s41599-026-06948-z`](https://doi.org/10.1057/s41599-026-06948-z); Liu et al. (2026) [`10.1057/s41599-026-08755-y`](https://doi.org/10.1057/s41599-026-08755-y) | 仅当转为人文社科/乡村文化工作流叙事且单位认可 SSCI 时考虑 |

---

## 三、面向作者的选刊决策树（纯客观指标 A 套 + 纯 OA SCIE）

1. **单位不限制 MDPI / IEEE Access，追求审稿快 + 强对口**：
   - **第一顺位**：***IEEE Access***（包装 AI 方法与零人工设计审计直接对口，1–1.5 个月）或 ***Applied Sciences*** / ***Electronics***（1.5–2.5 个月，扩散/适配器/消融模板最熟）。
   - **第二顺位**：***Symmetry***（放大刀版折面几何对称性）或 ***Mathematics***（放大拓扑矩阵/色域投影数学形式化）。
2. **单位对 MDPI 或 IEEE Access 有限制，追求高校普遍认可度**：
   - **第一顺位**：***Scientific Reports***（Nature 旗下 SCIE 3 区，口碑稳、接收完整开源数据与纯客观定量消融的 AI 设计方法论文）。
   - **第二顺位**：***PLOS ONE***（SCIE 3 区，极度欢迎开源基准 `PackDieline-Syn` + 严谨消融与统计复现，对无人工被试零歧视）。
3. **坚决排除**：*Applied Mathematics and Nonlinear Sciences*、*Journal of Information Systems Engineering & Management*（均已被 Scopus 剔除、非 SCI）及 *Frontiers in Computer Science*（目前为 ESCI 非 SCIE）。
