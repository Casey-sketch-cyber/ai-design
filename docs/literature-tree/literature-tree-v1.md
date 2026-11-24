# 深度学习驱动的包装设计文献树（以乡村农产品为测试场景）

- 版本：v1（Gemini 起草，2026-11-24；已按作者修订方向及 PR #1 审查意见完成二轮修订）
- 状态：草稿（待 Claude 复核）
- 检索截至：2026-11-24。检索记录见 [`literature/search-log.md`](../../literature/search-log.md)（共 37 组检索与核查），完整文献清单见 [`literature/papers.csv`](../../literature/papers.csv)（共 107 篇：已核对官方摘要 102 篇 `verified=yes, summary_source=abstract`；未核对或仅见元数据 5 篇单独标 `verified=no`），公开数据集与版权分析专报见 [`docs/DATASETS.md`](../DATASETS.md)。

**核心问题结论**：按作者 2026-11-24 修订方向——以**深度学习等 AI 方法做包装设计**为核心贡献（乡村农产品作为主要测试场景，OPC / 方案交付降为可选应用或讨论框架），且**仅使用纯客观评价指标（方案 A，不招募专家与消费者）**——本方向**可以发表开放获取（OA）的 SCIE 期刊论文**，并且纯客观指标 + SOTA 基线对比 + 模块消融实验的论证形式较契合工程与计算机综合类 SCIE 期刊的审稿规范。不过，经补全 **3D 网格纹理生成（跨视角、跨接缝一致性）**脉络（TEXTure、Text2Tex、TexFusion、Point-UV Diffusion、Paint3D、SyncMVD、Meta 3D TextureGen、RoCoTex、MVPaint）后必须客观承认：**折叠纸盒本质上是简单 3D 网格，其 UV 展开图即刀版，因此跨视角无缝贴图本身属于情况 2（别处已有较成熟解法），而非情况 3 空白**。真正尚未被现有 3D 纹理同步方法覆盖、且能构成可辩护方法贡献的空白，集中在**工业印刷与包装硬约束**：① 3D 外视角不可见的**粘口、插舌与防尘翼**处理；② **3 mm 出血线与模切/压痕安全区**约束；③ 各折叠面上的**文字物理正向朝上**排布；④ **无光照烘焙的平面印刷稿（albedo-only）**；⑤ **CMYK 印刷色域约束**；⑥ **密集中文小字法定标签（GB 7718-2025 / GB 28050-2025）**。最务实的深度学习主方法应采用**免训练的推理期同步与印刷硬约束引导**（在冻结扩散主干上结合刀版 UV 潜在同步、不可见面/出血安全区掩码、分面正向对齐、中文双轨文字解耦与 CMYK 软色域引导），以乡村农产品多品类折叠盒为测试场景，在公开数据底座与自建评测基准上完成多基线（含直接运行在纸盒网格上的 TexFusion / SyncMVD / Paint3D / MVPaint）定量对比与消融实验。首选纯 OA SCIE 期刊为 ***IEEE Access***、***Applied Sciences***、***Scientific Reports***、***PLOS ONE***、***Electronics*** 与 ***Symmetry***。

---

## 怎么读

- **方法来源**：遵循彭思达的科研选题与领域视野方法（[learning_research](https://github.com/pengsida/learning_research)、[选题课件](https://pengsida.net/files/research_topic_selection.pdf)）与 `findInnovation` 模板：按时间线梳理技术演变，提炼“难点—解法树”，再用“四种情况”审慎界定创新空间。
- **脉络结构（四条，以脉络一为核心主线）**：
  - **时间线一（核心主线，52 篇已核对）**：深度学习用于包装、平面版式、中文文字渲染、品牌风格/主体控制，以及 **3D 网格跨视角/跨接缝纹理生成**（追踪 CVPR / ICCV / ECCV / SIGGRAPH / SIGGRAPH Asia / NeurIPS / ICLR / IJCAI / IEEE TMM 里程碑）。
  - **时间线二（测试场景背景，15 篇已核对）**：农产品包装与乡村振兴（阐明乡村农产品作为强约束测试场景的价值，以及低质期刊扎堆风险）。
  - **时间线三（可选讨论框架，13 篇已核对）**：OPC / 一人公司与人机协作设计（作为单卡轻量化推理的落地应用讨论背景）。
  - **时间线四（客观评测与公开数据，22 篇已核对 + 14 个公开数据集）**：零人类被试的纯客观指标体系（方案 A）与公开数据集（详表见 [`docs/DATASETS.md`](../DATASETS.md)）。
- **核对规则（严格执行）**：`literature/papers.csv` 中所有标 `verified=yes` 的 102 篇文献均已获取并核对官方页面或 OpenAlex / arXiv 官方摘要（`summary_source=abstract`），**不存在仅凭检索摘录标 `verified=yes` 的条目**；未能打开摘要原文的 5 篇（2 篇未定位原文的种子文献 + 3 篇付费墙阻挡摘要的 Elsevier 文献）均单独标为 `verified=no`，不作为核心方法依据。

---

## 初步假设（H1—H4）检验结论（归结至深度学习方法论文可行性）

| 假设 | 结论 | 核心依据与对“深度学习方法论文”的启示 |
| --- | --- | --- |
| **H1（交付与技术缺口）**：现有“AIGC + 包装设计”多停在出图与审美评价，很少处理刀版、印刷、中文文字、法规、条码与品牌一致性 | **成立（但需精确界定与 3D 纹理生成工作的边界）** | 现有包装应用 SCI/SSCI 论文（Gao & Li 2025 *J. Eng. Des.*; Song & Bai 2025 *Applied Sciences*; Liu et al. 2026 *IEEE Access*; Zhao et al. 2026 *HSSC*）普遍停留在单面概念图或提示词调用。另一方面，CV/图形学顶会已发展出较成熟的 **3D 网格跨视角无缝纹理同步方法**（TexFusion ICCV 2023、SyncMVD SIGGRAPH Asia 2024、MVPaint CVPR 2025）和单面电商海报中文大字生成（PosterMaker CVPR 2025）。**因此，深度学习方法论文能够成立，但贡献点不能泛泛地写成“解决多面无缝贴图”，而必须精准落在现有 3D 纹理方法未覆盖的印刷与包装硬约束上（粘口/插舌等不可见面、出血与安全区、分面文字正向朝上、无光照平面稿、CMYK 色域、背面密集中文小字法规排版）。** |
| **H2（组织形态：OPC）**：现有研究少以“一人 + 多智能体整单交付”为研究对象 | **部分成立（按作者决策降为可选讨论框架）** | 2026 年已有 *Systems* (`10.3390/systems14080994`)、IEEE *IT Professional* (`10.1109/mitp.2026.3694520`) 与 *SEJ* (`10.1002/sej.70067`) 正式使用 `One-Person Company (OPC)`。在深度学习方法论文中，OPC 适合放在 Discussion 中说明“单卡 RTX 5090 免训练推理如何降低小微设计与独立设计师的使用门槛”，不喧宾夺主。 |
| **H3（场景价值）**：乡村农产品预算低、品类碎、缺设计资源 | **成立（适合作为高约束测试场景）** | 乡村农产品（茶叶、杂粮、果干、蜂蜜等）多采用标准折叠纸盒、自立袋与罐贴，且受 **GB 7718-2025 & GB 28050-2025**（2027-03-16 正式强制实施）强中文标签规范约束，可为深度学习包装方法提供清晰、具挑战性的公开测试场景。 |
| **H4（低质风险）**：“AI + 农产品包装”已有不少低影响力论文，简单出图难进 SCI | **成立** | 7 篇种子文献中有 2 篇发在已被 Scopus 剔除的期刊（*AMNS* 2024、*JISEM* 2024），其中 Songyong Hu (2024) 仅用传统 HSV/灰度共生矩阵特征分析却冠名 Deep Learning。深度学习方法论文必须包含清晰的约束建模机制、强基线（含 TexFusion/SyncMVD/PosterMaker/ControlNet）定量对比与消融实验，避免沦为简单工具拼接。 |

---

## 长远目标与路线图

**长远目标**：发展面向工业制造与商业流通的**可印刷、强约束、高文字精度的深度学习包装生成方法与开源客观评测基准**，并以乡村农产品包装作为典型测试场景验证其在小微品牌中的实用价值。

| 里程碑任务 | 内容 | 现状 | 判断 |
| --- | --- | --- | --- |
| **M0：单面包装概念图生成** | 扩散模型 + LoRA/ControlNet 生成单面效果图 | **已做到** | 已做到，不作为主贡献 |
| **M1：单面电商海报排版与大标题中文渲染** | 主体避让版式 + 单面中英文标题准确渲染 | **较成熟**（CGL-GAN / PosterLayout / AnyText2 / PosterMaker CVPR 2025 / PosterLLaVA IEEE TMM 2026） | 已有强基线 |
| **M2：3D 网格跨视角无缝纹理合成** | 在共享潜空间或纹理域同步多视角去噪，消除一般 3D 网格接缝 | **较成熟**（Text2Tex / TexFusion ICCV 2023 / Point-UV ICCV 2023 / Paint3D CVPR 2024 / SyncMVD SIGGRAPH Asia 2024 / MVPaint CVPR 2025） | 别处已有成熟解法（**必须纳入基线对比**） |
| **M3：印刷硬约束下的刀版包装生成与中文双轨解耦** | 在刀版 UV 展开图上同时满足：不可见面（粘口/插舌）留白、出血与安全区、分面文字正向朝上、无光照平面印刷稿、CMYK 色域约束、以及正面艺术字 + 背面密集中文法定标签双轨合成 | **未见系统解法**（现有 3D 纹理同步方法与 2D 海报方法均未联合处理上述印刷与法规约束） | **值得做（情况 2，核心方法切入点）** |
| **M4：零人工纯客观包装生成评测基准** | 基于高许可公开数据集与程序化合成标准折叠盒型构建开源评测集，以纯客观几何/OCR/印前/偏好代理指标做基线与消融 | **空白** | **值得做（与 M3 同步作为论文核心产出）** |

---

## 领域视野六问

| 问题 | 回答 | 依据 |
| --- | --- | --- |
| **里程碑论文有哪些** | ① 扩散与控制底座：LDM (CVPR 2022)、ControlNet (ICCV 2023)、SD3 (ICML 2024)、Diffusion-4K (CVPR 2025)；② 平面版式与电商海报：Content-aware Layout (TOG 2019)、CanvasVAE (ICCV 2021)、CGL-GAN (IJCAI 2022)、PosterLayout (CVPR 2023)、LayoutDM (CVPR 2023)、LayoutGPT (NeurIPS 2023)、COLE (2023)、LayerDiffusion (SIGGRAPH 2024)、PosterMaker (CVPR 2025)、CreatiPoster (2025)、PosterLLaVA (IEEE TMM 2026)；③ 中文视觉文字：TextDiffuser (NeurIPS 2023)、GlyphControl (NeurIPS 2023)、AnyText (ICLR 2024)、Glyph-ByT5 (ECCV 2024)、UDiffText (ECCV 2024)、AnyText2 (2024)、FonTS (ICCV 2025)；④ **3D 网格跨视角无缝纹理**：TEXTure (SIGGRAPH 2023)、Text2Tex (ICCV 2023)、TexFusion (ICCV 2023)、Point-UV Diffusion (ICCV 2023)、Paint3D (CVPR 2024)、SyncMVD (SIGGRAPH Asia 2024)、Meta 3D TextureGen (2024)、RoCoTex (2024)、MV-Adapter (ICCV 2025)、MVPaint (CVPR 2025)；⑤ 主体/风格一致：LoRA (2021)、DreamBooth (CVPR 2023)、IP-Adapter (2023)、InstantStyle (2024)、AnyDoor (CVPR 2024)；⑥ 客观指标：FID (2017)、LPIPS (2018)、CLIPScore (2021)、ImageReward/PickScore (2023)、HPS v2 (2023)、VQAScore (ECCV 2024)、DesignCritic (*IEEE Access* 2026)。 | 详见下文时间线一、四及 `literature/papers.csv` |
| **技术如何演变** | ① 2019–2022：GAN/VAE 二维版式框生成；② 2022–2023：扩散底座 + ControlNet/IP-Adapter，同时 3D 纹理从逐视角修补（TEXTure、Text2Tex）迈向**共享潜空间纹理同步去噪（TexFusion）**与点云-UV 粗细扩散（Point-UV）；③ 2024–2025：多视角同步去噪趋于成熟（SyncMVD、Paint3D 无光照 UV、RoCoTex、MVPaint、MV-Adapter），同时平面界完成多语种文字渲染（AnyText2、Glyph-ByT5-v2、PosterMaker）与透明分层/SVG 代码化（LayerDiffusion、StarVector）；④ 2025–2026：开始探索面向垂直制造与设计原则的约束引导、奖励对齐与自动审计（ARIA-Diff、DesignCritic）。 | 时间线一 52 篇顶会顶刊演进轨迹 |
| **领域的终极目标** | 输入产品类别、参数化刀版、品牌视觉资产与法定标签文本，生成兼具高视觉质量、多面折合无缝、文字零差错且可直接用于 CMYK 制版印刷的包装文件。 | 工业包装流程与生成式 AI 交叉目标 |
| **目前达到什么水平** | 在**单面电商海报中文大字**（PosterMaker、AnyText2）与**通用 3D 网格跨视角无缝贴图**（TexFusion、SyncMVD、Paint3D、MVPaint）上均已有较成熟的开源方案；但在**面向印刷制版的刀版硬约束（粘口/插舌不可见面、出血线、安全区、分面文字正向、CMYK 色域）与背面密集中文法规排版**的联合满足上，现有方法仍存在明显失效。 | 详见“难点—解法树” |
| **哪些重要问题未解决** | 五类包装专用技术难点：① 刀版印刷硬约束与 3D 纹理同步的衔接（不可见面、出血/安全区、分面文字朝向、无光照平稿）；② 密集中文小字法定标签与艺术标题的混合零错生成；③ 跨 SKU 品牌 Logo 与主色板一致性保持；④ 产品类别先验与地域文化风格的解耦控制；⑤ sRGB 视觉生成到 300 DPI CMYK 可印刷性与可编辑分层的转换。 | 详见“难点—解法树” |
| **当前热点** | 同步多视角 3D 纹理精修（SyncMVD / MVPaint）、商品海报图文解耦（PosterMaker / CreatiPoster / PosterLLaVA）、可印刷高分辨率与多图层生成（Diffusion-4K / LayerDiffusion）、以及无人工设计原则审计与奖励对齐（VQAScore / DesignCritic / ARIA-Diff）。 | 2024–2026 顶会与 SCIE 检索聚类 |

---

## 时间线一（核心主线）：深度学习用于包装、平面版式、文字渲染与 3D 网格纹理生成

近七年，视觉与图形学界分别在“2D 版式与中文文字渲染”和“3D 网格跨视角无缝纹理生成”两条支线上取得显著进展，但**尚未将两者与工业包装印刷硬约束（不可见面、出血安全区、分面正向排版、CMYK 色域、法定中文小字）打通**。

### 子线 1A：平面版式、中文文字渲染、品牌一致与多图层生成

| 年份 | 工作（带链接） | 里程碑 / 后续 | 解决了什么 | 留下什么（面向包装的空白） |
| --- | --- | --- | --- | --- |
| 2019 | Zheng et al., *Content-aware Graphic Layout* ([ACM TOG](https://doi.org/10.1145/3306346.3322971)) | 里程碑 | 内容语义驱动的深度杂志版式合成 | 仅单面二维矩形框 |
| 2021 | Li et al., *LayoutGAN* ([IEEE TPAMI](https://doi.org/10.1109/tpami.2019.2963663)) & Yamaguchi, *CanvasVAE* ([ICCV](https://doi.org/10.1109/iccv48922.2021.00543)) | 里程碑 | 矢量线框版式对抗训练；多图层矢量文档建模与 **Crello** 数据集 | 未耦合现代高保真扩散底座 |
| 2021–22 | Hu et al., *LoRA* ([arXiv/ICLR](https://doi.org/10.48550/arxiv.2106.09685)); Rombach et al., *LDM* ([CVPR 2022](https://doi.org/10.1109/cvpr52688.2022.01042)); Zhou et al., *CGL-GAN* ([IJCAI 2022](https://doi.org/10.24963/ijcai.2022/692)) | 里程碑 | 低秩轻量微调、潜在扩散底座、电商海报主体避让版式与 6 万级 **CGL** 数据集 | 仅 2D 单面构图，无文字像素与包装折面 |
| 2023 | Zhang et al., *ControlNet* ([ICCV](https://doi.org/10.1109/iccv51070.2023.00355)); Ruiz et al., *DreamBooth* ([CVPR](https://doi.org/10.1109/cvpr52729.2023.02155)); Ye et al., *IP-Adapter* ([arXiv](https://doi.org/10.48550/arxiv.2308.06721)) | 里程碑 | 线稿/分割空间控制、主体定制微调与免微调图像提示适配器 | 单独使用难以防止风格—内容语义泄露或跨折线断裂 |
| 2023 | Hsu et al., *PosterLayout / DS-GAN* ([CVPR](https://doi.org/10.1109/cvpr52729.2023.00583)); Inoue et al., *LayoutDM* ([CVPR](https://doi.org/10.1109/cvpr52729.2023.00980)); Feng et al., *LayoutGPT* ([NeurIPS](https://doi.org/10.52202/075280-0802)); Li et al., *RADM* ([CIKM](https://doi.org/10.1145/3583780.3615028)) | 里程碑 | **PKU PosterLayout** 基准与客观版式指标；离散扩散/LLM/关系感知可控排版 | 面向单面促销海报，不含包装刀版安全区与多面对应 |
| 2023 | Chen et al., *TextDiffuser* ([NeurIPS](https://doi.org/10.52202/075280-0410)); Yang et al., *GlyphControl* ([NeurIPS](https://doi.org/10.52202/075280-1912)); Jia et al., *COLE* ([arXiv](https://doi.org/10.48550/arxiv.2311.16974)); Jain et al., *VectorFusion* ([CVPR](https://doi.org/10.1109/cvpr52729.2023.00190)) | 里程碑 | 字符掩码/字形控制渲染（+ **MARIO-10M**）；层次化多图层可编辑生成；文本到 SVG 蒸馏 | 以英文短句为主，复杂中文密集排版与印刷矢量精度不足 |
| 2024 | Esser et al., *SD3 / Rectified Flow* ([ICML](https://doi.org/10.48550/arxiv.2403.03206)); Zhang & Agrawala, *LayerDiffusion* ([ACM TOG](https://doi.org/10.1145/3658150)); Wang et al., *InstantStyle* ([arXiv](https://doi.org/10.48550/arxiv.2404.02733)); Chen et al., *AnyDoor* ([CVPR](https://doi.org/10.1109/cvpr52733.2024.00630)) | 里程碑 | MMDiT 底座、潜在透明 Alpha 多图层生成、风格—内容注意力解耦、DINOv2 零样本对象植入 | 仍局限于 sRGB 屏幕域单面图像 |
| 2024 | Tuo et al., *AnyText* ([ICLR](https://doi.org/10.48550/arxiv.2311.03054)) & *AnyText2* ([arXiv](https://doi.org/10.48550/arxiv.2411.15245)); Liu et al., *Glyph-ByT5* ([ECCV](https://doi.org/10.1007/978-3-031-73226-3_21) / [arXiv](https://arxiv.org/abs/2403.09622)) & *v2* ([arXiv](https://doi.org/10.48550/arxiv.2406.10208)); Chen et al., *TextDiffuser-2* ([ECCV](https://doi.org/10.1007/978-3-031-72652-1_23)); Zhao & Lian, *UDiffText* ([ECCV](https://doi.org/10.1007/978-3-031-72751-1_13)) | 里程碑 | 中文等多语种视觉文字高精度渲染与逐行字体/颜色定制（+ **AnyWord-3M**） | 适合大标题艺术字，对背面 100–300 字小号中文法规文本仍易出现笔画粘连，且非矢量单黑印刷 |
| 2025 | Gao et al., *PosterMaker* ([CVPR](https://doi.org/10.1109/cvpr52734.2025.00757)); Rodriguez et al., *StarVector* ([CVPR](https://doi.org/10.1109/cvpr52734.2025.01508)); Zhang et al., *Diffusion-4K* ([CVPR](https://doi.org/10.1109/cvpr52734.2025.02185)); Shi et al., *FonTS* ([ICCV](https://doi.org/10.1109/iccv51701.2025.01716)); Hong et al., *CreatiPoster* ([arXiv](https://doi.org/10.48550/arxiv.2506.10890)) | 里程碑 | 商品海报文字与场景解耦（中文高精度）、MLLM 原生 SVG 代码生成、原生 4K 小波微调、RGBA 多图层协议 | 均面向 2D 数字海报或通用矢量图标，未处理折叠包装多面刀版与印刷色域 |
| 2026 | Yang et al., *PosterLLaVA* ([IEEE TMM](https://doi.org/10.1109/tmm.2026.3753095)); Cao & Chen, *ARIA-Diff* ([Mathematics](https://doi.org/10.3390/math14203759)); Liu et al. ([IEEE Access](https://doi.org/10.1109/access.2026.3723820)) | 里程碑 / 包装对照 | MLLM 输出 JSON 版式并导出 SVG；少步扩散奖励对齐；饼干包装人机协同案例 | 现有包装 SCI 应用文仍显著滞后于上述视觉生成架构 |

### 子线 1B：3D 网格跨视角与跨接缝一致性纹理生成（关键补全脉络）

> **为什么这条脉络至关重要**：折叠纸盒本质上是一个由若干平面面片组成的简单 3D 网格，其 UV 展开图即工业刀版。近年来图形学与 CV 顶会已发展出较成熟的 3D 网格无缝纹理同步方法，**任何面向“2D 刀版 ↔ 3D 包装一致生成”的研究都必须正面继承并与之定量对比**。

| 年份 | 工作（带链接） | 里程碑 / 后续 | 解决了什么（跨视角/跨接缝核心机制） | 留下什么（为何不能直接套用作包装印刷稿） |
| --- | --- | --- | --- | --- |
| 2023 | Richardson et al., *TEXTure* ([ACM SIGGRAPH](https://doi.org/10.1145/3588432.3591503)) | 里程碑 | 基于深度条件扩散迭代绘制并修补 3D 网格多视角纹理 | 顺序投影修补易累积视角不一致、棱边接缝与阴影烘焙 |
| 2023 | Chen et al., *Text2Tex* ([ICCV](https://doi.org/10.1109/iccv51070.2023.01701)) | 里程碑 | 引入动态生成状态掩码（新建/更新/保持）指导多视角深度修补，减少视角拉伸 | 仍属异步逐视角绘制，在锐利盒棱处难以保证全局图案连贯，且无法控制文字正向 |
| 2023 | Cao et al., *TexFusion* ([ICCV](https://doi.org/10.1109/iccv51070.2023.00385) / [arXiv](https://arxiv.org/abs/2310.13772)) | **核心里程碑** | **在共享的潜空间纹理图（latent texture map）上交错同步多视角 2D 去噪采样**，从机理上显著缓解跨视角接缝与不一致 | 面向通用 3D 渲染网格：① 忽略粘口/插舌等 3D 不可见面；② 无 3 mm 出血与模切安全区；③ 不控制各折面文字物理朝上与密集中文排版；④ 易残留视图光照倾向与 sRGB 超色域 |
| 2023 | Yu et al., *Point-UV Diffusion* ([ICCV](https://doi.org/10.1109/iccv51070.2023.00388) / [arXiv](https://arxiv.org/abs/2308.10490)) | 里程碑 | 粗阶段 3D 点云扩散保证全局一致 + 细阶段 UV 空间扩散提升高频细节 | 依赖点云/网格类特定训练分布，未适配印刷刀版布局与图文法规约束 |
| 2024 | Zeng et al., *Paint3D* ([CVPR](https://doi.org/10.1109/cvpr52733.2024.00407)) | 里程碑 | 粗细两阶段去光照 UV 扩散（UVHD）消除 3D 贴图中的光照烘焙与接缝 | 解决“无光照”痛点迈出关键一步，但未处理包装刀版不可见面、出血线、文字朝向、CMYK 色域与密集中文排版 |
| 2024 | Liu et al., *SyncMVD* ([SIGGRAPH Asia](https://doi.org/10.1145/3680528.3687621) / [arXiv](https://arxiv.org/abs/2311.12891)) | **核心里程碑** | **在去噪早期于共享纹理域同步所有渲染视角的扩散轨迹**（免训练），以较低成本获得全局一致的无缝 3D 纹理 | 与 TexFusion 同理：只优化 3D 可见表面的视觉连贯性，缺乏对包装粘口/插舌、出血/安全区、每面文字正向、CMYK 色域及法定中文文本的约束建模 |
| 2024 | Bensadoun et al., *Meta 3D TextureGen* ([arXiv](https://arxiv.org/abs/2407.02430)) & Kim et al., *RoCoTex* ([arXiv](https://arxiv.org/abs/2409.19989)) | 后续 | 前馈两阶段快速一致 3D 纹理生成；基于 SDXL + 多 ControlNet + 对称视角与区域提示的免训练一致纹理合成 | 面向游戏/影视 3D 资产展示，未考虑平面制版印刷规范与中文包装信息层级 |
| 2025 | Huang et al., *MV-Adapter* ([ICCV](https://doi.org/10.1109/iccv51701.2025.01520)) & Cheng et al., *MVPaint* ([CVPR](https://doi.org/10.1109/cvpr52734.2025.00063) / [arXiv](https://arxiv.org/abs/2411.02336)) | 里程碑 | 即插即用并行多视角注意力适配器；同步多视角生成 + 空间感知 3D 修补（补盲区）+ 接缝平滑 UV 精修 | MVPaint 的空间修补仅针对普通 3D 遮挡区，并不理解包装粘口必须留白免油墨、出血区需外扩延展等工业刀版语义 |

**本线技术范式小结**：如果只看“给纸盒 3D 网格贴上跨棱边无缝的纹理”，TexFusion (ICCV 2023)、Paint3D (CVPR 2024)、SyncMVD (SIGGRAPH Asia 2024) 与 MVPaint (CVPR 2025) 已经提供了成熟的共享潜空间/纹理域同步范式。**但把 3D 资产纹理直接当成包装印刷稿会立即暴露六大结构性缺口**（不可见面粘口/插舌、出血与安全区、分面文字物理朝上、无光照平面印刷稿、CMYK 色域、密集中文小字法规排版）。这决定了我们的研究定位属于**情况 2**（将成熟的多视角潜空间同步机制迁移并改造至带工业印刷硬约束的新场景），技术贡献必须明确落在上述六大缺口上。

---

## 时间线二（测试场景背景）：农产品包装设计与乡村振兴

本脉络提供具有代表性的高约束测试场景与行业参照（仅列出已核对摘要的文献；3 篇未能打开摘要原文的 Elsevier 文献及 2 篇未定位原文的种子文献均已在 `papers.csv` 标 `verified=no`）：

| 年份 | 工作（带链接） | 角色 | 核心结论与对测试场景的约束输入 |
| --- | --- | --- | --- |
| 2007–2009 | Silayoi & Speece ([EJM](https://doi.org/10.1108/03090560710821279)); Rundh ([Br. Food J.](https://doi.org/10.1108/00070700910992880)) | 领域奠基 | 包装同时承担“视觉传播”与“信息/物流结构”双职能，说明生成模型须兼顾审美与结构/信息合规 |
| 2020–2024 | Boz et al. ([Sustainability 2020](https://doi.org/10.3390/su12062192)); Tanaiutchawoot et al. ([PTS 2023](https://doi.org/10.1002/pts.2755)); Duan et al. ([Sustainability 2024](https://doi.org/10.3390/su16041381)); Giri & Chaulagai ([Cogent Food Agric. 2024](https://doi.org/10.1080/23311932.2024.2399793)) | 农业与可持续背景 | 农产品包装正向生态纸材与规范信息演进，对底材色域与清晰标签提出明确约束 |
| 2024 | Chen Tao et al. ([JISEM](https://doi.org/10.55267/iadt.07.15214)); Songyong Hu ([AMNS](https://doi.org/10.2478/amns-2024-1882)) | 种子2、3（弱来源） | 所在期刊已被 Scopus 剔除；Hu (2024) 仅做浅层 HSV/GLCM 特征，作反面警示 |
| 2025 | Song & Bai ([Applied Sciences](https://doi.org/10.3390/app15116033)); Chen & Fu ([AHFE](https://doi.org/10.54941/ahfe1006930)); Gao & Li ([J. Eng. Des.](https://doi.org/10.1080/09544828.2025.2489647)); Tang et al. ([JTAER](https://doi.org/10.3390/jtaer20030248)) | 种子4、5 + 同类参照 | 现有 AIGC 包装 SCI 论文均采用“评论/问卷 + AHP + 商用生图 + TOPSIS”，依赖人工评价且未解决刀版与印前硬约束 |
| 2026 | Zhao et al. ([HSSC](https://doi.org/10.1057/s41599-026-06948-z)); Liu et al. ([HSSC](https://doi.org/10.1057/s41599-026-08755-y)); Wu et al. ([ACSR](https://doi.org/10.2991/978-94-6239-737-8_27)); 李雪娜等 ([2026](https://doi.org/10.64216/3104-9664.26.04.014)) | 种子7 + 新国标背景 | 注：上述两篇 2026 *HSSC* 论文均已核实官方 Nature URL 及 OpenAlex 完整摘要（马奶酒包装与乡村工艺工作流），均偏向人文/文化转译而非深度学习算法架构；**GB 7718-2025 & GB 28050-2025**（2027-03-16 正式强制实施）为测试场景提供了明确的客观中文标签合规基准 |

---

## 时间线三（可选讨论框架）：OPC 一人公司与人机协同设计

> 按作者决策，本脉络仅作为论文引言/讨论部分的可选应用背景（论证轻量化免训练推理如何赋能单人设计实践），不作为核心算法主线。

| 年份 | 代表工作（带链接） | 核心洞见（均已核对官方摘要） |
| --- | --- | --- |
| 2023–2025 | Noy & Zhang ([Science 2023](https://doi.org/10.1126/science.adh2586)); Choi et al., *CreativeConnect* ([CHI 2024](https://doi.org/10.1145/3613904.3642794)); Zhou et al., *OptiMuse* ([CHI 2024](https://doi.org/10.1145/3613904.3642812)); Chen et al., *CoExploreDS* ([CHI 2025](https://doi.org/10.1145/3706598.3713869)); Rezazadeh & Bonehgazy ([arXiv:2511.09533](https://doi.org/10.48550/arxiv.2511.09533)) | 生成式 AI 提升个体创作效率，但专业设计是非线性、多约束过程 |
| 2026 | Liu & Yang ([Systems](https://doi.org/10.3390/systems14080994)); Kshetri ([IEEE IT Prof.](https://doi.org/10.1109/mitp.2026.3694520)); Li, Zeng, & Li ([SEJ](https://doi.org/10.1002/sej.70067)); Dell'Acqua et al. ([Org. Sci.](https://doi.org/10.1287/orsc.2025.21838)); Chen et al. ([J. Eng. Des.](https://doi.org/10.1080/09544828.2026.2616583)) | 2026 年同行评审期刊已正式确立 **`One-Person Company (OPC)`**（`1+N+AI`），并指出在“锯齿状技术前沿”之外（如硬工程/合规任务）需要确定性护栏与自动校验 |

---

## 时间线四（核心评测与公开数据）：纯客观指标体系（方案 A）与数据集

> 按作者决策，**不招募专家或消费者，全部评测严格采用方案 A（纯客观指标 + 多基线定量对比 + 消融实验）**。

| 客观指标维度 | 代表指标与出处（带链接，均已核对官方摘要） | 测量对象 |
| --- | --- | --- |
| **1. 分布与高频画质** | **FID** ([Heusel et al., NeurIPS 2017](https://doi.org/10.48550/arxiv.1706.08500))、**LPIPS** ([Zhang et al., CVPR 2018](https://doi.org/10.1109/cvpr.2018.00068))、**GLCM 细节度** ([Diffusion-4K, CVPR 2025](https://doi.org/10.1109/cvpr52734.2025.02185)) | 与公开真实包装库的分布距离、感知多样性及高分辨率细节 |
| **2. 无人工偏好代理** | **HPS v2** ([Wu et al., 2023](https://doi.org/10.48550/arxiv.2306.09341))、**ImageReward** ([Xu et al., NeurIPS 2023](https://doi.org/10.52202/075280-0700))、**PickScore** ([Kirstain et al., NeurIPS 2023](https://doi.org/10.52202/075280-1594))、**DesignCritic** ([Wu & Song, IEEE Access 2026](https://doi.org/10.1109/access.2026.3730226)) | 预训练人类偏好奖励模型自动打分与校准后的设计原则审计 |
| **3. 组合语义对齐** | **CLIPScore** ([Hessel et al., EMNLP 2021](https://doi.org/10.18653/v1/2021.emnlp-main.595))、**VQAScore** ([Lin et al., ECCV 2024](https://doi.org/10.1007/978-3-031-72673-6_20) / [arXiv](https://arxiv.org/abs/2404.01291)) | 衡量生成图对“产品品类 + 地域文化风格 + 指定元素”复合提示的忠实度 |
| **4. 中文文字渲染精度** | **OCR Sentence Acc / Word Acc / CER / `1 - NED`**（PaddleOCR，遵循 [AnyText-benchmark, ICLR 2024](https://doi.org/10.48550/arxiv.2311.03054) 协议） | 定量评估正面艺术大字与背面密集中文标签的字符准确度 |
| **5. 版式与刀版几何** | **mIoU / Alignment / Overlap / Saliency Occlusion (`R_occ`)**（[CGL-GAN IJCAI 2022](https://doi.org/10.24963/ijcai.2022/692); [PosterLayout CVPR 2023](https://doi.org/10.1109/cvpr52729.2023.00583)）+ **跨折线缝隙 Seam PSNR/LPIPS** + **3D 折叠多视角重投影 SSIM** + **不可见面污染率（Glue-Flap Contamination %）** + **3 mm 出血/安全区越界率** + **分面文字朝向正确率** | 定量检验相较于通用 3D 纹理同步方法（TexFusion / SyncMVD / MVPaint）在包装刀版印刷硬约束上的真实改进 |
| **6. 品牌与印前/法规硬指标** | **DINOv2 / CLIP-I 跨 SKU 相似度**、**色板 CIEDE2000 ($\Delta E_{00}$)**、**CMYK 色域溢出率（Out-of-Gamut %）**、**条码 `pyzbar` 解码率**、**GB 7718-2025/28050-2025 确定性规则通过率** | 呼应 [Jung (*Sustainability* 2026)](https://doi.org/10.3390/su18062943) 的“视觉层 vs. 可操作逻辑层”双层客观评测思路 |

> 14 个免费公开数据集详表与自采版权分析见专报 [`docs/DATASETS.md`](../DATASETS.md)。

---

## 按里程碑任务归类的代表性 pipeline

| 里程碑任务 | 代表性深度网络 Pipeline | 代表论文 |
| --- | --- | --- |
| **单面商品海报与中文文字生成** | `产品图 + 版式规划(MLLM/LayoutDM) → SceneGenNet + TextRenderNet / AnyText2 → 单面 RGB 海报` | PosterMaker (CVPR 2025); AnyText2 (2024); PosterLLaVA (IEEE TMM 2026) |
| **3D 网格跨视角无缝纹理生成** | `3D 网格多视角渲染(深度/法线) → 共享潜空间/纹理域同步多视角去噪(TexFusion / SyncMVD) → 去光照精修(Paint3D) / 遮挡修补与接缝平滑(MVPaint) → 3D 渲染 UV 贴图` | TexFusion (ICCV 2023); Paint3D (CVPR 2024); SyncMVD (SIGGRAPH Asia 2024); MVPaint (CVPR 2025) |
| **品牌风格/主体控制与透明分层** | `冻结扩散底座 + IP-Adapter/InstantStyle/AnyDoor + LayerDiffusion → 可分离风格化前景与背景` | IP-Adapter (2023); InstantStyle (2024); AnyDoor (CVPR 2024); LayerDiffusion (SIGGRAPH 2024) |
| **本课题目标空白（印刷硬约束刀版生成）** | `参数化刀版 UV 网格(可见面/粘口插舌不可见面/3 mm 出血与安全区/每面正向旋转角) → 免训练同步潜空间 UV 去噪(去光照+不可见面留白+出血延展+分面正向旋转对齐) + 可微 CMYK 软色域引导 + 双轨中文文字(艺术大字扩散 + 背面矢量法规层) → 可印刷 2D 刀版稿 & 3D 折叠多视图` | **现有 3D 纹理同步方法与 2D 海报方法均未联合覆盖的交叉空白** |

---

## 难点—解法树（深度学习做包装设计的五大具体技术难点）

| 具体技术难点 | 现有代表性方法 | 来自 | 在包装设计上的具体空白 | 别的数据域里有没有同内核的解法 |
| --- | --- | --- | --- | --- |
| **T1：包装上中文文字渲染错误** | AnyText (ICLR 2024)、AnyText2 (2024)、Glyph-ByT5-v2 (2024)、UDiffText (ECCV 2024)、PosterMaker (CVPR 2025)、FonTS (ICCV 2025) | CV/AI 顶会 | 现有方法面向单面大标题；包装背面需渲染 **100–300 字小号中文配料表与 1+6 营养表**，像素扩散难以满足零错字与单黑矢量印刷，需与正面艺术大字分轨协同处理 | 有：网页代码生成（Design2Code NAACL 2025）与多层协议（CreatiPoster 2025）通过结构化矢量/代码层规避密集小字像素畸变 |
| **T2：品牌元素与跨 SKU 一致性** | LoRA (2021)、DreamBooth (CVPR 2023)、IP-Adapter (2023)、InstantStyle (2024)、AnyDoor (CVPR 2024) | CV 顶会 | 多口味/多盒型系列生成时仍易出现 Logo 畸变、主色板 $\Delta E_{00}$ 偏移与参考图构图泄露 | 有：跨帧一致自注意力与层级风格解耦（InstantStyle） |
| **T3：2D 刀版展开图与 3D 折叠多面对应及印刷硬约束** | **TexFusion (ICCV 2023)、SyncMVD (SIGGRAPH Asia 2024)、Paint3D (CVPR 2024)、Point-UV (ICCV 2023)、RoCoTex (2024)、MVPaint (CVPR 2025)、MV-Adapter (ICCV 2025)** | CV/图形学顶会 | **通用跨视角无缝贴图在 3D 纹理领域已有较成熟方案**；但直接套用到纸盒网格上仍无法解决六大印刷硬约束：**① 粘口/插舌/防尘翼等 3D 不可见面的留白或上胶免油墨规则；② 3 mm 出血线外扩与模切/压痕安全区保护；③ 各折面折叠后文字物理朝上的旋转角约束；④ 严格无光照平面印刷稿；⑤ CMYK 印刷色域；⑥ 背面密集中文小字** | **有成熟同内核解法**：3D 网格纹理同步去噪（TexFusion / SyncMVD / MVPaint）已解决共享 UV 潜空间同步的核心机理，本课题属于将其改造迁移至印刷硬约束包装场景（**情况 2**） |
| **T4：按产品类别与地域文化风格解耦控制** | CGL-GAN (IJCAI 2022)、PosterLayout (CVPR 2023)、SD3 (ICML 2024)、InstantStyle (2024)、ARIA-Diff (*Mathematics* 2026) | CV 顶会 & SCIE | 农产品品类结构先验与地域文化纹样在提示词中易相互干扰，尚缺面向包装类别与风格的正交控制机制 | 有：属性解耦风格注入（InstantStyle）与受约束奖励对齐（ARIA-Diff） |
| **T5：生成结果的可印刷性与可编辑性** | LayerDiffusion (SIGGRAPH 2024)、Diffusion-4K (CVPR 2025)、StarVector (CVPR 2025)、PosterLLaVA (IEEE TMM 2026) | CV/图形学顶会 | 现有模型在 sRGB 屏幕域优化，缺少可微 CMYK 色域引导、300 DPI 高频细节保真与刀版分层矢量导出 | 有：可微色彩空间投影（sRGB→CMYK soft-gamut penalty）与双层客观评测（Jung 2026 *Sustainability*） |

---

## 候选课题（深度学习方法 + 纯客观指标 A 套）

| 候选 | 设定（输入 → 输出） | 要解决的具体问题或 failure case | 来源 |
| --- | --- | --- | --- |
| **DL-1（免训练优先）：面向印刷硬约束的刀版同步包装扩散框架（PrintDieLine-Sync / DieLineDiff）** | 输入：参数化 2D 刀版网格与功能区掩码（可见面、粘口/插舌不可见面、3 mm 出血、安全区、分面正向旋转角）+ 品类/文化文本 + 中文标题与法定标签 JSON → 输出：无光照 2D 可印刷 CMYK 刀版稿 + 几何一致 3D 折叠多视角图 | 针对直接把 TexFusion / SyncMVD / Paint3D 跑在纸盒网格上时产生的六大印刷失效（粘口污染、出血截断、侧面文字倒置、光照烘焙、CMYK 溢色、中文小字乱码），提出免训练推理期同步与约束引导框架 | 难点—解法树 + failure case（面向 3D 纹理同步的新约束迁移） |
| **DL-2：双轨图文解耦与印前引导的中文包装生成框架（PackLayText）** | 输入：包装品类与地域风格条件 + 中文艺术标题 + 结构化法定标签 JSON + 品牌色板 → 输出：高准艺术大字 + 零错矢量法规层 + 满足显著性避让与 CMYK 色域约束的分层包装 | T1（中文密集小字错乱）+ T4（类别与风格解耦）+ T5（CMYK 色域与版式重叠） | failure case + 工程问题转科学问题 |
| **DL-3：跨 SKU 品牌一致与品类-文化解耦适配器（BrandPack-Adapter）** | 输入：1 组品牌资产（Logo + 主色板 + 文化参考）+ 多规格/品类条件 → 输出：跨 SKU 色偏低、标识保持稳定的系列包装 | T2（跨 SKU 漂移与语义泄露）+ T4（品类—文化解耦） | 用户需求 + failure case |
| **DL-4：基于真实数据/多目标客观奖励的包装扩散对齐微调（PrintAlign-Diff）** | 输入：冻结扩散模型 + 真实公开包装分布（Open Food Facts / RPC）与客观印刷/OCR 奖励 → 输出：对齐后的包装专用轻量权重 | T1 + T5（减少 CMYK 溢色与排版违规） | 新锤子（ARIA-Diff / 扩散对齐） |
| **DL-5：传统感性工学 + 商用 Midjourney + 问卷/AHP** | 输入：调研问卷 + 提示词 → 输出：单面概念效果图 | 无深度学习方法创新且依赖人工评价 | 淘汰对照组 |

---

## 按四种情况判断

四种情况的定义（引自彭思达《科研选题》课件，2025-08-12，[PDF](https://pengsida.net/files/research_topic_selection.pdf)）：
1. “同样输入输出的任务，已经有了不错的解决方案，只是有些地方做得还不够好。”
2. “输入或输出有所变化的任务，已经有了不错的解决方案。或者在多个完全不同 data domain、技术问题内核相同的任务中，都有了不错的类似解决方案。”
3. “只有少数一两个完全不同 data domain、技术问题内核相同的任务，有不错的解决方案。”
4. “在各个不同领域的任务中，虽然有类似的技术问题，但都没有较好的解决方案。”

> 注：彭思达课件只指出“不同问题的创新空间区别很大”，并未对四种情况排序；下表判断由本研究基于补全后的 3D 纹理生成脉络与约束条件独立作出。

| 候选 | 属于哪种情况 | 理由 | 与已有工作的区别类型 | 初步结论 |
| --- | --- | --- | --- | --- |
| **DL-1：印刷硬约束刀版同步包装扩散框架（DieLineDiff，免训练优先）** | **情况 2**（已从情况 3 审慎回调） | 跨视角、跨接缝 3D 网格纹理同步（TexFusion、SyncMVD、Paint3D、MVPaint）在图形学领域已有较成熟方案，折叠纸盒 UV 本质上也是 3D 网格贴图；本课题属于**将成熟的多视角同步技术迁移到带有印刷硬约束的新场景**，核心贡献在于系统揭示并解决现有 3D 纹理同步方法在粘口/插舌不可见面、出血线与安全区、分面文字物理朝上、无光照平稿、CMYK 色域及密集中文小字上的结构性失效，且无需人工评价即可用客观指标清晰度量 | **克服现有方案在新约束下的局限 + 满足工业印刷新需求** | **优先（核心推荐，主方法优先采用免训练推理同步与引导）** |
| **DL-2：双轨图文解耦与印前引导中文包装框架（PackLayText）** | **情况 2** | 单面商品海报图文解耦在 PosterMaker (CVPR 2025) 和 CreatiPoster (2025) 已有强先例，迁移至中文预包装食品并引入密集中文法规矢量层保护与 CMYK 软色域引导，新约束清晰且所有指标均可自动评测 | **克服现有方案局限 + 满足新场景硬约束** | **优先（可与 DL-1 融为一体作为其图文与印前约束模块）** |
| **DL-3：跨 SKU 品牌一致与品类-文化解耦适配器** | **情况 2** | 风格/主体适配器（IP-Adapter / InstantStyle / AnyDoor）较成熟，扩展至跨 SKU 品牌色板 $\Delta E_{00}$ 与标识保持，适合作为增强分支 | **改善跨 SKU 可控性与一致性** | **备选（可作 DL-1 的可选扩展）** |
| **DL-4：客观奖励对齐扩散微调（PrintAlign-Diff）** | **情况 2** | 扩散对齐微调在通用生图与室内设计已有先例，需防范多目标奖励冲突导致画质退化，调参周期较长 | **更有效的微调对齐方法** | **备选（作为后续训练式加分项）** |
| **DL-5：感性工学 + Midjourney + 问卷/AHP** | **情况 1** | 已有大量成熟先例，且不符合“深度学习方法 + 零被试”约束 | 无实质方法区别 | **换题（淘汰）** |

---

## 未核实事项与下一步

### 未核实事项
1. **5 篇标为 `verified=no` 的条目**（详见 `literature/papers.csv`）：
   - 2 篇未定位独立 DOI 的种子文献：Tao Chen（Semarak Ilmu 标题）、SAUNIC 2026（寨子乡论文）。
   - 3 篇付费墙阻挡摘要的 Elsevier 文献：`10.1016/j.engappai.2025.112865`、`10.1016/j.foodcont.2023.110101`、`10.1016/j.jrurstud.2019.09.010`（均已确认 DOI 真实存在并跳转 Elsevier 官方页，但因未能读取摘要原文，按严格核对规则暂标 `verified=no`，不作为核心论证依据）。
2. **公开数据集拉取实测**：[`docs/DATASETS.md`](../DATASETS.md) 中 14 个数据集的下载速度与协议审批周期需在开工首周实测。

### 下一步
- [ ] 阅读修订后的 [`docs/INNOVATION.md`](../INNOVATION.md)，确认“**免训练推理期同步与印刷硬约束引导为主方法** + 纳入 TexFusion/SyncMVD/Paint3D/MVPaint 纸盒网格基线”的实验路线。
- [ ] 启动 1 周最小验证实验：在标准 ECMA A20 折叠纸盒 3D 网格（以刀版为 UV）上直接运行 SyncMVD / Paint3D 开源基线，定量暴露其在粘口/插舌、出血安全区、文字朝向、CMYK 色域与中文小字上的失效数据，并验证免训练印刷硬约束同步原型的客观增益。
