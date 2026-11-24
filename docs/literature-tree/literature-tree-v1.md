# 深度学习驱动的包装设计（面向乡村农产品场景）文献树

- 版本：v1（Gemini 起草并按 2026-11-24 修订版任务书全面重构）
- 状态：草稿（待 Claude 核查）
- 检索截至：2026-11-24。检索记录见 [`literature/search-log.md`](../../literature/search-log.md)（共 34 组中英文与顶会/数据集/期刊定向检索），完整文献清单见 [`literature/papers.csv`](../../literature/papers.csv)（共 100 篇，其中已核对 98 篇，未核对 2 篇单独标出），公开数据集与自采版权分析专报见 [`docs/DATASETS.md`](../DATASETS.md)。

**核心问题结论**：按作者 2026-11-24 修订方向——转向**“用人工智能（深度学习等）做包装设计”**（以 AI 方法本身为核心贡献，OPC/方案交付降为可选应用框架）且**只采用纯客观评价指标（方案 A，不招募专家与消费者）**——本方向**完全可以发表开放获取（OA）的 SCIE 期刊论文**，而且比偏主观问卷的传统设计论文**更契合 SCIE 工程/计算机综合期刊对“可复现定量对比 + 模块消融实验”的审稿标准**。文献树表明：计算机视觉顶会已经较好解决了单张自然图像生成（LDM/SD3）、单面电商海报图文版式（CGL-GAN/PosterLayout/PosterLLaVA）和单面商品大标题渲染（AnyText2/PosterMaker），但**在真实包装设计上仍有五大未解决的硬核技术难点**——尤其是**① 参数化刀版展开图与 3D 多面折痕连续性对应**、**② 包装密集中文文字（艺术大字 + 法规小字）的零差错混合渲染**与**③ 跨 SKU 品牌元素/主色板一致性及印刷色域约束**。最值得做的深度学习方法切入点是在开源扩散底座（SDXL/Flux/SD3）上，用单卡 RTX 5090 (32 GB) 提出**“刀版拓扑感知的多面一致条件扩散 + 中文双轨图文解耦与印前引导架构（PackDiff / DieLineDiff）”**，配套公开数据集（Open Food Facts / RPC / ABO / Crello / AnyWord-3M）与自建可开源合成 2D 刀版—3D 渲染基准，用 **FID / LPIPS / HPS v2 / VQAScore / OCR Acc & 1-NED / 版式 mIoU & Overlap / 跨折线缝隙 PSNR / 3D 重投影 SSIM / CMYK 溢出率 & GB 7718-2025 规则通过率**全套客观指标做 SOTA 基线对比与消融实验。首选纯 OA SCIE 期刊为 ***IEEE Access***、***Applied Sciences***、***Scientific Reports***、***PLOS ONE***、***Electronics*** 与 ***Symmetry***；主要风险是若仅简单微调一个 LoRA 或堆砌现成工具而无针对包装技术难点的架构/损失/引导创新与严谨消融，将被判定为“简单 A+B 应用”而拒稿。

---

## 怎么读

- **方法来源**：遵循彭思达的科研选题与领域视野方法（[learning_research](https://github.com/pengsida/learning_research)、[选题课件](https://pengsida.net/files/research_topic_selection.pdf)）与 `findInnovation` 模板：以时间线梳理技术演进，用“难点—解法树”定位技术空白，用“四种情况”判断选题创新空间。
- **脉络权重调整（按 2026-11-24 作者决策）**：
  - **时间线一（绝对主线，45 篇已核对）**：深度学习 / 生成式 AI 用于包装、平面、版式、文字渲染、品牌一致性与 3D 多面贴图（深度追踪 CVPR / ICCV / ECCV / SIGGRAPH / NeurIPS / ICLR / IJCAI / IEEE TMM 里程碑）。
  - **时间线二（应用场景，18 篇）**：农产品包装设计与乡村振兴（明确乡村包装的约束特征与低质文献警示）。
  - **时间线三（可选框架，14 篇）**：OPC / 一人公司与人机协作设计（作为论文讨论或落地部署框架，不再占核心技术篇幅）。
  - **时间线四（核心评测与数据，23 篇 + 14 个公开数据集）**：**纯客观无人工指标体系（方案 A）**与可用公开数据集（详表见 [`docs/DATASETS.md`](../DATASETS.md)）。
- **核对声明**：凡带可点开 DOI 链接的 98 篇文献均已通过 OpenAlex / Crossref API 核实标题、作者、年份与发表处；2 篇未能核实原文的种子文献单独标注 `verified=no`。

---

## 初步假设（H1—H4）检验结论（聚焦：深度学习方法论文能否成立）

| 假设 | 结论 | 核心依据与对“深度学习方法论文”的启示 |
| --- | --- | --- |
| **H1（交付与技术缺口）**：现有“AIGC + 包装设计”多停在出图与审美评价，很少解决刀版、印刷、中文文字、法规、条码与品牌一致性 | **高度成立（构成 DL 方法论文的核心问题源）** | 现有包装应用 SCI/SSCI 论文（Gao & Li 2025 *J. Eng. Des.*; Song & Bai 2025 *Applied Sciences*; Zhao et al. 2026 *HSSC*; Liu et al. 2026 *IEEE Access*）全部停留在单面概念图或商用 Midjourney 提示词调用；而 CV 顶会虽有电商海报生成（PosterMaker CVPR 2025）与 3D 贴图扩散（Paint3D CVPR 2024、MV-Adapter ICCV 2025），但**尚未解决工业折叠包装特有的“刚性参数化刀版多面拓扑 + 中文密集小字零错渲染 + CMYK/出血印前约束 + 多 SKU 品牌一致”联合建模问题**。**结论：深度学习方法论文完全成立，核心贡献应落在针对这些技术难点的模型结构/条件引导/解耦合成机制上。** |
| **H2（组织形态：OPC）**：现有研究少以“一人 + 多智能体整单交付”为研究对象 | **部分成立（按作者决策降为可选应用/讨论框架）** | 2026 年已有 *Systems* (`10.3390/systems14080994`)、IEEE *IT Professional* (`10.1109/mitp.2026.3694520`) 与 *SEJ* (`10.1002/sej.70067`) 确立 `One-Person Company (OPC)`（`1+N+AI`）学术概念。若以深度学习方法为主线，OPC 不宜喧宾夺主，最合适的位置是作为 **Discussion / Deployment Case（轻量化单卡 5090 推理如何支撑单人设计师低成本服务县域农产品包装）**，放大方法论文的实用价值。 |
| **H3（场景价值）**：乡村农产品预算低、品类碎、缺设计资源 | **成立（决定论文的类别条件与基准构造场景）** | 乡村农产品（茶、粮油、果干、蜂蜜、杂粮等）高频使用标准折叠纸盒/自立袋/罐贴，同时受 **GB 7718-2025 & GB 28050-2025**（2027-03-16 正式强制实施）强中文标签约束，是检验深度学习方法“按产品类别控制风格 + 中文文字零错 + 刀版/印前合规”的理想高难测试场（challenging cases）。 |
| **H4（低质风险）**：“AI + 农产品包装”已有不少弱刊论文，简单出图难进 SCI | **高度成立（警示必须做扎实的方法对比与消融）** | 7 篇种子文献中，2 篇发在已被 Scopus 剔除的弱刊（*AMNS* 2024、*JISEM* 2024），其中 Songyong Hu (2024) 虽题含 Deep Learning 实为传统 HSV/灰度共生矩阵浅层特征。要发高认可度 OA SCIE，**绝不能只做“微调一个 LoRA + 展示几张效果图”**，必须有明确的技术痛点定义、架构/损失/条件机制设计、公开/开源合成基准上的多基线定量对比与消融实验。 |

---

## 长远目标与路线图

**长远目标**：构建面向真实工业制造与商业流通的**结构感知、文字精确、风格可控、印前可生产的深度学习包装生成范式与客观评测基准**，以单卡可部署的轻量化开源模型赋能乡村农产品与小微品牌包装设计。

| 里程碑任务 | 内容 | 现状 | 判断 |
| --- | --- | --- | --- |
| **M0：通用文生图与风格微调** | 用扩散模型 + LoRA/ControlNet 生成单面包装概念图 | **已做到**（LDM / SD3 / ControlNet / IP-Adapter 已成通用基础） | 已做到（不可作主贡献） |
| **M1：二维单面电商海报与文字渲染** | 内容感知避让主体排版 + 单面中英文视觉标题准确渲染 | **已较好解决**（CGL-GAN / PosterLayout / AnyText2 / Glyph-ByT5-v2 / PosterMaker CVPR 2025 / PosterLLaVA IEEE TMM 2026） | 已有强基线（包装需向多面与密集小字突破） |
| **M2：包装专用五大技术难点联合攻关** | ① 刀版展开图与 3D 多面折痕连续性对应；② 中文视觉标题 + 密集法规小字零错混合生成；③ 跨 SKU 品牌 Logo/色板一致性；④ 按农产品类别与文化风格解耦控制；⑤ CMYK/出血/300DPI 可印刷性引导 | **未解决 / 高度割裂**（仅在远域 3D 网格贴图 Paint3D/MV-Adapter 或 2D 透明图层 LayerDiffusion 有局部前置组件） | **空白，最值得做（深度学习方法主贡献）** |
| **M3：零人工纯客观包装生成评测基准与开源数据集** | 基于公开数据 + 程序化合成 2D 刀版↔3D 渲染对构建开源 Benchmark，并用纯客观多维指标体系验证方法与消融 | **未见**（现有仅有 2D 海报基准 PKU PosterLayout/Crello 或 3D 通用物块 ABO，无包装刀版—文字—合规配对基准） | **空白，值得做（与 M2 耦合直接支撑无被试 SCIE 论文）** |
| **M4（可选延展）：单卡轻量化推理与 OPC 落地框架** | 将 M2 模型封装为单卡 RTX 5090 可运行的轻量工作流服务乡村农产品小微交付 | **待验证** | 作为论文应用章节或后续工程扩展 |

---

## 领域视野六问

| 问题 | 回答 | 依据 |
| --- | --- | --- |
| **里程碑论文有哪些** | ① 扩散与控制基石：LDM (CVPR 2022)、ControlNet (ICCV 2023)、SD3/Rectified Flow (ICML 2024)、Diffusion-4K (CVPR 2025)；② 版式与电商海报：Content-aware Layout (TOG 2019)、CanvasVAE (ICCV 2021)、CGL-GAN (IJCAI 2022)、PosterLayout/DS-GAN (CVPR 2023)、LayoutDM (CVPR 2023)、LayoutGPT (NeurIPS 2023)、COLE (2023)、LayerDiffusion (SIGGRAPH 2024)、PosterMaker (CVPR 2025)、CreatiPoster (2025)、PosterLLaVA (IEEE TMM 2026)；③ 中文视觉文字渲染：TextDiffuser (NeurIPS 2023)、GlyphControl (NeurIPS 2023)、AnyText (ICLR 2024)、Glyph-ByT5 (ECCV 2024)、UDiffText (ECCV 2024)、AnyText2 (2024)、FonTS (ICCV 2025)；④ 主体/风格一致与 3D 贴图：LoRA (2021)、DreamBooth (CVPR 2023)、IP-Adapter (2023)、InstantStyle (2024)、AnyDoor (CVPR 2024)、TEXTure (SIGGRAPH 2023)、Paint3D (CVPR 2024)、MV-Adapter (ICCV 2025)；⑤ 客观评价指标：FID (NeurIPS 2017)、LPIPS (CVPR 2018)、CLIPScore (EMNLP 2021)、ImageReward/PickScore (NeurIPS 2023)、HPS v2 (2023)、VQAScore (ECCV 2024)、DesignCritic (*IEEE Access* 2026)。 | 详见时间线一、四及 `literature/papers.csv` |
| **技术如何演变** | **第一代（2019–2022）**：GAN/VAE 离散版式框生成（LayoutGAN、CanvasVAE、CGL-GAN）；**第二代（2022–2023）**：潜在扩散 + 空间/主体适配器（LDM、ControlNet、DreamBooth、IP-Adapter、LayoutDM）；**第三代（2024–2025）**：MMDiT 底座 + 多语种字符感知渲染（AnyText/Glyph-ByT5/AnyText2/FonTS）+ 透明多图层（LayerDiffusion）+ 电商商品文字解耦（PosterMaker）+ 去光照 3D UV 贴图与多视角适配器（Paint3D、MV-Adapter）+ 原生 SVG 代码生成（StarVector）；**第四代前沿（2025–2026）**：从单面 RGB 视觉合成走向**多条件拓扑约束（刀版/3D UV）+ 图层与矢量代码混合解耦 + 客观原则驱动的推理引导/奖励对齐**。 | 综合时间线一 45 篇顶会顶刊演进轨迹 |
| **领域的终极目标** | 给定产品类别、结构刀版参数、品牌视觉资产、地域文化风格与法定文字清单，端到端或结构化生成**视觉高质、多面折合无缝、中文零错字、品牌零漂移、300 DPI CMYK 可直接制版印刷**的包装方案。 | 综合包装工业需求与 CV/图形学生成范式 |
| **目前达到什么水平** | 单面 RGB 电商海报与中英文艺术标题准确率已突破 85%–90%+（AnyText2 / PosterMaker CVPR 2025）；通用 3D 网格无光照贴图与多视角生成已实现即插即用（Paint3D / MV-Adapter ICCV 2025）；透明多图层与 4K 生成已跑通（LayerDiffusion / Diffusion-4K CVPR 2025）。但在**面向工业包装的“刀版折面拓扑连续性 + 密集小字零错 + 品牌多 SKU 锁定 + CMYK 可印刷性”联合求解**上仍远未达标。 | 见下文“难点—解法树”五大难点分析 |
| **哪些重要问题未解决** | 五大具体深度学习技术难点：① 包装密集中文文字（尤以小字配料/营养表与曲面透视）渲染错误；② 跨 SKU/多视图品牌 Logo 与色板漂移；③ 2D 参数化刀版展开图与 3D 折叠多面几何对应及跨折线接缝断裂；④ 按产品类别与地域文化风格的细粒度解耦控制；⑤ 屏幕 RGB 像素输出到 300 DPI CMYK 可印刷/可编辑多图层交付的鸿沟。 | 见“难点—解法树” |
| **当前热点** | ① 商业商品海报解耦生成（PosterMaker CVPR 2025、CreatiPoster 2025、PosterLLaVA IEEE TMM 2026）；② 即插即用多视角/3D 贴图扩散（Paint3D CVPR 2024、MV-Adapter ICCV 2025）；③ 字符与排版可控 DiT（AnyText2、FonTS ICCV 2025）；④ 无人工自动设计审计与奖励对齐（VQAScore ECCV 2024、DesignCritic *IEEE Access* 2026、ARIA-Diff *Mathematics* 2026）。 | 2024–2026 CVPR/ICCV/ECCV/SCIE 热点聚类 |

---

## 时间线一（主线）：深度学习用于包装、平面、版式、文字渲染与 3D 贴图

近七年，该主线从“二维矩形版式 GAN”演进到“可控多图层扩散 + 中文字符渲染 + 3D 无光照多视角贴图”，但**尚未针对“折叠包装刀版拓扑、密集中文合规排版与印刷色域约束”建立专用深度学习架构与基准**。

| 年份 | 工作（带链接） | 子方向 / 里程碑 | 解决了什么（Pipeline 与技术洞见） | 留下什么（面向包装的空白） |
| --- | --- | --- | --- | --- |
| 2019 | Zheng et al., *Content-aware Graphic Layout* ([ACM TOG / SIGGRAPH](https://doi.org/10.1145/3306346.3322971)) | 版式生成·里程碑 | 根据图像与文本语义生成杂志版式，开启内容感知布局建模 | 仅二维单面，无包装结构与文字实体 |
| 2021 | Li et al., *LayoutGAN* ([IEEE TPAMI](https://doi.org/10.1109/tpami.2019.2963663)) | 版式生成·里程碑 | 可微矢量线框渲染器实现几何—像素联合对抗训练 | 难以处理复杂嵌套层级与多面拓扑 |
| 2021 | Yamaguchi, *CanvasVAE* ([ICCV](https://doi.org/10.1109/iccv48922.2021.00543)) | 多层文档·里程碑 | 多图层矢量文档建模 + 发布 **Crello** 公开基准 | 生成画质受限于早期 VAE |
| 2021 | Hu et al., *LoRA* ([arXiv / ICLR 2022](https://doi.org/10.48550/arxiv.2106.09685)) | 轻量微调·里程碑 | 低秩适配大幅降低大模型微调显存，成单卡 5090 风格/类别微调基础 | 单独微调易过拟合且缺乏结构硬约束 |
| 2022 | Rombach et al., *LDM / Stable Diffusion* ([CVPR](https://doi.org/10.1109/cvpr52688.2022.01042)) | 文生图底座·里程碑 | 潜在空间扩散实现高分辨率文生图，奠定开源生成底座 | 文字畸变、空间不可控、RGB 像素无图层 |
| 2022 | Zhou et al., *CGL-GAN* ([IJCAI](https://doi.org/10.24963/ijcai.2022/692)) | 电商海报·里程碑 | 构图感知电商海报版式生成 + 发布 6 万级中文 **CGL** 数据集与主体避让指标 | 仅预测 2D 边界框，不生成图像与包装折面 |
| 2023 | Zhang et al., *ControlNet* ([ICCV](https://doi.org/10.1109/iccv51070.2023.00355)) | 空间控制·里程碑 | 零卷积条件分支注入线稿/深度/分割等空间约束 | 仅对齐像素轮廓，不保证多面跨缝连续与矢量文字 |
| 2023 | Ruiz et al., *DreamBooth* ([CVPR](https://doi.org/10.1109/cvpr52729.2023.02155)) | 主体一致·里程碑 | 少样本定制扩散模型实现特定主体跨场景生成 | 每品牌需单独微调，且难同步保持 Logo 几何零畸变 |
| 2023 | Ye et al., *IP-Adapter* ([arXiv](https://doi.org/10.48550/arxiv.2308.06721)) | 图像提示·里程碑 | 22M 解耦交叉注意力适配器实现免微调参考图注入，兼容 ControlNet | 易发生内容与风格耦合（语义泄露） |
| 2023 | Hsu et al., *PosterLayout (DS-GAN)* ([CVPR](https://doi.org/10.1109/cvpr52729.2023.00583)) | 电商版式·里程碑 | 显著性感知海报排版 + **PKU PosterLayout** 基准与标准客观版式指标体系 | 面向平面促销海报，未引入刀版安全区与多面对应 |
| 2023 | Inoue et al., *LayoutDM* ([CVPR](https://doi.org/10.1109/cvpr52729.2023.00980)) | 版式扩散·里程碑 | 离散状态扩散统一处理多类版式条件约束 | 仅处理矩形框集合，无图文像素联合渲染 |
| 2023 | Feng et al., *LayoutGPT* ([NeurIPS](https://doi.org/10.52202/075280-0802)) | LLM 规划·里程碑 | 用 LLM 生成 CSS 声明式空间布局指导下游扩散模型 | 缺乏确定性几何/印刷规则反向校验 |
| 2023 | Chen et al., *TextDiffuser* ([NeurIPS](https://doi.org/10.52202/075280-0410)) & Yang et al., *GlyphControl* ([NeurIPS](https://doi.org/10.52202/075280-1912)) | 文字渲染·里程碑 | 字符掩码/字形图条件注入扩散模型 + 发布 **MARIO-10M** | 主要针对英文大字，中文复杂笔画与小字仍高错 |
| 2023 | Richardson et al., *TEXTure* ([SIGGRAPH](https://doi.org/10.1145/3588432.3591503)) | 3D 贴图·里程碑 | 迭代多视角深度扩散为 3D 网格上色 | 易烘焙虚假光影，在纸盒锐利棱边易产生接缝与文字扭曲 |
| 2023 | Jia et al., *COLE* ([arXiv](https://doi.org/10.48550/arxiv.2311.16974)) & Jain et al., *VectorFusion* ([CVPR](https://doi.org/10.1109/cvpr52729.2023.00190)) | 多层/SVG·里程碑 | 层次化多模型生成可编辑分层海报；SDS 蒸馏像素扩散生成 SVG | 闭源数据/路径冗余，无包装刀版与印刷规范 |
| 2024 | Esser et al., *SD3 / Rectified Flow* ([ICML](https://doi.org/10.48550/arxiv.2403.03206)) | 底座跃迁·里程碑 | MMDiT + Rectified Flow 架构，大幅改善复杂提示理解与英文拼写 | 原生中文小字与包装多面展开拓扑依然失效 |
| 2024 | Tuo et al., *AnyText* ([ICLR](https://doi.org/10.48550/arxiv.2311.03054)) & *AnyText2* ([arXiv](https://doi.org/10.48550/arxiv.2411.15245)) | 中文文字·里程碑 | OCR 笔画编码 + WriteNet+AttnX 实现中英文逐行字体/颜色定制 + 发布 **AnyWord-3M** (Apache-2.0) | 适合包装大标题/艺术字，但对背面 200+ 字密集配料表与营养成分表仍存在字符粘连与错字 |
| 2024 | Liu et al., *Glyph-ByT5* ([ECCV](https://doi.org/10.1007/978-3-031-73226-3_21)) & *v2* ([arXiv](https://doi.org/10.48550/arxiv.2406.10208)) | 多语种字形·后续 | 字符级 ByT5 编码器耦合 SDXL 支持含中文在内的 10 语种高准文字渲染 | 仍属光栅像素渲染，无法矢量编辑或单黑印刷 |
| 2024 | Zhang & Agrawala, *LayerDiffusion* ([ACM TOG / SIGGRAPH](https://doi.org/10.1145/3658150)) | 透明图层·里程碑 | 潜在空间编码 Alpha 通道原生生成透明前景与多层构图 | 为包装元素分离提供关键底座，但未结合刀版布局约束 |
| 2024 | Wang et al., *InstantStyle* ([arXiv](https://doi.org/10.48550/arxiv.2404.02733)) & Chen et al., *AnyDoor* ([CVPR](https://doi.org/10.1109/cvpr52733.2024.00630)) | 风格解耦/对象植入·里程碑 | 注意力层减法实现风格与内容解耦；DINOv2 特征零样本植入对象 | 尚未针对“农产品类别先验 × 地域文化风格 × Logo 锁定”联合优化 |
| 2024 | Zeng et al., *Paint3D* ([CVPR](https://doi.org/10.1109/cvpr52733.2024.00407)) | 无光照 UV 贴图·里程碑 | 粗细两阶段去光照 UV 扩散消除 3D 表面阴影烘焙与接缝 | 面向通用 3D 物体，未利用折叠纸盒规则多面平面 UV 与印刷排版约束 |
| 2025 | Gao et al., *PosterMaker* ([CVPR](https://doi.org/10.1109/cvpr52734.2025.00757)) | 商品海报·里程碑 | 解耦 TextRenderNet (>90% 文本精度) 与 SceneGenNet，实现高精度中文商品海报生成 | **最接近包装设计的 CVPR 前沿基线**，但仅覆盖单面 2D 电商海报，无刀版折叠拓扑与法定背面排版 |
| 2025 | Huang et al., *MV-Adapter* ([ICCV](https://doi.org/10.1109/iccv51701.2025.01520)) | 多视角一致·里程碑 | 即插即用并行多视角注意力将 2D 扩散低成本升级为 3D 多视图一致生成 | 面向自由 3D 视角，尚未将折叠包装的**刀版邻接图（正/侧/背/顶共边关系）**显式编入跨面注意力 |
| 2025 | Rodriguez et al., *StarVector* ([CVPR](https://doi.org/10.1109/cvpr52734.2025.01508)) & Zhang et al., *Diffusion-4K* ([CVPR](https://doi.org/10.1109/cvpr52734.2025.02185)) | SVG 代码 / 4K 超分·里程碑 | MLLM 直接生成语义 SVG 代码（+ **SVG-Stack**）；小波微调原生 4K 图像生成（+ GLCM 细节指标） | 分别推进了矢量代码化与印刷级 300 DPI 分辨率，但未整合入包装展开图生成管线 |
| 2025 | Shi et al., *FonTS* ([ICCV](https://doi.org/10.1109/iccv51701.2025.01716)) & Hong et al., *CreatiPoster* ([arXiv](https://doi.org/10.48550/arxiv.2506.10890)) | 排版控制/多层协议·后续 | DiT 词级字体风格控制；RGBA 协议模型输出 JSON 图层与 10 万无版权多层语料 | 均止步于数字平面海报，缺印刷色域与 3D 包装对应 |
| 2026 | Yang et al., *PosterLLaVA* ([IEEE TMM](https://doi.org/10.1109/tmm.2026.3753095)) | MLLM 版式·里程碑 | 视觉指令微调 MLLM 输出 JSON 版式并导出可编辑 SVG 海报 | 单面二维海报范式，未扩展至包装展开多面与食品法规 |
| 2025–26 | Gao & Li ([J. Eng. Des. 2025](https://doi.org/10.1080/09544828.2025.2489647)); Wu ([IEEE Access 2025](https://doi.org/10.1109/access.2025.3596858)); Liu et al. ([IEEE Access 2026](https://doi.org/10.1109/access.2026.3723820)) | 包装应用对照 | 将评论挖掘/模糊注意力/人机协同引入茶/饼干等产品包装 | **鲜明对比**：现有包装类 SCI 论文几乎全停留在调用现成生图工具或浅层分类器，严重落后于 CV 顶会的前沿架构 |

**本线技术范式小结**：视觉生成顶会已清晰演化出四项可组合的现代深度模块——**① 轻量适配器（LoRA / IP-Adapter / InstantStyle / MV-Adapter）**、**② 字符与图文双网络解耦（AnyText2 / PosterMaker）**、**③ 潜在透明图层与去光照 UV 扩散（LayerDiffusion / Paint3D）**、**④ 结构化 JSON/SVG 代码层（PosterLLaVA / StarVector）**。但工业包装设计界尚未吸收这套现代深度学习武器库，形成了显著的**“顶会通用视觉组件先进 vs. 包装垂直领域停留在 Midjourney 提示词阶段”的代差红利**——这正是我们在单卡 RTX 5090 上以“领域专用深度学习架构 + 损失/引导创新”切入 SCIE 的最佳窗口！

---

## 时间线二（应用场景）：农产品包装设计与乡村振兴

该脉络为深度学习模型提供真实约束与应用场景：既涵盖消费感知与可持续材料背景，也暴露了低质期刊扎堆与 **GB 7718-2025/28050-2025** 强法规约束的现实背景。

| 年份 | 工作（带链接） | 角色 | 核心结论与对 DL 方法设计的约束输入 |
| --- | --- | --- | --- |
| 2007–2009 | Silayoi & Speece ([EJM](https://doi.org/10.1108/03090560710821279)); Rundh ([Br. Food J.](https://doi.org/10.1108/00070700910992880)) | 领域奠基 | 包装兼具“视觉美学吸引”与“标签信息/供应链结构”双职能 → 证明模型必须同时优化视觉质量与结构/信息合规 |
| 2020–2024 | Boz et al. ([Sustainability 2020](https://doi.org/10.3390/su12062192)); Duan et al. ([Sustainability 2024](https://doi.org/10.3390/su16041381)); Rivers et al. ([Food Control 2024](https://doi.org/10.1016/j.foodcont.2023.110101)) | 可持续与合规背景 | 农废材质（牛皮纸/秸秆纸）表面底色偏黄/低饱和度，且预包装食品标签高频违规 → 对扩散模型的印刷色域与强制信息保留提出硬约束 |
| 2024 | Chen Tao et al. ([JISEM](https://doi.org/10.55267/iadt.07.15214)); Songyong Hu ([AMNS](https://doi.org/10.2478/amns-2024-1882)) | 种子2、3（弱来源） | 均已被 Scopus 剔除；Hu (2024) 仅用传统 HSV/GLCM 特征却冠名 Deep Learning，属典型反面教材 |
| 2025 | Song & Bai ([Applied Sciences](https://doi.org/10.3390/app15116033)); Chen & Fu ([AHFE](https://doi.org/10.54941/ahfe1006930)); Tang et al. ([JTAER](https://doi.org/10.3390/jtaer20030248)) | 种子4、5 + 电商背景 | 种子 5 依赖 100 人问卷 + AHP/TOPSIS + Midjourney，无法在不招募被试时复现；Tang (2025) 证明包装色彩/字体/插画直接决定电商复购 |
| 2026 | Zhao et al. ([HSSC](https://doi.org/10.1057/s41599-026-06948-z)); Wu et al. ([ACSR](https://doi.org/10.2991/978-94-6239-737-8_27)); 李雪娜等 ([2026](https://doi.org/10.64216/3104-9664.26.04.014)) | 种子7 + 新国标线索 | 文化转译+主观评分已被占满；**GB 7718-2025 & GB 28050-2025**（2027-03-16 强制实施）带来明确的机器可执行法规约束（禁用“零添加”、字高 ≥1.8 mm、1+6 营养表、数字标签二维码），天然适合作为零被试客观评测指标 |

---

## 时间线三（可选应用/讨论框架）：OPC 一人公司与人机协同设计

> 按 2026-11-24 作者决策，本脉络降为**可选应用框架或论文 Discussion 章节**，用于说明所提轻量深度学习模型（单卡 RTX 5090 可推理）为何能赋能小微农业与单人设计师，不再作为主实验目标。

| 年份 | 代表工作（带链接） | 核心洞见（供论文引言/讨论引用） |
| --- | --- | --- |
| 2019–2023 | Räisänen & Tuovinen ([J. Rural Studies 2019](https://doi.org/10.1016/j.jrurstud.2019.09.010)); Noy & Zhang ([Science 2023](https://doi.org/10.1126/science.adh2586)) | 农村小微企业缺专业设计资源，AI 可显著降低个体专业任务门槛 |
| 2024–2025 | Zhou et al., *OptiMuse* ([CHI 2024](https://doi.org/10.1145/3613904.3642812)); Chen et al., *CoExploreDS* ([CHI 2025](https://doi.org/10.1145/3706598.3713869)); Rezazadeh & Bonehgazy ([arXiv:2511.09533](https://doi.org/10.48550/arxiv.2511.09533)) | 创意设计是非线性人机协同过程，单人创业者可借助智能体扩展产能 |
| 2026 | Liu & Yang ([Systems](https://doi.org/10.3390/systems14080994)); Kshetri ([IEEE IT Prof.](https://doi.org/10.1109/mitp.2026.3694520)); Li, Zeng, & Li ([SEJ](https://doi.org/10.1002/sej.70067)); Dell'Acqua et al. ([Org. Sci.](https://doi.org/10.1287/orsc.2025.21838)); Chen et al. ([J. Eng. Des.](https://doi.org/10.1080/09544828.2026.2616583)) | 学术界已正式确立 **`One-Person Company (OPC)`**（`1+N+AI` 社会技术系统），并证明在“锯齿状技术前沿”之外（如强结构/强法规任务）必须通过确定性工程约束与自动校验保障最低可行智能（MVI） |

---

## 时间线四（核心评测与数据）：零人工纯客观指标体系（方案 A）与公开数据集

> 按 2026-11-24 作者决策，**完全放弃方案 B（不招募专家与消费者）**，全部评价由可复现的**纯客观深度学习指标 + 基线定量对比 + 消融实验**支撑。

| 客观指标维度 | 代表指标与出处（带链接） | 测量对象 | 是否需人类被试 |
| --- | --- | --- | --- |
| **1. 图像分布与感知保真** | **FID** ([Heusel et al., NeurIPS 2017](https://doi.org/10.48550/arxiv.1706.08500))、**LPIPS** ([Zhang et al., CVPR 2018](https://doi.org/10.1109/cvpr.2018.00068)) | 生成包装图与真实公开包装库（Open Food Facts / RPC）的特征分布距离及感知多样性 | **否** |
| **2. 自动审美与偏好代理** | **HPS v2** ([Wu et al., 2023](https://doi.org/10.48550/arxiv.2306.09341))、**ImageReward** ([Xu et al., NeurIPS 2023](https://doi.org/10.52202/075280-0700))、**PickScore** ([Kirstain et al., NeurIPS 2023](https://doi.org/10.52202/075280-1594))、**DesignCritic** ([Wu & Song, IEEE Access 2026](https://doi.org/10.1109/access.2026.3730226)) | 基于预训练偏好奖励模型自动打分，被 CVPR/NeurIPS/SCIE 广泛认可为无人工视觉审美代理指标 | **否** |
| **3. 细粒度图文与类别/风格对齐** | **CLIPScore** ([Hessel et al., EMNLP 2021](https://doi.org/10.18653/v1/2021.emnlp-main.595))、**VQAScore** ([Lin et al., ECCV 2024](https://doi.org/10.1007/978-3-031-72673-6_20))、DINOv2 / CLIP-I 风格-内容解耦相似度 | 衡量生成图对“农产品类别 + 地域文化风格 + 指定元素”复杂组合条件的忠实度 | **否** |
| **4. 中文视觉文字渲染精度** | **OCR Sentence Accuracy / Word Accuracy / 字符错误率 CER / `1 - NED`**（PaddleOCR / AnyText-benchmark 协议，[Tuo et al., ICLR 2024](https://doi.org/10.48550/arxiv.2311.03054)） | 客观定量比较各基线模型在中文品名、卖点与配料表渲染上的拼写正确率 | **否** |
| **5. 版式与刀版几何约束指标** | **mIoU / Alignment / Overlap / Saliency Occlusion (`R_occ`) / Underlay (`R_und`)**（[CGL-GAN IJCAI 2022](https://doi.org/10.24963/ijcai.2022/692); [PosterLayout CVPR 2023](https://doi.org/10.1109/cvpr52729.2023.00583)）+ **刀版出血/安全区越界率（Bleed Violation Rate）** + **跨折线缝隙连续性（Seam Edge PSNR/LPIPS）** + **3D 折叠多视角重投影 SSIM** | 客观验证 2D 刀版展开图与 3D 折叠效果图的几何一致性、无缝连续性与版式规整度 | **否** |
| **6. 品牌一致与可印刷/法规硬指标** | **Logo 检测召回率 & 模板 SSIM**、**品牌色板 CIEDE2000 ($\Delta E_{00}$)**、**CMYK 色域溢出率（Out-of-Gamut %）**、**300 DPI 高频清晰度（GLCM，[Diffusion-4K CVPR 2025](https://doi.org/10.1109/cvpr52734.2025.02185)）**、**条码 `pyzbar` 自动解码率**、**GB 7718-2025 规则通过率** | 客观证明生成结果不仅好看，而且满足跨 SKU 一致性与工业印刷/法规硬要求（呼应 [Jung, *Sustainability* 2026](https://doi.org/10.3390/su18062943) 双层脱节评测） | **否** |

> 公开数据集详表（Open Food Facts、ABO、RPC、Crello、PKU PosterLayout、CGL、CreatiPoster 100K、AnyWord-3M、MARIO-10M、SVG-Stack 等 14 个数据集的规模、License、下载方式与自采版权风险）见专报 [`docs/DATASETS.md`](../DATASETS.md)。

---

## 按里程碑任务归类的代表性 pipeline

| 里程碑任务 | 代表性深度网络 Pipeline | 代表论文 |
| --- | --- | --- |
| **单面商品海报与中文文字生成** | `产品前景图 + 布局规划器(MLLM/LayoutDM) → SceneGenNet 背景生成 + TextRenderNet / WriteNet+AttnX 字形分支 → 单面 RGB 海报` | PosterMaker (CVPR 2025); AnyText2 (2024); PosterLLaVA (IEEE TMM 2026) |
| **品牌风格与主体控制** | `冻结扩散主干(SDXL/SD3/Flux) + 轻量适配器(LoRA / IP-Adapter 解耦层 InstantStyle / DINOv2 AnyDoor) → 风格或主体锁定图像` | IP-Adapter (2023); InstantStyle (2024); AnyDoor (CVPR 2024) |
| **3D 多视角与无光照 UV 贴图** | `3D 网格深度/法线/UV 坐标图 → 粗阶段多视角一致扩散(MV-Adapter) → 细阶段去光照 UV 空间修补扩散(Paint3D) → 无缝 3D 贴图` | TEXTure (SIGGRAPH 2023); Paint3D (CVPR 2024); MV-Adapter (ICCV 2025) |
| **透明多图层与高分辨率合成** | `潜在透明偏移编码(LayerDiffusion) / RGBA 协议 JSON(CreatiPoster) + 小波 4K 扩散(Diffusion-4K) → 可编辑高清分层文档` | LayerDiffusion (SIGGRAPH 2024); CreatiPoster (2025); Diffusion-4K (CVPR 2025) |
| **本课题目标空白（包装专用 DL 架构）** | `参数化刀版拓扑图(面邻接矩阵+折线/出血掩码) + 品类/文化解耦编码 + 品牌资产锁定 → 刀版拓扑感知跨面注意力扩散(无光照 UV 底图) + 双轨文字合成(大标题视觉文字分支 + 背面矢量法规排版层) + 印前 CMYK/安全区引导 → 2D 可印刷刀版图 & 3D 折叠多视图` | **当前文献空白（本课题核心方法靶点）** |

---

## 难点—解法树（深度学习做包装设计的五大具体技术难点）

| 具体技术难点 | 现有代表性深度学习方法 | 来自 | 在包装设计上的具体空白（为什么现有方法还不够好） | 别的数据域里有没有同内核的解法 |
| --- | --- | --- | --- | --- |
| **T1：包装上中文文字渲染错误**（大标题艺术字错笔画、背面密集配料表/营养表乱码、折面/曲面文字畸变） | AnyText (ICLR 2024)、AnyText2 (2024)、Glyph-ByT5-v2 (2024)、UDiffText (ECCV 2024)、PosterMaker (CVPR 2025)、FonTS (ICCV 2025) | CV/AI 顶会 | 现有方法仅验证**短句大标题**（1–15 字）；农产品包装正面需地域风格中文艺术大字，背面/侧面同时含 **100–300 字小号法规文本与表格**，像素扩散在密集小字上 CER 依然极高且不可单黑印刷；缺乏**“视觉艺术字扩散分支 + 法定信息矢量排版掩码保护”双轨协同架构** | 有：前端网页代码生成（Design2Code NAACL 2025）与多层海报协议（CreatiPoster 2025）采用结构化渲染避免密集小字像素崩坏 |
| **T2：品牌元素与跨 SKU 主体一致性**（多口味/多规格/多盒型下 Logo 变形、IP 漂移、主色板 $\Delta E_{00}$ 偏移） | LoRA (2021)、DreamBooth (CVPR 2023)、IP-Adapter (2023)、InstantStyle (2024)、AnyDoor (CVPR 2024) | CV 顶会 | 单用 IP-Adapter 易把参考图的旧构图/文字一起带入（语义泄露），单用 LoRA 需为每个县域小品牌重训；针对同一农产品品牌下 **3–6 个不同规格/口味 SKU 同步锁定 Logo 几何、主色板与文化纹样**，尚缺免重训的多条件色板+标志保持引导机制 | 有：视频/故事生成中的跨帧一致性注意力（Consistent Self-Attention）与 InstantStyle 层级注入 |
| **T3：多面展开图 / 刀版与 3D 包装效果图的对应**（折线处图案断裂、顶盖/侧面文字倒置、模切/糊口区误绘、3D 渲染阴影污染 2D 印稿） | ControlNet (ICCV 2023)、TEXTure (SIGGRAPH 2023)、Paint3D (CVPR 2024)、MV-Adapter (ICCV 2025) | CV/图形学顶会 | **最核心空白**：直接用 2D 扩散画展开图会导致物理折叠后相邻共边严重错位、侧面图文方向颠倒；而 Paint3D/MV-Adapter 面向任意 3D 雕塑网格，**未利用工业折叠纸盒的刚性平面折面拓扑（已知精确的边邻接映射与 90°/180° 旋转对应关系）**。把“刀版邻接边循环卷积 / 跨折面配对注意力 + 去光照 UV 约束”引入扩散模型，是极具技术含金量的原创新机制 | 有：全景图无缝生成中的 **Circular Padding / MultiDiffusion 跨边界特征对齐** 与 MV-Adapter 的并行多视角注意力（极少数远域有同内核解法 → **情况 3**！） |
| **T4：按产品类别与地域文化风格精准控制**（类别先验与地域纹样耦合导致“茶叶不像茶叶、文化符号僵硬贴图”） | CGL-GAN (IJCAI 2022)、PosterLayout (CVPR 2023)、SD3 (ICML 2024)、InstantStyle (2024)、ARIA-Diff (*Mathematics* 2026) | CV 顶会 & SCIE | 现有农产品包装研究靠人工拼凑长 Prompt（易同质化且跨品类泛化差）；缺乏将**产品品类结构先验（茶盒/果干袋/罐贴）**与**地域文化风格嵌入**在特征空间正交解耦、并结合客观原则奖励对齐的可控扩散微调方法 | 有：属性解耦风格迁移（InstantStyle）与奖励微调（Diffusion-DPO / Flow-GRPO NeurIPS 2025） |
| **T5：生成结果的可印刷性与可编辑性**（分辨率不足 300 DPI、RGB 荧光色转 CMYK 色域崩溃、不可分图层修改、图文越过出血/压痕线） | LayerDiffusion (SIGGRAPH 2024)、Diffusion-4K (CVPR 2025)、StarVector (CVPR 2025)、PosterLLaVA (IEEE TMM 2026) | CV/图形学顶会 | 现有模型全在 sRGB 屏幕域优化，无 CMYK 色域惩罚损失、无出血/模切安全区掩码约束、无图层化印前导出；生成图在视觉上好看，却无法直接制版 | 有：可持续城市设计双层评测（Jung 2026 *Sustainability*）与可微色彩空间投影（sRGB→CMYK soft-gamut loss） |

---

## 候选课题（全部定位为深度学习/AI 方法论文，纯客观指标 A 套）

| 候选 | 设定（输入 → 输出） | 要解决的具体技术难点或 failure case | 来源 |
| --- | --- | --- | --- |
| **DL-1：刀版拓扑感知的多面一致包装扩散模型（DieLineDiff）** | 输入：参数化 2D 刀版拓扑（面掩码、共边邻接表、旋转角、出血/安全区）+ 品类/文化提示 + 品牌参考图 → 输出：跨折线无缝连续、无光照 2D 印刷展开图 + 精确对应的 3D 折叠多视角图 | **T3（刀版与 3D 多面对应断裂、文字倒置）+ T2（跨面风格/色板一致性）+ T5（出血与安全区越界）** | 难点—解法树 + 新锤子（迁移 MV-Adapter / Paint3D / 跨缝特征对齐） |
| **DL-2：双轨图文解耦与印前引导的中文农产品包装生成框架（PackLayText）** | 输入：包装品类与文化文本 + 中文艺术标题 + 结构化法定标签 JSON（配料/营养/条码）+ 品牌色板 → 输出：高准中文艺术标题前景 + 零错矢量法规层 + 避让显著性主体且符合 CMYK 色域约束的完整分层包装图 | **T1（中文密集小字乱码）+ T4（类别与风格解耦）+ T5（可印刷 CMYK 色域与无重叠合规版式）** | failure case + 工程问题转科学问题（改进 PosterMaker / LayerDiffusion） |
| **DL-3：跨 SKU 品牌一致性与类别-文化解耦的可控包装生成网络（BrandPack-Adapter）** | 输入：1 张品牌核心资产（Logo + 3–5 主色 + 文化参考图）+ N 个不同产品类别/规格条件 → 输出：一套跨 SKU 风格/Logo/色板高度锁定且品类特征清晰的系列化包装图 | **T2（跨 SKU 品牌漂移与语义泄露）+ T4（按产品类别与文化解耦控制）** | 用户需求 + failure case（改进 IP-Adapter / InstantStyle） |
| **DL-4：基于可微 CMYK 色域损失与多目标奖励对齐的可印刷包装扩散微调（PrintAlign-Diff）** | 输入：预训练扩散模型 + 公开食品包装库 → 输出：经 CMYK 可微色域惩罚、OCR 正确率奖励与版式规则奖励对齐后的包装专用扩散权重 | **T1 + T5（用 Flow-GRPO / 奖励微调解决 RGB 色域溢出与排版违规）** | 新锤子（迁移 Flow-GRPO NeurIPS 2025 / ARIA-Diff *Mathematics* 2026） |
| **DL-5：传统“感性工学 + 商用 Midjourney + 问卷/AHP”包装研究** | 输入：调研问卷 + Midjourney 提示词 → 输出：概念效果图 | 无深度学习方法创新且依赖人工被试 | 对照排除项 |

---

## 按四种情况判断

四种情况的定义（引自彭思达《科研选题》课件，2025-08-12，[PDF](https://pengsida.net/files/research_topic_selection.pdf)）：
1. “同样输入输出的任务，已经有了不错的解决方案，只是有些地方做得还不够好。”
2. “输入或输出有所变化的任务，已经有了不错的解决方案。或者在多个完全不同 data domain、技术问题内核相同的任务中，都有了不错的类似解决方案。”
3. “只有少数一两个完全不同 data domain、技术问题内核相同的任务，有不错的解决方案。”
4. “在各个不同领域的任务中，虽然有类似的技术问题，但都没有较好的解决方案。”

> 注：彭思达课件只指出“不同问题的创新空间区别很大”，并未给四种情况排序；下表判断与优先级由本研究结合“纯客观指标 A 套 + 单卡 RTX 5090 + 零付费数据 + 目标 OA SCIE”独立作出。

| 候选 | 属于哪种情况 | 理由 | 与已有工作的区别类型 | 初步结论 |
| --- | --- | --- | --- | --- |
| **DL-1：刀版拓扑感知的多面一致包装扩散模型（DieLineDiff）** | **情况 3** | 技术内核（“已知刚性拓扑邻接关系下的跨区域无缝一致扩散 + 去光照 UV 生成”）目前仅在少数远域（通用 3D 网格 Paint3D/MV-Adapter、全景图无缝拼接）有解法；引入工业折叠包装的精确刀版邻接映射、90°/180° 折面朝向约束与出血/安全区掩码后，问题几何结构清晰且能用**纯客观 3D 重投影 SSIM + 跨折线缝隙 PSNR + 越界率**铁证胜出，完全不需人工评价！ | **克服现有方案局限 + 换一个角度看问题（从 2D 平面图升级为刀版拓扑感知多面生成）+ 更有效的方法** | **最优先（方法创新最强、最抗审稿、最适合纯客观指标）** |
| **DL-2：双轨图文解耦与印前引导的中文包装生成框架（PackLayText）** | **情况 2**（部分情况 3） | 商品海报文字+场景解耦在 PosterMaker (CVPR 2025) 和 CreatiPoster (2025) 已有强先例，迁移至中文农产品包装并增加“密集小字矢量层保护 + 刀版安全区 + CMYK 可微色域约束”，新约束清晰且所有增益（OCR CER、CMYK 溢出率、Overlap、GB 7718 规则通过率）均可 100% 客观定量验证 | **克服现有方案局限 + 满足新场景硬约束** | **优先（工程落地稳妥，可与 DL-1 模块化组合或独立成篇）** |
| **DL-3：跨 SKU 品牌一致与类别-文化解耦适配器（BrandPack-Adapter）** | **情况 2** | 风格/主体解耦适配器在 InstantStyle/IP-Adapter/AnyDoor 已有较成熟路线，迁移至多 SKU 包装系列并引入色板 $\Delta E_{00}$ 与 Logo 保持约束，区分度中等偏稳，但方法新意略逊于 DL-1 | **改善可控性与跨 SKU 一致性** | **备选（可作为 DL-1 的“大胆尝试”扩展分支）** |
| **DL-4：多目标客观奖励对齐扩散微调（PrintAlign-Diff）** | **情况 2** | 扩散在线奖励微调（Flow-GRPO / DPO / ARIA-Diff）在通用生图与室内设计已热，用于包装 CMYK/OCR 奖励对齐有一定新意，但训练稳定性与奖励 hacking（为了压低 CMYK 溢出导致画面灰暗）调试周期较长 | **更有效的对齐训练方法** | **备选（作后续或消融中的增强模块，不作首篇唯一押注）** |
| **DL-5：感性工学 + Midjourney + 问卷/AHP** | **情况 1** | 已被 2025–2026 多篇 SCI/SSCI 充分解决，且违背“深度学习方法 + 零被试”两条作者硬约束 | 无实质方法区别 | **坚决换题（淘汰）** |

---

## 未核实事项与下一步

### 未核实事项
1. **未核对的 2 篇种子文献**：Tao Chen（Semarak Ilmu 标题）与 SAUNIC 2026（寨子乡论文）在 OpenAlex / Crossref / 公开网页均未定位独立 DOI 或官方出版页，已标 `verified=no`。
2. **公开数据集下载实测**：`docs/DATASETS.md` 中 14 个公开数据集的规模与许可均来自官方仓库/HuggingFace/论文页公开说明，实际拉取（尤其是国内网络访问 HuggingFace / AWS S3 下载速度及 PKU PosterLayout 协议审批耗时）需在开工第一周实测确认。
3. **空白判断边界**：关于“尚未见将折叠纸盒参数化刀版邻接拓扑显式编入扩散模型跨面注意力的 SCIE/顶会论文”的结论基于 2026-11-24 在 OpenAlex / Crossref / arXiv 的 34 组检索，不含尚未公开的在审预印本。

### 下一步
- [ ] 阅读 [`docs/DATASETS.md`](../DATASETS.md)，确定首阶段下载的公开数据集子集（推荐优先拉取 **Open Food Facts + RPC + Crello + AnyWord-3M 小子集**，零申请直下）并搭建 Blender 参数化折叠盒型合成脚本。
- [ ] 阅读 [`docs/INNOVATION.md`](../INNOVATION.md)，确认首选深度学习方法切入点（推荐 **DL-1 刀版拓扑感知多面一致扩散**，或稳健版 **DL-2 双轨中文图文与印前约束生成**）并立刻启动 1 周最小验证实验。
- [ ] 阅读 [`docs/JOURNALS.md`](../JOURNALS.md)，按单位对 MDPI / IEEE Access / 综合 SCIE 刊的认定口径敲定 1–2 本目标纯 OA SCIE 期刊。
