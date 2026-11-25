# 文献检索记录（Search Log）

- 课题：**用人工智能（深度学习等）做包装设计（乡村农产品包装为应用场景，OPC/方案交付为可选框架）**
- 检索执行者：Gemini
- 检索日期：2026-11-24 至 2026-11-25（含初版 Q01–Q25、修订版 Q26–Q34、PR #1 返修 Q35–Q37，以及任务 06 方法深研 v3 专项检索 Q38–Q41）
- 数据来源约束：仅使用免费开放学术来源（OpenAlex API、Crossref API、arXiv、GitHub/HuggingFace 数据集公开页、出版社公开页、公开网页检索含知网/万方免费摘要线索），未使用 Scopus 或 Web of Science 付费权限。

> **核心结论前置**：转向**“用深度学习等 AI 方法做包装设计”**且**只用纯客观指标（A 套，不招募专家/消费者）**后，本课题**完全具备发表纯开放获取（OA）SCIE 期刊论文的可行性**，甚至比纯设计实践类论文更容易通过客观定量对比与消融实验满足 SCIE 工程/计算机综合刊的审稿门槛。核心方法贡献应锚定在**深度学习包装设计尚未解决好的关键技术难点**（尤其是“**参数化刀版拓扑感知的多面一致生成 + 视觉艺术字/矢量法定标签双轨解耦 + 可印刷性/合规性客观引导**”），配套自建可开源合成刀版—3D 配对基准与 Open Food Facts/RPC/AnyWord-3M 公开数据，首选 OA SCIE 靶刊为 ***Applied Sciences***、***Scientific Reports***、***PLOS ONE***、***IEEE Access***、***Electronics*** 与 ***Symmetry***。

---

## 一、系统检索记录表（中英文，共 41 组）

### 第一阶段：初始四脉络与种子文献检索（Q01–Q25）

| 编号 | 检索日期 | 数据库 / 来源 | 查询词 / 检索式 | 命中总数 | 筛选后保留数 | 覆盖脉络 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q01 | 2026-11-24 | OpenAlex API | `title_and_abstract.search:("graphic design" AND "layout generation")` | 62 | 4 | 脉络一：深度学习平面与包装设计 | 核实 LayoutGAN (TPAMI)、Content-aware Layout (TOG)、NDN 等 |
| Q02 | 2026-11-24 | OpenAlex API | `title_and_abstract.search:("layout generation" AND diffusion)` | 159 | 3 | 脉络一：深度学习平面与包装设计 | 核实 LayoutDM (CVPR 2023)、LayoutDiffusion (ICCV 2023) |
| Q03 | 2026-11-24 | OpenAlex API | `title_and_abstract.search:("visual text rendering" OR ("text rendering" AND diffusion))` | 112 | 5 | 脉络一：深度学习文字渲染 | 核实 AnyText (ICLR 2024)、Glyph-ByT5 (ECCV 2024)、Glyph-ByT5-v2、TextDiffuser-2、FonTS (ICCV 2025) |
| Q04 | 2026-11-24 | OpenAlex API | `title_and_abstract.search:(("graphic design" OR "poster design") AND ("large language model" OR "multi-agent" OR agentic))` | 226 | 5 | 脉络一：多层/LLM 设计生成 | 核实 COLE、OpenCOLE、PosterLLaVA (IEEE TMM 2026)、CreatiPoster (2025)、OptiMuse (CHI 2024) |
| Q05 | 2026-11-24 | OpenAlex API | `title_and_abstract.search:("packaging design" AND ("generative AI" OR AIGC OR Midjourney OR "Stable Diffusion" OR "diffusion model"))` | 77 | 6 | 脉络一 & 二：AIGC 包装设计 | 核实 *J. Eng. Des.* (2025) 茶包装、*HSSC* (2026) 马奶酒包装、*EAAI* (2026) 化妆品包装 |
| Q06 | 2026-11-24 | OpenAlex API | `title_and_abstract.search:(("vector graphics" OR SVG) AND generation AND (diffusion OR LLM))` | 176 | 3 | 脉络一：矢量与可编辑生成 | 核实 CanvasVAE (ICCV 2021)、VectorFusion (CVPR 2023)、StarVector (CVPR 2025) |
| Q07 | 2026-11-24 | Crossref API | `folding carton dieline structural packaging CAD automated design` | 4,157,442 | 2 | 脉络一：包装刀版与结构 | 未见“深度学习扩散模型 + 折叠纸盒参数化刀版多面拓扑一致生成”的顶会/SCIE 成熟方案 |
| Q08 | 2026-11-24 | Crossref API | `prepress printability packaging design automated workflow` | 3,052,120 | 2 | 脉络一 & 四：可印刷性约束 | 印前 CMYK 色域/出血/300DPI 约束尚未融入深度学习扩散生成损失与评测体系 |
| Q09 | 2026-11-24 | OpenAlex API | `title_and_abstract.search:("agricultural" AND "packaging design" AND "rural revitalization")` | 45+ | 5 | 脉络二：农产品包装与乡村振兴 | 多数发表在普通会议或低质期刊，印证 H4 |
| Q10 | 2026-11-24 | OpenAlex API | `title_and_abstract.search:("agricultural" AND "packaging design" AND ("artificial intelligence" OR AIGC OR "deep learning" OR "generative AI"))` | 30+ | 4 | 脉络二：AI + 农产品包装 | 核实 Songyong Hu (2024, AMNS)、Global NEST (2023) 等 |
| Q11 | 2026-11-24 | OpenAlex API | `title_and_abstract.search:("packaging design" AND "agricultural" AND ("purchase intention" OR "consumer" OR "sustainable"))` | 80+ | 6 | 脉络二：农产品包装背景 | 核实 Rundh (2009)、Silayoi & Speece (2007)、Boz et al. (2020)、Tang et al. (2025)、Duan et al. (2024) |
| Q12 | 2026-11-24 | OpenAlex API | `title_and_abstract.search:("GB 7718" OR ("prepackaged food" AND labeling AND China))` | 11 | 3 | 脉络二 & 四：食品法规客观规则 | 核实 2026 年中文新作（GB 7718-2025/28050-2025 合规审核）与 *Food Control* 2024 |
| Q13 | 2026-11-24 | OpenAlex API | `title_and_abstract.search:(农产品 AND 包装设计 AND (AIGC OR 人工智能 OR 生成式))` | 18 | 2 | 脉络二：中文检索 | 定位 2026 年地域农产品包装中文研究 |
| Q14 | 2026-11-24 | 公开网页检索（知网/万方线索） | `AIGC 农产品 包装设计 乡村振兴 论文 知网 万方 2024 2025 2026` | 网页聚合 | 2（仅作现状佐证） | 脉络二：中文核心动态 | 确认中文刊物 2024–2026 以提示词出图+地域文化案例为主，鲜见深度学习架构级创新 |
| Q15 | 2026-11-24 | OpenAlex API | `title_and_abstract.search:("one-person company" AND (AI OR "artificial intelligence" OR agent OR digital))` | 30 | 6 | 脉络三（可选）：OPC 术语 | 核实 *Systems* (2026)、IEEE *IT Professional* (2026)、*SEJ* (2026) 均正式使用 OPC |
| Q16 | 2026-11-24 | 公开网页 + arXiv | `"OPC" "一人公司" AI 智能体 学术论文 OR 研究 2025 2026` | 网页聚合 | 2 | 脉络三（可选）：OPC | 核实 `arXiv:2511.09533` (*Digital Co-Founders*) |
| Q17 | 2026-11-24 | OpenAlex API | `title_and_abstract.search:((solopreneur OR "solo entrepreneur" OR "micro-enterprise") AND ("generative AI" OR "AI agent"))` | 382 | 3 | 脉络三（可选） | 核实 *J. Rural Studies* (2019)、*Science* (2023)、*Org. Sci.* (2026) |
| Q18 | 2026-11-24 | OpenAlex API | `title_and_abstract.search:(("human-AI") AND ("co-creation" OR collaboration) AND ("graphic design" OR "product design"))` | 103 | 6 | 脉络三（可选） | 核实 CHI 2024/2025 与 *J. Eng. Des.* 2026 多智能体产品设计 |
| Q19 | 2026-11-24 | OpenAlex API | `title_and_abstract.search:("generative AI" AND ("graphic designers" OR "design practice" OR "creative work"))` | 685 | 4 | 脉络三（可选） | 核实 *Scientific Reports* (2026) 手工艺人采纳研究、WORKBank (2025) |
| Q20 | 2026-11-24 | OpenAlex API | `title_and_abstract.search:("design science research" AND evaluation AND framework)` | 2,535 | 3 | 脉络四：评测框架 | 核实 Hevner (2004)、Peffers (2007)、Venable FEDS (2016) |
| Q21 | 2026-11-24 | OpenAlex API | `title_and_abstract.search:("packaging design" AND (Kansei OR AHP OR TOPSIS))` | 207 | 5 | 脉络四：传统主观评测对比 | 核实 *PTS* (2024)、*BioResources* (2025)、*Heliyon* (2024)、*Scientific Reports* (2025) |
| Q22 | 2026-11-24 | OpenAlex API（精准核验） | 35 篇顶会里程碑标题核验（LDM/ControlNet/LayoutDM/LayoutGPT/Design2Code/CLIPScore 等） | 35 | 32 | 全脉络 | 全部核对官方 DOI 与摘要 |
| Q23 | 2026-11-24 | Crossref API（ISSN 定向） | 11 本候选 OA 期刊 2024–2026 相似文 | >18,000 | 18 | 期刊调研 | 核实 *Applied Sciences*、*Sustainability*、*PLOS ONE*、*Scientific Reports*、*Heliyon* 等 2024–2026 真实 DOI |
| Q24 | 2026-11-24 | 公开网页（国标公告） | `"GB 7718-2025" "GB 28050-2025" 预包装食品标签通则 实施日期 修订变化` | 网页聚合 | 2 项强制国标 | 脉络四：确定性法规指标 | 确认 2027-03-16 正式强制实施，可转化为纯客观零被试合规审计指标 |
| Q25 | 2026-11-24 | 公开网页 | Clarivate MJL / 中科院 2025 分区与预警名单 / AMNS 与 JISEM 收录状态 | 网页聚合 | 13 刊核实 | 期刊调研 | 确认 2025 预警名单仅 5 刊；AMNS 与 JISEM 已被 Scopus 剔除 |

### 第二阶段：按修订版任务书追加的深度学习技术难点、纯客观指标、公开数据集与 DL 期刊检索（Q26–Q34）

| 编号 | 检索日期 | 数据库 / 来源 | 查询词 / 检索式 | 命中总数 | 筛选后保留数 | 覆盖脉络 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q26 | 2026-11-24 | Crossref API + OpenAlex API | 深度学习中文文字渲染与电商海报顶会里程碑核验：`TextDiffuser`, `TextDiffuser-2`, `AnyText2`, `UDiffText`, `GlyphControl`, `CGL-GAN`, `PosterLayout (DS-GAN)`, `RADM`, `PosterMaker (CVPR 2025)`, `LayerDiffusion (SIGGRAPH 2024)` | 10 | 10 | 脉络一主线（技术难点 T1/T4/T5） | 全部核实正式 DOI；`PosterMaker` (CVPR 2025) 与 `AnyText2` 为商品海报中文文字强基线 |
| Q27 | 2026-11-24 | Crossref API + OpenAlex API | 品牌主体/风格一致性与 3D 贴图/多视角顶会里程碑核验：`LoRA`, `DreamBooth`, `IP-Adapter`, `InstantStyle`, `AnyDoor`, `SD3 (Rectified Flow)`, `TEXTure (SIGGRAPH 2023)`, `Paint3D (CVPR 2024)`, `MV-Adapter (ICCV 2025)`, `Diffusion-4K (CVPR 2025)` | 10 | 10 | 脉络一主线（技术难点 T2/T3/T5） | 全部核实正式 DOI；确认 3D 无光照 UV 贴图（Paint3D）与多视角适配器（MV-Adapter）尚未迁移至折叠纸盒刀版拓扑 |
| Q28 | 2026-11-24 | Crossref API + OpenAlex API | 纯客观无人工生成质量与偏好代理指标核验：`FID (Heusel 2017)`, `LPIPS (CVPR 2018)`, `ImageReward (NeurIPS 2023)`, `PickScore (NeurIPS 2023)`, `HPS v2`, `VQAScore (ECCV 2024)`, `DesignCritic (IEEE Access 2026)` | 7 | 7 | 脉络四（纯客观评测 A 套） | 构建完整零被试深度学习方法客观评测体系，*IEEE Access* 2026 *DesignCritic* 进一步证明纯客观自动设计审计可发 SCIE |
| Q29 | 2026-11-24 | Crossref API + 公开网页（HF/GitHub/AWS） | 公开商品与包装数据集核查：`Open Food Facts`, `ABO (CVPR 2022)`, `RPC (SCIS 2022)`, `SKU-110K (CVPR 2019)`, `Grocery Store Dataset (WACV 2019)`, `M5Product (CVPR 2022)`, `Product1M (ICCV 2021)` | 7 | 7 | 数据集专项（`docs/DATASETS.md`） | 全部核实规模、License（ODbL/CC-BY-SA、CC-BY 4.0、CC-BY-NC-SA 4.0 等）、下载方式与是否需申请 |
| Q30 | 2026-11-24 | 公开网页（HF/GitHub/论文页） | 公开版式、中文文字与矢量数据集核查：`Crello`, `PKU PosterLayout`, `CGL-Dataset`, `CreatiPoster 100K`, `AnyWord-3M`, `MARIO-10M`, `SVG-Stack` | 7 | 7 | 数据集专项（`docs/DATASETS.md`） | 全部核实 License（CDLA-Permissive-2.0、Apache-2.0、CC-BY-SA 4.0 等）与免费下载路径 |
| Q31 | 2026-11-24 | 公开网页（法规与判例分析） | `学术研究 爬取 电商 商品 包装图片 训练 AI 深度学习 著作权法 第二十四条 合理使用 版权风险 规避` | 网页聚合 | 完整合规评估 | 数据集专项（自采风险） | 明确《著作权法》第二十四条三步检验法、商标权/外观专利复合风险与反不正当竞争边界 |
| Q32 | 2026-11-24 | Crossref API（ISSN 定向） | 深度学习/CV 友好纯 OA SCIE 期刊 2024–2026 同类文：`IEEE Access` (`2169-3536`), `Electronics` (`2079-9292`), `Sensors` (`1424-8220`), `Symmetry` (`2073-8994`), `Mathematics` (`2227-7390`) | 5,278 | 10 | 期刊调研（DL 方法靶刊） | 检出 *IEEE Access* 2026 饼干包装 AI 设计 (`10.1109/access.2026.3723820`)、*DesignCritic* (`10.1109/access.2026.3730226`)、*Symmetry* 2025 包装设计 (`10.3390/sym18010019`)、*Mathematics* 2026 扩散设计奖励对齐 (`10.3390/math14203759`) |
| Q33 | 2026-11-24 | 公开网页 | `"IEEE Access" "Electronics" MDPI "Sensors" MDPI "Symmetry" MDPI "Mathematics" MDPI SCIE impact factor 2025 中科院分区 预警名单 APC 审稿周期` | 网页聚合 | 5 刊指标核实 | 期刊调研 | 核实 5 刊均不在 2025 中科院预警名单（注明 IEEE Access/Sensors/Symmetry 2020–2021 历史预警与移出记录） |
| Q34 | 2026-11-24 | 中文关键词复查 | `深度学习 包装设计 扩散模型 版式生成 中文文字渲染 公开数据集`（结合 OpenAlex & 公开网页） | 网页聚合 | 复核空白结论 | 脉络一与空白复核 | 确认中文研究仍集中在 Midjourney/SD 提示工程与 LoRA 风格微调，尚无“刀版印刷硬约束 + 中文双轨文字 + 纯客观指标基准”的系统性深度学习方法论文 |
| Q35 | 2026-11-24 | Direct HTTP + OpenAlex API + arXiv 官方页 | PR #1 审查返修专项：逐条核对原 `papers.csv` 中全部 44 条 `snippet` 条目及 5 篇被标疑 2026 DOI（`10.1057/s41599-026-08755-y`、`10.1057/s41599-026-06948-z`、`10.1109/access.2026.3723820`、`10.1109/access.2026.3730226`、`10.3390/math14203759`） | 44 | 41 升级为 `abstract`；3 篇付费墙 Elsevier 改标 `verified=no` | 文献质量审计 | 5 篇被标疑 2026 论文全部确认官方 DOI 有效并获取 OpenAlex 官方摘要；`papers.csv` 彻底消除 `verified=yes` 且 `summary_source=snippet` 的违规记录 |
| Q36 | 2026-11-24 | Crossref API + arXiv 官方页 | PR #1 审查返修专项：补全 3D 网格跨视角/跨接缝纹理生成脉络：`Text2Tex` (ICCV 2023)、`TexFusion` (ICCV 2023)、`Point-UV Diffusion` (ICCV 2023)、`SyncMVD` (SIGGRAPH Asia 2024)、`Meta 3D TextureGen` (2024)、`RoCoTex` (2024)、`MVPaint` (CVPR 2025) | 7 | 7（全部 `verified=yes, abstract`） | 脉络一子线 1B | 全部打开 arXiv 官方摘要页并核对正式会议 DOI，据此将 `DieLineDiff` 从情况 3 回调为情况 2 并新增纸盒网格 3D 纹理基线 |
| Q37 | 2026-11-24 | Direct HTTP 核验 | 核实期刊风险出处链接：`ewl.fenqubiao.com`、南京财经大学/甘肃工职大/福建理工 2020–2025 预警公告、`retractionwatch.com` 2024-09-30 & 2026-08-07 *Heliyon* On Hold 报道 | 6 页 | 5 页返回 HTTP 200 | 期刊调研风险出处 | 修正 *IEEE Access* 预警史为“2020 年中档、2021 年移出”；确认 *Heliyon* On Hold 于 2026-08-07 报道解除；全部风险项补齐可点击出处 |
| Q38 | 2026-11-25 | arXiv 官方页 + OpenAlex API | 任务 06（方法 v3 深研）：多窗扩散同步与 360° 无缝/环形填充脉络：`MultiDiffusion` (ICML 2023)、`Mixture of Diffusers` (2023)、`SyncDiffusion` (NeurIPS 2023)、`Diffusion360` (2023)、`PanFusion` (CVPR 2024)、`StitchDiffusion` (WACV 2024)、`SpotDiffusion` (WACV 2025)、`Tiled Diffusion` (CVPR 2025) | 8 | 8（全部 `verified=yes, abstract`） | 脉络一（v3 环带闭环与去滑窗路线） | 全部核对 arXiv 官方摘要与 OpenAlex 正式会议 DOI，证实 UNet/VAE 水平环形卷积与时间交错窗可替代高代价 MultiDiffusion/SyncDiffusion |
| Q39 | 2026-11-25 | arXiv 官方页 + OpenAlex API | 任务 06（方法 v3 深研）：跨视角/六面共享自注意力与区域注意力路由：`MVDiffusion` (NeurIPS 2023)、`CubeDiff` (ICLR 2025)、`StyleAligned` (CVPR 2024)、`ConsiStory` (SIGGRAPH 2024)、`StoryDiffusion` (NeurIPS 2024)、`MasaCtrl` (ICCV 2023)、`DenseDiffusion` (ICCV 2023)、`TexPainter` (SIGGRAPH 2024) | 8 | 8（全部 `verified=yes, abstract`） | 脉络一（v3 跨面一致性与分面提示路由） | 全部核对官方摘要与 DOI，确立“水平环形卷积环带 + 跨面共享自注意力盖面 + 空间交叉注意力路由”的免训练核心架构 |
| Q40 | 2026-11-25 | arXiv 官方页 + OpenAlex API | 任务 06（方法 v3 深研）：全局-局部软修补与高分辨率一致：`SDEdit` (ICLR 2022)、`RePaint` (CVPR 2022)、`Blended Latent Diffusion` (SIGGRAPH 2023)、`Differential Diffusion` (CGF 2025 / arXiv 2023)、`DemoFusion` (CVPR 2024) | 5 | 5（全部 `verified=yes, abstract`） | 脉络一（v3 局部 3D 接缝软修补） | 全部核对官方摘要与 DOI，支撑方案 P1 与 P2+P1 盖面 6 条非原生 3D 缝的逐像素软强度局部缝合 |
| Q41 | 2026-11-25 | OpenAlex API | 任务 06（方法 v3 深研）：可微无缝拼接客观评测指标：`TexTile: A Differentiable Metric for Texture Tileability` (CVPR 2024) | 1 | 1（`verified=yes, abstract`） | 脉络四（v3 补充接缝客观指标） | 核对官方摘要与 CVPR 2024 DOI，为超越传统像素色差 SDR 的跨折线结构连贯性客观评测提供顶会依据 |

---

## 二、7 篇种子文献逐篇核查与引文追踪记录

| 序号 | 任务书提供的种子信息 | 核查状态 | 核实后的完整题录与链接 | 发表处质量评估 | 引文追踪（前向/后向） |
| --- | --- | --- | --- | --- | --- |
| 1 | Tao Chen, *The Application of Artificial Intelligence-Assisted Technology in Packaging Design of Agricultural Products*（Semarak Ilmu 旗下期刊） | **未核对**（`verified=no`） | 在 OpenAlex、Crossref 与公开网页检索中均未检索到该精确标题的独立 DOI；同一作者组（Tao Chen & Mohamed Razeef Abdul Razak，马来西亚 UiTM）已核实的相近论文为 `10.55267/iadt.07.15214`（JISEM 2024）与 `10.28978/nesciences.1567827`（*Natural and Engineering Sciences* 2024） | Semarak Ilmu 旗下期刊多为马来西亚普通非 SCI 期刊，**弱来源** | 无法追踪 |
| 2 | *A Framework for AI-driven Rural Revitalization Strategies: Balancing Brand Image, Cultural Compliance and Consumer Behavior Focusing on Agri Products Packaging Designs*, `doi:10.55267/iadt.07.15214` | **已核对**（OpenAlex 摘要与元数据） | Chen Tao, Mohamed Razeef Abdul Razak, Yuyang Xia, Peng Mingqian (2024). *Journal of Information Systems Engineering & Management* (ISSN 2468-4376). [https://doi.org/10.55267/iadt.07.15214](https://doi.org/10.55267/iadt.07.15214) | **极弱来源 / 高风险期刊**：非 SCI/SSCI，且已被 Scopus 剔除（Discontinued，收录止于 2024）；纯定性访谈综述，不指导方法设计 | 引用 46 篇；被引 6 次 |
| 3 | Songyong Hu (2024), *Exploration of Agricultural Product Packaging Design Innovation in the Context of Deep Learning*, *Applied Mathematics and Nonlinear Sciences* 9(1) | **已核对**（OpenAlex 摘要与元数据） | Songyong Hu (2024). *Applied Mathematics and Nonlinear Sciences* 9(1). [https://doi.org/10.2478/amns-2024-1882](https://doi.org/10.2478/amns-2024-1882) | **极弱来源 / 口碑差期刊**：非 SCI/SSCI，已被 Scopus 剔除（Discontinued）；题为“Deep Learning”实则仅做传统 HSV 颜色与灰度共生矩阵浅层特征分析，反面印证深度学习做包装设计亟需真正的现代生成架构贡献 | 被引 3 次 |
| 4 | *Generative AI-Driven Optimization for Cultural Packaging Design: Translating Chinese Poetic Imagery into Tea Packaging Design*, AHFE 2025, `doi:10.54941/ahfe1006930` | **已核对**（Crossref + OpenAlex 摘要） | Zixiao Chen, RongRong Fu (2025). *AHFE International* Vol. 199. [https://doi.org/10.54941/ahfe1006930](https://doi.org/10.54941/ahfe1006930) | **普通国际会议论文集**，典型“ChatGPT+Midjourney + AHP/FCE”非算法应用范式 | 被引 1 次 |
| 5 | *Sustainable Application and Evaluation of the Novel Stingray Model in Non-Heritage Packaging: The Case of Clay Sculptures in Joon County*, *Applied Sciences* 2025, 15(11), 6033, `doi:10.3390/app15116033` | **已核对**（Crossref + OpenAlex 摘要 + 公开摘要页） | Qichao Song, Zhaoyi Bai (2025). *Applied Sciences*, 15(11), 6033. [https://doi.org/10.3390/app15116033](https://doi.org/10.3390/app15116033) | **SCIE 期刊（MDPI）**：依赖 100 人问卷 + AHP/TOPSIS + 商用 Midjourney，**非深度学习方法改进且高度依赖人工被试**，与作者修订后的“深度学习方法 + 零被试纯客观指标”形成鲜明对比 | 引用 67–70 篇；被引 6 次（追踪到 *Design Science* 2026、*Scientific Reports* 2026、*BioResources* 2025） |
| 6 | *Humanity in the Loop: Ethical AI-Enabled Packaging Design for Rural Agricultural Revitalization in Zhaizixiang, China*（SAUNIC 2026 会议论文） | **未核对**（`verified=no`） | OpenAlex、Crossref 与公开网页检索均未找到正式出版页或 DOI | 未知（未核实，不作为研究设计依据） | 无法追踪 |
| 7 | *An AIGC-Driven Teaching Model for Packaging Design Courses: Design and Experimental Study*（ADDT 2026, `doi:10.2991/978-94-6239-737-8_27`） | **已核对**（OpenAlex 摘要与元数据） | Shengnan Wu, Jiahui Wang, Na Wang, Weiqian Mao (2026). *Advances in Computer Science Research* (Atlantis Press). [https://doi.org/10.2991/978-94-6239-737-8_27](https://doi.org/10.2991/978-94-6239-737-8_27) | **普通会议论文集**（60 名学生教学实验），非深度学习方法论研究 | 引用 3 篇；被引 0 次 |
