# 项目进度（交接记录）

> 每个 AI 结束工作前在最上方追加一条记录，新的在上面。

## 当前状态

任务 18 已完成 RTX 3090 实验。五组均未过硬筛选，未锁定复验配置。原 E0 关口参照。种子 42 4/5，条带 v1 1.7209（上限 1.2528），未过条带 v1 与原像素。种子 1234 5/5，条带 v1 1.3026（上限 1.3832）。停在 Step 1，等 Claude 审查。 [盖面能量匹配双种子报告](experiments/E02/CAP-ENERGY.md)

任务 19 的独立研究见待办。

## 待办

- [x] Codex：任务 02，见 `tasks/02-codex-dielinediff-s0-s1.md`（S0 + S1，做完停下等审查）
- [x] Claude：审查任务 02 的 PR #2，通过并合并（2026-11-25），审查意见见 PR 留言
- [x] Codex：任务 03，见 `tasks/03-codex-e01b-prototype-fix.md`（E01b，做完停下等审查）
- [x] Claude：审查任务 03 的 PR #3，通过并合并（2026-11-25），审查意见见 PR 留言
- [x] Codex：任务 04，见 `tasks/04-codex-e01b-crease-fix.md`（做完停下等审查）
- [x] Claude：审查任务 04 的 PR #4，通过并合并（2026-11-25），审查意见见 PR 留言
- [x] Codex：任务 05，见 `tasks/05-codex-e01b-bottom-frame.md`（做完停下等审查）
- [x] Claude：审查任务 05 的 PR #6，通过并合并（2026-11-25）
- [x] Gemini：按 PR #5 中 Claude 第一轮 6 项意见修订 `docs/METHOD_V3_PROPOSALS.md` 并逐条回复（任务 06，分支 `gemini/method-v3-research`）
- [x] Claude：复审并合并 PR #5，完成 docs/METHOD_V3_CONSENSUS.md
- [x] Codex：完成任务 07 Step 0 和 Step 1（见 E02 报告，关口未过）
- [x] Claude：审查任务 07 的 PR #7，通过并合并（2026-11-25）
- [x] Codex：任务 08，见 `tasks/08-codex-v3-cap-fix.md`（盖面修复，已完成，关口未过）
- [x] Gemini：任务 09，见 `tasks/09-gemini-v3-cap-review.md`（已在 PR #7 留言审阅并补录 C5 相关 4 篇文献）
- [x] Claude：审查任务 08 的 PR #9 与任务 09 的 PR #8，均已合并（2026-11-25）
- [x] Codex：任务 10，见 [盖面几何与调和初始化报告](experiments/E02/CAPGEOM.md)
- [x] Claude：审查任务 10 的 PR #10，通过并合并（2026-11-25）
- [x] Gemini：任务 11，见 `tasks/11-gemini-v3-g2-stripe-discussion.md`（已在 PR #10 留言逐条回应 5 项立场）
- [x] Codex：任务 12，已完成 [CPU 条带诊断](experiments/E02/STRIPE-DIAG.md)
- [x] Claude：汇总任务 11、12，修正立场（暂不引入条带 v2），合并 PR #11、#12（2026-11-25）
- [x] Codex：任务 13，见 [盖面颜色与纹理延续报告](experiments/E02/CAPTEXTURE.md)
- [x] Gemini：任务 14，见 `tasks/14-gemini-respond-revised-stance.md`（已在 PR #12 留言回应立场修正并审阅任务 13 设计）
- [x] Claude：审查任务 13 的 PR #14 与任务 14 的 PR #13，均已合并（2026-11-26）
- [x] Codex：完成任务 15。[盖面均匀化与环带横幅报告](experiments/E02/UNIFORM-RING.md)
- [x] Gemini：任务 16，见 `tasks/16-gemini-beat-all-baselines.md`（已提交 `docs/BEAT_BASELINES_GEMINI.md`）
- [x] Claude：审查并合并 PR #16、#15，形成任务 17（2026-11-26）
- [x] Codex：完成任务 17。[两种子环带与盖面报告](experiments/E02/RING-M2M3.md)
- [x] Claude：审查并合并 PR #17（2026-11-26）
- [x] Codex：完成任务 18。[盖面能量匹配双种子报告](experiments/E02/CAP-ENERGY.md)
- [x] Gemini：任务 19，见 `tasks/19-gemini-cap-interior-ideas.md`（已提交 `docs/CAP_INTERIOR_GEMINI.md` 并补录 4 篇核对文献，分支 `gemini/cap-interior-ideas`）
- [x] Claude：审查并合并 PR #19（任务 18）、PR #18（任务 19，解决冲突），形成任务 20、21（2026-11-26）
- [ ] Codex：任务 20，见 `tasks/20-codex-joint-cap.md`（做完停下等审查）
- [ ] Gemini：任务 21，见 `tasks/21-gemini-c3-highres.md`
- [x] Claude：审查任务 01 的 PR #1（2026-11-24，退回）
- [x] Gemini：按 PR #1 审查意见完成 4 项必改与 2 项建议修改，并已在 PR #1 逐条回复
- [x] Claude：复审 PR #1，通过并合并（2026-11-24；顺手更正 Heliyon 解除 On Hold 日期为 2026-05-14）
- [x] 作者：已决定不招募专家或消费者，只用评价方案 A（2026-11-24）
- [x] 作者：指示在 AutoDL RTX 3090 上启动任务 02（S0 和 S1）
- [ ] 作者：确认单位对 MDPI / IEEE Access 的认定口径

## 已知问题

- `literature/papers.csv` 中共有 140 篇文献，其中 135 篇已核对官方摘要（`verified=yes, summary_source=abstract`，含任务 06 新增 22 篇、任务 09 新增 4 篇、任务 16 新增 3 篇与任务 19 新增 4 篇）；5 篇标为 `verified=no`（2 篇未定位独立 DOI 的种子文献 + 3 篇 Elsevier 付费墙阻挡摘要的文献），均已单独标注且未作为核心论证依据。
- 公开数据集的国内网络拉取速度与 PKU PosterLayout 协议审批周期需在实验启动第一周实测。
- `docs/JOURNALS.md` 中的 JIF 与中科院 2025 分区来自公开汇总数据，正式投稿前须在 Clarivate MJL 官网与本单位图书馆目录复核。

---

## 工作记录

### 2026-11-26 · claude · main
- **完成**：审查并合并 PR #19（任务 18）、PR #18（任务 19，解决 PROGRESS 冲突）；下达任务 20（Codex）和任务 21（Gemini）。
- **结论**：初图补足会被盖面去噪冲淡（E1 初图 9–12 mm 能量 1.996 → 成图 1.086），改初图的路线（含 C1）停止。下一轮改去噪与拓扑：提示词净化（J1）、SDXL base 混合潜变量替代 inpainting 模型（J2）、C2a 环内上托（J3）、C3 原生联合环盖生成（J4，主方案）。

### 2026-11-26 · Codex · 盖面能量匹配与 SDEdit 强度

- **诊断** 003、007 底盖的初图参考带偏疏，SDEdit 后 9–12 mm 能量再降 73.7% 和 71.5%。
- **筛选** 固定 R0，完成 E0–E4 的 RTX 3090 对照、逐毫米曲线与盖面原像素检查。五组均未过硬筛选，未锁定复验配置。
- **关口** 原 E0 关口参照。种子 42 4/5，条带 v1 1.7209（上限 1.2528），未过条带 v1 与原像素。种子 1234 5/5，条带 v1 1.3026（上限 1.3832）。停在 Step 1，等 Claude 审查。 [盖面能量匹配双种子报告](experiments/E02/CAP-ENERGY.md)

### 2026-11-26 · gemini · gemini/cap-interior-ideas（任务 19：底盖内部纹理补足机制与独立方案）
- **完成**：核查 `RING-M2M3.md`、`UNIFORM-RING.md`、`ring-m2m3-stripe-sources.csv`、种子 42/1234 原像素裁图与源码（`v3_uniform_texture.py`、`v3_ring_midscale_cap.py`、`v3_inpainting.py`、`E02_capfix.yaml`），撰写 [`docs/CAP_INTERIOR_GEMINI.md`](CAP_INTERIOR_GEMINI.md)，并核对补录 4 篇顶会文献（Gatys et al. CVPR 2016、Suvorov et al. LaMa WACV 2022、Chung et al. DPS ICLR 2023、Alaluf et al. Cross-Image Attention SIGGRAPH 2024）至 `literature/papers.csv`（136 → 140 篇）与 `literature/search-log.md`（Q44）。
- **核心发现与方案**：
  1. **五重机理解释与 3 条可验证预测**：除 inpainting 大掩码有效感受野衰减/背景填洞偏置、SDEdit 0.85 冲淡及 `U1` 初图三处隐性衰减（`0.60` 包络、双线性移相低通、`>16 px` 粗残差扣减）外，定位两项被忽略的主因——① C1a 提示词中的 `"seamless background"` 与远景地貌词主动诱导盖面内部简化；② `E01-003 / 007` 底盖存在显著的**各向异性笔触转向**（`0–3 mm` 竖向草茎/麦秆到 `9–12 mm` 弯为横向纹理，使沿折切向差分能量骤降 4–5 倍）。
  2. **任务 18（E1–E4）评估**：预判 `E1` 单配 `0.85` 降幅有限，`E2/E3/E4` 降强度至 `0.70–0.80` 有望压低 `v1`，但四边线性叠加初图在低强度下易暴露**双重曝光重影 / 拼贴感**（触犯新增目视硬约束），且 `E4` 需警惕 `3–6 mm` 同心锐度台阶。
  3. **3 个互补新方法（均适配 RTX 3090，严守只提 R 不压 E）**：**C1**（保方向样例块拼贴 Image Quilting + 泊松色度融合 + 去背景化 C1a 提示词净化）、**C2**（去噪环内 `x̂_0` 闭式单向局部纹理方差匹配 + 高分辨解码层跨图参考自注意力 K/V 注入）、**C3**（原生折线联合环盖生成 `1280×1536` + 三非原生边非对称 Differential Diffusion 融合，附双种子闸门 1 风险控管）。
  4. **排序与衔接路线**：按收益/成本/风险排序为 `C1 > C2a > E2/E4 > C2b > C3`，并按任务 18 实测落点给出分档衔接组合。
- **交接**：提交 PR 待 Claude 审查，并与任务 18 实测结果汇总。

### 2026-11-26 · claude · main
- **完成**：审查并合并 PR #17；下达任务 18（Codex）和任务 19（Gemini）。
- **注意**：E01-003 证明底盖“贴边密、内部疏”是盖面自身的问题；原则是提高参考带 R，不降低贴边 E。

### 2026-11-25 · Codex · 环带提示词调度与循环滚动

- **环带** 实测五种方案在两组种子的下界、画质、出血和法向剖面。保留 R0，四种新环带均有至少一项未过。
- **盖面和对照** 盖面比较 U1、U1-σ28，锁定 U1-σ28，以种子 1234 复验。两组 B2-regional-M2 作为补充对照。
- **关口** 种子 42 4/5，条带 v1 1.7209（上限 1.2528），未过 条带 v1 与原像素。种子 1234 5/5，条带 v1 1.3026（上限 1.3832）。停在 Step 1，等 Claude 审查。 [两种子环带与盖面报告](experiments/E02/RING-M2M3.md)

### 2026-11-26 · claude · main
- **完成**：审查并合并 PR #16、#15（解决冲突）；在 PR #15 回应 Gemini；下达任务 17。
- **注意**：U1 已使用 L2 归一，Gemini 的 M1a 只保留扩大高通尺度（σ = 28 px）这一项。

### 2026-11-25 · Codex · 盖面纹理均匀化与环带横幅抑制

- **环带** R1 相对 R0 的单侧下界变化为 +0.0635 / -0.0070（42 / 1234）。保留 R0，两种子均另测 B2-regional-R1。
- **盖面** 种子 42 比较同环带 G5、U1、U2，锁定 U1 后用种子 1234 复验。
- **关口** 种子 42 4/5，条带 v1 1.6841（上限 1.2528），未过 条带 v1 与原像素。种子 1234 5/5，条带 v1 1.3032（上限 1.3832）。停在 Step 1，等 Claude 审查。 [盖面均匀化与环带横幅报告](experiments/E02/UNIFORM-RING.md)

### 2026-11-26 · gemini · gemini/beat-all-baselines（任务 16：全关口超越对手方案研究）
- **完成**：核查 `stripe-diag.csv`（905 张归档图）、`captexture-stripe-sources.csv`、`results-captexture.csv`、任务 13 样例图与源码（`v3.py`、`v3_attention.py`、`sdxl.py`、`v3_captexture_inject.py`、`b2_controlnet.py`、`b4_paint3d.py`），撰写 [`docs/BEAT_BASELINES_GEMINI.md`](BEAT_BASELINES_GEMINI.md)，并核对补录 3 篇顶会/顶刊文献（Heitz & Neyret 2018、Efros & Freeman 2001、Hertz et al. 2023 Prompt-to-Prompt）至 `literature/papers.csv` 与 `literature/search-log.md`（Q43）。
- **核心发现与方案**：
  1. **揭示 `Paint3D` 与 `B2-regional` 条带 v1 低的真实机制**：两者并非全盒纹理均匀，而是在折线 `0–3 mm` 存在结构性 `E ≤ R` 负向偏移（`Paint3D` `med(a_E/a_R) = 0.07`、`89.3%` 至少一侧 `E ≤ R`；`B2-regional` 因 ControlNet 1 px 折线框 + `"clear margins"` 使 `10/10` 盒 `a_v1 = 1.000`），触发单侧截断 `max(0, E-R)=0`。我们应消除 `v3` 的 `E > R` 正向偏置而非模糊折线。
  2. **检验 H1–H4 并定位两处隐藏源码根因**：① 盖面 `v3_captexture_inject.py` 采用 L1 凸归一叠加四边独立高频，数学上必然引起 50%–75% 方差坍塌（即使任务 15 的 U1 取消余弦衰减也受此影响），且 `σ = 8 px`（`0.71 mm`）滤掉了 `1–3 mm` 中频纹理；② 环带 `sdxl.py:33` 强加 `"centered graphic elements with clear margins"`、背面加 `"calm background"`，叠加第 1 步起的 0 px 硬切分（`regional_weight = 1.0`）和 `11.5 mm` 窄缓冲。
  3. **提出 3 个免训练新方法（均适配 RTX 3090）**：**M1**（方差保持 L2 归一 + `σ=24–32 px` 多尺度盖面纹理注入，备选 M1b Image Quilting + Poisson）、**M2**（时间步粗到细全局—局部提示词调度 + 去边距净化 + H1 羽化）、**M3**（环带步间水平循环潜变量滚动 + H2 扩缓冲至 `1280×1152` 约 `22 mm`）。
  4. **评估 H4（折线安全区）**：直接做 `±3 mm` 高频衰减会被审稿人视为针对 `v1` 的定向优化且落入 `AUC = 0.500` 盲区，建议排最后暂不启用；给出若未来启用时的 4 条正当设计边界与 4 项反作弊检查。
- **交接**：提交 PR 待 Claude 审查，并结合任务 15 实测结果定夺下一轮交给 Codex 的组合。

### 2026-11-26 · claude · main
- **完成**：按作者要求（全部关口都要超过对手）分析条带来源，下达任务 16 给 Gemini。
- **数据**：G5 · 42 主要由 B-Btm 盖面侧（1.91）和 F-T 环带侧（1.29）控制；G5 · 1234 主要由 F-R 环带侧（右面左沿 1.23）控制，疑与分面提示词边界落在折线上有关。

### 2026-11-26 · claude · main
- **完成**：审查并合并 PR #14、#13（解决冲突）；下达任务 15。
- **注意**：任务 15 允许修改环带（负向提示词 R1），B2-regional 必须同样加入 R1 以保证公平。

### 2026-11-25 · Codex · v3 盖面颜色与纹理延续

- **完成**：实测四组 ILVR、纹理注入及两档高强度 SDEdit，比对 30 步原日程和冻结环带。
- **选型**：保留 G5。顶盖和底盖面内纹理硬约束通过，结果同时登记整盒条带、逐折拆解及长尾。
- **关口与交接**：客观条带 v1 仍未达标，原生折线目视通过。种子 1234 的盖侧 ΔE00 为 5.346，条带 v1 为 1.3455。停在 Step 1，待 Claude 审查。 [报告](experiments/E02/CAPTEXTURE.md)

### 2026-11-25 · gemini · gemini/v3-cap-texture-review（任务 14：PR #12 立场回应与任务 13 审阅）
- **完成**：核对任务 12 诊断报告（`STRIPE-DIAG.md`）、`stripe-diag-texture.csv`、`stripe-diag-lpips.csv` 及原像素裁图，结合 `v3_geometry.py`、`v3_capfix_geometry.py`、`v3_capgeom.py`、`v3_inpainting.py` 在 PR #12 留言逐条回应任务 14 的 4 个问题（见 https://github.com/Casey-sketch-cyber/ai-design/pull/12#issuecomment-7166031174 ）。
- **核心结论**：
  1. **同意修正此前判断**：原像素裁图（`E01-003/004/007` 底盖）与 `stripe-diag-texture.csv` 证实 `G2 · 0.85` 盖面已退化为近似平涂色块（纹理能量约 `B2-regional` 的 1/34–1/53），其低 ΔE00 优势很大程度上来自平涂，且底盖三缝 LPIPS 0/10 盒优于 `B2-regional`。
  2. **同意暂不引入条带 v2**：维持 `v1` 冻结作为关口，避免掩盖真实的纹理中断缺陷；同意在论文中如实披露 `v1` 对沿折线均匀亮暗带不敏感（`AUC = 0.500`）并保留原像素目视核查，同时建议种子 1234 复验继续附带 `ring_only_box_floor / cap_only_box_floor` 拆解。
  3. **任务 13 审阅与防踩坑提醒**：
     - 核算 2× 盖面潜空间（`176 × 144`，核心 `98 × 64`，每潜格约 `0.71 mm`）：`N = 4` 等效 `32×`（约 `2.8 mm/格`，偏紧），`N = 8` 等效 `64×`（约 `5.7 mm/格`，更匹配），预期“前 70% 步数”优于“全程”（必要时可备选 `N = 16` 或前 50% 步数）。
     - 提醒 Codex 在 G4 实现中注意 `q(z_harm, prev_timestep)` 时间步对齐、确定性 DDIM 噪声一致性（复用固定 `ε_0` 或在 `x̂_0` 空间替换）及 `area` 下采样 + 仅写回 `~known_mask`。
     - 认同 `0.5 × B2-regional` 十盒均值下限（不宜再抬高或逐盒卡顶盖），并提出 3 条防“加噪点凑能量”补充护栏：面内能量上界 `≤ 2.0 × 环带均值` + 底三缝跨缝纹理能量差改善、`ImageReward ≥ -1.855`、`2×` 下采样能量留存率诊断。
  4. **论文贡献定位**：提炼“折叠纸盒四边非原生拓扑边界下的频域解耦盖面合成”，明确相对 Blended Latent Diffusion、SDEdit 与 ILVR 的增量（由四边冻结环带 PDE 调和场合成无现成参考图的低频先验 `z_harm`，与全步四边上下文修复解耦高频纹理生成），将 `G1 → G2 → G4` 组织为因果完整的消融主线。
- **未完成**：待 Codex 完成任务 13 并提交 PR 后，按 Claude 安排跟进。

### 2026-11-25 · claude · main
- **完成**：审查并合并 PR #12、#11（解决冲突）；修正立场，下达任务 13、14。
- **注意**：G2 盖面几乎平涂；任务 13 增加硬约束，盖面纹理能量不得低于 B2-regional 的 0.5 倍。条带 v1 对沿折线的均匀亮暗带不敏感，论文中需说明，并保留原像素目视检查。

### 2026-11-25 · Codex · CPU 条带诊断

- **完成**：复算 905 张历史原图，完成 288 条合成对照、30 张真实峰值裁图、40 盒的七条非原生缝 LPIPS 拆分。
- **证据**：无带 A 组 30/32 条目标折线高于旧参考值，四式均对沿线均匀亮暗带失灵。v2-a 减少 A 组越线，也漏掉历史任务 04 的真条带。G2 两档的贴边项和参考项均有变化。
- **交接**：原指标、阈值和既有关口保持冻结。候选对照和底盖 LPIPS 差异提交 Claude、Gemini 汇总。 [报告](experiments/E02/STRIPE-DIAG.md)

### 2026-11-25 · gemini · gemini/v3-g2-discussion（任务 11：PR #10 讨论）
- **完成**：核对任务 10 数据（`results-capgeom.csv`、`capgeom-stripe-sources.csv`、`CAPGEOM.md`、样例图）与代码（`v3_capgeom*.py`、`v3_inpainting.py`、`metrics/physical.py`），在 PR #10 留言逐条回应 Claude 的 5 项立场（见 https://github.com/Casey-sketch-cyber/ai-design/pull/10#issuecomment-7160816593 ）。
- **核心结论**：
  1. 同意以 `G2`（当前起点 `0.85`）作为候选主配置，赞同在 10 条调参集上改为“核心连续性/出血主目标 + 门禁约束”的分层选型规则。
  2. 认同 `stripe_index v1` 存在单面平滑分母脆弱性（另有 `G1` 种子 1234 单环带 `ring_only_box_floor` 均值 `1.280 > 1.253`、5/10 盒超标作为独立反例），赞同 Step 2 前新增并冻结 `v2` 且并列保留 `v1`；但提醒任务 12 须拆出分子/分母核查 `0.75 → 0.85` 中 `Btm:cap` 从 `2.934` 升至 `4.044` 的真实来源，优先看好 `v2-a`（六面中位数）与 `v2-b`（能量下限）组合，对 `v2-c`（同面 `10–15 mm`）持保留态度。
  3. 给出防止“改尺子迎合自己”的 5 条方法学边界（开发/测试集严格隔离、独立模型无关证据、保留 `v1` 双口径披露、以合成对照与历史真实条带区分度选型、冻结后不再改动）及 SCIE 论文透明写法。
  4. 拆解 `7-seam LPIPS` 发现：`G2 · 0.85` 的 `L–B` 缝（`0.1664`）和顶盖（`0.3932`）已优于或持平 `B2-regional`，差距集中在高频前景密集的底盖（`bottom_seam_lpips = 0.4511` vs `0.3606`）；建议 Step 2 前仅补测 `G2 · 0.90`（必要时 `0.92`）及 1–2 潜格软 `paint_mask` 过渡。
  5. 强烈同意在 Step 2 前对单一优胜 `G2` 配置用种子 1234 复验前 10 条（不调参），并给出 Codex 1–2 天收口路线。
- **未完成**：待 Codex 提交任务 12（`STRIPE-DIAG.md`）后，结合分子/分母与合成对照结果补充 `v2` 公式定型意见。

### 2026-11-25 · claude · main
- **完成**：审查并合并 PR #10；下达任务 11（Gemini 讨论）和任务 12（Codex CPU 诊断）。
- **注意**：条带指数 v1 的分母取自同一面内部，盖面越平滑越容易被放大；是否新增 v2 由讨论决定，v1 不改。

### 2026-11-25 · Codex · v3 盖面几何与调和初始化

- **完成**：十盒比较贴边、两档调和初始化和时间步对齐的共享注意力。按任务 08 规则保留 G1。逐像素核对冻结环带、相邻侧面与出血，指标定义与阈值哈希通过。
- **关口**：七缝 LPIPS、原口径出血和客观条带指数未达标。只保留环带侧贡献的条带下界为 1.2244，上限为 1.2528。种子 1234 的盖侧 ΔE00 和全盒条带指数分别为 14.960、1.3483。种子 42 的峰值贡献以盖面侧为主，种子 1234 则以环带侧为主。
- **交接**：停在 Step 1，待 Claude 审查。逐项票数、长尾和逐折数据见 [任务 10 报告](experiments/E02/CAPGEOM.md)。

### 2026-11-25 · claude · main
- **完成**：审查并合并 PR #9、PR #8（解决 PROGRESS 冲突）；下达任务 10。
- **注意**：盖面非原生外切边的出血改为从紧贴的冻结边带裁取，这是 3D 相邻侧面的真实延续。

- [x] Codex：任务 08，见 `tasks/08-codex-v3-cap-fix.md`（盖面修复，已完成，关口未过）
- [x] Gemini：任务 09，见 `tasks/09-gemini-v3-cap-review.md`（已在 PR #7 留言审阅并补录 C5 相关 4 篇文献）

### 2026-11-25 · claude · main
- **完成**：审查并合并 PR #7；下达任务 08（Codex 盖面修复）和任务 09（Gemini 审阅）。
- **注意**：指标已于任务 07 冻结，盖面修复不得改动指标或环带阶段。

### 2026-11-24 · codex · v3 Step 0 与 Step 1

- **Step 0**：拆分原生与非原生缝、分析出血失分，五盒重跑去 Outside 锚定实验。在同一 755 张历史原图完成新口径回算和反例健全性检查。生成前锁定指标提交与参数哈希。
- **Step 1**：十盒 B2-regional、两档 v3 及共享注意力候选均已在 RTX 3090 运行。以六项规则选 1280×1024 无共享版，实测 21.57 秒/盒、峰值约 10.18 GiB。
- **关口**：两套出血评分胜过 B2-regional 和 B2-post，CLIP 与开销通过。非原生缝和全盒条带未过。环绕缝改善，盖侧非原生缝是主要差距。
- **交接**：停在 Step 1，提交 Claude 审查。[完整报告](experiments/E02/REPORT.md)

### 2026-11-25 · claude · main
- **完成**：第二轮审查并合并 PR #5；写 `docs/METHOD_V3_CONSENSUS.md` 和任务 07。
- **注意**：Step 1 关口中 CLIPScore 为单侧判定（不低于 B2 减 0.05）。

### 2026-11-25 · gemini · gemini/method-v3-research（PR #5 第一轮审查修订与逐条回应）
- **完成**：
  - 合并 `origin/main`（含已合并的 PR #6 任务 05 结果），逐条落实 Claude 在 PR #5 的 6 项审查意见并更新 [`docs/METHOD_V3_PROPOSALS.md`](METHOD_V3_PROPOSALS.md)：
    1. **顶/底盖面改为顺序条件外扩，取消双向平均**：吸取任务 05 正面—顶面与背面—右面亮边教训，P2 修订版改为“阶段一环带完整生成并冻结 → 阶段二顶/底盖面居中、四周拼上冻结环带的 4 条对应 3D 邻边（只读不写回）、四角自由生成后丢弃，用 Blended Latent Diffusion（+ 可选 StyleAligned K/V 加分项）顺序生成”；评估后将 P1 盖面局部软修补移出默认必选，仅作残余退路。
    2. **新增 `B2-regional` 受控基线**：同一刀版画布 + 相同分面提示词路由（Regional Prompting）+ 无白墙钳位 + 同源 LUT 与统一印前复位，剥离“提示词贡献”与“几何拓扑贡献”。
    3. **补齐环形卷积 4 项实现细节**：给出任意管式盒宽度/高度按 64 px（8 潜格）整除的通用取整公式（将高分辨档修正为 `1472 × 1152`，与 `1280 × 1024` 两档同测并配 DINOv2 图内自相似度重复检测）；确认 UNet 全卷积（含 stride=2 下采样）与 VAE 解码器同步水平环形填充（且不沿 X 轴分块 tiling）；垂直 Y 轴弃用 `reflect` 改保留原生 `zeros` + 上下 10–12 mm 缓冲裁切。
    4. **明确新指标 M1–M4 的冻结纪律与反例检查**：原物理口径永久保留在主表，主结论须在原/新口径同时成立；明确规定 T0（Step 0 阶段，v3 生成启动前）完成健全性检查（含 M3 有纹理边缘逐像素复制与重采样伪出血反例）并在历史图上定稿冻结，之后严禁改动。
    5. **落实开放问题共识与标注规范**：记录 4 项开放问题共识（优先轻量 Regional Prompting、双档分辨率同测、统一 `preflight` 口径、继承任务 05 底面刀版方向）；全文耗时与显存均明确标为“估计”，待 Step 0 / Step 1 实测确认；P4 与 P5 明确留作后备不进首轮；保留 Step 0（D1–D3）在 v3 实现前执行。
- **未完成**：待 Claude 复审 PR #5 并汇总成 `docs/METHOD_V3_CONSENSUS.md`，交作者确认后下达 Codex。

### 2026-11-25 · claude · main
- **完成**：合并 PR #6；对 PR #5（Gemini v3 方案）给出第一轮审查意见：盖面改为以环带为条件的顺序外扩生成；新增 B2-regional 基线；补充环形卷积实现细节；新指标需冻结定义并保留原口径。

### 2026-11-25 · gemini · gemini/method-v3-research（任务 06：方法深研 v3 提案初版）
- **完成**：
  - 完整核读 E01/E01b 报告、PR #2/#3/#4 审查留言、`b2_controlnet.py` / `ours_tf_v2.py` / `windows_v2.py` / `metrics/` 源码及可视化样本，交付 `docs/METHOD_V3_PROPOSALS.md` 初版（机制归因 + 判别实验 D1–D5 + 5 个候选方案 P1–P5 + 评测偏差分析与 M1–M4）。
  - 新增并逐篇核对 **22 篇** v3 相关核心论文官方摘要与 DOI，更新 `literature/papers.csv`（总计 129 篇：124 篇 `verified=yes, abstract`，5 篇 `verified=no`，0 违规）与 `literature/search-log.md`（新增 Q38–Q41）。
- **注意 / 坑**：① SyncDiffusion 原版梯度反传迁移至 SDXL 极易在 24 GB 上 OOM 且耗时飙升，不宜照搬；② 新增评测指标须与旧物理口径并行保留，不可替换旧口径。

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
