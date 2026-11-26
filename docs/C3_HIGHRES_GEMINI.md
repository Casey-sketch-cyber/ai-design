# 任务 21：C3 联合环盖画布（约 2 MP）的去重复与稳态设置，及失败后备方案（Gemini）

- **首选设置**：**两段式低分辨联合生成 + 放大细化（Coarse-to-Fine `896×1088 → 1280×1536`）**——Stage 1a 在 SDXL 原生舒适面积 `896×1088`（`0.975 MP`）用水平环形卷积 + 三层纵向软羽化区域提示词跑 28 步定全局布局，Stage 1b 双三次放大至 `1280×1536` 后用 `SDEdit (strength=0.47, 14 步)` 细化，RTX 3090 耗时约 **18–20 s/盒**（≤ 25 s）、显存约 **11.5 GiB**，完全兼容 `v3_circular.py` 与 `v3_attention.py` 且不引入非整数空洞伪影。
- **备选设置**：**ScaleCrafter 早步各向异性重膨胀（`dilation=(2, 1)`，单次 `1280×1536`）**——仅在去噪前 33% 步（步 0–9/30）对 SDXL 深层 `down_blocks.2`、`mid_block`、`up_blocks.0` 的 `3×3` ResNet 卷积施加纵向空洞 `dilation=(2, 1)`，直接复用 `v3_circular.py` 动态读 `dilation` 的原生逻辑，耗时约 **19–21 s/盒**、零额外显存。
- **后备方案（若 C3 闸门 1 止损）**：**主后备 B1（不改 R0 的“先单边开放外扩、再三非原生边非对称缝合”）**——在冻结 R0 环带上，先拆掉左右对侧三堵墙、仅沿原生折线（`B-Btm` / `F-T`）用 `SDXL base + Blended Latent Diffusion` 做单边开放外扩（消除四墙夹逼与 90° 侧边转向），再以原生折线 `0–15 mm` 锁定（`strength=0`）仅对三条非原生切边做 `Differential Diffusion (0.68)` 缝合；**次后备 B2（方向感知 `x̂_0` 切向频谱上托 + 解码高分辨层跨图 K/V 注入）**。

---

- 执行者：Gemini | 日期：2026-11-26 | 对应任务书：`tasks/21-gemini-c3-highres.md` | 分支：`gemini/c3-highres`
- 核心几何基准（核对 `code/dielinediff/dieline/geometry.py:35-37`）：盒型 `70 × 45 × 160 mm`，周长 `230 mm`。在 `width_px = 1280`（`160` 潜格，`5.565 px/mm`）下，盒身 `160 mm = 112` 潜行（`896 px`），盖深 `45 mm = 32` 潜行（`256 px`），上下各加 `8` 潜行缓冲（`64 px ≈ 11.5 mm`），联合画布为 **`1280 × 1536`**（潜空间 `160 × 192`，面积 **`1.966 MP ≈ 1.88 × 1024²`**，横向倍率 **`1.25×`**、纵向倍率 **`1.50×`**，纵横比 `0.833:1`）。

---

## 1. 六种高分辨率去重复路线的系统对比与落地配置

### 1.1 六种路线对比（面向 `1280 × 1536` 约 2 MP 联合环盖画布）

| 路线（依据文献） | 免训练 | 与水平环形卷积（`v3_circular.py`）兼容 | 与区域提示词（`v3_attention.py`）兼容 | RTX 3090 Stage 1 实估耗时（≤25 s 要求） | 对 `1.25× × 1.50×`（1.97 MP）的适配性评估 |
| --- | :---: | :---: | :---: | :---: | --- |
| **两段式低分辨联合生成 + 放大细化**（DiffuseHigh, AAAI 2025; SDEdit, ICLR 2022; DemoFusion 精简思想） | 是 | **完全兼容**（两阶段均开 `circular_ring_codec`） | **完全兼容**（两阶段同拓扑 `RegionRouting`） | **`~18–20 s/盒`**（通过：`896×1088` 28 步 ~8.5 s + `1280×1536` 14 步 ~9.5 s + 编解码 ~1 s） | **最优（首选）**：宏观构图完全在 `0.975 MP` 原生舒适区内成形，消除 2 MP 物体重复与 `M3b` 式跨种子崩塌；不改卷积核，无非整数缩放空洞伪影。 |
| **ScaleCrafter**（早步各向异性空洞重膨胀，ICLR 2024） | 是 | **完全兼容**（`v3_circular.py:14-30` 已原生按 `layer.dilation` 算水平环形填充） | **完全兼容**（潜空间特征图尺寸不变） | **`~19–21 s/盒`**（通过：单次 30 步，几乎零额外开销） | **次优（备选）**：仅纵向扩至 `1.50×`、横向维持 `R0` 已验证的 `1.25×`，施以 `dilation=(2, 1)` 仅扩纵向感受野；唯一风险是 `1.5×` 并非整数 `2`，须严格限于深层前 33% 步以防轻微网格纹理。 |
| **HiDiffusion**（RAU-Net + MSW-MSA，ECCV 2024） | 是 | **部分兼容**（RAU-Net 兼容；**MSW-MSA 分窗自注意力在 X 轴切断环形闭环**，需改写为环形移窗） | **有隐患**（RAU-Net 额外 2× 下采样使 `mid_block` 缩至 `20×24`，窄侧面 `L/R` 仅剩约 4 潜格，易糊） | `~15–17 s/盒`（通过） | **不推荐用于 2 MP**：HiDiffusion 为 `4 MP–16 MP`（2048²/4096²）设计；在 `1.97 MP` 上额外 2× 下采样把瓶颈压到比 SDXL 原生 `32×32` 更小（面积缩半），容易损伤窄侧面构图精度。 |
| **FouriScale**（空洞卷积 + 频域低通双分支，ECCV 2024） | 是 | **有隐患**（2D FFT 隐含双轴周期，Y 轴非周期需 padding-then-crop 进一步胀大画布） | 兼容 | **`~32–38 s/盒`**（**超时**：每步需双分支 UNet 前向 + FFT 低通引导） | **不推荐**：Stage 1 即超 25 s 预算，挤压 Stage 2 后整盒必超 60 s 红线。 |
| **DemoFusion**（渐进放大 + 残差 + 多窗空洞采样，CVPR 2024） | 是 | 需将局部滑窗改造为 X 轴环形滑窗 | 需逐窗裁剪区域掩码 | **`~35–45 s/盒`**（**超时**：多阶段 + 重叠滑窗） | **不推荐完整版**：算力过重；仅保留其“原生分辨率定构图 + 轻度放大细化”的核心思想，即本报告首选方案。 |
| **MultiDiffusion / SyncDiffusion 分块**（ICML 2023 / NeurIPS 2023） | 是 | 可环形滑窗 | 可逐窗裁剪 | **`~32–90 s/盒`**（**严重超时**；SyncDiffusion 含 LPIPS 梯度反传） | **不推荐**：高重叠开销，且 MultiDiffusion 缺全局感受野反而易加重重复（任务 06 已论证弃用）。 |

### 1.2 首选设置（可直接填入配置与 `v3.py`）

**核心思想**：把 `1280 × 1536` 的 Stage 1 拆为“原生舒适面积布局（1a）+ 环形轻度放大细化（1b）”，兼顾零重复、跨种子稳态与 3090 速度预算。

```yaml
# code/configs/E02_joint_cap.yaml (Preferred: C3-Coarse2Fine)
joint_stage1:
  mode: coarse_to_fine
  target_width_px: 1280
  cap_buffer_px: 64               # 上下各 64 px (8 潜行 ≈ 11.5 mm)，总高按盒深动态对齐至 64 倍数（当前盒 1536 px）
  stage1a_scale: 0.70             # 约 896 × 1088（0.975 MP，潜空间 112 × 136，同比例且对齐 64 px）
  stage1a_steps: 28
  stage1a_guidance: 5.0
  stage1a_lut_projection: false   # 色域投影留到 Stage 1b 末段一次完成，避免两次投影过饱和
  upsample_space: rgb_bicubic     # 1a 整图解码（开 circular_ring_codec）→ 双三次升至 1280×1536 → VAE 重编码
  stage1b_total_steps: 30         # 沿用原 30 步 DDIM 日程
  stage1b_strength: 0.47          # 执行末 14 步（start=16, t ≈ 467 → 0）：锁定 1a 全局构图防重复，充分补回 5.56 px/mm 锐度
  stage1b_guidance: 5.0
  stage1b_projection_tail: 0.30   # 仅在 1b 末 30%（最后 9 步）每 3 步执行原 LUT 色域投影
  stage1b_projection_interval: 3
  stage1b_time_ids_original: [1536, 1280]
```

- **为何选 `rgb_bicubic` 放大而非纯潜空间双线性放大**：SDXL 4 通道潜变量存在高频非线性耦合，直接双线性放大潜变量易产生轻度油画色斑；在 `full_vae_canvas` + `horizontal_circular` 下先解码 `896×1088` RGB（约 `0.25 s`）、双三次放大到 `1280×1536`（边界 X 轴先做 16 px 环形填充再缩放裁回，保持 `L–B` 像素零接缝）、再 VAE 编码（约 `0.25 s`），画质显著更锐利，与 **DiffuseHigh (AAAI 2025)** 及 **SDXL Refiner** 实践一致。
- **实现位置（`code/dielinediff/methods/v3.py`）**：
  - 在 `v3_geometry.py` 新增 `build_joint_ring_layout(topology, width_px, cap_buffer_px=64)`，动态按 `ceil(box["width"] * latent_w / circumference)` 计算盖深，返回 `RingLayout` 兼容结构外加 `cap_rects = {"T": ..., "Btm": ...}`；
  - 在 `v3.py` 新增 `_prepare_joint_ring(contexts, pooled, generator, cfg)`：先按 `stage1a_size`（`896×1088`）调用现有 `_denoise(..., circular_ring_codec, region_attention)`，随后执行环形安全双三次放大与编码，再复用 `v3_capgeom_init.py` 的 `sdedit_schedule` / `sdedit_latents` 执行 14 步 Stage 1b 去噪。
- **实估开销**：`896×1088` 28 步约 `8.5 s` + 编解码放大 `0.6 s` + `1280×1536` 14 步约 `9.5 s` = **Stage 1 共 `~18.6 s/盒`**，峰值显存 **`~11.5 GiB`**；加上 Stage 2 三边非对称缝合（`strength ≤ 0.70`，21 步 × 2 盖，无调和初图求解则省 9 s，约 `22–26 s`），**整盒预计 `~42–46 s/盒`（远低于 `60 s` 红线）**。

### 1.3 备选设置（ScaleCrafter 早步各向异性重膨胀，可直接填入配置）

若 Codex 在闸门 1 希望同步对照一个**单次前向、零放大步骤**的方案，采用针对 `1.25× × 1.50×` 定制的 **ScaleCrafter 早步各向异性空洞卷积**：

```yaml
# code/configs/E02_joint_cap.yaml (Fallback: C3-ScaleCrafter-Aniso)
joint_stage1:
  mode: single_pass_redilation
  target_width_px: 1280
  cap_buffer_px: 64               # 1280 × 1536
  steps: 30
  guidance: 5.0
  redilation:
    enabled: true
    dilation_hw: [2, 1]           # 仅纵向扩张 2×（匹配 Y 轴 1.50× 延展），X 轴维持 1（保留 R0 1280 宽的稳态）
    apply_steps: [0, 9]           # 仅前 10/30 步（前 33% 布局成形期）；第 10 步起恢复 (1, 1) 保精细纹理与零网纹
    target_blocks:
      - down_blocks.2             # 40 × 48 潜特征（ResNet 3×3 卷积，跳过 1×1 conv_shortcut 与 Downsample2D）
      - mid_block                 # 40 × 48 瓶颈层（ResNet 3×3 卷积）
      - up_blocks.0               # 40 × 48 解码深层（ResNet 3×3 卷积）
    min_kernel_size: 3            # 仅作用于 kernel_size == (3, 3) 且 stride == (1, 1) 的 nn.Conv2d
    keep_circular_x: true
```

- **与 `v3_circular.py` 的原生零冲突实现**：检查 `code/dielinediff/methods/v3_circular.py:14-31`，`_horizontal_conv2d` 已通过 `total_y = (layer.kernel_size[0] - 1) * layer.dilation[0]` 动态计算上下填充并把 `layer.dilation` 直接传入 `F.conv2d`。因此只需在 `v3_circular.py` 新增约 25 行的 `anisotropic_redilation(unet, dilation=(2, 1), prefixes=...)` 上下文管理器，在 `index <= 9` 时临时将匹配层的 `layer.dilation` 设为 `(2, 1)`、`layer.padding` 设为 `(2, 1)`（出块恢复 `(1, 1)`），即可同时保持 X 轴环形卷积与 Y 轴扩展感受野。
- **实估开销**：`1280×1536` 30 步约 **`19–21 s/盒`**，峰值显存 **`~11.5 GiB`**。

---

## 2. 纵向非循环边界分析与缓冲区宽度建议

1. **延展区边缘是否会出现边框伪影？**
   - **会出现在整张画布最外侧 `0–10 mm`（顶部 `y ∈ [0, 64 px]` 和底部 `y ∈ [1472, 1536 px]`），但不会伤及核心评分区**：
     - **对原生折线 `F-T` 和 `B-Btm`（零风险）**：在 `1280 × 1536` 中，`F-T` 位于 `y = 320 px`（距画布顶边 **`57.5 mm`**），`B-Btm` 位于 `y = 1216 px`（距画布底边 **`57.5 mm`**）！它们被完全推入画布深处，远离 Y 轴零填充边界与 SDXL 边缘构图偏置（这比任务 17 `M3b` 的 `23 mm` 缓冲更深一倍以上，且不会像 `M3b` 那样在空缓冲里催生虚假横幅）。
     - **对盖面外切边（`B-T` 顶边与 `F-Btm` 底边，低风险）**：这两条边距画布顶底恰好隔着缓冲区，且在 **Stage 2 中会以 `strength = 0.70` 重绘缝合**，其外侧 3 mm 出血也由冻结环带直接裁出（`compose_capgeom_atlas`），因此画布最外沿轻微的零填充边界效应会在裁切缓冲 + Stage 2 缝合中被消除。
2. **缓冲区宽度推荐：`cap_buffer_px = 64 px`（8 潜行 ≈ `11.5 mm`），不建议盲目扩到 `128 px`**：
   - `64 px`（8 潜行）已大于 SDXL 编码器最深层 `3×3` 卷积及 VAE 解码器单侧有效感受野（约 4–6 潜行），足以隔离 Y 轴 `padding_mode="zeros"` 带来的暗边/亮边伪影；
   - 若将缓冲扩到 `128 px`，总高将升至 `1664 px`（面积升至 `2.13 MP`），既无益于已在画布深处 `57.5 mm` 的 `F-T / B-Btm`，又徒增 8% 耗时与重复风险；
   - **关键几何保护**：在 `v3_geometry.py` 沿用任务 20 规定的动态公式 `cap_depth_px = ceil(width_mm * ppm / 8) * 8`，`total_h = _aligned(body_px + 2 * (cap_depth_px + 64))`，不写死 `256`；且 Stage 1a 的 `896×1088` 与 Stage 1b 的 `1280×1536` 保持同一套相对比例坐标。

---

## 3. 纵向相邻的区域提示词设计（防止顶/底盖复制 F 面主体物）

在 `1280 × 1536` 联合画布中，顶盖 `T` 紧贴 `F` 上方、底盖 `Btm` 紧贴 `B` 下方。若直接给整个 `F` 用 `front_prompt`（如 `"prominent illustration of golden millet heads..."`）、给 `T` 用 `{side_prompt}`，在 `~2 MP` 下容易出现两种失败：① 顶盖 `T` 拷贝 `F` 的主体物（如再画一轮太阳或一丛孤立主花）；② `F` 与 `T` 在 `F-T` 折线处出现 0 px 提示词硬阶跃，反而重新制造 `F-T` 条带！

### 具体三层纵向路由与提示词隔离方案（直接对应 `sdxl.py`、`v3_attention.py`、`v3.py`）

1. **保持全局 `pooled` 嵌入不含主体物名词（检查 `v3.py:114`）**：
   - SDXL 的 `pooled_prompt_embeds`（CLIP-G 池化向量）作用于**全画布每个像素**，不经过空间区域路由！
   - 核查 `v3.py:104-114`：当前 `pooled[0]` 来自 `style = f"{STYLE}{continuity}, {brief['main_color']}"`，不含主体物，这是正确的；**严禁**把 `front_prompt` 塞进 `pooled[0]`，否则主体物会绕过 `RegionRouting` 直接泄露到 `T` 与 `Btm`。
2. **`F` 面纵向三段拆分，主体物严格锁在 `F` 中部（`sdxl.py` + `v3_geometry.py`）**：
   - 将 `F` 面纵向 `112` 潜行（`160 mm`）拆分为三区：
     - **`F_hero`（中部 60%，`y ∈ [y_body0 + 22, y_body1 - 22]`，距 `F-T` 与 `F-Btm` 均留出 ~31 mm 安全距离）**：唯一接收 `brief["front_prompt"]`（主体物 + 主视觉）；
     - **`F_top`（上部 20%，`31 mm`）与 `F_btm`（下部 20%，`31 mm`）**：**去掉 `front_prompt` 主体物**，改用连续纹理/氛围延续词（与相邻 `T` / `Btm` 共享同一套延续词！）；
   - **这样设计的双重好处**：① 主体物被牢牢锁在 `F` 中腹，距离顶盖 `T` 超过 `31 mm`，有效阻断 `T` 复制主体物的跨区泄露；② **`F-T` 折线两侧（`F_top` 与 `T`）以及 `B-Btm` 折线两侧（`B` 下部与 `Btm`）使用的是完全一致的提示词，在 `F-T` 与 `B-Btm` 折线上不存在任何提示词切换边界！**
3. **盖面区与缓冲区的精确提示词分配（`sdxl.py`）**：
   - **`T`（`F` 上方）与 `F_top`**：使用无主体物、无 `"seamless background"` 的上缘连续纹理词：
     `"{style}, {texture_motif}, {main_color}, continuous upper artwork extension across the fold, same brushwork and consistent detail density, no standalone subject, no frame"`
   - **`Btm`（`B` 下方）与 `B` 下沿**：使用任务 20 J1 的均质前景延续词（去掉远景山丘词与 `"seamless background"`）：
     `"{style}, {texture_motif}, {main_color}, continue the same foreground foliage, pattern and brushwork evenly over the whole surface, uniform detail density, same stroke direction as the adjacent artwork"`
   - **`L / R` 上下方及 `B` 上方、`F` 下方的丢弃缓冲区**：继承相邻侧面上/下沿的连续纹理词，避免在缓冲与 `T / Btm` 的左右交界处产生突兀色块污染自注意力。
   - **盖面负面提示词（在 `v3_attention.py` 对 `T / Btm` 单独路由负向文本，或加入全局负向）**：追加 `"plain background, empty area, blank space, blurry, duplicate subject, second sun, repeated emblem, picture frame, rectangular border"`。
4. **纵向软羽化过渡（`v3_attention.py`）**：
   - 在 `F_hero ↔ F_top`、`F_hero ↔ F_btm` 交界（位于 `F` 内部 31 mm 处，远离折线！）施加 **`±6` 潜行（约 `8.6 mm`）的线性交叉注意力输出混合**：对过渡带像素同时计算相邻两区域的 `scaled_dot_product_attention` 并按行权重 `α(y) ∈ [0, 1]` 加权求和（仅增加 ~15 行代码，开销 `< 0.2 s`）。这样从 `F` 中腹主体物到 `F_top` 再跨过 `F-T` 进入 `T` 全程连续无台阶。

---

## 4. 若 C3 在闸门 1 止损时的后备方案（不改初图、不压低贴边 E）

任务 18 已证伪初图侧修补，任务 20 已安排 `J1`（提示词净化）、`J2`（SDXL base 混合潜变量）、`J3`（C2a 各向同性方差上托）。如果 `J4 (C3)` 在闸门 1 止损且 `J1–J3` 仍差临门一脚（例如 `E01-003 / 007` 底盖仍因四墙夹逼发生竖草转横、切向能量 `R` 偏低），提出 **两个全新的、完全不动 R0 环带且不改初图的后备方案**：

---

### 后备方案 B1（首选后备）：冻结 R0 下的“先单边开放外扩、再三非原生边非对称缝合”

#### 1. 机制（把 C3 的拓扑红利搬回冻结 R0，零环带风险、零 2 MP 重复风险）
重新审视为什么现有 Stage 2 盖面在 `B-Btm` 上总出现 `0–3 mm` 竖草密、`9–12 mm` 变疏/横转：
- 现有 `build_capgeom_layout` 从第 0 步起就把盖面关进**四面冻结的死框**里，其中左右两条边（`L`、`R` 旋转 90°）把横置草叶直接顶进 `Btm` 内部，对边（`F` 旋转 180°）把倒置纹理顶回来——四壁夹逼迫使扩散模型在中心折中妥协、将草叶弯横并简化；
- **B1 拆解这一四壁冲突，且完全保持 R0 环带不动**：
  1. **Step 2a（单边开放原生外扩，消除四壁夹逼与 90° 侧边干扰）**：构造一张舒适尺寸的 `~1024 × 1024` 单边外扩画布——**仅在顶部放置 `B` 面底部 `45 mm` 的冻结上下文**（对顶盖则仅在底部放置 `F` 面顶部 `45 mm`），**左右与对边全部开放（不设任何冻结墙！）**；使用 **`J2`（SDXL base + Blended Latent Diffusion，去除 9 通道 inpainting 的填背景偏置）+ `J1` 均质延续提示词**，让 `B` 的草叶/麦穗沿单一自然方向笔直向下自由生长 `55 mm`（覆盖整个 `45 mm` 底盖深度 + 10 mm 富余）。因没有左右 90° 冲突墙，也没有 9 通道 inpainting 填洞偏置，`B-Btm` 在 `0–20 mm` 内将保持均匀密度与一致笔触方向（`E ≈ R`）；
  2. **Step 2b（三非原生边非对称 Differential Diffusion 缝合）**：把 Step 2a 生成的原生延展底盖原样放入现有四边紧贴画布（`build_capgeom_layout`），**完全沿用任务 20 第 2.3 节的强度图**——锁定原生折线 `B-Btm`（或 `F-T`）及其内侧 `0–15 mm`（`strength = 0.0`），仅在距三条非原生切边 `0–15 mm` 内平滑升至 `0.68–0.70` 完成三边无缝缝合，并导出 3 mm 冻结环带外切出血。

#### 2. 依据文献（全部核对官方摘要与 DOI）
- **Avrahami, Fried & Lischinski (2023)**，*Blended Latent Diffusion*，*ACM SIGGRAPH 2023*（已在库）：支撑 Step 2a 使用无填背景偏置的 SDXL base 模型执行潜空间条件外扩。
- **Levin & Fried (2025)**，*Differential Diffusion: Giving Each Pixel Its Strength*，*CGF 2025 / arXiv:2306.00950*（已在库）：支撑 Step 2b 锁定原生折线 `0–15 mm`、仅缝合三条非原生边的逐像素强度图。
- **Suvorov et al. (2022)**，*LaMa*，*WACV 2022*，`10.1109/WACV51458.2022.00323`（已在库）：解释四面封闭大掩码修复内部的感受野退化。

#### 3. RTX 3090 成本
- 无需 `harmonic_fill`（直接省去 `9.2 s/盒` CPU 泊松求解！）；
- `R0` 环带 `12.4 s` + 2 盖 ×（Step 2a 开放外扩 22 步 `~10 s` + Step 2b 三边缝合 18 步 `~9.5 s`）= **整盒约 `51–53 s/盒`（≤ 60 s）**，峰值显存 **`~10.3 GiB << 20 GiB`**。

#### 4. 一天内最小验证与判定标准
- **最小验证**：固定 `R0` 环带，在种子 42 前 10 盒上跑完 B1（约 15 分钟 GPU），再直接用种子 1234 复验（约 15 分钟）。
- **判定标准**：① 两种子五条冻结关口全过（种子 42 `v1 ≤ 1.2528`，`E01-003 / 007` `B-Btm` 盖面侧均 `< 1.30`；种子 1234 `v1 ≤ 1.3832`）；② 0–20 mm 逐毫米曲线中 `9–12 mm / 0–3 mm` 比值 `≥ 0.80`（且 `0–3 mm` 能量不低于 `E0`，验证未压低 `E`）；③ 三条非原生边缝合后 `cap6 ΔE00 ≤ 4.8`、`7-seam LPIPS ≤ 0.355`、`M3 ≥ 55%`；④ 原像素 10/10 无拼贴、重影、内框环，法向剖面无均匀带。

---

### 后备方案 B2（次选后备，与 J2/J3 直接叠加）：方向感知 `x̂_0` 切向频谱上托 + 解码高分辨层跨图 K/V 注入

#### 1. 机制（补齐 J3 各向同性方差匹配的方向盲区）
若 `J2 + J3` 已经抬高了盖面内部总体方差，但 `E01-003 / 007` 仍因草茎在 `9–12 mm` 弯横导致**沿折切向能量 `R` 偏低**：
1. **方向感知单向频谱上托（升级 J3）**：在第 6–22 步每隔 2 步对 `x̂_0` 盖面内部（`nearest > 3 mm`，严禁碰 `0–3 mm`）计算**沿原生折线切向的一维高通标准差 `σ_tan`**（而非各向同性标量 std），当 `σ_tan < 0.90 × σ_tan(冻结侧面 9–12 mm)` 时，仅对切向高通分量施加温和单向增益（`γ = 0.20–0.25`，保持均值与法向分量不变），以 **DPS (ICLR 2023)** 的 `x̂_0` 引导形式反推等效 `ε̃`；
2. **高分辨解码层跨图参考自注意力 K/V 注入**：在第 8–24 步向 SDXL 解码器高分辨自注意力层（`up_blocks.1` 与 `up_blocks.2`）注入冻结侧面内部（`3–18 mm`）同时间步的 `K/V`（权重 `0.35`），让盖面中心直接参考并延续侧面内部的真实竖向笔触（**Cross-Image Attention, SIGGRAPH 2024**; **StyleAligned, CVPR 2024**）。

#### 2. 依据文献（均已在库并核对摘要）
- **Alaluf et al. (2024)**，*Cross-Image Attention for Zero-Shot Appearance Transfer*，*SIGGRAPH 2024*，[https://doi.org/10.1145/3641519.3657423](https://doi.org/10.1145/3641519.3657423)
- **Chung et al. (2023)**，*Diffusion Posterior Sampling for General Noisy Inverse Problems*，*ICLR 2023*，[https://doi.org/10.48550/arxiv.2209.14687](https://doi.org/10.48550/arxiv.2209.14687)
- **Hertz et al. (2024)**，*Style Aligned Image Generation via Shared Attention*，*CVPR 2024*

#### 3. RTX 3090 成本与最小验证
- 额外耗时约 **`+3.0 s/盒`**（总计 `~55–57 s/盒`，若将 `harmonic_fill` 容差放宽至 `1e-4` 可回落至 `~51 s/盒`），峰值显存 **`~10.8 GiB`**；
- 1 天内可在固定 `R0` 与 `J2` 上用种子 42/1234 前 10 盒完成验证，判定标准同 B1（额外验证 `2×` 下采样能量留存率与 `ImageReward ≥ -1.75` 以防高频过锐化）。

---

## 5. 给 Codex 执行任务 20 `J4 (C3)` 的直接落地检查清单

1. **优先用本报告第 1.2 节首选设置（`896×1088` 28 步 → `1280×1536` 14 步 `strength=0.47`）执行 J4 闸门 1**，同步检查 `M4` 重复度与 `ring_only_box_floor`；
2. **务必启用第 3 节的 `F` 面三段纵向拆分（`F_hero` 仅占中部 60%，`F_top` 与 `T` 同词）+ `±6` 潜行纵向软羽化**，且不要把 `front_prompt` 放进全局 `pooled` 嵌入；
3. 缓冲区固定 `cap_buffer_px = 64`（`~11.5 mm`），总高按盒盖深动态对齐 64 倍数；
4. 若首选设置过闸门 1 即继续 Stage 2；若首选设置仅因 `1a→1b` 放大略损锐度导致 `L–B` 轻微超容差，可改测第 1.3 节备选设置（ScaleCrafter `dilation=(2, 1)` 步 0–9）；若两者均未过闸门 1，则按规则止损并直接切入**后备方案 B1（冻结 R0 单边开放外扩 + 三边非对称缝合）**。
