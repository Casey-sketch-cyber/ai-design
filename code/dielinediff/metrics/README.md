# E01 自动评测

`Evaluator.measure` 接受与 `region_map.png` 完全同尺寸的 8 位 sRGB 刀版。带透明度的输入合成至白纸色。每张图使用同一个几何文件、印刷 ICC 和阈值。外观面按 `topology.json` 的观看方向旋转后提供给学习指标。图像尺寸错误会直接报错，避免在评测时重新缩放或修补基线输出。

| 输出字段 | 定义与阈值 | 方向 |
| --- | --- | --- |
| `seam_sdr` | 在 12 条 3D 邻接边两侧取对应 2 px 深度的条带，按折叠关系逆序并对齐。先求各缝两侧对应像素的平均 CIEDE2000，再除以六个外观面内水平和竖直相邻像素的平均 CIEDE2000。分母下限为 1.0，避免近乎纯色时比值发散。 | 低 |
| `seam_lpips` | 两侧各取 32 px，拼成接缝为中轴的样本，与两侧各自连续的同宽面内条带分别比较。输入沿接缝长度缩放到 224 px，宽不足 64 px 时扩到 64 px，使用官方 LPIPS AlexNet v0.1，先均衡两侧，再平均各缝。 | 低 |
| `upright_rate` | OpenAI CLIP ViT-B/32 对“正立、倒置、横躺”提示词零样本分类。六个外观面中正立标签占比，0 至 1。 | 高 |
| `ocr_upright_rate` | PP-OCRv5 移动版模型对各正立坐标中的可识别文本行辅助判断。置信度 ≥0.5 且成功给出 0°/180° 的行中，0° 的比例。无证据时留空。 | 高 |
| `glue_ink_pct` | 粘口中任一 RGB 通道小于 250 的像素百分比。留 5 级余量容纳保存或量化误差。 | 低 |
| `flap_edge_pct` | 插舌、防尘翼内部相邻像素的 L*/100 变化大于 0.10 的百分比，跨区域边界不参与。阈值筛选强纹理，补充 OCR 对非文字内容的覆盖。 | 低 |
| `flap_ocr_count` | 每个插舌和防尘翼连通区中置信度 ≥0.5 的非空文字行个数。 | 低 |
| `bleed_fill_pct` | 映射到相邻外观面的出血像素中，非白（任一通道 <250）且与最近外观像素的 CIEDE2000 落在 [0.8, 9] 的百分比。只纳入离外观面在出血宽度内的像素。下限用于排除逐像素复制，上限用于识别明显色差。 | 高 |
| `lighting_residual` | 各外观面 L* 在高斯模糊后逐像素空间梯度幅值的均值，再对六面取均值，单位为 L*/px。高斯 σ 为单面短边的 8%。 | 低 |
| `oog_pct`、`roundtrip_delta_e00` | 对全部可印刷像素（外观面、插舌、防尘翼与出血）执行真实 ICC sRGB→CMYK→sRGB 变换。报告 CIEDE2000 > 2 的比例及平均 CIEDE2000。相对比色并启用黑场补偿，测量阶段直接使用 LittleCMS。 | 低 |
| `clip_score` | 每个正立面的 CLIP 图文余弦相似度下截于 0 后乘 2.5，最后六面平均。遵循原版 CLIPScore 的文本前缀，面图在白背景方形中居中以保留版面。 | 高 |
| `hps_v2` | HPS v2.0 官方 `HPS_v2_compressed.pt` 的图文特征余弦相似度，六面平均。HPS 和 CLIP 使用同一面级提示词。 | 高 |
| `elapsed_seconds`、`peak_memory_mib` | 生成器传入的实测单盒时间和峰值显存。记录 MiB，并在运行记录中注明测量口径。评测时间不含在方法耗时中。 | 低 |

阈值用于 E01 的固定比较。低频亮度代理也会响应刻意使用的设计渐变。出血指标会把本来适合延伸的均匀底色计为复制，因此适合与实际展开图共同解读。CLIP 正立率只能衡量零样本分类器的判断，无方向图案和图文旋转会降低可靠性。CMYK 指标估计所选 ICC 印刷条件的颜色往返误差，实际机台可能有不同表现。

## 印刷特性文件

E01 固定为 ISO Coated v2 (ECI)，FOGRA39L，文件名 `ISOcoated_v2_eci.icc`，SHA-256 为 `128dc02f7246cc3807af0323695379f64151a8f27a587736acc59f8b6ce894b8`。ECI 提供免费公开下载，原作者授权为 HEIDELBERG license，档案镜像记录了使用与再分发条款。源文件从 Debian 归档获取并校验，实验仓库不附 ICC 二进制。 [ECI 下载与说明](https://eci.org/doku.php_id=en_downloads.html) · [Debian 文件与授权](https://sources.debian.org/data/non-free/i/icc-profiles/2.1-2/debian/copyright)

```bash
export PYTHONPATH=code
python -m dielinediff.color build --output "$XDG_CACHE_HOME/icc/iso-coated-v2-lut.npy"
```

生成 33×33×33 规则采样表，ICC 转换使用 8 位通道，采样前将节点舍入成相邻的 8 位输入。PyTorch 以三线性插值进行投影，并保留对输入 RGB 的可导性。`--icc` 可使用已下载的相同文件，评测器会按 LUT 元数据校验特性文件。另用 Ghostscript SWOP ICC 时应视为另一次印刷条件，不与 E01 的 FOGRA39L 数字混用。

## 模型与健全性检查

学习后端默认全部开启。指定 `learned=()` 可先检查确定性几何与颜色指标，关闭的字段为空值。PaddleOCR 可独立装入 CPU 虚拟环境，用 `E01_OCR_PYTHON` 指定。将 `TMPDIR`、`XDG_CACHE_HOME`、`HF_HOME`、`TORCH_HOME` 与 `PADDLE_PDX_CACHE_HOME` 指向项目数据盘。

PaddleOCR 使用 PaddlePaddle 3.2.2 CPU 和 PaddleOCR 3.3.2。模型版本为 PP-OCRv5 mobile 检测 `0d63e78`、识别 `682f205` 及 PP-LCNet x1.0 行方向 `cd237a4`。可预下载各发行方模型的 `inference.*` 文件至 `$PADDLE_PDX_CACHE_HOME/official_models/<模型名>`。HPS 权重使用发行方 `xswu/HPSv2` 的 `697403c` 提交，通过 OpenCLIP 2.32.0 的同构 ViT-H/14 严格载入，并校验 SHA-256。

```bash
PYTHONPATH=code python -m dielinediff.metrics.sanity \
  --geometry "$OUTPUT_ROOT/geometry" \
  --atlas "$OUTPUT_ROOT/geometry/selfcheck/selfcheck-atlas.png" \
  --lut "$XDG_CACHE_HOME/icc/iso-coated-v2-lut.npy" \
  --output "$OUTPUT_ROOT/metric-check/summary.json"
```

检查使用棋盘格加上定向文字、错位、涂墨、空白出血、复制边缘、阴影和高饱和色。CLIPScore 和 HPS 使用匹配与不匹配描述。LUT 梯度以恒等映射及其雅可比检查。此脚本也验证成本字段可传到指标数据，实际 GPU 峰值与耗时由生成器采集。

指标实现依据见 [LPIPS](https://github.com/richzhang/PerceptualSimilarity)、[OpenAI CLIP](https://github.com/openai/CLIP)、[CLIPScore](https://aclanthology.org/2021.emnlp-main.595/)、[HPS v2](https://github.com/tgxs002/HPSv2) 和 [PaddleOCR](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)。


## E01b 原生刀版与印前复位

E01b 分别用旧组的 1024 × 1160 与新版的 1592 × 1792 图集及各自几何进行评分，使用同一个 FOGRA39L ICC 文件。各图以已有分辨率读取。接缝 SDR 和 LPIPS 仍采用 2 px 和 32 px 条带，同一像素带宽在旧组约为 0.48 mm 和 7.75 mm，在新版约为 0.31 mm 和 5.00 mm。比较分数时应考虑这一尺度差异。

`raw` 为已有图集，`preflight` 统一把粘口及刀版外置为纸白，把插舌和防尘翼置为简报中的 `main_color`。六个可见面和出血像素维持原值。按原图报告粘口油墨覆盖率。复位对印刷区域颜色的影响计入复位版 ICC 往返指标，固定区域之外的版面与图文质量使用相同分数。

`image_reward` 使用官方 ImageReward v1.0 权重及官方 BLIP 推理头，在六个正立面上各评分一次，取均值。每面按 E01 的白底居中规则补成方图，交给官方图像预处理。提示词与 CLIPScore 和 HPS 一致，ImageReward 按官方 BERT 配置截到 35 token。模型使用 Apache-2.0 许可证。 [官方模型与权重](https://huggingface.co/zai-org/ImageReward) · [官方代码](https://github.com/zai-org/ImageReward/tree/78f8051059b34fef8e0032ccb40561c23a2ddbbb)

在 E01 环境中运行以下命令准备 E01b 的额外评分依赖、固定修订的权重和 tokenizer。默认把资源放在 `$XDG_CACHE_HOME/image-reward`，可用 `E01B_IMAGE_REWARD_HOME` 覆盖。加上 `--check-only` 可离线核对资源和依赖。

```bash
source code/scripts/e01_env.sh
"$DIELINEDIFF_ROOT/envs/e01/bin/python" -m dielinediff.metrics.prepare_image_reward
```


## E01b 毫米口径与折边条带

折线与出血按统一的 4 px/mm 物理网格计算，默认 RTE 为 992 × 1120。各面先按原生 UV 矩形裁取，再在统一网格中双线性重采样。出血候选像素按设计的 3 mm 外轮廓和离外观面最远 3 mm 筛选。候选像素由两套刀版各自的毫米原点和比例配准，只从原生标签 10 读颜色。对掩码与掩码加权颜色使用相同的高斯抗混叠及双线性采样，采样后按有效权重归一化。没有原生出血支持的候选像素视为未填充。统一插值可能使被复制的颜色在印刷面与出血侧产生不同的局部平均，从而计入少量像素。旧分辨率自检图的原生逐像素复制对照由 0% 变为约 1.13%，纯白对照仍为 0%。C1 色差和 LPIPS 每边分别取 0.5 mm（2 px）与 5 mm（20 px）。顶面、底面指标分别汇总其各自的四条 3D 接缝。SDR 使用同一全盒面内色差作为分母。图文偏好与 CLIP 方向仍从原生 UV 正立面计算。色域指标沿用原生分辨率和同一 ICC。

折边条带只评价原刀版上连接两块可见面的真实压痕。默认 RTE 有五条，两侧均从该面内部取 0–3 mm 窄带，同面参考取 9–12 mm，沿折线方向的长度和对齐相同。能量是沿折线方向 RGB 在 [0,1] 内一阶差分的平方均值。单侧分数为 `1 + max(E_edge - E_ref, 0) / (E_ref + (3/255)^2)`。能量下限用于避免近乎平坦区域把微小量化误差放大，同面参考吸收贯穿版面的高频图案。折线取两侧平均，整盒取五条折线的最大值。数值越高，越需要检查集中于折线附近的高频能量。另记录背面—底面一条用于排查。均匀沿折线延伸的纯色带和局部密集版面可能逃过指标或触发高值，因此同时审核原像素裁图。门槛仍是最好基线均值的 1.2 倍。

印前复位仅改变标签 0、7、8、9，裁取的可见面和带掩码的出血采样与这些像素互斥。于是原图与复位版共用毫米接缝、条带、出血和方向值，色域往返按两套图分别计算。保留原像素口径可用 `E01bEvaluator(..., physical=False)`。

```bash
python -m dielinediff.metrics.physical_sanity \
  --geometry "$DIELINEDIFF_ROOT/outputs/geometry" \
  --atlas "$DIELINEDIFF_ROOT/outputs/geometry/selfcheck/selfcheck-atlas.png" \
  --output "$DIELINEDIFF_ROOT/outputs/e01b/metric-checks/physical-sanity.json"
```


### E01b B2-post 与指定折线关口

B2-post 从 E01 归档 B2 的原尺寸图开始，使用 v2 加载的同一 `.npy` LUT 和同一三线性 RGB 投影函数全幅映射一次，量化为 8 位后得到 `raw`。对该结果执行共用印前复位得到 `preflight`。固定色块因此以复位配置为准，B2 原始归档字节保持原样。B2-post 图文、接缝与印刷指标均从映射后的图重新计算。

任务 05 同时检查 `stripe_index` 与 `back_bottom_stripe_index`。前者取五条可见压痕的最大值，后者只取刀版上相连的背面—底面折线两侧平均。两者各自的上限为相应最好基线均值的 1.2 倍，基线包含 B2-post。两个指数均从可见面计算，印前复位保持其原值。原像素折线裁图仍需单独审阅。
