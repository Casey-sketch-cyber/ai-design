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
