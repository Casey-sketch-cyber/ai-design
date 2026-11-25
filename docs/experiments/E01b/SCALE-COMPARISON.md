# 旧口径与统一物理尺度对照

接缝色差与 LPIPS 采用 0.5 mm 和 5 mm 窄带，出血 3 mm。物理评测使用 4 px/mm，图文质量沿用原图裁面。E01 留出集按简报平均三个种子。

| 批次 | 方法 | 变体 | 指标 | 简报 | 旧均值 | 新均值 | 新−旧 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| E01 holdout | b2 | raw | bleed_fill_pct | 40 | 46.17 | 46.995 | +0.82515 |
| E01 holdout | b2 | raw | seam_lpips | 40 | 0.4469 | 0.4358 | -0.011099 |
| E01 holdout | b2 | raw | seam_sdr | 40 | 5.032 | 5.8911 | +0.85907 |
| E01 holdout | b3 | raw | bleed_fill_pct | 40 | 0.0053316 | 11.522 | +11.516 |
| E01 holdout | b3 | raw | seam_lpips | 40 | 0.49724 | 0.45277 | -0.044474 |
| E01 holdout | b3 | raw | seam_sdr | 40 | 12.811 | 14.034 | +1.2226 |
| E01 holdout | b4 | raw | bleed_fill_pct | 40 | 43.456 | 42.038 | -1.4179 |
| E01 holdout | b4 | raw | seam_lpips | 40 | 0.44718 | 0.40235 | -0.044834 |
| E01 holdout | b4 | raw | seam_sdr | 40 | 4.8546 | 5.2725 | +0.41796 |
| E01 holdout | v1 | raw | bleed_fill_pct | 40 | 2.5244 | 4.9606 | +2.4362 |
| E01 holdout | v1 | raw | seam_lpips | 40 | 0.50433 | 0.48304 | -0.021296 |
| E01 holdout | v1 | raw | seam_sdr | 40 | 8.2713 | 8.4196 | +0.1483 |
| E01 holdout | v1_no_projection | raw | bleed_fill_pct | 40 | 2.4552 | 4.8563 | +2.4011 |
| E01 holdout | v1_no_projection | raw | seam_lpips | 40 | 0.50267 | 0.48259 | -0.020081 |
| E01 holdout | v1_no_projection | raw | seam_sdr | 40 | 8.1662 | 8.3234 | +0.15723 |
| E01 holdout | v1_no_sync | raw | bleed_fill_pct | 40 | 34.421 | 34.057 | -0.36418 |
| E01 holdout | v1_no_sync | raw | seam_lpips | 40 | 0.54995 | 0.51746 | -0.032489 |
| E01 holdout | v1_no_sync | raw | seam_sdr | 40 | 10.92 | 11.295 | +0.3748 |
| E01b task03 pilot | b2 | preflight | bleed_fill_pct | 5 | 53.623 | 55.74 | +2.1168 |
| E01b task03 pilot | b2 | preflight | seam_lpips | 5 | 0.3882 | 0.35801 | -0.030183 |
| E01b task03 pilot | b2 | preflight | seam_sdr | 5 | 2.7836 | 3.628 | +0.84443 |
| E01b task03 pilot | b2 | raw | bleed_fill_pct | 5 | 53.623 | 55.74 | +2.1168 |
| E01b task03 pilot | b2 | raw | seam_lpips | 5 | 0.3882 | 0.35801 | -0.030183 |
| E01b task03 pilot | b2 | raw | seam_sdr | 5 | 2.7836 | 3.628 | +0.84443 |
| E01b task03 pilot | b3 | preflight | bleed_fill_pct | 5 | 0 | 14.841 | +14.841 |
| E01b task03 pilot | b3 | preflight | seam_lpips | 5 | 0.56142 | 0.52107 | -0.040353 |
| E01b task03 pilot | b3 | preflight | seam_sdr | 5 | 16.407 | 17.939 | +1.5315 |
| E01b task03 pilot | b3 | raw | bleed_fill_pct | 5 | 0 | 14.841 | +14.841 |
| E01b task03 pilot | b3 | raw | seam_lpips | 5 | 0.56142 | 0.52107 | -0.040353 |
| E01b task03 pilot | b3 | raw | seam_sdr | 5 | 16.407 | 17.939 | +1.5315 |
| E01b task03 pilot | b4 | preflight | bleed_fill_pct | 5 | 20.666 | 19.84 | -0.82645 |
| E01b task03 pilot | b4 | preflight | seam_lpips | 5 | 0.36849 | 0.3202 | -0.04829 |
| E01b task03 pilot | b4 | preflight | seam_sdr | 5 | 7.0813 | 7.3089 | +0.22764 |
| E01b task03 pilot | b4 | raw | bleed_fill_pct | 5 | 20.666 | 19.84 | -0.82645 |
| E01b task03 pilot | b4 | raw | seam_lpips | 5 | 0.36849 | 0.3202 | -0.04829 |
| E01b task03 pilot | b4 | raw | seam_sdr | 5 | 7.0813 | 7.3089 | +0.22764 |
| E01b task03 pilot | v1 | preflight | bleed_fill_pct | 5 | 2.7327 | 5.0351 | +2.3024 |
| E01b task03 pilot | v1 | preflight | seam_lpips | 5 | 0.49012 | 0.46177 | -0.028352 |
| E01b task03 pilot | v1 | preflight | seam_sdr | 5 | 8.7854 | 9.0543 | +0.26886 |
| E01b task03 pilot | v1 | raw | bleed_fill_pct | 5 | 2.7327 | 5.0351 | +2.3024 |
| E01b task03 pilot | v1 | raw | seam_lpips | 5 | 0.49012 | 0.46177 | -0.028352 |
| E01b task03 pilot | v1 | raw | seam_sdr | 5 | 8.7854 | 9.0543 | +0.26886 |
| E01b task03 pilot | v2 | preflight | bleed_fill_pct | 5 | 38.52 | 36.77 | -1.75 |
| E01b task03 pilot | v2 | preflight | seam_lpips | 5 | 0.42461 | 0.39735 | -0.027268 |
| E01b task03 pilot | v2 | preflight | seam_sdr | 5 | 4.1506 | 4.6192 | +0.46862 |
| E01b task03 pilot | v2 | raw | bleed_fill_pct | 5 | 38.52 | 36.77 | -1.75 |
| E01b task03 pilot | v2 | raw | seam_lpips | 5 | 0.42461 | 0.39735 | -0.027268 |
| E01b task03 pilot | v2 | raw | seam_sdr | 5 | 4.1506 | 4.6192 | +0.46862 |
