# RTE 纸盒刀版

在仓库根目录生成 70 × 45 × 160 mm 纸盒，并检查其折叠方向。

```bash
PYTHONPATH=code python -m dielinediff.dieline --output outputs/geometry
PYTHONPATH=code python -m dielinediff.render selfcheck --geometry outputs/geometry --output outputs/selfcheck
PYTHONPATH=code python -m dielinediff.render render --geometry outputs/geometry --texture outputs/selfcheck/selfcheck-atlas.png --output outputs/render
PYTHONPATH=code python -m unittest discover -s code/tests -p test_dieline.py
```

刀版采用反向插舌式布局。上盖连接正面，下盖连接背面。插舌倒角考虑纸厚，网格为按标称外轮廓构造的六个零厚度贴图面。SVG 包含裁切、压痕、3 mm 出血和 3 mm 安全线四个图层，可通过命令行修改纸盒尺寸和翼片尺寸。

`region_map.png` 是单通道无损标签图，编号见 `region_labels.json`。`bleed_owner.png` 用相同的 1–6 号标签将出血像素归属于距离最近的外观面。其余像素值为 0。

`topology.json` 的 `rect` 和 `safe_rect` 为左上原点、右下排除的像素矩形。`uv` 以左下为原点，次序为 `[u_min, v_min, u_max, v_max]`。`upright_rotation` 为将面转到折叠后正立方向所需的逆时针角。每条接边由两对面名和边名及 `reverse` 表示。水平边从左向右，垂直边从上向下。

`box.obj` 采用 +Y 朝上、+Z 朝正面的坐标，每个面有两三角片和独立的 UV。OBJ 随附单个 MTL 和白色占位图，便于第三方贴图系统沿用刀版 UV。检查渲染使用 trimesh 读取 OBJ，在 CPU 上执行无光照正交投影。
