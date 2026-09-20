# 绘图脚本

本目录集中维护 12 个绘图入口及共用工具。所有命令从项目根目录执行。
绘图统一读取 `results/revision6_source.json`；14 个图件 CSV 和 `dense_trajectory_examples.json` 与脚本同放在本目录。
作者提供的四个 SVG 原图与正式 PDF 位于 `paper/figures/`。
Matplotlib 重绘产生的工作导出位于 `artifacts/figures/`。
`paper/` 不随 GitHub 源码分发；作者 SVG 导出与完整材料构建仅适用于具有论文文件的本地工作区。

| 文件 | 用途 |
|---|---|
| `make_revision6_protocol_figure.py` | 协议、评价设计及 Figure 3 |
| `make_revision5_figures.py` | Figure 4：估计量与参考条件 |
| `make_dense_trajectory_examples.py` | Figure 5：匿名轨迹示例 |
| `make_supplementary_figures.py` | 补充图与证据 CSV；S1/S2 调用作者 SVG 导出器 |
| `make_revision6_decision_figures.py` | S4：决策边界敏感性 |
| `make_revision6_validity_figure.py` | 当前流程的有效性图入口 |
| `make_revision3_figures.py` | 当前入口依赖的有效性图函数 |
| `make_revision4_figures.py` | Figure 4 入口依赖的样式和绘图函数 |
| `export_current_protocol_svgs.py` | 作者提供的两幅正文 SVG 转矢量 PDF |
| `export_current_supplementary_svgs.py` | 作者提供的 S1/S2 SVG 转矢量 PDF |
| `manuscript_fonts.py` | Figure 3、4、5、S3、S4 的 Linux Libertine O 字体配置 |
| `vector_export.py` | 共用矢量 PDF 导出工具 |

## 使用

单独重绘图件，例如：

```bash
.venv/bin/python -B scripts/figures/make_dense_trajectory_examples.py
```

核对作者 SVG 与导出 PDF 是否一致：

```bash
.venv/bin/python -B scripts/figures/export_current_protocol_svgs.py --check
.venv/bin/python -B scripts/figures/export_current_supplementary_svgs.py --check
```

完整材料构建由 `scripts/build_revision6_publication.py` 调用这些入口，
并将实际引用的生成 PDF 从 `artifacts/figures/` 同步到 `paper/figures/`。
作者 SVG 导出器直接更新原图旁的 `*_embed.pdf`。
Matplotlib 绘图入口也会在 `artifacts/figures/` 生成 SVG、普通 PDF、PNG/TIFF；
这些是可清理的工作文件，常规检查只要求正式目录中的 11 个 PDF 和 4 个原始 SVG。
共用函数文件由相应入口导入，无需逐个运行。
