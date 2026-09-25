# 项目站点改版

参考站点：[Wearable IMU Activity Segmentation Pipeline](https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/)。
PersonaGuard 采用相同的 MkDocs Material 框架及文档式导航，配合深蓝双层页头、浅蓝渐变首屏、流程卡片、证据摘要、演示预览和资源入口。

## 信息结构

| 页面 | 地址 | 内容 |
|---|---|---|
| 概览 | `/` | 问题 → 方法 → 证据 → 演示 → 资源 |
| 方法 | `/method/` | Fig. 1、五条规则、执行细节、原有 RQ1–RQ3 |
| 评估设计 | `/evaluation/` | Fig. 2、MER-PS 二次分析、外部研究记录 |
| 研究结果 | `/results/` | Fig. 3–5、完整数值与对照表、四案例交互 |
| 讨论 | `/discussion/` | 证据边界、研究局限、后续研究 |
| 浏览器演示 | `/demo/` | 11 个记录、本地规则执行、假设条件、JSON 下载 |
| 复现与资源 | `/reproduce/` | 代码、规则、正式结果、复现命令 |

中文页面位于对应的 `/zh/` 路径。语言切换保留当前章节；两种语言均在构建时生成，关闭 JavaScript 仍可阅读正文。
桌面使用双层导航和章节目录，手机使用 Material 抽屉菜单。站内搜索涵盖中英文内容，支持跟随系统主题及手动切换。

## 内容与边界

- 全部研究章节、RQ、五张主图、完整结果表、四案例正式判定及复审要求从原站迁移。
- 首页波形明确标为报告时间差的示意，非实验数据；卡片中的 R3 来自正式 `signed_calibration` 案例。
- 首屏及结果摘要不将技术指标改善写成实际用户收益；参考报告不是个体情绪真值。
- 正式数值仍由 `results/revision6_source.json` 生成；检查覆盖所有双语源片段。
- 浏览器演示保留原规则引擎与数据，输入不上传。原始案例和假设条件分别标注。
- 不发布论文正文、原始数据、刺激媒体或本地研究历史。

## 源码与发布

- `website/`：MkDocs 页面、双语源片段、共享样式和交互脚本。
- `mkdocs.yml`、`requirements-site.txt`：框架配置与固定的直接依赖。
- `docs/assets/`：已有公开规则数据和图表，生成器继续沿用这些路径。
- `dist/project-site/`：临时构建结果；`make site` 检查后复制到 `docs/`。
- `docs/`：GitHub Pages 发布产物与仓库文档；不手工修改生成 HTML。

发布保持 `main/docs`，不更改 Pages 设置。构建使用公开资源白名单，不复制工作区其他研究材料。
旧 `playground.html` 会携带查询参数跳转到 `demo/`；首页详细结果、图表及规则锚点跳转到相应章节。

## 检查

```bash
make site-check PYTHON=.venv-site/bin/python
make browser-check check
```

站点构建使用 `mkdocs build --strict`，检查 14 个双语正文页面和旧地址入口的本地链接、锚点、资源与标题。
浏览器/Python 差分检查覆盖全部 11 个案例、字段取值、2,000 组组合条件、优先级与非法输入。
浏览器视觉与交互检查另行覆盖桌面和手机、中英文、主题切换、搜索、案例切换及 JSON 导出；不将 Chromium 检查等同于全浏览器实机测试。
