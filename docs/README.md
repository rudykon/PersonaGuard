# 项目文档导航

这里提供当前代码的安装、运行、数据准备与分享说明。
正式汇总结果以 `results/revision6_source.json` 为准。

| 任务 | 入口 |
|---|---|
| 安装环境、运行检查或复现实验 | [复现指南](REPRODUCING.md) |
| 查找协议、特征和分析命令 | [脚本索引](../scripts/README.md) |
| 准备授权数据与缓存 | [数据说明](../data/DATASET.md) |
| 查看协议规则和案例 | [规则](../configs/protocol/protocol_rules.json)、[案例](../configs/protocol/protocol_replay_cases.json) |
| 整理 GitHub 发布内容 | [上传前检查](GITHUB.md) |

代码按协议执行、数据与特征、内容与反馈分析、测量与校准四条流程组织。
各入口的输入条件见脚本索引，命令参数使用对应脚本的 `--help` 查看。

协议配置见 [configs/protocol](../configs/protocol/README.md)，汇总数值和证据记录见 [results](../results/README.md)。

## 学术项目主页

线上页面：https://rudykon.github.io/PersonaGuard/

网站采用 **MkDocs Material**，参考 Wearable IMU 项目站点的页面结构和视觉风格。
导航分为概览、方法、研究证据（评估、结果、讨论）、浏览器演示、复现与资源。
提供独立中英文地址、站内搜索、深浅色主题和手机导航。

源文件位于 [website](../website/)，配置见 [mkdocs.yml](../mkdocs.yml)。
`docs/index.html` 及各分区 HTML 是生成产物，**请编辑源文件后重新构建**。
GitHub Pages 保持从 `main` 分支的 `/docs` 发布，不需要更改仓库设置。

```bash
python3 -m venv .venv-site
.venv-site/bin/pip install -r requirements-site.txt
make site PYTHON=.venv-site/bin/python
make site-serve PYTHON=.venv-site/bin/python
```

预览地址以 MkDocs 输出为准。只构建检查、不更新 `docs/`：

```bash
make site-check PYTHON=.venv-site/bin/python
make browser-check
```

内容维护：

- [双语内容](../website/overrides/content/)：首页和全部研究章节，使用 `data-en` / `data-zh` 保持逐段对应，构建时生成可独立阅读和搜索的两种语言。
- [共享样式](../website/stylesheets/personaguard.css)：导航、首屏、研究图表、演示页面、移动端和深色主题。
- [case-copy.js](assets/case-copy.js)：四案例的双语解释；`expected_action` 必须与正式判定一致。
- [cases.js](assets/cases.js)、[audit-data.js](assets/audit-data.js)：生成数据，不手工编辑。
- [图表说明](assets/figures/README.md)：五张完整主图及其授权边界。

修改正式规则、案例或数值来源后，依次运行数据生成和网站构建：

```bash
python3 -B scripts/build_project_page_data.py
python3 -B scripts/build_browser_demo_data.py
make site PYTHON=.venv-site/bin/python
```

构建检查全部源章节中的 `data-number`、案例判定、中英文页面、站内链接、锚点与资源。
`playground.html?case=...` 和原首页的详细研究锚点兼容跳转到新页面。
浏览器演示的执行边界见 [PLAYGROUND.md](PLAYGROUND.md)，页面迁移说明见 [PROJECT_PAGE.md](PROJECT_PAGE.md)。
