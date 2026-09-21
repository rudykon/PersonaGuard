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

页面源文件为 [index.html](index.html)，样式和交互位于 `assets/`。
按论文顺序组织为 Overview、Method、Evaluation、Results、Discussion、Resources。
首屏从“让常用功能更容易找到”的界面需求开场，桌面显示三种菜单选择的静态示意；手机保留场景文字。菜单依据外部文献案例，仅作说明情境。连续自评工作案例在评估区另行介绍。
四层结果、论文主图、主要发现及必要数值默认可见；四案例交互位于结果综合部分，默认选中画像解释。
使用原生 HTML/CSS/JavaScript，无需前端框架。GitHub Pages 从 `main` 分支的 `/docs` 发布。

```bash
python3 -m http.server 8000 --directory docs
python3 -B scripts/build_project_page_data.py --check
```

浏览 `http://localhost:8000`。页面内容分为以下几层：

- [case-copy.js](assets/case-copy.js)：按案例 ID 编写的双语解释、综合表摘要与复审要求。`expected_action` 必须与正式判定一致。
- [cases.js](assets/cases.js)：生成文件，不手工编辑。保存解析器输出、正式规则、来源记录中的责任角色与复审触发条件，以及由论文数值生成函数读取公开结果后生成的指标。
- [figures](assets/figures/README.md)：五张完整主图的网页导出、授权范围和文件摘要。

修改公开协议、案例或数值来源后，运行 `python3 -B scripts/build_project_page_data.py` 更新数据与页面中的 `data-number` 数值。
`--check` 核对案例 ID、双语文案、预期行动、生成文件及页面数值。来源字段由 `results/evidence_traceability.json` 连接到案例，不新增规则条件。
修改案例解释时，同时维护无 JavaScript 默认记录及综合表；新复审文案是对现有要求的摘要，不是新实验结果或许可。

完整论文题目以普通字号保留在首屏，章节采用短标题。校准示例、执行细节、原始字段、额外敏感性说明、后续计划与命令默认折叠。案例支持键盘切换，移动端保留六栏导航和滚动高亮。
页面不在浏览器中审查任意新输入，也不提供未公开的论文下载入口。

栏目与论文章节的对应关系、模块迁移和检查结果见 [本轮改版说明](PROJECT_PAGE.md)。
