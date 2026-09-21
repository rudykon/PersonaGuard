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

## 论文海报项目页

线上页面：https://rudykon.github.io/PersonaGuard/

页面源文件为 [index.html](index.html)，样式和交互位于 `assets/`。
阅读顺序为项目定义、审查示例、五个问题、方法检查与局限、项目资源；默认展示画像复用案例。
使用原生 HTML/CSS/JavaScript，无需前端依赖或构建。GitHub Pages 从 `main` 分支的 `/docs` 发布。

```bash
python3 -m http.server 8000 --directory docs
python3 -B scripts/build_project_page_data.py --check
```

浏览 `http://localhost:8000`。页面内容分为两层：

- [case-copy.js](assets/case-copy.js)：按案例 ID 编写的双语通俗说明。`expected_action` 用于核对其对应的正式判定；修改时同时核对页面中的无 JavaScript 默认示例。
- [cases.js](assets/cases.js)：生成文件，保存解析器输出、正式规则条件与优先级，不手动编辑。

修改公开协议或案例后，运行 `python3 -B scripts/build_project_page_data.py` 更新数据；
`--check` 同时检查生成文件、案例 ID、双语说明是否完整，以及说明对应的正式判定是否改变。
首屏仅预告案例，手机端显示简短入口；完整说明只在案例区展示。
规则首次展开仅解释理由及对下一步的影响，正式条件通过源码链接查看。
执行顺序、检查设计与数量、实验数值和命令默认折叠。页面展示已记录案例，支持键盘选择；不在浏览器中审查任意新输入。
页面不包含论文附件、`references/` 或私有数据。
