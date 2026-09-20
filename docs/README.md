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
