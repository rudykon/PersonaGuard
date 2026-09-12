# 当前代码的复现指南

命令从项目根目录执行。项目区分仓库检查、合成结构测试和授权数据分析，
各运行入口见[脚本索引](../scripts/README.md)。

## 1. 仓库检查

需要 Python 3.10+，无需 GPU、研究数据或网络访问：

```bash
python3 -B -m unittest -v tests.test_repository_layout
python3 -B scripts/check_repository.py
```

已安装 GNU Make 时，等价命令为 `make check` 和 `make github-check`。
测试使用临时文件验证仓库审计工具；审计读取当前 Git 候选文件，不修改索引、不提交或上传。

## 2. 合成数据结构测试

已有研究环境时直接使用；没有环境时可先准备 NumPy：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install 'numpy>=1.26'
make smoke PYTHON=.venv/bin/python
```

依赖安装可能访问网络，合成测试本身离线运行。
它检查分组折、阈值汇总与证据图结构，输出 `uses_gated_data: false`。
合成测试不生成或复现经验结果。

## 3. 研究环境

完整研究依赖沿用当前版本约束：

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
```

依赖包含 CUDA 12.1 对应的 PyTorch，以及表征提取所用的模型工具。
轻量检查无需安装全部研究依赖。数据访问和缓存目录见[数据说明](../data/DATASET.md)。

## 4. 从授权数据运行最终分析

按实际需要运行以下流程：

1. 准备开发数据、刺激清单、信号缓存和模型表征。
2. 运行内容先验、生理残差或试后稀疏反馈分析。
3. 运行参考轨迹、校准、估计量敏感性和最终测量审查。
4. 汇总正式结果并执行证据来源与协议重放检查。

测量链中的 `run_revision2_analyses.py` 至 `run_revision6_analyses.py` 共同提供最终结果依赖的分析与共用函数。
这些脚本保留既有产物目录和字段约定，以便与来源记录对应。
所有运行均需核对各脚本的信息时点、参与者/视频留出条件、输入缓存和输出目录。
实验重跑不会由 `make check` 或 `make smoke` 自动触发。

`paper_support/revision6_source.json` 是已保存正式分析记录的统一数值接口。
该文件的完整重建与本地材料核查需要 `artifacts/`、特征来源清单及其他本地输入。
这些输入不会随 GitHub 克隆自动获得。

## 5. 本地完整性核查

具备正式分析产物和本地材料时，可运行：

```bash
make publication-check PYTHON=.venv/bin/python
```

该命令只核查，不重新训练或刷新正式结果。完整测试发现也可能包含依赖本地材料的测试，
因此首次使用应从第 1 节的独立仓库检查开始。
