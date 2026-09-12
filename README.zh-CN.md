# 🧭 From Evidence to Action

<p align="center">
  <img src="docs/brand-mark.svg" width="520" alt="PersonaGuard brand mark">
</p>

**面向 HCI 系统的个性化决策证据审查协议**

[English](README.md) · **简体中文**

可执行的证据到行动审查协议与 HCI 研究复现代码。

本项目关注：**现有证据能够支持哪些个性化决策，又应在何处止步？**
我们将证据类型、评估条件和决策规则组织为可执行协议，覆盖离线估计、可选生理传感、
行为校准，以及画像保留与迁移。协议明确记录何时可在已评估边界内个性化、
何时应保留比较方案，以及何时需要补充证据或开展用户研究。

项目以连续效价—唤醒度数据为工作实例，展示协议如何组织证据与决策。
现有验证支持结构符合性、跨来源可表示性和确定性重放；独立分析者复用、协议可用性与用户收益仍待评估。

## 🗺️ 项目导航

| 想做什么 | 从这里开始 |
|---|---|
| 安装环境或复现实验 | [复现指南](docs/REPRODUCING.md) |
| 查找某项分析的运行命令 | [脚本索引](scripts/README.md) |
| 查看协议规则与审查案例 | [协议规则](paper_support/protocol_rules.json) · [重放案例](paper_support/protocol_replay_cases.json) |
| 查找当前流程的使用说明 | [文档导航](docs/README.md) |
| 准备 GitHub 发布 | [上传前检查](docs/GITHUB.md) |

## 🚀 快速开始

在项目根目录使用 **Python 3.10+** 运行以下命令。
仓库检查只使用 Python 标准库，无需 GPU、研究数据或网络访问。

```bash
# 运行独立的仓库检查
python3 -B -m unittest -v tests.test_repository_layout

# 只读检查 Git 候选文件与本地链接
python3 -B scripts/check_repository.py
```

审计读取工作区文件，不修改 Git 索引，也不执行提交或上传。
已安装 GNU Make 时，`make github-check` 可执行同一审计。

已有包含 NumPy 的研究环境后，可运行合成数据结构测试：

```bash
make smoke PYTHON=.venv/bin/python
```

该测试用合成数据检查分组折、阈值汇总与证据图结构，不复现经验结果。
完整分析需要相应的授权输入和缓存，具体流程见[脚本索引](scripts/README.md)。

## 🗂️ 目录与证据流

```text
CHI2027/
├── paper_support/  汇总结果、协议规则与案例、图件源码及源数据 CSV
├── src/merps/      可复用算法与模型
├── scripts/        分析、生成、验证与打包入口
├── tests/          单元测试及研究结果一致性检查
├── configs/        实验配置与刺激清单
├── docs/           当前流程、复现指南与发布说明
├── data/           本地数据与缓存；仅选定说明文件纳入 Git
├── checkpoints/    本地模型权重
├── artifacts/      本地实验结果、检查报告与编译输出
└── .local/         本机凭据与交接记录
```

汇总结果与审查决策沿同一证据链生成：

```text
研究输入与正式分析
        ↓
paper_support/revision6_source.json
        ↓
结果摘要、源数据表与审查输出
```

当前维护的代码覆盖协议重放、特征准备、内容先验、生理残差、稀疏反馈与测量审查。
各流程的运行入口和输入要求见[脚本索引](scripts/README.md)。

## 🔧 环境与共享说明

`requirements.txt` 保存完整研究环境的依赖约束，包括固定的 CUDA/PyTorch 版本；
`pyproject.toml` 提供本地包安装入口。轻量检查无需先安装全部训练依赖。
数据授权、环境要求和运行成本见[复现指南](docs/REPRODUCING.md)。

- **本地材料**：原始数据、刺激视频、参与者级完整输出、模型权重、凭据和备份不纳入普通仓库发布。
- **支撑材料**：汇总数据、源图和匿名示例按各自来源与许可核查分享范围。
- **许可证**：本项目尚未设置整体开源许可证；材料可读取不代表已获再分发授权。
- **发布准备**：Git 历史、第三方许可与匿名审稿身份需单独检查，详见[上传前检查](docs/GITHUB.md)。

<a id="results"></a>

## 📊 当前结果与来源

下方摘要由唯一数值源自动生成，与汇总结果及审查记录保持一致。
可展开查看，或直接阅读[数值源文件](paper_support/revision6_source.json)。

<details>
<summary>展开自动生成的详细结果</summary>

<!-- REVISION6_RESULTS:START -->
## Revision 6 唯一数值源与正式结论

以下结果由 `paper_support/revision6_source.json` 自动生成；汇总表格、结果摘要与图件不得直接读取早期 revision 目录。

- 锁定协议已在 4 个路线级案例上完成结构重放，产生 4 种不同决策；100 次顺序扰动结果不变，12/12 个非法记录被拒绝，5/5 个路线局部性探针、30/30 个正向授权边界探针及4/4 个声明优先级探针通过。这些检查只证明 schema、哈希、边型与传播规则的结构可复现性，不证明 substantive validity、分析者一致性或跨领域通用性。
- 同一声明式 resolver 还处理了 6 项独立作者团队的公开 HCI 研究所形成的 7 条路线，无需 schema 扩展且没有 fallback；其中 5 条被许可在已评估边界内个性化，2 条保留比较方案或进入用户研究。五条规则逐条消融均改变至少一个动作。这是跨来源可表示性证据，不是独立分析者一致性或实质正确性证明。
- 原始刺激审计确认 15/15 个文件存在且身份匹配。在 held-out participant $\times$ held-out released-video 的开发集 nested OOF 中，元数据时间先验 MAE 为 31.333，冻结 CLIP+SigLIP 内容条件化先验为 30.493（增益 .840）。该结果只覆盖已发布的 15 个视频支持点，不是新视频总体主张。
- 在同一内容预测之上，主因果 EEG/fNIRS 残差门槛下 MAE 为 30.499、增益 -.006，该受评价残差未证明增量，审计建议停用并保留内容。逐样本精确回退适用于残差停用后的输出；另一个时间令牌条件在主门槛下已选中零残差。不得把完整模态的受评价结果写成已等于基线，也不作等效或生理信号无信息解释。
- 每个试次结束后使用一个 SAM 效价--唤醒度对时，Gaussian retrieval + functional residual 稀疏恢复 MAE 为 26.289，相对已知视频群体先验增益 2.127 [.557,3.878]。这是试后稀疏反馈，不是零交互实时预测。
- 严格抗混叠主分析中，video mean MAE 为 .763 s；blockwise residual stack 为 .794 s，相对 video mean 的增益为 -.031 s [-.081,.014]。结论是“所评估流程未证明增量”，不是等效性或生理信号无效。
- 五折 outer、四折 inner 的 grouped CV 共重复 5 组参与者划分；每组均重建 reference 与 target。主 sensing 路径增益范围为 -.031---.001 s，0/5 组划分为正。
- Signed calibration 的 1/2/4/8-video 增益为 .021/.028/.045/.063 label units。阈值曲线只显示 24 名参与者的精确计数与比例，不再提供条件式 Wilson bands；它们不建立等效、实际价值或用户收益。
- 原概化理论符号已移除，输出重命名为 relative/absolute shared-reference 描述性方差分数；重叠 leave-one-out 参考诱发的依赖未被常规 ANOVA 模型刻画，因此不再使用 .70 gate，也不作常规概化系数或可靠性解释。
- 同试次 own-versus-donor 结果仅作为共享设备、共享时钟和共享任务条件下的界面依赖跨轴对齐证据；不把它解释为稳定个人属性。
- 设计结论聚焦何时不应个性化：当增量、稳定性、后果终点或治理证据不足时，保留透明无传感器基线并记录 abstention。
<!-- REVISION6_RESULTS:END -->

</details>
