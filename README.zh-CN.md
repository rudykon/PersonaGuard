# PersonaGuard

<p align="center">
  <img src="docs/brand-mark.svg" width="520" alt="PersonaGuard brand mark">
</p>

**面向 HCI 系统的个性化决策证据审查协议**

[English](README.md) · 简体中文

本项目以可执行协议判断现有证据能够支持哪些个性化决策。
以连续效价—唤醒度估计为实例，覆盖内容先验、生理传感、稀疏反馈及画像保留与迁移。

## 快速开始

需要 Python 3.10+ 和 Git，在项目根目录运行：

```bash
# 仓库测试与文件检查，无需研究数据或 GPU
make check github-check

# 使用 Python 标准库重放内置审查案例
python3 -B scripts/run_protocol_replay.py --resolve-case-set configs/protocol/protocol_replay_cases.json
```

未安装 Make 时，检查命令为 `python3 -B -m unittest -v tests.test_repository_layout tests.test_export_github`
和 `python3 -B scripts/check_repository.py`。
NumPy 合成测试与完整研究环境见[复现指南](docs/REPRODUCING.md)。

## 项目结构

```text
src/merps/       可复用算法
scripts/        分析、检查、导出及绘图源数据
configs/        协议规则、案例与刺激清单
results/        汇总结果与证据记录
tests/          单元测试及研究一致性检查
docs/           复现与 GitHub 上传说明
data/           仅发布数据获取和来源说明
```

| 任务 | 入口 |
|---|---|
| 安装依赖、复现实验 | [复现指南](docs/REPRODUCING.md) |
| 查找分析与绘图命令 | [脚本索引](scripts/README.md) |
| 查看协议规则和案例 | [协议说明](configs/protocol/README.md) |
| 查看结果和来源 | [结果说明](results/README.md) |
| 导出可上传 GitHub 的源码包 | [上传指南](docs/GITHUB.md) |

## 结果与共享范围

[`results/revision6_source.json`](results/revision6_source.json) 是唯一汇总数值源。
现有检查支持结构符合性与确定性重放；独立分析者复用、协议可用性与用户收益仍待评估。

原始数据、模型权重、实验缓存、中英文论文、凭据和备份留在本地。
完整分析需要授权数据及相应上游产物。本项目尚未选定整体开源许可证。

<a id="results"></a>

<details>
<summary>查看自动生成的详细结果</summary>

<!-- REVISION6_RESULTS:START -->
## Revision 6 唯一数值源与正式结论

以下结果由 `results/revision6_source.json` 自动生成；汇总表格、结果摘要与图件不得直接读取早期 revision 目录。

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
