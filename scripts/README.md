# 当前代码入口

本目录维护最终证据审查流程、数据准备、分析与验证工具。
所有命令从项目根目录运行；环境要求见[复现指南](../docs/REPRODUCING.md)。

## 快速检查

| 任务 | 命令 | 要求 |
|---|---|---|
| 仓库结构与工具检查 | `make check` | Python 3.10+；只使用标准库 |
| Git 候选文件审计 | `make github-check` | 只读扫描，不提交或上传 |
| 合成数据结构测试 | `make smoke PYTHON=.venv/bin/python` | 需要 NumPy；不读取受限数据 |

## 协议与结果

- `run_protocol_replay.py`：解析类型化记录、执行决策规则并进行结构重放；自有案例使用 `--resolve-case-set`，具体参数见 `--help`。
- `generate_evidence_traceability.py`、`refresh_evidence_traceability.py`：核查证据图与正式产物的来源哈希。
- `build_revision6_source.py`：汇集正式分析记录，构建 `paper_support/revision6_source.json`。
- `summarize_algorithm_experiments.py`：汇总内容先验、生理残差与稀疏反馈实验。
- `run_revision6_synthetic_smoke.py`：用合成输入检查分组折、阈值汇总和证据图结构。
- `check_repository.py`：审计 Git 候选文件、私有路径、大文件及双语首页链接。

## 数据与特征

| 阶段 | 入口 |
|---|---|
| 授权开发数据下载 | `download_data.sh` |
| 信号缓存与手工特征 | `prepare_innovation_cache.py` |
| CBraMod EEG 表征 | `extract_cbramod_embeddings.py` |
| 刺激身份审计与低层特征 | `prepare_stimulus_features.py` |
| 音视频基础模型特征 | `prepare_foundation_features.py` |
| CLIP、SigLIP、DINOv2 视觉特征 | `prepare_visual_backbone_features.py` |
| 多骨干特征归档 | `build_multibackbone_archive.py` |

数据目录与访问要求见[数据说明](../data/DATASET.md)，刺激清单位于 `configs/stimuli/`。

## 内容、生理与反馈分析

| 分析 | 入口 |
|---|---|
| 元数据时间先验 | `run_content_prior_experiments.py` |
| 音视频内容先验 | `run_av_content_prior.py` |
| 元数据条件下的因果生理残差 | `run_causal_physio_residual.py` |
| 内容条件下的因果生理残差 | `run_av_physio_residual.py` |
| 内容骨干选择与敏感性 | `run_content_backbone_router.py` |
| 已知视频的试后稀疏反馈恢复 | `run_sparse_anchor_optimization.py` |

这些分析使用各自的信息条件与比较方案。实验运行需要相应的授权数据、特征缓存和模型来源。

## 测量与校准分析链

以下入口共同生成最终结果所依赖的证据。文件中的 revision 编号对应固定的产物接口。

| 阶段 | 入口 |
|---|---|
| 参考轨迹与动态分解 | `run_normative_dynamics.py` |
| 参与者—视频画像矩阵 | `run_phase_trait_analysis.py` |
| 跨视频校准与折内选择 | `run_phase_personalization.py` |
| 模拟恢复、扰动和测量敏感性 | `run_measurement_robustness.py` |
| 共用统计推断与参考敏感性 | `run_revision2_analyses.py` |
| 嵌套分组交叉验证与参考重建 | `run_revision3_analyses.py` |
| 估计量、DTW 目标与带符号校准 | `run_revision4_analyses.py` |
| 身份排除重采样与校准负担 | `run_revision5_analyses.py` |
| 匹配供体、类别异质性、抗混叠及阈值审查 | `run_revision6_analyses.py` |

`src/merps/features.py` 提供当前流程所需的 MATLAB 读取与生理特征函数；
`src/merps/innovation/` 提供数据索引、信号缓存、动态分解和校准工具。

## 本地材料工具

`build_revision6_publication.py`、`generate_revision6_publication.py`、
`generate_submission_readiness.py`、`generate_word_count_report.py`、
`prepare_dense_trajectory_examples.py`、`verify_references.py`、
`validate_submission_pdf.py`、`build_anonymous_supplement.py` 和
`build_manuscript_review_bundle.py` 用于本地结果整理、核验与材料构建。
它们依赖各自的本地输入；不属于新克隆的默认检查命令。
