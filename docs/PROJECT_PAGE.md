# 项目页结构与检查

页面保留论文的论证顺序，以本研究分析的 MER-PS 视频观看与连续自评任务为主线。
完整论文题目仍在首屏，以 17 px 普通信息字号显示；论文正文不随网页发布。

## 栏目与论文章节

| 页面栏目 | 对应论文章节 | 页面内容 |
|---|---|---|
| Research Questions / 研究问题（Overview） | Introduction + Background and Related Work | 视频观看与摇杆报告任务、时间差异及其可能用途、原有 RQ1–RQ3 |
| Method / 方法 | Evidence-to-Action Audit Protocol | Fig. 1；分别审查解释时间差、部署校正、保存画像；规则入口和校准详情 |
| Evaluation / 评估设计 | Evaluation Methods | Fig. 2；MER-PS 二次分析与已有界面研究记录，区分证据生成与规则检查 |
| Results / 研究结果 | Results | 按原顺序展示规则测试、画像解释、新增信息的作用、审查结论 |
| Discussion / 讨论 | Discussion + Conclusion | 研究启示与局限；结尾回到观看视频、报告感受的人 |
| Resources / 资源 | Data and Code Availability 的公开部分 | 代码与规则、公开结果、复现指南 |

### 结果与主图

| 网页标题 | 论文章节 | 图表与结论 |
|---|---|---|
| Rule Tests / 规则测试 | Route Discrimination and Structural Conformance | Fig. 3；四条本研究记录、七条外部作者编码记录，包含五条有条件许可 |
| Profile Interpretation / 画像解释 | Interpretation Boundaries of the Candidate Profile | Fig. 4 A–C；测量与参考条件限制画像解释 |
| Added Information / 新增信息的作用 | Incremental Evidence Across Information Conditions | Fig. 4D、Fig. 5；先看传感与校准，再看额外轨迹分析，保留不同目标、信息时点和对照 |
| Review Outcomes / 审查结论 | Evidence-to-Action Synthesis | 四用途综合表、四案例交互、责任角色与复审条件 |

## 浏览器本地运行

新增 [运行案例](playground.html) 页面，主页首屏及每个案例均可进入。
原始案例展示与正式判定保持不变。新页面使用同一份生成的规则和证据词表，允许修改条件后本地执行；明确显示原始行动、假设条件、命中规则与优先级轨迹。
默认选择与当前主线一致的有符号校准案例，支持 4 条工作案例及 7 条外部作者编码记录；结果可下载为 JSON。
实现、隐私边界和差分测试见 [运行案例说明](PLAYGROUND.md)。

## 本次主线与依据

首屏问题改为“记录对齐了，就更懂你了吗？”。原乘车码情境与手机示意已移除。

- 开头：从参与者观看视频、用二维摇杆持续报告感受讲起。示意只解释采集任务，不复刻原始实验界面，不绘制虚构数据曲线。
- 概述和方法：同一份报告可以用于解释时间差、估计个人校正或形成供以后使用的画像；这些用途需要分别提供证据。
- 评估：明确本文对 MER-PS 已发布数据进行二次分析，没有新增采集或部署实验；已有界面研究记录用于检查同一套审查规则。
- 结果综合：回到报告感受的人，区分校正后的参考误差改善与尚未检验的用户收益，连接正式校准案例及其复审要求。
- 结尾：让记录更接近参考只回答测量问题；是否调整系统、是否长期保存，仍需分别判断。

公开依据：

| 内容 | 来源与边界 |
|---|---|
| 24 名参与者、15 个视频、连续愉悦度与唤醒度记录 | [MER-PS 数据说明](https://huggingface.co/datasets/MER-PS/MER-PS-trainval)、[项目数据说明](../data/DATASET.md) |
| 报告相对参考曲线的时间差异 | [证据记录](../results/evidence_traceability.json) 中的画像解释、计算与参考条件；参考报告不是个体情绪真值 |
| 个人校正的用途与行动 | [正式案例](../configs/protocol/protocol_replay_cases.json) 中的 `signed_calibration`；改善的是参考误差，尚未验证用户收益 |
| 校正数值与其他结果 | [数值来源](../results/revision6_source.json)，保持原样 |

场景来自真实研究任务，不编造某个参与者的经历、原话或产品效果。
既有外部菜单研究仍保留在正式结果区，与 MER-PS 工作案例分开。
首屏任务示意使用原生 HTML/CSS/SVG，桌面及手机均显示；未加入刺激视频、原始数据、参与者身份或新实验判定。

## 数据与公开范围

- 正式规则、解析器、案例判定、`case-copy.js`、生成数据 `cases.js` 和数值来源均未修改。
- 原有 RQ、五张主图、结果图表、案例模块和讨论中的科学局限均保留。
- 所有 `data-number` 标记与 `results/revision6_source.json` 的生成结果一致。
- 规则顺序仍为 R4 → R2 → R3 → R1 → R5。R1 不是许可；R5 需满足自身条件且没有更高优先级约束；无规则匹配时请求补充证据。
- 外部案例由作者整理，不是独立分析者对审查结论的验证。执行一致不等于现实决策改善。
- 五张网页主图沿用既有授权范围，见 [图表说明](assets/figures/README.md)。
- 不加入论文 PDF、论文目录、`references/`、受限数据或完整研究产物，也不创建论文下载入口。

## 中英文桌面与移动端检查

本地 Chromium，宽度 320、390、720、721、1050、1051、1440 px，高度 844 px；分别检查英文与中文。

| 检查 | 结果 |
|---|---|
| 两种语言 × 七种屏宽 × 四案例 | 56 组通过，正式行动与复审条件一致 |
| 数据任务示意与来源 | 所有宽度可见；正文标签至少 16 px，明确示意和参考边界 |
| 主线呼应与页面摘要 | 双语一致；旧乘车码情境已移除 |
| 六栏、四层结果、五张主图 | 顺序保留，主图默认可见并完整加载 |
| 导航、锚点与滚动高亮 | 通过，目标不被页头遮挡 |
| 默认及全部详情展开 | 无整页横向溢出 |
| 正文、案例和论文题目字号 | 正文 17 px；案例证据至少 16 px；论文题目 17 px |
| 键盘、语言偏好、复制命令、无 JavaScript | 通过 |
| 页面数据与正式判定 | `scripts/build_project_page_data.py --check` 通过 |
| 浏览器脚本与资源错误 | 无错误 |

检查针对浏览器视口，不代表手机实机或 Safari/Firefox 测试。
