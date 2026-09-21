# 项目页结构与检查

页面保留论文的论证顺序，使用短标题和网页摘要，以界面设计需求作为开场。
完整论文题目仍在首屏，以 17 px 普通信息字号显示；论文正文仅作为内容依据，不随网页发布。

## 栏目与论文章节

| 页面栏目 | 对应论文章节 | 页面内容 |
|---|---|---|
| Research Questions / 研究问题（Overview） | Introduction + Background and Related Work | 常用功能更容易找到的需求、三种菜单选择，以及原有 RQ1–RQ3 |
| Method / 方法 | Evidence-to-Action Audit Protocol | Fig. 1、针对具体用途的下一步、人工判断与程序职责；规则入口和校准详情 |
| Evaluation / 评估设计 | Evaluation Methods | Fig. 2、规则测试与数据分析；明确菜单情境与连续自评工作案例的关系 |
| Results / 研究结果 | Results | 按原顺序展示规则测试、画像解释、新增信息的作用、审查结论 |
| Discussion / 讨论 | Discussion + Conclusion | 研究启示与主要局限各一段；详细限制和后续计划可展开 |
| Resources / 资源 | Data and Code Availability 的公开部分 | 代码与规则、公开结果、复现指南 |

### 结果与主图

| 网页标题 | 论文章节 | 图表与结论 |
|---|---|---|
| Rule Tests / 规则测试 | Route Discrimination and Structural Conformance | Fig. 3；四条本研究记录、七条外部作者编码记录，包含五条有条件许可 |
| Profile Interpretation / 画像解释 | Interpretation Boundaries of the Candidate Profile | Fig. 4 A–C；测量与参考条件限制画像解释 |
| Added Information / 新增信息的作用 | Incremental Evidence Across Information Conditions | Fig. 4D、Fig. 5；先看传感与校准，再看额外轨迹分析，保留不同目标、信息时点和对照 |
| Review Outcomes / 审查结论 | Evidence-to-Action Synthesis | 四用途综合表、四案例交互、责任角色与复审条件 |

## 本次场景衔接

- 首屏替换为“设计团队希望常用功能更容易找到 → 固定、系统调整、用户定制三种选择 → PersonaGuard 帮助逐项判断”。项目名、短标题、论文题目和三个按钮保持原位。
- 桌面端使用原生 HTML/CSS 展示三个静态菜单。三种方案采用相同样式，不标优劣、分数或通过标记；1050 px 及以下隐藏示意，保留场景文字与来源说明。
- 概述直接引出原有研究问题。连续自评与时间差异的介绍移至评估区，明确菜单是设计问题的说明情境，24 名参与者、15 个视频属于论文的另一工作案例。
- 方法说明强调输出是针对具体用途的下一步，自动调整与长期复用分别判断。结果说明对应个人特征解释、是否增加传感、指标改善能否支持自动调整，以及能否留到下次。
- 六栏顺序、四层结果、五张主图、完整四案例模块与原有双语交互保持不变。

### 菜单情境的依据

来源为 [外部文献案例记录](../configs/protocol/external_reuse_cases.json) 中的 `findlater2004menus`，即 *A Comparison of Static, Adaptive, and Adaptable Menus*（DOI: `10.1145/985692.985704`）。
本次用正式解析器核对两条作者整理记录：

| 记录 | 正式判定 | 命中规则 |
|---|---|---|
| `findlater_adaptive_menu` | `RETAIN_EVALUATED_COMPARATOR` | R2 |
| `findlater_adaptable_menu` | `PROCEED_WITHIN_EVALUATED_BOUNDARY` | R5 |

结果区的菜单说明限于这些作者编码记录和已评估条件，不表示所有自动调整都无效，或所有用户定制均获许可。
首屏也注明这是文献案例组织的说明情境，不是已部署的 PersonaGuard 菜单产品。
示意中的功能名称和排列只用于解释三种选择，不是原始研究界面的复刻或实验结果。

### 上一轮精修记录（16ac179）

上一轮默认说明文字按可见 `p`、`dd`、`li`、`figcaption` 统计，排除关闭的详情、标题、表格和图内文字：中文从 2,983 个汉字减少至 1,566，英文从 1,583 个空白分隔词减少至 889。
这些数字属于上一轮删改统计，不作为本轮场景改写后的阅读量。正文与案例字号继续保持原样。

## 数据与公开范围

- 正式规则、解析器、案例判定、`case-copy.js`、生成数据 `cases.js` 和数值来源均未修改。
- 所有 `data-number` 标记保留；其值与 `results/revision6_source.json` 的生成结果一致。
- 规则顺序仍为 R4 → R2 → R3 → R1 → R5。R1 不是许可；R5 需满足自身条件且没有更高优先级约束；无规则匹配时请求补充证据。
- 外部案例是作者整理的记录，不是独立分析者对审查结论的验证。执行一致也不等于判断正确或现实决策改善。
- 五张网页主图沿用既有文件及公开范围，详见 [图表说明](assets/figures/README.md)。
- 不加入论文 PDF、论文目录、`references/`、受限数据或完整研究产物，也不创建论文下载入口。

## 中英文桌面与移动端检查

本地 Chromium，宽度 320、390、768、1024、1051、1440 px，高度 844 px；分别检查英文与中文。

| 检查 | 结果 |
|---|---|
| 六栏及四层结果顺序、五张主图 | 保留；主图默认可见、完整加载、可打开大图 |
| 两种语言 × 六种屏宽 × 四案例 | 48 组通过，行动、责任角色和复审条件与来源一致 |
| 菜单场景、来源说明与双语标签 | 桌面三种选择可见；手机隐藏示意，来源说明仍可读；无伪交互控件 |
| 导航、锚点与滚动高亮 | 通过；目标不被固定页头遮挡 |
| 默认及全部详情展开 | 无整页横向溢出 |
| 正文、案例和论文题目字号 | 正文 17 px；案例证据至少 16 px；首屏论文题目 17 px |
| 键盘、语言偏好、复制命令 | 通过 |
| 无 JavaScript | 保留综合表与默认记录 |
| 图片、资源及浏览器脚本错误 | 0 个失败 |
| 页面生成数据与正式判定检查 | `scripts/build_project_page_data.py --check` 通过 |
| 数值标记、图源顺序及页内链接 | 全部保留并通过检查 |

人工复核了浏览器生成的英文桌面与中文手机首屏截图。图内标注仍为原图英文，正文、图注和替代文本可切换中英文。
这里记录的是浏览器视口检查，不代表手机实机或 Safari/Firefox 检查。
