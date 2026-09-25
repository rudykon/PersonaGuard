# 在浏览器运行案例

入口：[在线运行](https://rudykon.github.io/PersonaGuard/playground.html)，或主页的“运行案例”按钮。
默认选择 MER-PS 有符号校准案例，支持全部 4 条工作案例与 7 条作者编码的外部研究记录。

选择案例后，页面立即重放原始条件。修改选项后点击“运行审查”，即可比较原始行动与本次结果，展开查看五条规则的命中情况和所检查的字段。
“假设补充了用户研究”只改变评价方式与用户结果，不改用途或其他证据；校准案例此时从 R3 转为 R1，而非直接得到许可。
继续将范围改为匹配时，原来的后果性部署用途仍不满足 R5 的可逆用途要求；无规则匹配时返回补充证据。
有条件许可可从已有的外部研究记录重放，例如用户可调菜单。

## 本地运行与研究边界

- 规则判断完全在浏览器 JavaScript 中执行，无服务器计算、API 密钥、远程运行服务或额外依赖。加载页面后，修改条件与运行不会发出网络请求。
- 条件只存在当前页面内存中，不上传，也不保存到浏览器持久存储；语言偏好沿用站点已有的本地存储。
- 程序检查字段、允许取值和声明的规则，不验证证据真实性、充分性或作者编码是否正确。
- 修改后的条件明确标为假设；原始证据说明和来源仅作为对照。修改不产生新的实证结果或真实部署许可。
- 结果下载由浏览器生成 JSON，包含原始案例结果、选定条件、变化列表、命中规则、解析轨迹和来源文件 SHA-256。它是演示报告，不是新的正式案例文件。
- 未加入原始数据、刺激媒体、论文 PDF 或 `references/`。

## 文件与单一来源

| 文件 | 用途 |
|---|---|
| `configs/protocol/protocol_rules.json` | 唯一正式规则与证据词表来源 |
| `configs/protocol/*cases.json` | 已有公开案例输入，保持原样 |
| `scripts/build_browser_demo_data.py` | 使用正式 Python 校验器检查来源、生成网页输入与原始判定 |
| `docs/assets/audit-data.js` | 生成文件，不手工修改 |
| `docs/assets/audit-engine.js` | 解释 `all`、`any`、`eq`、`in`、`not_in`，按正式优先级选择第一个匹配规则 |
| `docs/playground.html`、`assets/playground.js`、`assets/playground.css` | 双语表单、结果解释与移动端展示 |
| `tests/test_browser_audit.py` | 与 Python 解析器的差分测试、边界与非法输入检查 |

页面启动时重放所有原始案例，若结果与生成时的 Python 判定不一致，则不开放交互。
每次运行都计算当前条件；条件更改后隐藏旧结果并提示重新运行，避免把旧结论当成新条件的结果。
执行顺序始终为 R4 → R2 → R3 → R1 → R5，R1 不等于许可，无匹配不默认通过。

## 生成与检查

```bash
python3 -B scripts/build_browser_demo_data.py
make browser-check
python3 -m http.server 8000 --directory docs
```

浏览 `http://localhost:8000/playground.html`。静态页面也可直接从磁盘打开。
`make browser-check` 需要 Python 3.10+ 和 Node.js，无需 npm 安装、研究数据、GPU 或网络。
它检查生成文件、11 个原始案例、各字段的全部允许取值、2,000 组组合条件、优先级冲突、无匹配和非法输入。
这些检查验证网页与 Python 的执行一致性，不验证现实决策质量。
