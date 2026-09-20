# 上传 GitHub

## 导出源码包（推荐）

在项目根目录运行：

```bash
make check github-check
make github-export
```

生成 `dist/CHI2027-github.zip`，解压后的 `CHI2027/` 即为仓库根目录。
打包读取当前工作区，包含未提交的公开文件；不带 `.git/` 和历史提交。
导出前自动检查候选文件与文档链接，任何错误都会终止打包。
同名 ZIP 属于生成产物，再次运行会更新它。

| 上传内容 | 本地保留 |
|---|---|
| `src/`、`scripts/`、`tests/`、`configs/` | `paper/`、`paper_zh/`、`archive/`、`references/` |
| `results/`、`docs/`、双语 README、构建配置 | `artifacts/`、`checkpoints/`、`.venv/`、`.local/` |
| `data/DATASET.md`、`data/stimuli/PROVENANCE.md` | 原始数据、媒体、特征数组、凭据、旧 Git 历史 |

## 上传解压后的目录

在 GitHub 创建空仓库，然后在解压后的 `CHI2027/` 中运行：

```bash
git init -b main
git add .
git diff --cached --stat
git commit -m "Prepare research code for GitHub"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

先将 URL 替换为目标仓库地址。网页上传时也应上传解压后的目录内容，
包括 `.gitignore`，使 README 和源码可以直接浏览。

## 沿用当前 Git 仓库

原始研究工作区的 Git 历史包含论文文件；`.gitignore` 和取消跟踪只影响后续提交，不能清除历史。
仅发布代码时，应导入上面的干净源码包；不要把原始研究仓库的旧历史推送到代码仓库。
沿用旧历史时，先确认目标远程与历史内容：

```bash
git remote -v
git status --short
git diff --stat
```

当前工作区的检查可能提示此前已删除的文件，提交时应一并记录这些删除。
完整研究与论文一致性检查依赖本地材料；新克隆从 `make check` 开始。
数据与第三方材料沿用各自授权条件；项目整体许可证尚未选定。
