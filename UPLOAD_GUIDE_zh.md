# GitHub 投稿上传步骤

1. 本公开仓库不含数据、模型与逐行预测。将单独交付的 `MSH-private-reviewer-assets.zip` 上传投稿系统，供获授权的审稿人下载；不要把该压缩包放入公开 GitHub 仓库。SSA 已由作者说明在 GitHub 公开；如有原仓库链接，可补到 README。作者和论文引用可在确定后补齐，软件许可证由作者决定。
2. 在 GitHub 新建一个**空仓库**，不要预先勾选 README、`.gitignore` 或 License。本目录已经包含前两项；License 需由作者决定。
3. 在本目录打开终端，执行：

```bash
git init
git branch -M main
git add .
git status
git commit -m "Release reproducible manuscript code and frozen audits"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

把 `OWNER/REPOSITORY` 改成你的账户和仓库名。首次推送需要完成 GitHub 身份验证。推送前核对 `git status` 中的文件清单，确认没有 `assets/`、实验数据、模型或逐行预测文件。

4. 运行 README 中的核验命令，包括新版 M5 的重新训练。论文投稿使用的版本应在 GitHub 上创建 tag / release，例如 `v1.0-submission`，并在稿件 Code Availability 中引用该版本的地址。若期刊要求永久 DOI，可将该 release 归档到认可的代码存档服务，再引用实际生成的 DOI。

审稿人下载私有材料包和公开仓库后，可在仓库根目录执行 `python reviewer_run.py --assets-zip MSH-private-reviewer-assets.zip`。需使用 Python 3.11 并允许安装依赖；脚本自动创建隔离环境、运行三个阶段，并将核验日志写入 `results/reviewer_run/`。

建议的 Code Availability 英文占位句（填写真实地址与版本后使用）：

> The analysis code is available at [GitHub repository URL], release [version]. The processed input data, frozen estimators, and archived predictions are provided as supplementary files through the journal submission system for authorized review. Data provenance and the roles of the literature and experimental splits are documented in the repository.

不要把 M2–M5 表述为同一搜索预算下的严格消融。新版 M5 是相似度加权 XGBoost 源模型加 SVR 校准器，24 组 R² 为 0.880729；旧的 0.882191 仅保留在原始溯源记录中。详细限制见 `README.md` 与第三阶段的 provenance 文档。
