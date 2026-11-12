# 使用说明

这个 skill 只对外使用三个名称：`Lit-only`、`Lit-exp`、`SW-Lit`。

## 1. 安装依赖

在解压后的 skill 根目录打开 PowerShell：

```powershell
python -m pip install -r requirements.txt
```

建议使用 Python 3.11 或更新版本。XGBoost、scikit-learn 等依赖版本变化可能带来最后几位浮点差异，运行清单会自动核对基准值。

## 2. 一次运行全部流程

```powershell
python scripts/run_pipeline.py `
  --literature "D:\data\literature.xlsx" `
  --experiment "D:\data\round4.xlsx" `
  --output "D:\results\lit-transfer" `
  --mode all
```

`--literature` 可以使用原始文献数据库 `.xlsx`，也可以使用已清洗的长表 `.csv`。`--experiment` 可以使用包含完整第四轮 33 组、每组 3/7/28 d 强度的工作簿，或标准化的已清洗长表 `.csv`。

## 3. 分别运行

只运行纯文献内部验证：

```powershell
python scripts/run_pipeline.py --literature "D:\data\literature.xlsx" --output "D:\results\lit-only" --mode lit-only
```

运行纯文献模型对固定 9 组外部实验的验证：

```powershell
python scripts/run_pipeline.py --literature "D:\data\literature.xlsx" --experiment "D:\data\round4.xlsx" --output "D:\results\lit-exp" --mode lit-exp
```

运行相似度加权文献模型：

```powershell
python scripts/run_pipeline.py --literature "D:\data\literature.xlsx" --experiment "D:\data\round4.xlsx" --output "D:\results\sw-lit" --mode sw-lit
```

后两种模式会自动先运行 `Lit-only`，因为它们需要完全相同的数据划分和已经选定的 XGBoost 超参数。

## 4. 关键输出

- `RUN_MANIFEST.json`：输入哈希、划分、固定种子、9 组排名和三项基准核验。
- `Lit-only/Lit-only_metrics.csv`：Train、Validation、Test 指标。
- `Lit-only/Lit-only_candidate_validation_summary.csv`：五类候选算法的验证集结果和超参数。
- `Lit-exp/Lit-exp_predictions.csv` 与 `Lit-exp_metrics.csv`：固定 9 组、27 个点的外部预测与指标。
- `SW-Lit/SW-Lit_similarity_weights.csv`：每条文献记录的相似度、门控状态和权重。
- `SW-Lit/SW-Lit_predictions.csv` 与 `SW-Lit_metrics.csv`：相似度加权模型的外部预测与指标。

对齐的 9 组 rank 为 `1、17、33、11、23、4、30、8、26`。第二包冻结结果的参考值为：`Lit-only` Test R²=0.9013374756，`Lit-exp` pooled R²=0.2155723464，`SW-Lit` pooled R²=0.3741462335。实际重跑值和参考值的差异会写入 `RUN_MANIFEST.json`；只有核验通过时才可称为精确复现。

如果输出目录已经存在，程序默认拒绝覆盖；确认需要替换时再加 `--overwrite`。
