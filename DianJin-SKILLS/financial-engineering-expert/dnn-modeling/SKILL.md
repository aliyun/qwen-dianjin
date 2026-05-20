---
name: dnn-modeling
category: 建模训练
description: 【DNN深度学习建模】当用户需要"深度学习"、"DNN建模"、"神经网络"、"MLP"、"跑个DNN"、"深度模型"时使用。基于 PyTorch 实现多层全连接网络（MLP）进行二分类建模。支持 BatchNorm、Dropout、学习率调度、早停。复用平台已有的三段式数据切分、AUC/KS/BCR评估体系、稳定性分析。前提条件：需要已有数据集。与 xgb-modeling 的区别：本 Skill 使用深度神经网络，适合高维特征交互场景；xgb-modeling 使用 XGBoost 树模型。
next_steps:
  - skill: dnn-tuning
    text: DNN调参优化
    prompt: 帮我调整网络结构和训练参数
  - skill: model-comparison
    text: 多算法对比
    prompt: 与XGBoost和LR对比一下效果
---

# DNN 深度学习建模 (portable)

基于 PyTorch 实现 MLP (Multi-Layer Perceptron) 进行二分类建模。

**核心流程**：特征标准化 → MLP 训练（BatchNorm + Dropout + Early Stopping） → 概率预测 → 三段式评估

**适用场景**：
- 高维特征交互建模
- 特征间存在复杂非线性关系
- 数据量充足（>10k 样本）
- 对模型性能有极致追求（可与 XGBoost 做 ensemble）

---

## 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|------|:----:|--------|------|
| `--data_path` / `-d` | ✅ | - | 数据文件路径（parquet/csv） |
| `--target` / `-t` | ✅ | - | 目标变量列名（0/1 二分类） |
| `--time_col` | | `busi_dt` | 时间列名 |
| `--train_filter` | | 自动切分 | 训练集筛选条件（pandas query） |
| `--oot_filter` | | 按时间切出 | OOT 跨时间测试集条件 |
| `--oot_ratio` | | `0.20` | 未传 `--oot_filter` 时按时间切 OOT 的比例 |
| `--val_ratio` | | `0.25` | 从 train_full 切 val 的比例 |
| `--random_seed` | | `42` | 随机种子 |
| `--exclude_cols` | | - | 排除列，逗号分隔 |
| `--features` | | - | 指定特征列表，逗号分隔；不传则自动推断 |
| `--hidden_dims` | | `128,64,32` | 隐藏层维度，逗号分隔 |
| `--dropout` | | `0.3` | Dropout 比率 |
| `--learning_rate` | | `0.001` | 学习率 |
| `--batch_size` | | `512` | 批次大小 |
| `--epochs` | | `100` | 最大训练轮次 |
| `--patience` | | `10` | 早停耐心轮数 |
| `--weight_decay` | | `1e-4` | 权重衰减（L2 正则化） |
| `--pos_weight` | | `auto` | 正样本权重（auto=自动计算） |
| `--model_name` | | 自动生成 | 模型名称 |
| `--report_output` | | 自动生成 | 报告输出路径 |
| `--output_dir` | | `./outputs/<ts>` | 产物输出目录 |
| `--config` | | - | JSON 配置文件路径 |

---

## 执行方式

默认参数运行：

```bash
python scripts/modeling.py \
  --data_path ./data.parquet --target y_label \
  --time_col busi_dt \
  --exclude_cols "cust_code,busi_dt" \
  --output_dir ./outputs/dnn_run
```

自定义网络结构：

```bash
python scripts/modeling.py \
  --data_path ./data.parquet --target y_label \
  --hidden_dims "256,128,64" \
  --dropout 0.4 \
  --learning_rate 0.0005 \
  --epochs 200 \
  --output_dir ./outputs/dnn_run
```

---

## 常用场景

### 场景一：默认参数快速建模

```bash
python scripts/modeling.py --data_path ./data.parquet --target y_label --output_dir ./outputs/dnn_run
```

### 场景二：自定义网络结构

```bash
python scripts/modeling.py --data_path ./data.parquet --target y_label \
  --hidden_dims "256,128,64" --dropout 0.4 --learning_rate 0.0005 \
  --output_dir ./outputs/dnn_run
```

### 场景三：指定特征建模

```bash
python scripts/modeling.py --data_path ./data.parquet --target y_label \
  --features "feat1,feat2,feat3" --output_dir ./outputs/dnn_run
```

---

## 输出产物

1. **建模报告**（Markdown）— 含数据切分、网络结构、训练曲线、三段式评估指标、稳定性分析
2. **模型文件**（.pt）— PyTorch state_dict + 模型配置
3. **训练日志**（JSON）— 每 epoch 的 loss/AUC/KS 记录
4. **result.json** — 结构化产物清单

---

## 与其他建模 Skill 的对比

| 维度 | dnn-modeling | xgb-modeling | lr-modeling |
|------|-------------|--------------|-------------|
| 算法 | MLP (PyTorch) | XGBoost | LR + WoE |
| 特征处理 | StandardScaler | 原始值 | WoE 分箱 |
| 非线性能力 | 强（多层激活） | 强（树结构） | 弱 |
| 可解释性 | 弱 | 中（SHAP） | 强（系数） |
| 训练速度 | 慢 | 快 | 很快 |
| 适用数据量 | >10k | 任意 | 任意 |
| 评估体系 | AUC/KS/BCR/PSI | AUC/KS/BCR/PSI | AUC/KS/BCR/PSI |

---

## 上下游关系

- **前置**：`data-profiling` → `feature-analysis`（特征筛选）
- **后续**：`dnn-tuning`（调参优化）、`model-comparison`（多算法对比）
- **平行**：与 `xgb-modeling` / `lr-modeling` 可做横向对比

---

## 注意事项

1. **数据要求**：目标变量必须为 0/1 二分类
2. **数据量**：建议训练样本 > 10k，DNN 在小样本上易过拟合
3. **缺失值**：脚本自动使用中位数填充 + 添加缺失指示列
4. **标准化**：自动对数值特征做 StandardScaler 标准化
5. **早停**：当 val loss 连续 `patience` 轮不下降时自动停止
6. **不提供调参**：需要调参请切 `dnn-tuning`（搜索网络架构 + 训练参数）
7. **GPU**：自动检测 CUDA，无 GPU 时回退到 CPU
8. **模型保存**：模型文件保存到 `<output_dir>/models/`
