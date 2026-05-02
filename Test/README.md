# Test 目录文件说明

## 核心管线（按执行顺序）

| 文件 | 说明 | 运行方式 |
|:----|:----|:---------|
| `data_construct.py` | 合并原始数据分表为一张大表 | `python data_construct.py` |
| `features_engineering.py` | 特征工程：聚合泵/塔频率、提取压力、计算温差 | `python features_engineering.py [--input file]` |
| — | **特征工程输出** → `data_feature_engineered_v5.xlsx` | — |

### 模型训练（4个基线模型）

| 文件 | 模型 | 说明 |
|:----|:----|:------|
| `RF-train.py` | Random Forest | 随机森林回归，100棵树，max_depth=15 |
| `XGboost-train.py` | XGBoost | 500棵树，learning_rate=0.05 |
| `ANN-train.py` | ANN (MLP) | 5层全连接网络，BatchNorm+Dropout，早停 |
| `LSTM-train-final.py` | LSTM | 2层LSTM，seq=30，hidden=128，梯度裁剪 |

所有模型训练脚本均支持 `--input` 参数指定数据文件。

---

## 卡尔曼滤波模块

### 配置文件

| 文件 | 说明 |
|:----|:------|
| `kalman_config.py` | 滤波特征定义、Q/R噪声参数、物理范围约束、输出路径 |

### 核心实现

| 文件 | 说明 |
|:----|:------|
| `kalman_filter.py` | 多变量卡尔曼滤波器：predict/update/RTS平滑器/状态约束 (652行) |
| `kalman_utils.py` | 数据处理、评估指标（平滑度/SNR/相关性）、可视化工具 (1284行) |

### 执行脚本

| 文件 | 说明 | 运行方式 |
|:----|:------|:---------|
| `kalman_integration.py` | 全流程集成：加载→滤波→保存→可视化 | `python kalman_integration.py [--input file]` |
| `kalman_model_evaluation.py` | 4模型对比评估（子进程调用各训练脚本） | `python kalman_model_evaluation.py` |
| `kalman_parameter_tuning.py` | Q/R噪声参数自动调优 | `python kalman_parameter_tuning.py` |
| `kalman_diagnose.py` | 滤波后数据深度诊断（方差保持/互信息/差分分析） | `python kalman_diagnose.py` |
| `quick_kalman_test.py` | current_flow异常跳变检测与修复 | `python quick_kalman_test.py` |

---

## 数据文件

| 文件 | 大小 | 来源 | 用途 |
|:----|:---:|:----|:-----|
| `data_deep_learning_final_v3.xlsx` | 4.7 MB | `data_construct.py` 输出 | 原始合并大表，`features_engineering.py` 的输入 |
| `data_feature_engineered_v5.xlsx` | 1.3 MB | `features_engineering.py` 输出 | 8个特征 + system_cop（去current_flow + 去chiller_count + 个体设备替换聚合），**模型训练的原始基线数据** |
| `data_feature_engineered_v5_fixed.xlsx` | 1.3 MB | `quick_kalman_test.py` 生成 | 修复 current_flow 异常跳变后的特征数据 |
| `data_kalman_filtered.xlsx` | 1.8 MB | `kalman_integration.py` 中间输出 | 6维特征滤波后的完整数据（含未滤波列） |
| `data_feature_engineered_v5_kalman.xlsx` | 1.8 MB | `kalman_integration.py` 最终输出 | 滤波后特征数据，**卡尔曼滤波版的模型输入** |

---

## 评估结果

| 文件 | 说明 |
|:----|:------|
| `kalman_model_evaluation_report.md` | 滤波前后4模型R²对比报告（含详细数据表） |
| `model_results_original.json` | 原始数据上的模型训练结果缓存 |
| `model_results_filtered.json` | 滤波数据上的模型训练结果缓存 |

---

## 消融实验

| 目录 | 内容 |
|:----|:------|
| `ablation_exp1/` | 特征重要性消融：逐一删除关键特征，观察性能变化 |
| `ablation_exp2/` | 物理过滤消融：对比"剔除异常样本" vs "不做过滤" |
| `ablation_exp3/` | LSTM序列长度消融：对比不同 seq_length 的效果 |

---

## 文档

| 文件 | 说明 |
|:----|:------|
| `特征相关性分析.md` | 各特征与 system_cop 的相关性分析报告 |
| `README.md` | 本文件，目录结构说明 |

---

## 快速运行流程

```bash
# 完整管线（使用pytorch环境）
conda activate pytorch
cd Test

# 1. 数据准备（已取消物理过滤，仅保留COP>0）
python data_construct.py

# 2. 特征工程 → data_feature_engineered_v5.xlsx
python features_engineering.py

# 3. 卡尔曼滤波（可选，推荐）
python kalman_integration.py --input data_feature_engineered_v5.xlsx --output data_feature_engineered_v5_kalman_v8.xlsx

# 4. 模型训练
python RF-train.py --input data_feature_engineered_v5_kalman_v8.xlsx
python XGboost-train.py --input data_feature_engineered_v5_kalman_v8.xlsx
python ANN-train.py --input data_feature_engineered_v5_kalman_v8.xlsx
python LSTM-train-final.py --input data_feature_engineered_v5_kalman_v8.xlsx
```
