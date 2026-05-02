# 基于深度学习的冷水机组COP性能预测

**Graduation Project — Pei Jiaxuan**

## 项目简介

本项目使用深度学习与传统机器学习方法，对数据中心冷水机组的系统性能系数（COP）进行建模预测。通过物理知识指导的特征工程、卡尔曼滤波数据降噪，系统对比了 Random Forest、XGBoost、ANN、LSTM 四种模型，并通过三组消融实验验证关键设计决策。

## 最终性能（卡尔曼滤波数据）

| 模型 | R² | MAE | RMSE |
|------|:--:|:---:|:----:|
| **LSTM** | **0.8836** | 0.2541 | 0.3614 |
| XGBoost | 0.8697 | 0.2351 | 0.3758 |
| Random Forest | 0.8631 | 0.2271 | 0.3852 |
| ANN | 0.7949 | 0.3074 | 0.4716 |

- 训练数据：数据中心真实传感器数据，~14,800 样本，1 分钟间隔
- 输入特征：8 个物理特征（环境参数 + 设备运行参数）
- 目标变量：system_cop（系统制冷能效比）

## 项目结构

```
.
├── 05data/              # 原始传感器数据（20+ Excel分表）
├── Test/                # 工作目录（所有脚本）
│   ├── data_construct.py        # 数据合并、清洗、COP计算
│   ├── features_engineering.py  # 特征工程 → 8特征输出
│   ├── RF-train.py              # 随机森林训练
│   ├── XGboost-train.py         # XGBoost训练
│   ├── ANN-train.py             # 全连接神经网络训练
│   ├── LSTM-train-final.py      # LSTM训练
│   ├── kalman_filter.py         # 卡尔曼滤波器核心实现
│   ├── kalman_integration.py    # 卡尔曼滤波管线集成
│   ├── VAE-train.py             # 原始VAE训练
│   ├── vae_dualhead_train.py    # 双头VAE训练
│   ├── ablation_exp1/           # 消融实验1：特征有效性
│   ├── ablation_exp2/           # 消融实验2：物理过滤
│   └── ablation_exp3/           # 消融实验3：LSTM序列长度
├── pic/                 # 生成图表的输出目录
├── 实验总结.md           # 完整实验流程与结果总结
├── 消融实验.md           # 三组消融实验报告
├── 故障诊断.md           # VAE性能诊断报告
└── 卡尔曼滤波技术文档.md  # 卡尔曼滤波技术细节
```

## 快速开始

```bash
cd Test
conda activate pytorch

# 1. 数据准备
python data_construct.py                     # 合并原始数据
python features_engineering.py               # 特征工程

# 2. 卡尔曼滤波（推荐）
python kalman_integration.py \
    --input data_feature_engineered_v5.xlsx \
    --output data_feature_engineered_v5_kalman_v8.xlsx

# 3. 模型训练
python RF-train.py --input data_feature_engineered_v5_kalman_v8.xlsx
python XGboost-train.py --input data_feature_engineered_v5_kalman_v8.xlsx
python ANN-train.py --input data_feature_engineered_v5_kalman_v8.xlsx
python LSTM-train-final.py --input data_feature_engineered_v5_kalman_v8.xlsx

# 4. VAE 性能诊断
python VAE-train.py
python vae_dualhead_train.py
python vae_dualhead_diagnosis.py
```

## 关键设计

- **严格防止数据泄露**：归一化只在训练集拟合，严禁 COP 计算相关特征入模
- **统一划分方式**：所有模型使用 `sklearn.model_selection.train_test_split`（80/10/10，random_state=42）
- **固定随机种子**：所有实验 seed=42，确保可复现
- **5 变量卡尔曼滤波**：对连续传感器信号降噪，RTS 后向平滑消除相位滞后

## 实验亮点

| 消融实验 | 结论 |
|:-------|:-----|
| 特征有效性 | 供回水温差（temp_diff）是决定性特征，移除后 RF R² 下降 0.12、LSTM 下降 0.48 |
| 物理过滤 | 不做物理范围过滤性能更好（RF +0.0278），保留所有 COP>0 数据 |
| LSTM 序列长度 | seq=30 为最优选择 |
| VAE 诊断 | 双头 VAE 基于 COP 负残差检测性能退化，COP-残差相关性 r=+0.6327 |

## 环境要求

- Python 3.x（Conda pytorch 环境）
- PyTorch、scikit-learn、XGBoost、filterpy
- pandas、numpy、matplotlib、scipy
