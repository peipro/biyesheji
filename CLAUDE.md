# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **强制**：全程中文回答

---

## 项目
裴佳轩毕业设计：**基于深度学习的冷水机组COP性能预测**

## 快速运行
```bash
cd Test
conda activate pytorch
python data_construct.py       # 合并原始数据
python features_engineering.py # 特征工程 → data_feature_engineered_v5.xlsx
python RF-train.py             # 训练随机森林
python XGboost-train.py        # 训练XGBoost
python ANN-train.py            # 训练ANN
python LSTM-train-final.py     # 训练LSTM
```
图表输出到 `../pic/`

## 卡尔曼滤波
```bash
pip install filterpy
D:/Anaconda/envs/pytorch/python.exe kalman_integration.py --test --input data_feature_engineered_v5.xlsx  # 测试模式
D:/Anaconda/envs/pytorch/python.exe kalman_integration.py --input data_feature_engineered_v5.xlsx --output data_feature_engineered_v5_kalman_v8.xlsx  # 完整模式
D:/Anaconda/envs/pytorch/python.exe RF-train.py --input data_feature_engineered_v5_kalman_v8.xlsx
D:/Anaconda/envs/pytorch/python.exe XGboost-train.py --input data_feature_engineered_v5_kalman_v8.xlsx
D:/Anaconda/envs/pytorch/python.exe ANN-train.py --input data_feature_engineered_v5_kalman_v8.xlsx
D:/Anaconda/envs/pytorch/python.exe LSTM-train-final.py --input data_feature_engineered_v5_kalman_v8.xlsx
```
核心文件：`kalman_filter.py` `kalman_integration.py` `kalman_config.py`，对5个连续变量（temperature/humidity/temp_diff/蒸发压力/冷凝压力）做标准卡尔曼滤波 + RTS平滑。**注意**：current_flow已在特征优化中移除，非滤波目标变量system_cop保持不变。
详见：[卡尔曼滤波技术文档.md](../卡尔曼滤波技术文档.md)

## VAE性能诊断
```bash
python VAE-train.py                                    # 训练VAE → vae_best.pth
python vae_performance_diagnosis.py                    # 性能诊断 → pic/vae_diagnosis/ 共8张图
```
核心文件：`VAE-train.py` `vae_performance_diagnosis.py`，基于VAE重构误差的无监督异常检测。
- 输入9维（8个物理特征+system_cop），8维潜在空间
- β=0.1预热50 epoch（低KL权重保障重构质量）
- 三种阈值：95%分位、99%分位、均值+3σ

## 核心规则
- **目标变量**：`system_cop`
- **输入特征（8个）**：`temperature` `humidity` `temp_diff` `lxj_evap_press_avg` `lxj_cond_press_avg` `A4冷冻泵_f` `A1冷却泵_f` `A4冷却塔_f`
- ⚠️ **严禁** `total_power_kw` `calc_Q_kw` 入特征 → 标签泄露（COP = calc_Q_kw / total_power_kw）
- 过滤：只保留 `system_cop > 0`（消融实验2验证：不过滤性能更好）
- 固定随机种子 `42`，归一化只在训练集拟合

## 基线性能（最终方案：无过滤 + 卡尔曼滤波，80/10/10划分，sklearn划分统一）
| 模型 | R² | MAE | RMSE |
|------|:--:|:---:|:----:|
| **LSTM** | **0.8836** | 0.2541 | 0.3614 |
| **XGBoost** | 0.8697 | 0.2351 | 0.3758 |
| **Random Forest** | 0.8631 | 0.2271 | 0.3852 |
| **ANN** | 0.7949 | 0.3074 | 0.4716 |

**注：**
- 最终特征集为8个物理特征，经两轮优化：
  1. 去除 current_flow（相关系数仅0.0013）
  2. 去除 chiller_running_count，改用实际运行的个体设备特征
- **数据过滤**：只保留 system_cop > 0（消融实验2验证）
- **卡尔曼滤波**：对5个连续变量滤波，3/4模型受益（LSTM提升最大 +0.0182，XGBoost +0.0106）
- **归一化**：严格只在训练集拟合归一化器（ANN/LSTM已修复），树模型无需归一化
- **数据划分**：统一使用 sklearn train_test_split（80/10/10），保证各模型对比公平
- **字体**：图表使用 Microsoft YaHei + SimHei 后备，支持 R² 上标正常显示

## 运行环境## 运行环境

本项目必须在Conda的"pytorch"环境下运行

解释器路径: D:\Anaconda\envs\pytorch\python.exe

## 目录结构

- `05data/` - 原始数据分表
- `Test/` - 工作目录，所有脚本运行在此
- `Test/ablation_exp1/` - 特征有效性消融实验（RF+LSTM，卡尔曼数据）
- `Test/ablation_exp2/` - 物理过滤消融实验（RF+LSTM，卡尔曼数据）
- `Test/ablation_exp3/` - LSTM序列长度消融实验（卡尔曼数据）
- `消融实验.md` - 消融实验完整报告（卡尔曼滤波数据）
- `卡尔曼滤波技术文档.md` - 卡尔曼滤波技术文档
- `pic/` - 图表输出

## 当前状态
- ✅ 所有基线模型训练完成（最佳LSTM R²=0.8836 @ 卡尔曼滤波数据，sklearn划分统一）
- ✅ 卡尔曼滤波集成完成（5变量滤波，v8输出）
- ✅ 三组消融实验已完成（基于卡尔曼滤波数据重新运行）
- ✅ 卡尔曼滤波技术文档已撰写
- ✅ VAE性能诊断实验已完成（详见故障诊断.md）
- ✅ 数据泄露修复完成（ANN/LSTM归一化严格先划分后拟合）
- ✅ 早停+验证集机制已为XGBoost/RF添加（RF新增OOB Score）
- ✅ 图表字体修复（R²上标正常显示）
- ✅ 数据划分统一为 sklearn train_test_split
- ⏳ 待执行：论文写作

