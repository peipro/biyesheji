# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **强制**：全程中文回答，在本地的PyTorch环境运行python脚本

---

## 项目概述
裴佳轩毕业设计：**基于深度学习的冷水机组COP性能预测**（故障诊断尚未开始）

目标：预测冷水机组系统COP（性能系数=制冷量/功耗），对比不同模型性能。

## 快速运行命令
```bash
cd Test
python data_construct.py       # 合并原始数据 → data_deep_learning_final.xlsx
python features_engineering.py # 特征工程 → data_feature_engineered_v4.xlsx
python RF-train.py             # 训练随机森林
python XGboost-train.py        # 训练XGBoost
python ANN-train.py            # 训练ANN
python LSTM-train-final.py     # 训练LSTM最终版本
```
所有图表输出到 `../pic/` 目录。

## 核心信息
- **目标变量**：预测 `system_cop` = 总制冷量 / 总功耗
- **输入特征（9个）**：`temperature` `humidity` `temp_diff` `chiller_running_count` `lxj_evap_press_avg` `lxj_cond_press_avg` `A_Chilled_Pump_avg` `A_Cooling_Pump_avg` `A_Tower_avg`
- ⚠️ **红线禁忌**：严禁 `total_power_kw` `calc_Q_kw` 入特征 → 直接导致标签泄露
- **数据规模**：14000行，1分钟采样，约10天连续监测
- **评估指标**：R² ↑ MAE ↓ RMSE ↓

## 当前模型性能（基线实验）
| 模型 | 文件 | 测试集R² | 特征重要性 | 框架 |
|------|------|----------|------------|------|
| Random Forest | RF-train.py | 0.8248 | ✓ 原生支持（不纯度） | sklearn |
| XGBoost | XGboost-train.py | 0.8068 | ✓ 原生支持（增益） | xgboost |
| ANN (改进后) | ANN-train.py | 0.7196 | ✓ Permutation | PyTorch |
| LSTM-final | LSTM-train-final.py | 0.7901 | ✓ Permutation | PyTorch |

**性能排名**：Random Forest > XGBoost > LSTM > ANN

## 开发红线规则
- 归一化**必须**只在训练集拟合 → 防止数据泄露
- 物理过滤：只保留 `total_power_kw > 30kW` 且 `0.5 < system_cop < 12`
- 固定随机种子 `42` → 保证实验可重复
- B区设备全0已移除 → 避免引入噪声
- **必须**在conda管理的PyTorch环境运行所有Python脚本

## 项目结构
```
毕设/
├── 05data/              # 原始数据（20+个Excel文件）
├── Test/                # 最终训练代码
│   ├── data_construct.py       # 多源数据合并对齐
│   ├── features_engineering.py # 特征聚合工程
│   ├── RF-train.py             # 随机森林训练
│   ├── XGboost-train.py        # XGBoost训练
│   ├── ANN-train.py            # ANN训练
│   └── LSTM-train-final.py     # LSTM最终版本训练
├── pic/                 # 输出可视化图表
├── model/               # 预留：保存训练好的模型
├── 实验总结.md          # 完整实验总结：数据清洗、调优、模型对比、创新点
├── 消融实验.md          # 消融实验方案设计（5组实验）
├── 故障诊断.md          # 无标签故障诊断研究方案
└── CLAUDE.md            # 本文件
```

## 数据处理流水线
```
原始多源Excel(20+文件)
  ↓ data_construct.py
合并对齐 + 物理过滤 → data_deep_learning_final.xlsx (~14000行)
  ↓ features_engineering.py
特征聚合 + 移除噪声 → data_feature_engineered_v4.xlsx (9特征+目标COP)
  ↓
四种模型训练对比 → 输出指标 + 可视化
```

## 特征重要性分析方法
不同类型模型采用不同方法：
| 模型 | 方法 | 原理 |
|------|------|------|
| Random Forest | 不纯度平均减少（Gini） | 基于树节点分裂时不纯度减少 |
| XGBoost | 增益（Gain） | 基于特征分裂带来的平均增益 |
| ANN / LSTM | Permutation Importance | 逐个打乱特征，观察R²下降幅度 |

Permutation Importance与模型结构无关，保证深度学习模型也能输出特征重要性，便于跨模型对比。

## 消融实验（待执行）
按 `消融实验.md` 设计，运行5组验证实验，以Random Forest为基线：

1. **实验1**：物理特征有效性验证（移除temp_diff、chiller_running_count对比）
2. **实验2**：B区域特征移除必要性（保留vs移除全0特征对比）
3. **实验3**：物理合理性过滤有效性（过滤vs不过滤对比）
4. **实验4**：数据划分策略对比（随机划分vs时序划分），需在四个模型上都测试
5. **实验5**：LSTM序列长度影响（10/20/30/50对比）

**基线配置**：Random Forest，固定seed=42，80-20随机划分，使用全部9特征，移除B区，启用物理过滤。

## 故障诊断研究（待执行）
研究方向：无标签异常检测（只有正常运行数据，无故障标签）

推荐实施方案：
1. **阶段一**：Isolation Forest（传统方法基线）+ 预测残差方法（复用现有模型）
2. **阶段二**：VAE变分自编码器（基于重构误差的无监督检测）
3. **阶段三**：结合物理规则混合诊断系统

详见 `故障诊断.md`。

## 常用开发命令

单个模型训练示例：
```bash
cd Test
python RF-train.py          # 训练随机森林并输出评估指标
python LSTM-train-final.py  # 训练LSTM并输出特征重要性
```

转换markdown到docx：
```bash
python md_to_docx_v5.py  # 使用pypandoc转换实验总结文档
```

## 完成进度
- ✅ 数据清洗合并特征工程完整流水线
- ✅ 四个模型实现调优指标补全
- ✅ LSTM添加Permutation特征重要性可视化
- ✅ ANN优化提升R²
- ✅ 实验总结文档完成
- ✅ 消融实验方案设计完成
- ✅ 故障诊断思路规划完成
- ⏳ 待执行：消融实验、故障诊断实验、论文写作

## 关键开发提示
- 所有Python脚本工作目录应为 `Test/`，数据文件生成在 `Test/` 目录下
- 绘图自动保存到 `../pic/`，如需显示请确保环境支持matplotlib显示
- 深度学习模型（ANN、LSTM）使用PyTorch，训练完成自动绘图
- 始终固定random_state=42保证实验可复现
- 特征工程输出版本：当前使用 `data_feature_engineered_v4.xlsx`
- 运行消融实验时，保持其他条件一致，只改变实验变量，记录R²/MAE/RMSE三个指标
