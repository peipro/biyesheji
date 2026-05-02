"""
卡尔曼滤波效果诊断脚本
分析滤波前后数据的统计变化，诊断为什么神经网络效果下降
"""
import pandas as pd
import numpy as np
import json

# 加载数据
original = pd.read_excel("data_feature_engineered_v5_fixed.xlsx")
filtered = pd.read_excel("data_feature_engineered_v5_kalman.xlsx")

features = ['temperature', 'humidity', 'temp_diff', 'current_flow',
            'lxj_evap_press_avg', 'lxj_cond_press_avg']

target = 'system_cop'

print("=" * 80)
print("卡尔曼滤波诊断报告")
print("=" * 80)

# 1. 基本统计对比
print("\n--- 1. 基本统计对比 ---")
for col in features + [target]:
    orig = original[col]
    filt = filtered[col]
    diff = filt - orig
    print(f"\n{col}:")
    print(f"  原始均值: {orig.mean():.4f}  滤波均值: {filt.mean():.4f}  偏移: {diff.mean():.4f}")
    print(f"  原始标准差: {orig.std():.4f}  滤波标准差: {filt.std():.4f}  变化: {(filt.std()/orig.std()-1)*100:+.2f}%")
    print(f"  原始差分标准差: {orig.diff().std():.4f}  滤波差分标准差: {filt.diff().std():.4f}  平滑度: {(1-filt.diff().std()/orig.diff().std())*100:+.1f}%")
    print(f"  滤波改变量: 均值={diff.mean():.4f}, 标准差={diff.std():.4f}, 最大={diff.max():.4f}, 最小={diff.min():.4f}")
    print(f"  MAE(滤波前后差异): {np.abs(diff).mean():.4f}")

# 2. 与目标变量的相关性
print("\n--- 2. 与 system_cop 的相关性分析 ---")
orig_corrs = {}
filt_corrs = {}
for col in features:
    orig_corr = original[col].corr(original[target])
    filt_corr = filtered[col].corr(filtered[target])
    orig_corrs[col] = orig_corr
    filt_corrs[col] = filt_corr
    change = (filt_corr - orig_corr)
    print(f"  {col}: 原始r={orig_corr:.4f}  滤波r={filt_corr:.4f}  变化={change:+.4f}")

# 3. 特征之间的相关性结构变化
print("\n--- 3. 特征间相关性结构变化 ---")
orig_corr_matrix = original[features].corr().values
filt_corr_matrix = filtered[features].corr().values
corr_diff = np.abs(orig_corr_matrix - filt_corr_matrix)
print(f"  相关性矩阵平均绝对变化: {corr_diff.mean():.4f}")
print(f"  相关性矩阵最大绝对变化: {corr_diff.max():.4f}")

# 找出变化最大的特征对
n = len(features)
max_diff = 0
max_pair = ""
for i in range(n):
    for j in range(i+1, n):
        d = abs(orig_corr_matrix[i,j] - filt_corr_matrix[i,j])
        if d > max_diff:
            max_diff = d
            max_pair = f"{features[i]} vs {features[j]}"
print(f"  变化最大的特征对: {max_pair} ({max_diff:.4f})")

# 4. 信号能量分析（方差保持度）
print("\n--- 4. 信号能量保持分析 ---")
for col in features:
    orig_var = original[col].var()
    filt_var = filtered[col].var()
    keep_ratio = filt_var / orig_var * 100
    print(f"  {col}: 原始方差={orig_var:.4f}  滤波后方差={filt_var:.4f}  保持率={keep_ratio:.1f}%")

# 5. 时间序列的差分分析（动态特性保持）
print("\n--- 5. 动态特性保持（差分相关性） ---")
for col in features:
    orig_diff = original[col].diff().dropna()
    filt_diff = filtered[col].diff().dropna()
    diff_corr = orig_diff.corr(filt_diff)
    print(f"  {col}: 差分相关性={diff_corr:.4f}")

# 6. 检查是否过度平滑了重要特征
print("\n--- 6. 互信息变化（特征与目标的关系） ---")
from sklearn.feature_selection import mutual_info_regression

# 采样计算（互信息计算较慢）
sample_size = min(5000, len(original))
np.random.seed(42)
idx = np.random.choice(len(original), sample_size, replace=False)

X_orig = original[features].iloc[idx]
X_filt = filtered[features].iloc[idx]
y = original[target].iloc[idx]

mi_orig = mutual_info_regression(X_orig, y, random_state=42)
mi_filt = mutual_info_regression(X_filt, y, random_state=42)

print(f"\n  {'特征':<25} {'原始MI':<10} {'滤波MI':<10} {'变化':<10}")
print(f"  {'-'*55}")
for i, col in enumerate(features):
    change = (mi_filt[i] - mi_orig[i]) / (mi_orig[i] + 1e-10) * 100
    print(f"  {col:<25} {mi_orig[i]:<10.4f} {mi_filt[i]:<10.4f} {change:+.1f}%")

print(f"\n  平均互信息变化: {(mi_filt.mean()/mi_orig.mean()-1)*100:+.1f}%")

# 7. 总体评价
print("\n" + "=" * 80)
print("诊断结论")
print("=" * 80)

# 检查标准差保持率
var_keep_ratios = [(filtered[col].var() / original[col].var()) for col in features]
avg_var_keep = np.mean(var_keep_ratios)

# 检查差分相关性
diff_corrs = [original[col].diff().dropna().corr(filtered[col].diff().dropna()) for col in features]
avg_diff_corr = np.mean(diff_corrs)

print(f"""
  平均方差保持率: {avg_var_keep*100:.1f}%
  平均差分相关性: {avg_diff_corr:.4f}

  方差保持率 < 50% 表示过度平滑
  差分相关性 < 0.7 表示动态特性被破坏

  当前状态: {'✅ 滤波强度适中' if avg_var_keep > 0.5 else '❌ 过度平滑'}
            {'✅ 动态特性保持良好' if avg_diff_corr > 0.7 else '❌ 动态特性被破坏'}
""")
