import pandas as pd
import numpy as np

# 1. 加载 14000 行原始大表
# 确保该表包含 lxj1_eva_press, lxj2_eva_press 等列
df = pd.read_excel("data_deep_learning_final.xlsx")
print(f"✅ 原始数据读取成功，行数: {len(df)}")

# --- A. 泵与冷却塔按区域聚合 ---
groups = {
    'A_Chilled_Pump_avg': ['A1冷冻泵_f', 'A2冷冻泵_f', 'A3冷冻泵_f', 'A4冷冻泵_f'],
    'B_Chilled_Pump_avg': ['B1冷冻泵_f', 'B2冷冻泵_f', 'B3冷冻泵_f'],
    'A_Cooling_Pump_avg': ['A1冷却泵_f', 'A2冷却泵_f', 'A3冷却泵_f', 'A4冷却泵_f'],
    'B_Cooling_Pump_avg': ['B1冷却泵_f', 'B2冷却泵_f', 'B3冷却泵_f'],
    'A_Tower_avg': ['A1冷却塔_f', 'A2冷却塔_f', 'A3冷却塔_f', 'A4冷却塔_f'],
    'B_Tower_avg': ['B1冷却塔_f', 'B2冷却塔_f', 'B3冷却塔_f']
}

for new_col, raw_cols in groups.items():
    existing_cols = [c for c in raw_cols if c in df.columns]
    if existing_cols:
        df[new_col] = df[existing_cols].mean(axis=1)

# --- B. 冷水机特征整合 (基于 lxj1.xlsx 格式) ---

# 1. 识别运行状态 (根据文件，列名应包含 chiller_run_status)
status_cols = [c for c in df.columns if 'chiller_run_status' in c]
if not status_cols:
    # 兜底：如果没有 status 列，用频率 > 5Hz 判断
    f_cols = [c for c in df.columns if 'lxj' in c and ('_f' in c or 'frequency' in c)]
    run_mask = (df[f_cols] > 5).astype(int)
else:
    run_mask = (df[status_cols] == 1).astype(int)

df['chiller_running_count'] = run_mask.sum(axis=1)

# 2. 压力特征提取 (匹配文件中的 eva_press 和 con_press)
# 脚本会匹配如 'lxj1_eva_press' 这样的列
evap_cols = [c for c in df.columns if 'eva_press' in c and 'lxj' in c]
cond_cols = [c for c in df.columns if 'con_press' in c and 'lxj' in c]

print(f"🔍 匹配到蒸发压力列: {evap_cols}")
print(f"🔍 匹配到冷凝压力列: {cond_cols}")

# 计算运行机组的平均压力
if evap_cols:
    df['lxj_evap_press_avg'] = df[evap_cols].where(run_mask.values == 1).mean(axis=1)
    df['lxj_evap_press_avg'] = df['lxj_evap_press_avg'].fillna(method='ffill').fillna(0)

if cond_cols:
    df['lxj_cond_press_avg'] = df[cond_cols].where(run_mask.values == 1).mean(axis=1)
    df['lxj_cond_press_avg'] = df['lxj_cond_press_avg'].fillna(method='ffill').fillna(0)

# --- C. 物理温差 ---
if 'return_temp' in df.columns and 'supply_temp' in df.columns:
    df['temp_diff'] = df['return_temp'] - df['supply_temp']
else:
    df['temp_diff'] = 0

# --- D. 导出精简集 ---
core_features = [
    'date_time', 'total_power_kw', 'calc_Q_kw', 'system_cop',
    'temp_diff', 'current_flow', 'temperature', 'humidity',
    'chiller_running_count', 'lxj_evap_press_avg', 'lxj_cond_press_avg'
]
core_features += list(groups.keys())

final_cols = [c for c in core_features if c in df.columns]
df_refined = df[final_cols].copy().dropna()

df_refined.to_excel("data_feature_engineered_v3.xlsx", index=False)
print(f"✅ 特征工程完成！最终列数: {len(df_refined.columns)}，行数: {len(df_refined)}")