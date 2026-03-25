import pandas as pd
import numpy as np
from pyexpat import features


def refine_and_abstract_features(df):
    # --- 1. 数据清洗：处理负数冷量和 COP ---
    df['calc_Q_kw'] = df['calc_Q_kw'].apply(lambda x: max(x, 0))
    df['system_cop'] = df['system_cop'].apply(lambda x: max(x, 0))

    # --- 2. 模块化抽象：输送系统 (泵与塔) ---
    # 定义分区前缀
    pump_types = {
        'chilled_A': [c for c in df.columns if 'A' in c and '冷冻泵' in c],
        'chilled_B': [c for c in df.columns if 'B' in c and '冷冻泵' in c],
        'cooling_A': [c for c in df.columns if 'A' in c and '冷却泵' in c],
        'cooling_B': [c for c in df.columns if 'B' in c and '冷却泵' in c],
        'tower_all': [c for c in df.columns if '冷却塔' in c]
    }

    # 计算每个模块的平均频率
    for key, cols in pump_types.items():
        if cols:
            df[f'{key}_avg_freq'] = df[cols].mean(axis=1)
            # 统计开启台数 (频率 > 5Hz 视为开启)
            df[f'{key}_on_count'] = (df[cols] > 5).sum(axis=1)

    # --- 3. 模块化抽象：机组压力与电流 ---
    # 计算全场冷机的平均蒸发/冷凝压力（只考虑运行中的机器）
    eva_cols = [c for c in df.columns if 'eva_press' in c]
    con_cols = [c for c in df.columns if 'con_press' in c]
    cur_cols = [c for c in df.columns if 'current' in c]
    for col in eva_cols + con_cols + cur_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['chillers_avg_eva_press'] = df[eva_cols].replace(0, np.nan).mean(axis=1).fillna(0)
    df['chillers_avg_con_press'] = df[con_cols].replace(0, np.nan).mean(axis=1).fillna(0)
    df['chillers_total_current'] = df[cur_cols].sum(axis=1)

    # --- 4. 选取最终用于模型训练的“精简特征” ---
    final_model_features = [
        'date_time',
        # 需求端
        'temperature', 'humidity', 'current_flow', 'supply_temp', 'return_temp',
        # 输送端模块
        'chilled_A_avg_freq', 'chilled_B_avg_freq',
        'cooling_A_avg_freq', 'cooling_B_avg_freq',
        'tower_all_avg_freq',
        # 生产端模块
        'chillers_avg_eva_press', 'chillers_avg_con_press', 'chillers_total_current',
        # 标签
        'system_cop'
    ]
    return df[final_model_features]

df_raw = pd.read_excel('final_training_features.xlsx')
df_refined=refine_and_abstract_features(df_raw)

output_path='training-data.xlsx'
df_refined.to_excel(output_path,index=False)

