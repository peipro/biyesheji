import pandas as pd
import os

# --- 1. 配置基础信息 ---
input_file = "data.xlsx"  # 假设你的文件名
output_file = "final_training_features.xlsx"


def extract_features(df):
    # --- A. 基础必选特征 (环境、系统侧负荷、计算结果) ---
    base_cols = [
        'date_time',
        'temperature', 'humidity',
        'current_flow', 'supply_temp', 'return_temp',
        'total_power_kw', 'calc_Q_kw', 'system_cop'
    ]

    # --- B. 冷机侧特征 (每台冷机保留关键的 5 个物理指标) ---
    chiller_features = []
    for i in range(1, 5):
        prefix = f'lxj{i}_'
        cols = [
            f'{prefix}cooling_water_inlet_temp',
            f'{prefix}cooling_water_outlet_temp',
            f'{prefix}eva_press',
            f'{prefix}con_press',
            f'{prefix}current'
        ]
        # 检查这些列是否存在于表格中
        existing_cols = [c for c in cols if c in df.columns]
        chiller_features.extend(existing_cols)

    # --- C. 输送系统特征 (所有 A/B 区清洗后的频率) ---
    pump_tower_features = [c for c in df.columns if '_freq_cleaned' in c]

    # --- D. 汇总提取 ---
    final_cols = base_cols + chiller_features + pump_tower_features

    # 提取数据并进行最后的清洗
    # 1. 仅提取存在的列
    available_cols = [c for c in final_cols if c in df.columns]
    feature_df = df[available_cols].copy()

    # 2. 处理缺失值：机器学习不能有 NaN
    # 使用前向填充（用上一时刻状态填补），剩下的用 0 填补
    feature_df = feature_df.fillna(method='ffill').fillna(0)

    return feature_df


# --- 2. 执行提取 ---
if os.path.exists(input_file):
    df_raw = pd.read_excel(input_file)
    df_final = extract_features(df_raw)

    # 保存结果
    df_final.to_excel(output_file, index=False)
    print(f"特征提取完成！共提取 {df_final.shape[1]} 个特征。")
    print(f"文件已保存至: {output_file}")
else:
    print("找不到原始数据文件，请检查文件名。")