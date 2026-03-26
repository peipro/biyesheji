import pandas as pd
import numpy as np
import os

# --- 1. 配置 ---
INPUT_FILE = "data_all_merged_optimized.xlsx"
OUTPUT_FILE = "data_plus_features.xlsx"  # 生成增强后的特征表


def aggregate_and_enhance_features():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到输入文件 {INPUT_FILE}")
        return

    print(f"正在读取数据并执行【物理+时序】特征增强...")
    df = pd.read_excel(INPUT_FILE)

    # --- 步骤 1：基础信息提取 ---
    base_cols = [
        'date_time', 'power_consume', 'calc_Q_kw', 'current_flow',
        'return_temp', 'supply_temp', 'temperature', 'humidity',
        'total_power_kw', 'system_cop'
    ]
    available_base_cols = [c for c in base_cols if c in df.columns]
    new_df = df[available_base_cols].copy()

    # --- 步骤 2：聚合泵与塔的特征 (原有逻辑保留) ---
    device_groups = {
        'chilled_pump': [c for c in df.columns if '冷冻泵_freq_cleaned' in c],
        'cooling_pump': [c for c in df.columns if '冷却泵_freq_cleaned' in c],
        'cooling_tower': [c for c in df.columns if '冷却塔_freq_cleaned' in c]
    }

    for group_name, cols in device_groups.items():
        if not cols: continue
        new_df[f'{group_name}_count'] = (df[cols] > 1).sum(axis=1)
        temp_group = df[cols].replace(0, np.nan)
        new_df[f'{group_name}_avg_freq'] = temp_group.mean(axis=1).fillna(0)

    # --- 步骤 3：聚合冷水机组特征 (原有逻辑保留) ---
    def get_running_avg(row, suffix):
        running_indices = [i for i in range(1, 5) if row.get(f'lxj{i}_chiller_run_status') == 1]
        if not running_indices: return 0
        vals = [row[f'lxj{idx}_{suffix}'] for idx in running_indices if f'lxj{idx}_{suffix}' in row]
        return np.mean(vals) if vals else 0

    new_df['chiller_count'] = (df[[c for c in df.columns if 'chiller_run_status' in c]] == 1).sum(axis=1)
    new_df['chiller_avg_eva_press'] = df.apply(lambda r: get_running_avg(r, 'eva_press'), axis=1)
    new_df['chiller_avg_con_press'] = df.apply(lambda r: get_running_avg(r, 'con_press'), axis=1)
    new_df['chiller_avg_freq'] = df.apply(lambda r: get_running_avg(r, 'frequency'), axis=1)

    # ========================================================
    # --- 核心改进：【物理逻辑增强】 ---
    # ========================================================
    # 1. 显性化温差：COP与温差高度非线性相关
    new_df['temp_diff'] = new_df['return_temp'] - new_df['supply_temp']

    # 2. 显性化压差：压差决定了压缩机的实际做功强度
    new_df['press_diff'] = new_df['chiller_avg_con_press'] - new_df['chiller_avg_eva_press']

    # 3. 组合特征：单机负荷因子 (频率与台数的耦合)
    new_df['load_intensity'] = new_df['chiller_avg_freq'] * new_df['chiller_count']

    # 4. 环境应力：考虑湿度对冷却塔散热的影响
    new_df['env_index'] = new_df['temperature'] * (new_df['humidity'] / 100)

    # ========================================================
    # --- 核心改进：【时序滞后增强】 ---
    # ========================================================
    # 暖通系统有热惯性，加入上一时刻(10min前)的状态可以大幅提升静态模型(RF)的精度
    lag_targets = ['chiller_avg_freq', 'chiller_avg_con_press', 'temperature', 'temp_diff']
    for col in lag_targets:
        if col in new_df.columns:
            new_df[f'{col}_lag1'] = new_df[col].shift(1)

    # --- 步骤 4：数据清洗与异常过滤 ---
    # 1. 删掉由于 shift 产生的空行
    new_df = new_df.dropna().copy()

    # 2. 物理合理性过滤：剔除过渡工况和传感器异常
    # 比如 COP 应该在合理范围，温差应该大于0等
    new_df = new_df[
        (new_df['system_cop'] > 1.0) & (new_df['system_cop'] < 8.0) &
        (new_df['temp_diff'] > 0)
        ]

    # 保存
    new_df.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ 增强版特征工程完成！")
    print(f"最终样本数: {len(new_df)}，特征总数: {len(new_df.columns)}")
    print(f"新增关键特征: temp_diff, press_diff, load_intensity, 以及各类 lag1 特征")


if __name__ == "__main__":
    aggregate_and_enhance_features()