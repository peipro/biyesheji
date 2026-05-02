import pandas as pd
import numpy as np
import argparse
import sys
import os

# 解析命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description='特征工程脚本')
    parser.add_argument('--input', type=str, default='data_deep_learning_final_v3.xlsx',
                       help='输入数据文件路径 (默认: data_deep_learning_final_v3.xlsx)')
    parser.add_argument('--output', type=str, default='data_feature_engineered_v5.xlsx',
                       help='输出数据文件路径 (默认: data_feature_engineered_v5.xlsx)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='显示详细输出 (默认: True)')
    return parser.parse_args()

# 主函数
def main():
    args = parse_arguments()

    # 1. 加载合并后的原始大表
    input_file = args.input
    output_file = args.output

    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        print("请先运行 data_construct.py 或指定正确的输入文件")
        sys.exit(1)

    df = pd.read_excel(input_file)
    print(f"原始数据读取成功: {input_file}")
    print(f"行数: {len(df)}, 列数: {len(df.columns)}")

    # --- A. 泵与冷却塔按区域聚合 ---
    # 只聚合A区所有设备，计算区域平均频率作为特征
    groups = {
        # A区设备
        'A_Chilled_Pump_avg': ['A1冷冻泵_f', 'A2冷冻泵_f', 'A3冷冻泵_f', 'A4冷冻泵_f'],
        'A_Cooling_Pump_avg': ['A1冷却泵_f', 'A2冷却泵_f', 'A3冷却泵_f', 'A4冷却泵_f'],
        'A_Tower_avg': ['A1冷却塔_f', 'A2冷却塔_f', 'A3冷却塔_f', 'A4冷却塔_f'],
    }

    for new_col, raw_cols in groups.items():
        existing_cols = [c for c in raw_cols if c in df.columns]
        if existing_cols:
            df[new_col] = df[existing_cols].mean(axis=1)

    # --- B. 冷水机特征整合 ---

    # 1. 识别运行状态
    status_cols = [c for c in df.columns if 'chiller_run_status' in c]
    if not status_cols:
        # 兜底：如果没有 status 列，用频率 > 5Hz 判断
        f_cols = [c for c in df.columns if 'lxj' in c and ('_f' in c or 'frequency' in c)]
        run_mask = (df[f_cols] > 5).astype(int)
    else:
        run_mask = (df[status_cols] == 1).astype(int)

    df['chiller_running_count'] = run_mask.sum(axis=1)

    # 2. 压力特征提取
    evap_cols = [c for c in df.columns if 'eva_press' in c and 'lxj' in c]
    cond_cols = [c for c in df.columns if 'con_press' in c and 'lxj' in c]

    if args.verbose:
        print("匹配到蒸发压力列:", evap_cols)
        print("匹配到冷凝压力列:", cond_cols)

    # 计算运行机组的平均压力
    if evap_cols:
        df['lxj_evap_press_avg'] = df[evap_cols].where(run_mask.values == 1).mean(axis=1)
        df['lxj_evap_press_avg'] = df['lxj_evap_press_avg'].ffill().fillna(0)

    if cond_cols:
        df['lxj_cond_press_avg'] = df[cond_cols].where(run_mask.values == 1).mean(axis=1)
        df['lxj_cond_press_avg'] = df['lxj_cond_press_avg'].ffill().fillna(0)

    # --- C. 物理温差 ---
    if 'return_temp' in df.columns and 'supply_temp' in df.columns:
        df['temp_diff'] = df['return_temp'] - df['supply_temp']
    else:
        df['temp_diff'] = 0

    # --- D. 导出精简集 ---
    core_features = [
        'date_time', 'total_power_kw', 'calc_Q_kw', 'system_cop',
        'temp_diff', 'temperature', 'humidity',
        'lxj_evap_press_avg', 'lxj_cond_press_avg',
        'A4冷冻泵_f', 'A1冷却泵_f', 'A4冷却塔_f'
    ]

    final_cols = [c for c in core_features if c in df.columns]
    df_refined = df[final_cols].copy().dropna()

    # 保存到输出文件
    df_refined.to_excel(output_file, index=False)
    print(f"特征工程完成！")
    print(f"输出文件: {output_file}")
    print(f"最终列数: {len(df_refined.columns)}, 行数: {len(df_refined)}")
    print("已包含A区所有设备特征（B区数据正确合并但不参与特征聚合）")

if __name__ == '__main__':
    main()
