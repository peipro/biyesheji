"""
卡尔曼滤波配置文件
定义需要滤波的特征、噪声参数和滤波器配置
"""

import numpy as np

# ==================== 特征配置 ====================

# 需要卡尔曼滤波的连续变量特征（5个）
# 注意：system_cop 是目标变量，不对目标变量进行滤波，避免数据泄露
# 注意2：current_flow 已在特征优化中移除（相关系数仅0.0013）
FILTER_COLS = [
    'temperature',        # 环境温度 (°C)
    'humidity',           # 环境湿度 (%RH)
    'temp_diff',          # 冷冻水进出水温差 (°C)
    'lxj_evap_press_avg', # 平均蒸发压力 (bar)
    'lxj_cond_press_avg'  # 平均冷凝压力 (bar)
]

# 状态名称（与FILTER_COLS对应）
STATE_NAMES = FILTER_COLS

# 不需要滤波的离散/控制变量特征及目标变量
NON_FILTER_COLS = [
    'system_cop',       # 系统能效比 (目标变量，不可滤波)
    'A4冷冻泵_f',        # A4冷冻泵频率 (控制信号/离散)
    'A1冷却泵_f',        # A1冷却泵频率 (控制信号/离散)
    'A4冷却塔_f'         # A4冷却塔频率 (控制信号/离散)
]

# 所有特征（用于验证）
ALL_FEATURES = FILTER_COLS + NON_FILTER_COLS

# ==================== 状态约束 ====================
# 各特征的物理合理范围，用于滤波后裁剪异常值
STATE_BOUNDS = {
    'temperature':       (5.0, 45.0),
    'humidity':          (5.0, 100.0),
    'temp_diff':         (-1.0, 10.0),
    'lxj_evap_press_avg': (1.5, 5.0),
    'lxj_cond_press_avg': (2.5, 12.0),
}

# ==================== 噪声参数配置 ====================
# 过程噪声协方差矩阵 Q (6×6)
# Q/R 的设计原则：
#   - Q 反映系统真实变化速率，用 diff_std² 的量级
#   - R 反映观测噪声水平，应 > Q 才能让滤波器平滑
#   - R/Q 比值越大 → 滤波越强（更信任模型而非观测）
#   - R/Q 比值约 5~20 对一般传感器去噪合适

INITIAL_Q = np.diag([
    0.001,    # temperature
    0.025,    # humidity
    0.001,    # temp_diff (保护相关性,使用最小滤波)
    0.0001,   # lxj_evap_press_avg
    0.001,    # lxj_cond_press_avg
])

# 观测噪声协方差矩阵 R (6×6)
# 策略：对关键特征极轻滤波(R/Q<1)，对噪声特征正常滤波(R/Q≈2-3)
# temp_diff 是预测最重要的特征，必须保护其与cop的相关性
# 压力传感器噪声较大，可以正常滤波
INITIAL_R = np.diag([
    0.0005,   # temperature: R/Q=0.5,轻滤波
    0.0125,   # humidity: R/Q=0.5,轻滤波
    0.0003,   # temp_diff: R/Q=0.3,最轻滤波保护相关性
    0.0002,   # lxj_evap_press_avg: R/Q=2,正常滤波(噪声较大)
    0.002,    # lxj_cond_press_avg: R/Q=2,正常滤波(噪声较大)
])

# 初始状态协方差矩阵 P (5×5)
INITIAL_P = np.eye(5) * 10.0

# 状态转移矩阵 F (5×5)
STATE_TRANSITION_MATRIX = np.eye(5)

# 观测矩阵 H (5×5)
OBSERVATION_MATRIX = np.eye(5)

# ==================== 滤波器配置 ====================

KALMAN_CONFIG = {
    'use_adaptive_noise': False,     # 关闭自适应噪声，使用固定Q/R避免过度平滑
    'adaptive_window_size': 100,
    'adaptive_learning_rate': 0.05,
    'warmup_steps': 100,
    'save_intermediate': False,
    'visualize': True,
    'verbose': True
}

# ==================== 数据预处理配置 ====================

DATA_FILTER_CONDITIONS = {
    'min_power': 30.0,
    'min_cooling': 1.0,
    'min_cop': 0.5,
    'max_cop': 12.0
}

TIME_CONFIG = {
    'time_column': 'date_time',
    'sampling_interval': '1T',
    'resample_method': 'mean'
}

# ==================== 验证配置 ====================

VALIDATION_CONFIG = {
    'smoothness_threshold': 0.1,
    'snr_improvement_threshold': 0.2,
    'correlation_preservation_threshold': 0.95,
    'max_runtime_seconds': 30
}

# ==================== 输出配置 ====================

OUTPUT_CONFIG = {
    'filtered_data_file': 'data_kalman_filtered_v8.xlsx',
    'feature_engineered_file': 'data_feature_engineered_v5_kalman_v8.xlsx',
    'evaluation_report_file': 'kalman_evaluation_report_v8.md',
    'visualization_dir': '../pic/kalman_filter_v8',
    'log_file': 'kalman_filter_v8.log'
}

if __name__ == '__main__':
    print("卡尔曼滤波配置验证")
    print(f"需要滤波的特征: {len(FILTER_COLS)}个 → {FILTER_COLS}")
    print(f"不需要滤波的特征: {len(NON_FILTER_COLS)}个 → {NON_FILTER_COLS}")
    print(f"Q矩阵形状: {INITIAL_Q.shape}")
    print(f"R矩阵形状: {INITIAL_R.shape}")
    print(f"\nQ对角线: {np.diag(INITIAL_Q)}")
    print(f"R对角线: {np.diag(INITIAL_R)}")
    print(f"R/Q比值: {np.diag(INITIAL_R) / np.diag(INITIAL_Q)}")
    print(f"\n状态约束: {STATE_BOUNDS}")
    print("配置验证完成!")
