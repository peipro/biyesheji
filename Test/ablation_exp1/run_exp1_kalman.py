"""
消融实验1：物理特征有效性验证（卡尔曼滤波数据）
使用 sklearn train_test_split，RF 和 LSTM 两种模型
对比：8特征基线 vs 移除关键特征后的性能变化
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import sys, codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

DATA = "data_feature_engineered_v5_kalman_v8.xlsx"
ALL_FEATURES = [
    'temperature', 'humidity', 'temp_diff',
    'lxj_evap_press_avg', 'lxj_cond_press_avg',
    'A4冷冻泵_f', 'A1冷却泵_f', 'A4冷却塔_f'
]
DEVICE_FEATURES = ['A4冷冻泵_f', 'A1冷却泵_f', 'A4冷却塔_f']

df = pd.read_excel(DATA)
y = df['system_cop']

def run_rf(features, label):
    """在指定特征集上训练RF并返回测试集指标"""
    X = df[features]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, oob_score=True)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"[RF] {label} | 特征数={len(features)} | R²={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} OOB={rf.oob_score_:.4f}")
    return r2, mae, rmse

# RF: 四种配置
print("=" * 60)
print("消融实验1(RF) — 卡尔曼滤波数据 — sklearn划分")
print("=" * 60)
rf_base = run_rf(ALL_FEATURES, "基线(8特征)")
rf_no_td = run_rf([f for f in ALL_FEATURES if f != 'temp_diff'], "-temp_diff")
rf_no_dev = run_rf([f for f in ALL_FEATURES if f not in DEVICE_FEATURES], "-设备频率(3)")
rf_no_both = run_rf([f for f in ALL_FEATURES if f != 'temp_diff' and f not in DEVICE_FEATURES], "-temp_diff-设备频率")

print(f"\n{'='*60}")
print("RF 消融实验结果汇总")
print(f"{'='*60}")
print(f"  基线(8特征):       R²={rf_base[0]:.4f}  MAE={rf_base[1]:.4f}  RMSE={rf_base[2]:.4f}")
print(f"  移除temp_diff:     R²={rf_no_td[0]:.4f}  ΔR²={rf_no_td[0]-rf_base[0]:+.4f}")
print(f"  移除设备频率(3):   R²={rf_no_dev[0]:.4f}  ΔR²={rf_no_dev[0]-rf_base[0]:+.4f}")
print(f"  移除两者:          R²={rf_no_both[0]:.4f}  ΔR²={rf_no_both[0]-rf_base[0]:+.4f}")
