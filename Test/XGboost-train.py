import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# --- 1. 配置与加载 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel("data_feature_engineered_v3.xlsx")

# 特征列保持绝对一致
feature_cols = [
    'temperature', 'humidity', 'temp_diff', 'chiller_running_count',
    'lxj_evap_press_avg', 'lxj_cond_press_avg',
    'A_Chilled_Pump_avg', 'B_Chilled_Pump_avg',
    'A_Cooling_Pump_avg', 'B_Cooling_Pump_avg',
    'A_Tower_avg', 'B_Tower_avg'
]
X = df[feature_cols]
y = df['system_cop']

# --- 2. 划分数据集 ---
# 采用 8:2 划分，random_state 固定保证可重复性
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. 定义并训练 XGBoost 模型 ---
# 参数说明：
# n_estimators: 迭代次数（树的数量）
# max_depth: 树的最大深度，通常 3-10
# learning_rate: 学习率，防止过拟合
# subsample: 训练每棵树时使用的样本比例
model_xgb = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

print("🚀 开始训练 XGBoost 模型...")
model_xgb.fit(X_train, y_train)

# --- 4. 预测与评估 ---
y_pred = model_xgb.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 XGBoost 模型评估结果:")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# --- 5. 特征重要性可视化 ---
plt.figure(figsize=(10, 6))
xgb.plot_importance(model_xgb, importance_type='weight', max_num_features=12, height=0.5)
plt.title(f"XGBoost Feature Importance (R2: {r2:.4f})")
plt.show()

# --- 6. 预测效果对比图 ---
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:150], label='Actual', color='blue', alpha=0.7)
plt.plot(y_pred[:150], label='XGBoost Predict', color='orange', linestyle='--')
plt.title("XGBoost Model COP Prediction Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()