import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 1. 加载数据
df = pd.read_excel("data_feature_engineered_v4.xlsx")
feature_cols = [
    'temperature', 'humidity', 'temp_diff', 'chiller_running_count',
    'lxj_evap_press_avg', 'lxj_cond_press_avg',
    'A_Chilled_Pump_avg', 'A_Cooling_Pump_avg', 'A_Tower_avg'
]
X = df[feature_cols]
y = df['system_cop']

print(f"数据加载完成: 样本数 {len(df)}, 特征数 {len(feature_cols)}")

# 2. 划分数据集 (随机打乱以保证公平对比)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

# 3. 训练随机森林
# n_estimators: 森林中树的数量; max_depth: 限制深度防止过拟合
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
print("开始训练随机森林...")
rf_model.fit(X_train, y_train)

# 4. 预测与评估
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"R2 Score: {r2_rf:.4f}")
print(f"MAE: {mae_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")

# 5. 可视化特征重要性
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
feat_importances.sort_values().plot(kind='barh', color='skyblue')
plt.title(f"Random Forest Feature Importance (R2: {r2_rf:.4f})")
plt.show()
# 6. 增加：真实值与预测值对比图
plt.figure(figsize=(12, 6))

# 为了看清趋势，选取前150个测试样本进行展示
plt.plot(y_test.values[:150], label='真实值 (Actual)', color='royalblue', linewidth=2, alpha=0.8)
plt.plot(y_pred_rf[:150], label='RF预测值 (Predicted)', color='darkorange', linestyle='--', linewidth=2)

plt.title(f"Random Forest: Actual vs Predicted COP (R2: {r2_rf:.4f})", fontsize=14)
plt.xlabel("测试样本编号 (Sample Index)", fontsize=12)
plt.ylabel("系统能效 (System COP)", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()