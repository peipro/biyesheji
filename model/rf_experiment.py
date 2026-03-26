import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. 加载数据
df = pd.read_excel("../data_plus_features.xlsx")

# 2. 特征选择 (排除目标变量和强相关中间变量)
# 排除 date_time, 目标 cop, 以及用于计算 cop 的 Q 和 功率
exclude_cols = ['date_time', 'system_cop', 'calc_Q_kw', 'power_consume', 'total_power_kw']
features = [c for c in df.columns if c not in exclude_cols]

X = df[features]
y = df['system_cop']

print(f"参与训练的特征有: {features}")

# 3. 划分数据集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 创建随机森林模型并训练
# n_estimators=100 表示森林里有100棵树
rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
rf_model.fit(X_train, y_train)

# 5. 模型预测
y_pred = rf_model.predict(X_test)

# 6. 评估模型性能
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- 随机森林模型评估结果 ---")
print(f"R² (拟合优度): {r2:.4f}")  # 越接近1越好
print(f"MAE (平均绝对误差): {mae:.4f}")
print(f"RMSE (均方根误差): {rmse:.4f}")

# 7. 特征重要性分析 (实验核心)
importances = rf_model.feature_importances_
feat_importance = pd.DataFrame({'feature': features, 'importance': importances})
feat_importance = feat_importance.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feat_importance)
plt.title('Feature Importance Analysis (Random Forest)')
plt.show()

# 8. 预测结果可视化对比
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label='Actual COP', color='blue', alpha=0.7)
plt.plot(y_pred[:100], label='Predicted COP', color='red', linestyle='--', alpha=0.8)
plt.legend()
plt.title('COP Prediction: Actual vs Predicted (First 100 samples)')
plt.ylabel('COP')
plt.show()