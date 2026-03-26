import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载增强后的数据
df = pd.read_excel("../data_plus_features.xlsx")

# 2. 精确定义【瘦身版】特征列表
# 剔除了 supply_temp, return_temp, 以及所有中间计算量
exclude_cols = [
    'date_time', 'system_cop', 'calc_Q_kw', 'power_consume', 'total_power_kw',
    'supply_temp', 'return_temp', 'current_flow' # 彻底删掉计算公式的原始项
]

features = [c for c in df.columns if c not in exclude_cols]
X = df[features]
y = df['system_cop']

print(f"✅ 实验启动！当前剔除了强相关冗余，剩余特征数: {len(features)}")
print(f"关键输入特征包括: {features[:10]} ...")

# 3. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 训练模型 (稍微增加深度提升学习能力)
rf_final = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
rf_final.fit(X_train, y_train)

# 5. 评估
y_pred = rf_final.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"\n--- 剔除冗余后的 RF 评估结果 ---")
print(f"R² (拟合优度): {r2:.4f}")

# 6. 重新查看特征重要性
importances = rf_final.feature_importances_
feat_imp = pd.DataFrame({'feature': features, 'importance': importances}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feat_imp)
plt.title('Feature Importance (After Removing Redundancy)')
plt.show()