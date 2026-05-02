import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser(description='随机森林训练')
parser.add_argument('--input', type=str, default='data_feature_engineered_v5.xlsx',
                    help='输入数据文件路径 (默认: data_feature_engineered_v5.xlsx)')
args = parser.parse_args()

# 1. 加载数据
df = pd.read_excel(args.input)
feature_cols = [
    'temperature', 'humidity', 'temp_diff',
    'lxj_evap_press_avg', 'lxj_cond_press_avg',
    'A4冷冻泵_f', 'A1冷却泵_f', 'A4冷却塔_f'
]
X = df[feature_cols]
y = df['system_cop']

print(f"数据加载完成: 样本数 {len(df)}, 特征数 {len(feature_cols)}")

# 2. 随机划分数据集 80%训练, 10%验证, 10%测试
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

# 3. 训练随机森林
# n_estimators: 森林中树的数量; max_depth: 限制深度防止过拟合
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, oob_score=True)
print("开始训练随机森林...")
rf_model.fit(X_train, y_train)
print(f"OOB Score: {rf_model.oob_score_:.4f}")

# 4. 预测与评估（验证集 + 测试集）
for name, X_eval, y_true in [("训练集", X_train, y_train), ("验证集", X_val, y_val), ("测试集", X_test, y_test)]:
    y_pred = rf_model.predict(X_eval)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} — R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# 用测试集指标做最终报告
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# 5. 可视化特征重要性
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
feat_importances.sort_values().plot(kind='barh', color='skyblue')
plt.title(f"Random Forest Feature Importance (R2: {r2_rf:.4f})")
plt.tight_layout()
plt.savefig("../pic/baseline/rf_feature_importance.png", dpi=150, bbox_inches='tight')
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
plt.savefig("../pic/baseline/rf_pred_comparison.png", dpi=150, bbox_inches='tight')
plt.show()