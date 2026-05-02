import pandas as pd
import numpy as np
import argparse
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

# --- 1. 配置与加载 ---
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser(description='XGBoost训练')
parser.add_argument('--input', type=str, default='data_feature_engineered_v5.xlsx',
                    help='输入数据文件路径 (默认: data_feature_engineered_v5.xlsx)')
args = parser.parse_args()

df = pd.read_excel(args.input)

# 特征列
feature_cols = [
    'temperature', 'humidity', 'temp_diff',
    'lxj_evap_press_avg', 'lxj_cond_press_avg',
    'A4冷冻泵_f', 'A1冷却泵_f', 'A4冷却塔_f'
]
X = df[feature_cols]
y = df['system_cop']

# --- 2. 随机划分数据集 80%训练, 10%验证, 10%测试 ---
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

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
    random_state=42,
    eval_metric='rmse',
    early_stopping_rounds=20
)

print("开始训练 XGBoost 模型（使用早停，验证集监控过拟合）...")
model_xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
best_iter = model_xgb.best_iteration + 1 if model_xgb.best_iteration is not None else 'N/A'
print(f"早停结束，最佳迭代次数: {best_iter}")

# --- 4. 预测与评估 ---
# 使用早停后的最佳模型（XGBRegressor内部已自动使用最佳迭代次数）
for name, X_eval, y_true in [("训练集", X_train, y_train), ("验证集", X_val, y_val), ("测试集", X_test, y_test)]:
    y_pred = model_xgb.predict(X_eval)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} — R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

y_pred = model_xgb.predict(X_test)

# --- 5. 特征重要性可视化 ---
plt.figure(figsize=(10, 6))
xgb.plot_importance(model_xgb, importance_type='gain', max_num_features=12, height=0.5)
plt.title(f"XGBoost Feature Importance (R2: {r2:.4f})")
plt.tight_layout()
plt.savefig("../pic/baseline/xgb_feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()

# --- 6. 预测效果对比图 ---
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:150], label='Actual', color='blue', alpha=0.7)
plt.plot(y_pred[:150], label='XGBoost Predict', color='orange', linestyle='--')
plt.title(f"XGBoost Model COP Prediction Comparison (R2: {r2:.4f})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../pic/baseline/xgb_pred_comparison.png", dpi=150, bbox_inches='tight')
plt.show()