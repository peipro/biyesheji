import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

# --- 1. 环境配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 加载数据 ---
# 使用与 LSTM 相同的特征工程后数据集
df = pd.read_excel("data_feature_engineered_v3.xlsx")

# 定义特征列 (与 LSTM 完全一致，剔除功率和冷量防止泄露)
feature_cols = [
    'temperature', 'humidity', 'temp_diff', 'chiller_running_count',
    'lxj_evap_press_avg', 'lxj_cond_press_avg',
    'A_Chilled_Pump_avg', 'B_Chilled_Pump_avg',
    'A_Cooling_Pump_avg', 'B_Cooling_Pump_avg',
    'A_Tower_avg', 'B_Tower_avg'
]
target_col = 'system_cop'

X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

# --- 3. 数据预处理 ---
# 神经网络对量纲极其敏感，必须标准化
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_norm = scaler_x.fit_transform(X)
y_norm = scaler_y.fit_transform(y)

# 转换为 Tensor
X_tensor = torch.FloatTensor(X_norm)
y_tensor = torch.FloatTensor(y_norm)

# 随机打乱并划分数据集 (MLP 不需要滑动窗口，直接点对点预测)
torch.manual_seed(42)
indices = torch.randperm(len(X_tensor))
split = int(len(X_tensor) * 0.8)

train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

# 构建 DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)


# --- 4. 定义全连接网络 (MLP) 结构 ---
class MLP_Predictor(nn.Module):
    def __init__(self, input_dim):
        super(MLP_Predictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),  # 输入层 -> 隐藏层1
            nn.ReLU(),
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(128, 64),  # 隐藏层1 -> 隐藏层2
            nn.ReLU(),
            nn.Linear(64, 32),  # 隐藏层2 -> 隐藏层3
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出层
        )

    def forward(self, x):
        return self.layers(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP_Predictor(len(feature_cols)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# --- 5. 训练模型 ---
epochs = 100
print(f"🚀 开始训练 MLP 模型 (设备: {device})...")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.6f}")

# --- 6. 评估与可视化 ---
model.eval()
with torch.no_grad():
    # 预测全量测试集
    y_pred_norm = model(X_test.to(device)).cpu().numpy()

# 逆标准化还原
y_pred = scaler_y.inverse_transform(y_pred_norm)
y_true = scaler_y.inverse_transform(y_test.numpy())

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print("\n📊 MLP 模型评估结果:")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")

# --- 7. 新增：特征重要性分析 (Permutation Importance) ---
print("🧐 正在计算特征重要性...")
importances = {}

# 计算基准分数 (R2)
baseline_r2 = r2

for i, col in enumerate(feature_cols):
    # 复制一份测试数据
    X_test_permuted = X_test.clone()

    # 随机打乱当前特征列的顺序
    perm_indices = torch.randperm(X_test_permuted.size(0))
    X_test_permuted[:, i] = X_test_permuted[perm_indices, i]

    # 用打乱后的数据预测
    with torch.no_grad():
        y_perm_pred_norm = model(X_test_permuted.to(device)).cpu().numpy()
        perm_r2 = r2_score(y_test.numpy(), y_perm_pred_norm)

    # 重要性 = 基准 R2 - 打乱后的 R2
    importances[col] = baseline_r2 - perm_r2

# --- 8. 拆分绘图展示 ---

# 窗口 1: 预测对比图
plt.figure(figsize=(10, 6))
plt.plot(y_true[:150], label='Actual COP', color='royalblue', linewidth=1.5, alpha=0.8)
plt.plot(y_pred[:150], label='MLP Predicted', color='red', linestyle='--', linewidth=1.5)
plt.title(f"MLP Model: Actual vs Predicted COP (R2: {r2:.4f})")
plt.xlabel("Sample Index")
plt.ylabel("System COP")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 窗口 2: 特征重要性条形图
plt.figure(figsize=(10, 6))
# 转换为 Series 方便绘图
importance_series = pd.Series(importances).sort_values()
importance_series.plot(kind='barh', color='salmon')
plt.title("MLP Model: Feature Importance (Permutation Method)")
plt.xlabel("Importance Score (R2 Drop)")
plt.ylabel("Features")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()