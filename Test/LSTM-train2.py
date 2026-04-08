import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
df = pd.read_excel("data_feature_engineered_v4.xlsx")

# 严禁加入：total_power_kw, calc_Q_kw (防止数据泄露)
feature_cols = [
    'temperature', 'humidity', 'temp_diff', 'chiller_running_count',
    'lxj_evap_press_avg', 'lxj_cond_press_avg',
    'A_Chilled_Pump_avg', 'B_Chilled_Pump_avg',
    'A_Cooling_Pump_avg', 'B_Cooling_Pump_avg',
    'A_Tower_avg', 'B_Tower_avg'
]
target_col = 'system_cop'

# 2. 预处理与时序正确划分 (保持时间顺序)
scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
X_norm = scaler_x.fit_transform(df[feature_cols])
y_norm = scaler_y.fit_transform(df[[target_col]])

def create_sequences(x, y, seq_length=30):
    xi, yi = [], []
    for i in range(len(x) - seq_length):
        xi.append(x[i : i + seq_length])
        yi.append(y[i + seq_length])
    return np.array(xi), np.array(yi)

# 正确做法：先按时序划分，再创建序列
# 训练集：前80%按时间顺序，测试集：后20%未来数据
split = int(len(X_norm) * 0.8)
X_train_raw, X_test_raw = X_norm[:split], X_norm[split:]
y_train_raw, y_test_raw = y_norm[:split], y_norm[split:]

# 分别对训练集和测试集创建序列
WINDOW_SIZE = 30
X_train, y_train = create_sequences(X_train_raw, y_train_raw, WINDOW_SIZE)
X_test, y_test = create_sequences(X_test_raw, y_test_raw, WINDOW_SIZE)

print(f"窗口大小: {WINDOW_SIZE}")
print(f"训练集序列数: {len(X_train)}, 测试集序列数: {len(X_test)}")

X_train, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test)
y_train, y_test = torch.FloatTensor(y_train), torch.FloatTensor(y_test)

# 3. 模型定义
class COP_LSTM(nn.Module):
    def __init__(self, input_dim):
        super(COP_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, 64, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = COP_LSTM(len(feature_cols)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

# 4. 训练
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
print("🚀 开始真实物理特征训练...")
for epoch in range(60):
    model.train()
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1:02d} 完成")

# --- 5. 评估与绘图 (拆分展示) ---
from sklearn.metrics import r2_score, mean_absolute_error

model.eval()
with torch.no_grad():
    y_pred_norm = model(X_test.to(device)).cpu().numpy()

# 在真实物理尺度上计算指标，和其他模型公平对比
y_pred = scaler_y.inverse_transform(y_pred_norm)
y_true = scaler_y.inverse_transform(y_test.numpy())

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print(f"\n✅ 最终测试集评估结果:")
print(f"R² Score: {r2:.4f} (越接近1越好)")
print(f"MAE: {mae:.4f} (越小越好)")

# 图表 1: 预测效果对比
plt.figure(figsize=(12, 6))
plt.plot(y_true[:300], label='Actual COP', color='royalblue', alpha=0.8)
plt.plot(y_pred[:300], label='LSTM Predict', color='darkorange', linestyle='--')
plt.title(f"LSTM Model: Actual vs Predicted COP (Window Size = {WINDOW_SIZE}, R² = {r2:.4f}, MAE = {mae:.4f})")
plt.xlabel("Time Step (minute)")
plt.ylabel("System COP")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() # 第一个窗口弹出，关闭后会显示第二个

# 特征重要性分析 (Permutation Importance)
importance = {}
for i, col in enumerate(feature_cols):
    X_test_sh = X_test.clone()
    X_test_sh[:, :, i] = X_test_sh[torch.randperm(X_test_sh.size(0)), :, i]
    with torch.no_grad():
        sh_r2 = r2_score(y_test.numpy(), model(X_test_sh.to(device)).cpu().numpy())
    importance[col] = r2 - sh_r2

# 图表 2: 特征重要性
plt.figure(figsize=(10, 6))
pd.Series(importance).sort_values().plot(kind='barh', color='darkgreen')
plt.title("LSTM Model: Feature Importance Analysis")
plt.xlabel("Importance Score (R2 Drop)")
plt.tight_layout()
plt.show()