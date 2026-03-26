import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

# --- 1. 数据准备 ---
# 加载你刚刚生成的特征工程后的表
df = pd.read_excel("data_feature_engineered.xlsx")

# 自动识别特征列 (排除时间轴和目标变量)
target_col = 'system_cop'
feature_cols = [c for c in df.columns if c not in ['date_time', target_col]]

print(f"输入特征维度: {len(feature_cols)}")
print(f"特征列表: {feature_cols}")

# 数据标准化
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_data = scaler_x.fit_transform(df[feature_cols])
y_data = scaler_y.fit_transform(df[[target_col]])


# --- 2. 构造滑动窗口序列 ---
def create_sequences(x, y, seq_length=30):
    """用过去 30 分钟的数据预测当前时刻"""
    xi, yi = [], []
    for i in range(len(x) - seq_length):
        xi.append(x[i: i + seq_length])
        yi.append(y[i + seq_length])
    return np.array(xi), np.array(yi)


WINDOW_SIZE = 30  # 窗口长度：30分钟
X_all, y_all = create_sequences(x_data, y_data, WINDOW_SIZE)

# 划分训练集与测试集 (8:2)
split = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

# 转为 PyTorch 张量
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)


# --- 3. 定义 LSTM 模型结构 ---
class COP_Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3):
        super(COP_Predictor, self).__init__()
        # batch_first=True 对应数据维度 [batch, seq, feature]
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出作为特征
        last_time_step = lstm_out[:, -1, :]
        return self.fc(last_time_step)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = COP_Predictor(len(feature_cols)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 4. 训练模型 ---
epochs = 50
print(f"开始在 {device} 上训练...")

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    if epoch % 5 == 0:
        print(f"Epoch {epoch:02d} | Loss: {train_loss / len(train_loader):.6f}")

# --- 5. 评估与可视化 ---
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test.to(device)).cpu().numpy()

# 逆标准化还原真实物理值
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test.numpy())

# 计算指标
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print("-" * 30)
print(f"📊 测试集表现:")
print(f"R² Score: {r2:.4f} (越接近 1 越好)")
print(f"MAE: {mae:.4f}")

# 绘制对比曲线 (截取前 300 个点观察细节)
plt.figure(figsize=(12, 6))
plt.plot(y_true[:300], label='Actual COP', color='blue', alpha=0.7)
plt.plot(y_pred[:300], label='Predicted COP', color='red', linestyle='--')
plt.title(f"LSTM COP Prediction (R2: {r2:.4f})")
plt.xlabel("Time Steps (min)")
plt.ylabel("COP")
plt.legend()
plt.show()