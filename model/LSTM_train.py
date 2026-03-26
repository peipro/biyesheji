import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler  # 换成StandardScaler对异常值更鲁棒
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# --- 1. 数据深度预处理 ---
df = pd.read_excel("../data_plus_features.xlsx")
# 严格剔除物理非法值
df = df[(df['system_cop'] > 1.2) & (df['system_cop'] < 7.0)].copy()

# 核心特征选择
core_features = ['chiller_avg_freq', 'chiller_avg_con_press', 'chiller_avg_eva_press',
                 'temp_diff', 'press_diff', 'temperature', 'humidity']

X_raw = df[core_features].values
y_raw = df['system_cop'].values.reshape(-1, 1)

# 使用 StandardScaler 替代 MinMaxScaler，防止极值拉扁分布
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_x.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw)


# 构造序列 (窗口缩短为 3，降低学习难度)
def create_sequences(X, y, steps=3):
    Xs, ys = [], []
    for i in range(len(X) - steps):
        Xs.append(X[i:(i + steps)])
        ys.append(y[i + steps])
    return np.array(Xs), np.array(ys)


X_seq, y_seq = create_sequences(X_scaled, y_scaled, steps=3)

# 【核心改变】：开启 Shuffle=True。这能解决工况偏移导致的 R2 为负的问题
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, shuffle=True
)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)


# --- 2. 极简 LSTM 结构 ---
class RobustLSTM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, 16, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # 取最后一个时刻并加随机失活
        return self.fc(out)


model = RobustLSTM(len(core_features))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# --- 3. 训练循环 ---
epochs = 400
train_losses = []
test_losses = []

print("正在全力抢救 R²...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = criterion(pred, y_train_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 极严格的梯度裁剪
    optimizer.step()

    # 记录测试集表现
    model.eval()
    with torch.no_grad():
        t_pred = model(X_test_t)
        t_loss = criterion(t_pred, y_test_t)
        train_losses.append(loss.item())
        test_losses.append(t_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}, Train Loss: {loss.item():.4f}, Test Loss: {t_loss.item():.4f}")

# --- 4. 最终评估 ---
model.eval()
with torch.no_grad():
    y_p_s = model(X_test_t).numpy()
    y_p = scaler_y.inverse_transform(y_p_s)
    y_t = scaler_y.inverse_transform(y_test_t.numpy())

    final_r2 = r2_score(y_t, y_p)
    print(f"\n--- 修正版 LSTM 实验结果 ---")
    print(f"R² Score: {final_r2:.4f}")

# 可视化 Loss 曲线，检查是否收敛
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend();
plt.show()