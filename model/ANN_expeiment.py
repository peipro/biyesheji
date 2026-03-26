import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  # 换成归一化，对COP这种正数更稳
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

# 1. 极简预处理
df = pd.read_excel("../data_plus_features.xlsx").dropna()
exclude = ['date_time', 'system_cop', 'calc_Q_kw', 'power_consume', 'total_power_kw', 'supply_temp', 'return_temp',
           'current_flow']
features = [c for c in df.columns if c in df.columns and c not in exclude]

X = df[features].values
y = df['system_cop'].values.reshape(-1, 1)

# 改用 MinMaxScaler 限制在 [0, 1] 之间，防止梯度爆炸
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_s = scaler_x.fit_transform(X)
y_s = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.2, random_state=42)


# 2. 极简网络 (去掉复杂的层，先求跑通)
class SimpleNet(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, 32),
            nn.Tanh(),  # Tanh 比 ReLU 在回归任务中更平滑
            nn.Linear(32, 1)
        )

    def forward(self, x): return self.net(x)


model = SimpleNet(X_train.shape[1])
# 重点：学习率调低到 0.0001，防止 R2 变成负数
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

# 3. 训练
for epoch in range(1000):  # 增加轮数，小步快跑
    model.train()
    optimizer.zero_grad()
    pred = model(torch.tensor(X_train, dtype=torch.float32))
    loss = criterion(pred, torch.tensor(y_train, dtype=torch.float32))
    loss.backward()
    # 增加梯度裁剪，防止爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

# 4. 评估 (一定要反缩放！)
model.eval()
with torch.no_grad():
    y_p_s = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    y_p = scaler_y.inverse_transform(y_p_s)
    y_t = scaler_y.inverse_transform(y_test)

    final_r2 = r2_score(y_t, y_p)
    print(f"修正后的 R2: {final_r2:.4f}")