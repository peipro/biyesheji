"""
消融实验1(LSTM)：物理特征有效性验证（卡尔曼滤波数据）
使用 sklearn train_test_split，seq_len=30
对比：8特征基线 vs 移除关键特征后的性能变化
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import copy
import random
import sys, codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

DATA = "data_feature_engineered_v5_kalman_v8.xlsx"
ALL_FEATURES = [
    'temperature', 'humidity', 'temp_diff',
    'lxj_evap_press_avg', 'lxj_cond_press_avg',
    'A4冷冻泵_f', 'A1冷却泵_f', 'A4冷却塔_f'
]
DEVICE_FEATURES = ['A4冷冻泵_f', 'A1冷却泵_f', 'A4冷却塔_f']

SEQ_LENGTH = 30; HIDDEN_DIM = 128; DROPOUT = 0.1
BATCH_SIZE = 64; LR = 0.001; WEIGHT_DECAY = 1e-5
EPOCHS = 100; PATIENCE = 20; CLIP_GRAD_NORM = 5.0

df = pd.read_excel(DATA)
y_raw = df[['system_cop']].values

class COP_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, output_dim))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def create_sequences(x, y, seq_len):
    xi, yi = [], []
    for i in range(len(x) - seq_len):
        xi.append(x[i:i + seq_len])
        yi.append(y[i + seq_len])
    return np.array(xi), np.array(yi)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_lstm(features, label):
    """在指定特征集上训练LSTM并返回测试集指标"""
    X_raw = df[features].values.astype(np.float32)
    X_seq, y_seq = create_sequences(X_raw, y_raw, SEQ_LENGTH)
    y_seq = y_seq.reshape(-1, 1)

    X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_x.fit(X_train.reshape(-1, X_train.shape[2]))
    X_train = scaler_x.transform(X_train.reshape(-1, X_train.shape[2])).reshape(len(X_train), SEQ_LENGTH, -1)
    X_val   = scaler_x.transform(X_val.reshape(-1, X_val.shape[2])).reshape(len(X_val), SEQ_LENGTH, -1)
    X_test  = scaler_x.transform(X_test.reshape(-1, X_test.shape[2])).reshape(len(X_test), SEQ_LENGTH, -1)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds   = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = COP_LSTM(len(features), HIDDEN_DIM, 1, DROPOUT).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    early_stop = 0
    for epoch in range(EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * bx.size(0)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss; best_state = copy.deepcopy(model.state_dict()); early_stop = 0
        else:
            early_stop += 1
            if early_stop >= PATIENCE: break

    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    y_true = y_test.reshape(-1, 1)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[LSTM] {label} | 特征数={len(features)} | R²={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} | Best Val Loss={best_val_loss:.4f}")
    return r2, mae, rmse

print("=" * 60)
print("消融实验1(LSTM) — 卡尔曼滤波数据 — sklearn划分")
print("=" * 60)
lstm_base   = run_lstm(ALL_FEATURES, "基线(8特征)")
lstm_no_td  = run_lstm([f for f in ALL_FEATURES if f != 'temp_diff'], "-temp_diff")
lstm_no_dev = run_lstm([f for f in ALL_FEATURES if f not in DEVICE_FEATURES], "-设备频率(3)")
lstm_no_both= run_lstm([f for f in ALL_FEATURES if f != 'temp_diff' and f not in DEVICE_FEATURES], "-temp_diff-设备频率")

print(f"\n{'='*60}")
print("LSTM 消融实验结果汇总")
print(f"{'='*60}")
print(f"  基线(8特征):       R²={lstm_base[0]:.4f}  MAE={lstm_base[1]:.4f}  RMSE={lstm_base[2]:.4f}")
print(f"  移除temp_diff:     R²={lstm_no_td[0]:.4f}  ΔR²={lstm_no_td[0]-lstm_base[0]:+.4f}")
print(f"  移除设备频率(3):   R²={lstm_no_dev[0]:.4f}  ΔR²={lstm_no_dev[0]-lstm_base[0]:+.4f}")
print(f"  移除两者:          R²={lstm_no_both[0]:.4f}  ΔR²={lstm_no_both[0]-lstm_base[0]:+.4f}")
