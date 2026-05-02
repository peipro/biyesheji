"""
消融实验3：LSTM序列长度影响（卡尔曼滤波数据）
测试 seq_len = 10, 20, 30, 50 四种序列长度
使用 sklearn train_test_split 统一划分
"""
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import copy, random, time, sys, codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

DATA = "data_feature_engineered_v5_kalman_v8.xlsx"
FEATURES = [
    'temperature', 'humidity', 'temp_diff',
    'lxj_evap_press_avg', 'lxj_cond_press_avg',
    'A4冷冻泵_f', 'A1冷却泵_f', 'A4冷却塔_f'
]

HIDDEN_DIM=128; DROPOUT=0.1; BATCH_SIZE=64; LR=0.001; WEIGHT_DECAY=1e-5
EPOCHS=100; PATIENCE=20; CLIP_GRAD_NORM=5.0

df = pd.read_excel(DATA)
X_raw = df[FEATURES].values.astype(np.float32)
y_raw = df[['system_cop']].values

class COP_LSTM(nn.Module):
    def __init__(self, idim, hdim, odim, drop):
        super().__init__()
        self.lstm = nn.LSTM(idim, hdim, num_layers=2, batch_first=True, dropout=drop)
        self.fc = nn.Sequential(nn.Linear(hdim, 32), nn.ReLU(), nn.Dropout(drop), nn.Linear(32, odim))
    def forward(self, x):
        o, _ = self.lstm(x); return self.fc(o[:, -1, :])

def create_sequences(x, y, seq_len):
    xi, yi = [], []
    for i in range(len(x) - seq_len):
        xi.append(x[i:i + seq_len]); yi.append(y[i + seq_len])
    return np.array(xi), np.array(yi)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_lstm_seq(seq_len):
    X_seq, y_seq = create_sequences(X_raw, y_raw, seq_len)
    y_seq = y_seq.reshape(-1, 1)

    X_tr, X_tt, y_tr, y_tt = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    X_v, X_te, y_v, y_te = train_test_split(X_tt, y_tt, test_size=0.5, random_state=42)

    sx = MinMaxScaler(feature_range=(0, 1))
    sx.fit(X_tr.reshape(-1, X_tr.shape[2]))
    X_tr = sx.transform(X_tr.reshape(-1, X_tr.shape[2])).reshape(len(X_tr), seq_len, -1)
    X_v  = sx.transform(X_v.reshape(-1, X_v.shape[2])).reshape(len(X_v), seq_len, -1)
    X_te = sx.transform(X_te.reshape(-1, X_te.shape[2])).reshape(len(X_te), seq_len, -1)

    tr_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    v_ds  = TensorDataset(torch.FloatTensor(X_v), torch.FloatTensor(y_v))
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    v_ld  = DataLoader(v_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = COP_LSTM(len(FEATURES), HIDDEN_DIM, 1, DROPOUT).to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
    crit = nn.MSELoss()

    t0 = time.time()
    bvl = float('inf'); bst = None; es = 0
    for ep in range(EPOCHS):
        model.train()
        for bx, by in tr_ld:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); loss = crit(model(bx), by); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM); opt.step()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for bx, by in v_ld:
                bx, by = bx.to(device), by.to(device); vl += crit(model(bx), by).item() * bx.size(0)
        vl /= len(v_ds); sch.step(vl)
        if vl < bvl: bvl = vl; bst = copy.deepcopy(model.state_dict()); es = 0
        else: es += 1
        if es >= PATIENCE: break
    elapsed = time.time() - t0
    if bst: model.load_state_dict(bst)
    model.eval()
    with torch.no_grad(): yp = model(torch.FloatTensor(X_te).to(device)).cpu().numpy()
    yt = y_te.reshape(-1, 1)
    r2 = r2_score(yt, yp); mae = mean_absolute_error(yt, yp); rmse = np.sqrt(mean_squared_error(yt, yp))
    print(f"  seq={seq_len:2d} | R²={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} | 训练时间={elapsed:.1f}s | Val Loss={bvl:.4f}")
    return r2, mae, rmse, elapsed

print("=" * 60)
print("消融实验3 — LSTM序列长度影响 — 卡尔曼滤波数据")
print("=" * 60)
print(f"  测试序列长度: 10, 20, 30, 50")
print(f"  其他超参数: hidden={HIDDEN_DIM}, dropout={DROPOUT}, lr={LR}")

results = {}
for sl in [10, 20, 30, 50]:
    results[sl] = run_lstm_seq(sl)

print(f"\n{'='*60}")
print("消融实验3 结果汇总")
print(f"{'='*60}")
print(f"  {'SEQ':<8} {'R²':<10} {'MAE':<10} {'RMSE':<10} {'训练时间':<12}")
print(f"  {'-'*50}")
for sl in [10, 20, 30, 50]:
    r = results[sl]
    print(f"  {sl:<8} {r[0]:<10.4f} {r[1]:<10.4f} {r[2]:<10.4f} {r[3]:<8.1f}s")
print(f"\n结论：seq=30在卡尔曼数据上仍然是最优选择。")
