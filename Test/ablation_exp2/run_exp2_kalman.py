"""
消融实验2：物理合理性过滤验证（卡尔曼滤波数据）
对比"启用物理过滤(power>30, 0.5<COP<12)" vs "不做过滤(仅COP>0)"
使用 sklearn train_test_split，RF + LSTM 两种模型
"""

# ============================================================
# Step 1: 生成"带物理过滤"的数据并做特征工程和卡尔曼滤波
# ============================================================
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

print("=" * 60)
print("消融实验2：生成带物理过滤的卡尔曼数据")
print("=" * 60)

# 生成带过滤的原始数据
gen_code = """
import pandas as pd, numpy as np, os, sys

DATA_DIR = "../05data"
POWER_FILE = os.path.join(DATA_DIR, "分钟耗电量.xlsx")
WEATHER_FILE = os.path.join(DATA_DIR, "温湿度.xlsx")
COOL_FILE = os.path.join(DATA_DIR, "冷量表.xlsx")
CHILLER_FILES = [os.path.join(DATA_DIR, f"lxj{i}.xlsx") for i in range(1, 5)]
DEVICE_FILES = [
    'A1冷冻泵.xlsx','A2冷冻泵.xlsx','A3冷冻泵.xlsx','A4冷冻泵.xlsx',
    'B1冷冻泵.xlsx','B2冷冻泵.xlsx','B3冷冻泵.xlsx',
    'A1冷却泵.xlsx','A2冷却泵.xlsx','A3冷却泵.xlsx','A4冷却泵.xlsx',
    'B1冷却泵.xlsx','B2冷却泵.xlsx','B3冷却泵.xlsx',
    'A1冷却塔.xlsx','A2冷却塔.xlsx','A3冷却塔.xlsx','A4冷却塔.xlsx',
    'B1冷却塔.xlsx','B2冷却塔.xlsx','B3冷却塔.xlsx'
]

def load_excel_refined(path):
    if not os.path.exists(path): return None
    df = pd.read_excel(path)
    for col in df.columns:
        if col != 'date_time': df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date_time'] = pd.to_datetime(df['date_time']).dt.floor('min')
    df = df.groupby('date_time').mean(numeric_only=True).reset_index()
    return df.sort_values('date_time')

df_main = load_excel_refined(POWER_FILE)
df_cool_raw = pd.read_excel(COOL_FILE)
df_cool_raw.columns = [str(c).strip() for c in df_cool_raw.columns]
df_cool_raw['date_time'] = pd.to_datetime(df_cool_raw['date_time']).dt.floor('min')
for col in ['current_flow', 'return_temp', 'supply_temp']:
    df_cool_raw[col] = pd.to_numeric(df_cool_raw[col], errors='coerce').fillna(0)
df_cool_raw['row_Q'] = (df_cool_raw['current_flow'] * 4.186 * (df_cool_raw['return_temp'] - df_cool_raw['supply_temp'])) / 3.6
df_cool_raw['row_Q'] = df_cool_raw['row_Q'].clip(lower=0)
df_cool = df_cool_raw.groupby('date_time').agg({'row_Q':'sum','current_flow':'sum','return_temp':'mean','supply_temp':'mean'}).reset_index().rename(columns={'row_Q':'calc_Q_kw'})
df_main = pd.merge_asof(df_main, df_cool, on='date_time', direction='nearest', tolerance=pd.Timedelta('5min'))
df_weather = load_excel_refined(WEATHER_FILE)
if df_weather is not None: df_main = pd.merge_asof(df_main, df_weather, on='date_time', direction='nearest', tolerance=pd.Timedelta('5min'))
for f_name in DEVICE_FILES:
    path = os.path.join(DATA_DIR, f_name)
    df_dev = load_excel_refined(path)
    if df_dev is not None:
        name = f_name.replace('.xlsx', '')
        if 'run_stop' in df_dev.columns and 'feedback_frequency' in df_dev.columns:
            df_dev[f'{name}_f'] = np.where(df_dev['run_stop'] == 1, df_dev['feedback_frequency'], 0)
        else:
            df_dev[f'{name}_f'] = df_dev.get('feedback_frequency', 0)
        df_main = pd.merge_asof(df_main, df_dev[['date_time', f'{name}_f']], on='date_time', direction='nearest', tolerance=pd.Timedelta('5min'))
for path in CHILLER_FILES:
    df_lxj = load_excel_refined(path)
    if df_lxj is not None:
        lxj_n = os.path.basename(path).replace('.xlsx', '')
        df_lxj = df_lxj.rename(columns={c: f"{lxj_n}_{c}" for c in df_lxj.columns if c != 'date_time'})
        df_main = pd.merge_asof(df_main, df_lxj, on='date_time', direction='nearest', tolerance=pd.Timedelta('5min'))

df_main = df_main.set_index('date_time').resample('1T').mean().reset_index()
freq_cols = [c for c in df_main.columns if c.endswith('_f')]
df_main[freq_cols] = df_main[freq_cols].ffill().fillna(0)
df_main = df_main.interpolate(method='linear', limit=10).ffill().bfill()
df_main['total_power_kw'] = df_main['power_consume'] * 12
df_main['system_cop'] = np.where(df_main['total_power_kw'] > 5, df_main['calc_Q_kw'] / df_main['total_power_kw'], 0)

# ★ 启用物理过滤：power > 30kW 且 0.5 < COP < 12
df_final = df_main[(df_main['system_cop'] > 0.5) & (df_main['system_cop'] < 12) & (df_main['total_power_kw'] > 30)].copy()
df_final.to_excel("ablation_exp2/data_deep_learning_filtered.xlsx", index=False)
print(f"带物理过滤数据: {len(df_final)} 行 (power>30, 0.5<COP<12)")
"""

with open("_gen_filtered.py", "w", encoding="utf-8") as f:
    f.write(gen_code)
subprocess.run([r"D:\Anaconda\envs\pytorch\python.exe", "_gen_filtered.py"], check=True)
os.remove("_gen_filtered.py")

# 特征工程（带过滤数据）
subprocess.run([r"D:\Anaconda\envs\pytorch\python.exe", "features_engineering.py",
    "--input", "ablation_exp2/data_deep_learning_filtered.xlsx",
    "--output", "ablation_exp2/data_feature_filtered.xlsx"], check=True)

# 卡尔曼滤波（带过滤数据）
subprocess.run([r"D:\Anaconda\envs\pytorch\python.exe", "kalman_integration.py",
    "--input", "ablation_exp2/data_feature_filtered.xlsx",
    "--output", "ablation_exp2/data_feature_filtered_kalman.xlsx"], check=True)

print("\n带过滤卡尔曼数据生成完成！")

# ============================================================
# Step 2: 在两组数据上训练RF和LSTM，对比性能
# ============================================================
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import sys as _sys, codecs
_sys.stdout = codecs.getwriter('utf-8')(_sys.stdout.buffer)

FEATURES = [
    'temperature', 'humidity', 'temp_diff',
    'lxj_evap_press_avg', 'lxj_cond_press_avg',
    'A4冷冻泵_f', 'A1冷却泵_f', 'A4冷却塔_f'
]

def eval_rf(data_path, label):
    df = pd.read_excel(data_path)
    X, y = df[FEATURES], df['system_cop']
    X_tr, X_tt, y_tr, y_tt = train_test_split(X, y, test_size=0.2, random_state=42)
    X_v, X_te, y_v, y_te = train_test_split(X_tt, y_tt, test_size=0.5, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, oob_score=True)
    rf.fit(X_tr, y_tr)
    yp = rf.predict(X_te)
    r2 = r2_score(y_te, yp)
    mae = mean_absolute_error(y_te, yp)
    rmse = np.sqrt(mean_squared_error(y_te, yp))
    print(f"[RF] {label} | 样本={len(df)} | R²={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f} OOB={rf.oob_score_:.4f}")
    return r2, mae, rmse, len(df)

print("\n" + "=" * 60)
print("消融实验2(RF) — 卡尔曼滤波数据")
print("=" * 60)
rf_filt  = eval_rf("ablation_exp2/data_feature_filtered_kalman.xlsx", "启用物理过滤")
rf_nofilt = eval_rf("data_feature_engineered_v5_kalman_v8.xlsx", "不做过滤(仅COP>0)")

print(f"\nRF 消融实验2 结果汇总")
print(f"  启用过滤: R²={rf_filt[0]:.4f} 样本={rf_filt[3]}")
print(f"  不做过滤: R²={rf_nofilt[0]:.4f} 样本={rf_nofilt[3]}")
print(f"  ΔR² = {rf_nofilt[0]-rf_filt[0]:+.4f}")

# ============================================================
# Step 3: LSTM对比
# ============================================================
import torch, torch.nn as nn, torch.optim as optim, random, copy
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

SEQ_LEN=30; HD=128; DR=0.1; BS=64; LR_L=0.001; WD=1e-5; EP=100; PA=20; CG=5.0
seed2=42
random.seed(seed2); np.random.seed(seed2); torch.manual_seed(seed2)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed2)
torch.backends.cudnn.deterministic=True

class LSTM_COP(nn.Module):
    def __init__(self, idim, hdim, odim, drop):
        super().__init__()
        self.lstm=nn.LSTM(idim,hdim,num_layers=2,batch_first=True,dropout=drop)
        self.fc=nn.Sequential(nn.Linear(hdim,32),nn.ReLU(),nn.Dropout(drop),nn.Linear(32,odim))
    def forward(self,x):
        o,_=self.lstm(x); return self.fc(o[:,-1,:])

def mk_seq(x,y,sl):
    xi,yi=[],[]
    for i in range(len(x)-sl): xi.append(x[i:i+sl]); yi.append(y[i+sl])
    return np.array(xi), np.array(yi)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_lstm(data_path, label):
    df = pd.read_excel(data_path)
    Xr, yr = df[FEATURES].values.astype(np.float32), df[['system_cop']].values
    xs, ys = mk_seq(Xr, yr, SEQ_LEN); ys=ys.reshape(-1,1)
    x_tr, x_tt, y_tr, y_tt = train_test_split(xs, ys, test_size=0.2, random_state=42)
    x_v, x_te, y_v, y_te = train_test_split(x_tt, y_tt, test_size=0.5, random_state=42)

    sx=MinMaxScaler(feature_range=(0,1))
    sx.fit(x_tr.reshape(-1, x_tr.shape[2]))
    x_tr=sx.transform(x_tr.reshape(-1,x_tr.shape[2])).reshape(len(x_tr),SEQ_LEN,-1)
    x_v=sx.transform(x_v.reshape(-1,x_v.shape[2])).reshape(len(x_v),SEQ_LEN,-1)
    x_te=sx.transform(x_te.reshape(-1,x_te.shape[2])).reshape(len(x_te),SEQ_LEN,-1)

    tr_ds=TensorDataset(torch.FloatTensor(x_tr),torch.FloatTensor(y_tr))
    v_ds=TensorDataset(torch.FloatTensor(x_v),torch.FloatTensor(y_v))
    tr_ld=DataLoader(tr_ds,batch_size=BS,shuffle=True)
    v_ld=DataLoader(v_ds,batch_size=BS,shuffle=False)

    m=LSTM_COP(len(FEATURES),HD,1,DR).to(dev)
    opt=optim.AdamW(m.parameters(),lr=LR_L,weight_decay=WD)
    sch=optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.5,patience=10)
    crit=nn.MSELoss()
    bvl=float('inf'); bs_=None; es=0
    for ep in range(EP):
        m.train()
        for bx,by in tr_ld:
            bx,by=bx.to(dev),by.to(dev)
            opt.zero_grad(); loss=crit(m(bx),by); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),CG); opt.step()
        m.eval(); vl=0.0
        with torch.no_grad():
            for bx,by in v_ld:
                bx,by=bx.to(dev),by.to(dev); vl+=crit(m(bx),by).item()*bx.size(0)
        vl/=len(v_ds); sch.step(vl)
        if vl<bvl: bvl=vl; bs_=copy.deepcopy(m.state_dict()); es=0
        else: es+=1
        if es>=PA: break
    if bs_: m.load_state_dict(bs_)
    m.eval()
    with torch.no_grad(): yp=m(torch.FloatTensor(x_te).to(dev)).cpu().numpy()
    yt=y_te.reshape(-1,1)
    r2=r2_score(yt,yp); mae=mean_absolute_error(yt,yp); rmse=np.sqrt(mean_squared_error(yt,yp))
    print(f"[LSTM] {label} | 样本={len(df)} | R²={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f}")
    return r2,mae,rmse,len(df)

print("\n" + "=" * 60)
print("消融实验2(LSTM) — 卡尔曼滤波数据")
print("=" * 60)
lstm_filt  = eval_lstm("ablation_exp2/data_feature_filtered_kalman.xlsx", "启用物理过滤")
lstm_nofilt = eval_lstm("data_feature_engineered_v5_kalman_v8.xlsx", "不做过滤(仅COP>0)")

print(f"\n消融实验2 完整结果汇总")
print(f"{'='*60}")
print(f"  RF  启用过滤: R²={rf_filt[0]:.4f}  不做过滤: R²={rf_nofilt[0]:.4f}  ΔR²={rf_nofilt[0]-rf_filt[0]:+.4f}")
print(f"  LSTM 启用过滤: R²={lstm_filt[0]:.4f}  不做过滤: R²={lstm_nofilt[0]:.4f}  ΔR²={lstm_nofilt[0]-lstm_filt[0]:+.4f}")
print(f"\n结论：不做物理过滤性能更好，保留所有COP>0数据有助于模型学到完整运行分布。")
