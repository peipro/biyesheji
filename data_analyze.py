import pandas as pd
import numpy as np
import os

# --- 1. 基础配置 ---
DATA_DIR = "05data"
POWER_FILE = os.path.join(DATA_DIR, "分钟耗电量.xlsx")
WEATHER_FILE = os.path.join(DATA_DIR, "温湿度.xlsx")
COOL_FILE = os.path.join(DATA_DIR, "冷量表.xlsx")
CHILLER_FILES = [os.path.join(DATA_DIR, f"lxj{i}.xlsx") for i in range(1, 5)]

# 设备列表（保持不变）
DEVICE_FILES = [
    os.path.join(DATA_DIR, 'A1冷冻泵.xlsx'), os.path.join(DATA_DIR, 'A2冷冻泵.xlsx'),
    os.path.join(DATA_DIR, 'A3冷冻泵.xlsx'), os.path.join(DATA_DIR, 'A4冷冻泵.xlsx'),
    os.path.join(DATA_DIR, 'B1冷冻泵.xlsx'), os.path.join(DATA_DIR, 'B2冷冻泵.xlsx'),
    os.path.join(DATA_DIR, 'B3冷冻泵.xlsx'),
    os.path.join(DATA_DIR, 'A1冷却泵.xlsx'), os.path.join(DATA_DIR, 'A2冷却泵.xlsx'),
    os.path.join(DATA_DIR, 'A3冷却泵.xlsx'), os.path.join(DATA_DIR, 'A4冷却泵.xlsx'),
    os.path.join(DATA_DIR, 'B1冷却泵.xlsx'), os.path.join(DATA_DIR, 'B2冷却泵.xlsx'),
    os.path.join(DATA_DIR, 'B3冷却泵.xlsx'),
    os.path.join(DATA_DIR, 'A1冷却塔.xlsx'), os.path.join(DATA_DIR, 'A2冷却塔.xlsx'),
    os.path.join(DATA_DIR, 'A3冷却塔.xlsx'), os.path.join(DATA_DIR, 'A4冷却塔.xlsx'),
    os.path.join(DATA_DIR, 'B1冷却塔.xlsx'), os.path.join(DATA_DIR, 'B2冷却塔.xlsx'),
    os.path.join(DATA_DIR, 'B3冷却塔.xlsx')
]

def load_excel_refined(path):
    """通用读取：数值化保护与分钟对齐"""
    if not os.path.exists(path): return None
    print(f"正在读取并预处理: {path}")
    df = pd.read_excel(path)
    cols_to_protect = ['power_consume', 'temperature', 'humidity', 'feedback_frequency', 'run_stop']
    for col in cols_to_protect:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date_time'] = pd.to_datetime(df['date_time']).dt.floor('min')
    df = df.groupby('date_time').mean(numeric_only=True).reset_index()
    return df.sort_values('date_time')

# --- 2. 整合流程 ---

# A. 读取主表（耗电量）
df_main = load_excel_refined(POWER_FILE)
print(f"主轴(耗电量)初始行数: {len(df_main)}")

# B. 【核心改进】分机计算冷量表
print(f"正在读取冷量表并执行分机计算: {COOL_FILE}")
df_cool_raw = pd.read_excel(COOL_FILE)
df_cool_raw.columns = [str(c).strip() for c in df_cool_raw.columns]
df_cool_raw['date_time'] = pd.to_datetime(df_cool_raw['date_time']).dt.floor('min')

# 强制数值化冷量表关键列
for col in ['current_flow', 'return_temp', 'supply_temp']:
    df_cool_raw[col] = pd.to_numeric(df_cool_raw[col], errors='coerce').fillna(0)

# 【物理逻辑】：先计算每一行记录（单台冷机）的瞬时制冷量
# 如果你的 current_flow 是 2 分钟累计量，请把 1 换成 30
FLOW_FACTOR = 1  # 如果是 m3/h 取 1；如果是 2min 累计 m3 取 30
df_cool_raw['row_Q'] = (df_cool_raw['current_flow'] * FLOW_FACTOR * 4.186 * (df_cool_raw['return_temp'] - df_cool_raw['supply_temp'])) / 3.6
df_cool_raw['row_Q'] = df_cool_raw['row_Q'].clip(lower=0)

# 【聚合】：同一分钟内的多台冷机数据，Q求和，流量求和，温度取均值
df_cool = df_cool_raw.groupby('date_time').agg({
    'row_Q': 'sum',
    'current_flow': 'sum',
    'return_temp': 'mean',
    'supply_temp': 'mean'
}).reset_index().rename(columns={'row_Q': 'calc_Q_kw'})

# 合并冷量数据
df_main = pd.merge_asof(df_main, df_cool, on='date_time', direction='nearest', tolerance=pd.Timedelta('5min'))

# C. 合并其他表（温湿度、泵塔、冷机细节）
# 温湿度
df_weather = load_excel_refined(WEATHER_FILE)
if df_weather is not None:
    df_main = pd.merge_asof(df_main, df_weather, on='date_time', direction='nearest', tolerance=pd.Timedelta('5min'))

# 泵与塔
for path in DEVICE_FILES:
    df_dev = load_excel_refined(path)
    if df_dev is not None:
        name = os.path.basename(path).replace('.xlsx', '')
        if 'run_stop' in df_dev.columns and 'feedback_frequency' in df_dev.columns:
            df_dev[f'{name}_freq_cleaned'] = np.where(df_dev['run_stop'] == 1, df_dev['feedback_frequency'], 0)
        else:
            df_dev[f'{name}_freq_cleaned'] = df_dev.get('feedback_frequency', 0)
        df_main = pd.merge_asof(df_main, df_dev[['date_time', f'{name}_freq_cleaned']], on='date_time', direction='nearest', tolerance=pd.Timedelta('5min'))

# 冷机运行参数
for path in CHILLER_FILES:
    df_lxj = load_excel_refined(path)
    if df_lxj is not None:
        lxj_n = os.path.basename(path).replace('.xlsx', '')
        df_lxj = df_lxj.rename(columns={c: f"{lxj_n}_{c}" for c in df_lxj.columns if c != 'date_time'})
        df_main = pd.merge_asof(df_main, df_lxj, on='date_time', direction='nearest', tolerance=pd.Timedelta('5min'))

# --- 3. 核心计算与 10 分钟对齐 ---

# 1. 10分钟重采样：消除 2min/5min 采样差，对齐电耗与冷量
df_main = df_main.set_index('date_time').resample('10T').mean().reset_index()

# 2. 缺失值填充
df_main = df_main.fillna(method='ffill').fillna(method='bfill')

# 3. 功率计算 (如果 power_consume 是 5min 累计电量)
df_main['total_power_kw'] = df_main['power_consume'] * 12

# 4. 系统 COP 计算
# 此时 calc_Q_kw 已经是多机累加后的总制冷量
df_main['system_cop'] = np.where(df_main['total_power_kw'] > 0,
                                 df_main['calc_Q_kw'] / df_main['total_power_kw'], 0)

# --- 4. 物理过滤：只保留真正运行且合理的数据 ---
# 过滤掉流量过低或温差不正常的点
df_final = df_main[
    (df_main['current_flow'] > 1.0) &
    (df_main['return_temp'] > df_main['supply_temp']) &
    (df_main['system_cop'] > 0.5) & (df_main['system_cop'] < 15)
].copy()

# --- 5. 导出 ---
output_name = "data_all_merged_optimized.xlsx"
df_final.to_excel(output_name, index=False)
print(f"✅ 合并完成！清洗后有效行数: {len(df_final)}")