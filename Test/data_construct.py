import pandas as pd
import numpy as np
import os

# --- 1. 基础配置 ---
DATA_DIR = "../05data"
POWER_FILE = os.path.join(DATA_DIR, "分钟耗电量.xlsx")
WEATHER_FILE = os.path.join(DATA_DIR, "温湿度.xlsx")
COOL_FILE = os.path.join(DATA_DIR, "冷量表.xlsx")
CHILLER_FILES = [os.path.join(DATA_DIR, f"lxj{i}.xlsx") for i in range(1, 5)]

DEVICE_FILES = [
    'A1冷冻泵.xlsx', 'A2冷冻泵.xlsx', 'A3冷冻泵.xlsx', 'A4冷冻泵.xlsx',
    'B1冷冻泵.xlsx', 'B2冷冻泵.xlsx', 'B3冷冻泵.xlsx',
    'A1冷却泵.xlsx', 'A2冷却泵.xlsx', 'A3冷却泵.xlsx', 'A4冷却泵.xlsx',
    'B1冷却泵.xlsx', 'B2冷却泵.xlsx', 'B3冷却泵.xlsx',
    'A1冷却塔.xlsx', 'A2冷却塔.xlsx', 'A3冷却塔.xlsx', 'A4冷却塔.xlsx',
    'B1冷却塔.xlsx', 'B2冷却塔.xlsx', 'B3冷却塔.xlsx'
]


def load_excel_refined(path):
    """通用读取：数值化保护与分钟对齐"""
    if not os.path.exists(path): return None
    print("正在读取并预处理:", path)
    df = pd.read_excel(path)
    # 强制数值化，解决乘法报错
    for col in df.columns:
        if col != 'date_time':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['date_time'] = pd.to_datetime(df['date_time']).dt.floor('min')
    df = df.groupby('date_time').mean(numeric_only=True).reset_index()
    return df.sort_values('date_time')


# --- 2. 整合流程 ---

# A. 读取主表（耗电量）
df_main = load_excel_refined(POWER_FILE)
print(f"主轴(耗电量)初始行数: {len(df_main)}")

# B. 分机计算冷量表
print(f"正在读取冷量表: {COOL_FILE}")
df_cool_raw = pd.read_excel(COOL_FILE)
df_cool_raw.columns = [str(c).strip() for c in df_cool_raw.columns]
df_cool_raw['date_time'] = pd.to_datetime(df_cool_raw['date_time']).dt.floor('min')

for col in ['current_flow', 'return_temp', 'supply_temp']:
    df_cool_raw[col] = pd.to_numeric(df_cool_raw[col], errors='coerce').fillna(0)

# 计算瞬时制冷量 (kW)
df_cool_raw['row_Q'] = (df_cool_raw['current_flow'] * 4.186 * (
            df_cool_raw['return_temp'] - df_cool_raw['supply_temp'])) / 3.6
df_cool_raw['row_Q'] = df_cool_raw['row_Q'].clip(lower=0)

df_cool = df_cool_raw.groupby('date_time').agg({
    'row_Q': 'sum',
    'current_flow': 'sum',
    'return_temp': 'mean',
    'supply_temp': 'mean'
}).reset_index().rename(columns={'row_Q': 'calc_Q_kw'})

# 合并冷量数据
df_main = pd.merge_asof(df_main, df_cool, on='date_time', direction='nearest', tolerance=pd.Timedelta('5min'))

# C. 合并其他表
# 温湿度
df_weather = load_excel_refined(WEATHER_FILE)
if df_weather is not None:
    df_main = pd.merge_asof(df_main, df_weather, on='date_time', direction='nearest', tolerance=pd.Timedelta('5min'))

# 泵与塔 (关键点：如果没有匹配到，先留空)
for f_name in DEVICE_FILES:
    path = os.path.join(DATA_DIR, f_name)
    df_dev = load_excel_refined(path)
    if df_dev is not None:
        name = f_name.replace('.xlsx', '')
        if 'run_stop' in df_dev.columns and 'feedback_frequency' in df_dev.columns:
            df_dev[f'{name}_f'] = np.where(df_dev['run_stop'] == 1, df_dev['feedback_frequency'], 0)
        else:
            df_dev[f'{name}_f'] = df_dev.get('feedback_frequency', 0)
        df_main = pd.merge_asof(df_main, df_dev[['date_time', f'{name}_f']], on='date_time', direction='nearest',
                                tolerance=pd.Timedelta('5min'))

# 冷机运行参数
for path in CHILLER_FILES:
    df_lxj = load_excel_refined(path)
    if df_lxj is not None:
        lxj_n = os.path.basename(path).replace('.xlsx', '')
        df_lxj = df_lxj.rename(columns={c: f"{lxj_n}_{c}" for c in df_lxj.columns if c != 'date_time'})
        df_main = pd.merge_asof(df_main, df_lxj, on='date_time', direction='nearest', tolerance=pd.Timedelta('5min'))

# --- 3. 核心计算与高频化 ---

# 1. 重采样到均匀1分钟时间粒度
# 方案B：保持1分钟间隔，频率采用前向填充避免错标运行设备为停机
df_main = df_main.set_index('date_time').resample('1T').mean().reset_index()

# 2. 缺失值智能填充
# 对于频率列 (_f 结尾)，先使用前向填充延续设备状态，剩余缺失（序列开头）填充 0
# 这样避免把运行中的设备在插值点被错误标记为停机
freq_cols = [c for c in df_main.columns if c.endswith('_f')]
df_main[freq_cols] = df_main[freq_cols].ffill().fillna(0)

# 对于其他物理列，使用线性插值补全短缺失
df_main = df_main.interpolate(method='linear', limit=10).ffill().bfill()

# 3. 功率计算 (5min 累计能耗 * 12)
df_main['total_power_kw'] = df_main['power_consume'] * 12

# 4. 系统 COP 计算
df_main['system_cop'] = np.where(df_main['total_power_kw'] > 5,
                                 df_main['calc_Q_kw'] / df_main['total_power_kw'], 0)

# --- 4. 仅过滤停机数据（不做物理范围过滤） ===
# 消融实验2验证：保留所有COP>0数据性能更好（R²提升+0.03）
# 过滤掉COP<=0的停机/空载工况即可
df_final = df_main[df_main['system_cop'] > 0].copy()

# --- 5. 导出 ---
output_name = "data_deep_learning_final_v3.xlsx"
df_final.to_excel(output_name, index=False)
print("合并完成！最终有效行数:", len(df_final))