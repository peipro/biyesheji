import pandas as pd
import numpy as np
import os

# --- 1. 基础配置：确保所有路径都指向 05data 文件夹 ---
DATA_DIR = "05data"

# 定义主轴表 (5分钟采集)
POWER_FILE = os.path.join(DATA_DIR, "分钟耗电量.xlsx")
# 定义环境与负荷表 (2分钟采集)
WEATHER_FILE = os.path.join(DATA_DIR, "温湿度.xlsx")
COOL_FILE = os.path.join(DATA_DIR, "冷量表.xlsx")

# 定义 4 台冷机 (注意：这里已包含路径)
CHILLER_FILES = [
    os.path.join(DATA_DIR, "lxj1.xlsx"),
    os.path.join(DATA_DIR, "lxj2.xlsx"),
    os.path.join(DATA_DIR, "lxj3.xlsx"),
    os.path.join(DATA_DIR, "lxj4.xlsx")
]

# 定义 A 区和 B 区所有泵组与冷却塔
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

def load_excel_safe(path):
    """读取Excel并转换时间格式"""
    if not os.path.exists(path):
        print(f"找不到文件: {path}")
        return None
    print(f"正在读取: {path}")
    df = pd.read_excel(path)
    df['date_time'] = pd.to_datetime(df['date_time'])
    return df.sort_values('date_time')

# --- 2. 开始整合 ---

# 以“分钟耗电量”为主表
df_main = load_excel_safe(POWER_FILE)

# 合并温湿度和冷量
df_weather = load_excel_safe(WEATHER_FILE)
df_cool = load_excel_safe(COOL_FILE)

if df_weather is not None:
    df_main = pd.merge_asof(df_main, df_weather, on='date_time', direction='nearest')
if df_cool is not None:
    df_main = pd.merge_asof(df_main, df_cool, on='date_time', direction='nearest')

# 循环处理泵和塔：清洗频率 (run_stop=1 逻辑)
for path in DEVICE_FILES:
    df_dev = load_excel_safe(path)
    if df_dev is not None:
        # 获取不带路径和后缀的文件名作为列前缀
        dev_name = os.path.basename(path).replace('.xlsx', '')
        # 逻辑清洗
        df_dev[f'{dev_name}_freq_cleaned'] = np.where(df_dev['run_stop'] == 1, df_dev['feedback_frequency'], 0)
        # 异步对齐
        df_main = pd.merge_asof(df_main, df_dev[['date_time', f'{dev_name}_freq_cleaned']],
                                on='date_time', direction='nearest')

# 循环处理冷机
for path in CHILLER_FILES:
    df_lxj = load_excel_safe(path)
    if df_lxj is not None:
        lxj_name = os.path.basename(path).replace('.xlsx', '')
        # 重命名列防止冲突
        df_lxj = df_lxj.rename(columns={col: f"{lxj_name}_{col}" for col in df_lxj.columns if col != 'date_time'})
        df_main = pd.merge_asof(df_main, df_lxj, on='date_time', direction='nearest')

print("--- 正在清洗数据类型并处理异常值 ---")

# 定义需要参与物理计算的所有列
cols_to_fix = ['current_flow', 'return_temp', 'supply_temp', 'power_consume']

for col in cols_to_fix:
    if col in df_main.columns:
        # errors='coerce' 的作用：如果这一行是文字或乱码转不成数字，就强制把它变成 NaN
        df_main[col] = pd.to_numeric(df_main[col], errors='coerce')

# 顺手填补一下转换产生的空值（用前一个有效值填充），防止计算出 NaN 导致后续模型训练报错
df_main[cols_to_fix] = df_main[cols_to_fix].fillna(method='ffill').fillna(0)
# --- 3. 核心计算 ---
print("--- 正在计算总功率与系统 COP ---")
# 功率 (kW)
df_main['total_power_kw'] = df_main['power_consume'] * 12
# 制冷量 (kW) - 假设字段名: current_flow, return_temp, supply_temp
df_main['calc_Q_kw'] = (df_main['current_flow'] * 4.186 * (df_main['return_temp'] - df_main['supply_temp'])) / 3.6
# COP
df_main['system_cop'] = df_main.apply(
    lambda x: x['calc_Q_kw'] / x['total_power_kw'] if x['total_power_kw'] > 0 else 0, axis=1
)

# --- 4. 保存 ---
output_name = "data.xlsx"
df_main.to_excel(output_name, index=False)
print(f"✅ 完成！文件已保存为: {output_name}")