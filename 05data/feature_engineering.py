import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def load_all_data():
    """加载所有数据（简化版）"""
    import glob
    data = {}
    
    device_files = glob.glob('*冷冻泵.xlsx') + glob.glob('*冷却泵.xlsx') + glob.glob('*冷却塔.xlsx')
    for file in device_files:
        try:
            df = pd.read_excel(file)
            data[os.path.splitext(file)[0]] = df
        except:
            pass
    
    system_files = ['A冷冻.xlsx', 'A冷却.xlsx', 'B冷冻.xlsx', 'B冷却.xlsx']
    for file in system_files:
        try:
            df = pd.read_excel(file)
            data[os.path.splitext(file)[0]] = df
        except:
            pass
    
    lxj_files = glob.glob('lxj*.xlsx')
    for file in lxj_files:
        try:
            df = pd.read_excel(file)
            data[os.path.splitext(file)[0]] = df
        except:
            pass
    
    energy_files = ['冷量表.xlsx', '功率-制冷站房.xlsx', '电流-制冷站房.xlsx', '电学量值.xlsx']
    for file in energy_files:
        try:
            df = pd.read_excel(file)
            data[os.path.splitext(file)[0]] = df
        except:
            pass
    
    other_files = ['温湿度.xlsx', '流量表.xlsx']
    for file in other_files:
        try:
            df = pd.read_excel(file)
            data[os.path.splitext(file)[0]] = df
        except:
            pass
    
    return data

def preprocess_device_data(df, file_name):
    """预处理设备数据"""
    try:
        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time'])
            df.set_index('date_time', inplace=True)
        
        # 标准化设备名称
        if 'device_name' in df.columns:
            df['device_name'] = df['device_name'].str.strip()
        
        return df
    except Exception as e:
        print(f"预处理文件 {file_name} 时出错: {e}")
        return None

def create_temporal_features(df):
    """创建时间相关特征"""
    features = pd.DataFrame(index=df.index)
    
    features['hour'] = df.index.hour
    features['minute'] = df.index.minute
    features['second'] = df.index.second
    features['day'] = df.index.day
    features['weekday'] = df.index.weekday
    features['weekend'] = (df.index.weekday >= 5).astype(int)
    features['month'] = df.index.month
    
    return features

def create_cyclic_features(df):
    """创建周期性特征（用于处理时间的周期性）"""
    features = pd.DataFrame(index=df.index)
    
    # 小时周期性
    features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # 日间周期性
    features['minute_sin'] = np.sin(2 * np.pi * df.index.minute / 60)
    features['minute_cos'] = np.cos(2 * np.pi * df.index.minute / 60)
    
    return features

def compute_statistical_features(df, window_sizes=[5, 10, 20, 60]):
    """计算统计特征"""
    features = pd.DataFrame(index=df.index)
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            for window in window_sizes:
                features[f'{column}_mean_{window}'] = df[column].rolling(window=window).mean()
                features[f'{column}_std_{window}'] = df[column].rolling(window=window).std()
                features[f'{column}_min_{window}'] = df[column].rolling(window=window).min()
                features[f'{column}_max_{window}'] = df[column].rolling(window=window).max()
    
    return features

def create_derived_features(df):
    """创建衍生特征"""
    features = pd.DataFrame(index=df.index)
    
    # 温度差特征
    if 'supply_temp' in df.columns and 'return_temp' in df.columns:
        features['temp_diff'] = df['supply_temp'] - df['return_temp']
    
    # 压力差特征
    if 'supply_pressure' in df.columns and 'return_pressure' in df.columns:
        features['pressure_diff'] = df['supply_pressure'] - df['return_pressure']
    
    # 频率差特征
    if 'feedback_frequency' in df.columns and 'set_frequency' in df.columns:
        features['frequency_diff'] = df['feedback_frequency'] - df['set_frequency']
    
    # 功率相关特征
    if 'a_current' in df.columns and 'b_vol' in df.columns:
        features['estimated_power'] = df['a_current'] * df['b_vol']
    
    return features

def aggregate_device_data(data_dict):
    """聚合设备数据"""
    device_features = {}
    
    for name, df in data_dict.items():
        if any(keyword in name.lower() for keyword in ['冷冻泵', '冷却泵', '冷却塔', '冷冻', '冷却']):
            df_processed = preprocess_device_data(df, name)
            if df_processed is not None and 'device_name' in df_processed.columns:
                df_processed['feature_source'] = name
                device_features[name] = df_processed
    
    return device_features

def merge_related_datasets(data_dict):
    """合并相关数据集"""
    merged_data = {}
    
    # 设备运行数据合并
    pump_data = [data_dict[name] for name in data_dict if '泵' in name]
    tower_data = [data_dict[name] for name in data_dict if '塔' in name]
    lxj_data = [data_dict[name] for name in data_dict if 'lxj' in name.lower()]
    
    if pump_data:
        pump_df = pd.concat(pump_data, ignore_index=True)
        merged_data['pumps'] = pump_df
    
    if tower_data:
        tower_df = pd.concat(tower_data, ignore_index=True)
        merged_data['towers'] = tower_df
    
    if lxj_data:
        lxj_df = pd.concat(lxj_data, ignore_index=True)
        merged_data['lxj'] = lxj_df
    
    return merged_data

def standardize_features(df):
    """标准化特征"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # 处理缺失值
    imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    # 标准化
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df

def main():
    print("开始数据处理和特征工程...")
    
    # 1. 加载数据
    data = load_all_data()
    print(f"加载了 {len(data)} 个文件")
    
    # 2. 设备数据预处理和聚合
    device_features = aggregate_device_data(data)
    print(f"预处理了 {len(device_features)} 个设备特征数据集")
    
    # 3. 合并相关数据集
    merged_data = merge_related_datasets(data)
    print(f"合并了 {len(merged_data)} 个相关数据集")
    
    # 4. 保存预处理后的数据
    processed_data_dir = 'processed_data'
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    
    for name, df in merged_data.items():
        output_path = os.path.join(processed_data_dir, f'{name}_processed.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"保存了 {name} 的预处理数据到 {output_path}")
    
    print("\n特征工程完成！")
    print(f"预处理后的数据保存在: {os.path.abspath(processed_data_dir)}")
    
    return merged_data

if __name__ == "__main__":
    merged_data = main()