import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_data():
    """
    加载所有Excel文件并返回数据字典
    """
    data = {}
    
    # 设备运行数据
    device_files = glob.glob('*冷冻泵.xlsx') + glob.glob('*冷却泵.xlsx') + glob.glob('*冷却塔.xlsx')
    for file in device_files:
        try:
            df = pd.read_excel(file)
            data[os.path.splitext(file)[0]] = df
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
    
    # 系统级数据
    system_files = ['A冷冻.xlsx', 'A冷却.xlsx', 'B冷冻.xlsx', 'B冷却.xlsx']
    for file in system_files:
        if os.path.exists(file):
            try:
                df = pd.read_excel(file)
                data[os.path.splitext(file)[0]] = df
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
    
    # 冷箱数据
    lxj_files = glob.glob('lxj*.xlsx')
    for file in lxj_files:
        try:
            df = pd.read_excel(file)
            data[os.path.splitext(file)[0]] = df
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
    
    # 能量监测数据
    energy_files = [
        '冷量表.xlsx', '分钟耗电量.xlsx', '小时耗电量.xlsx', 
        '天耗电量.xlsx', '功率-制冷站房.xlsx', '电流-制冷站房.xlsx',
        '电学量值.xlsx'
    ]
    for file in energy_files:
        if os.path.exists(file):
            try:
                df = pd.read_excel(file)
                data[os.path.splitext(file)[0]] = df
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
    
    # 环境和流量数据
    other_files = ['温湿度.xlsx', '流量表.xlsx']
    for file in other_files:
        if os.path.exists(file):
            try:
                df = pd.read_excel(file)
                data[os.path.splitext(file)[0]] = df
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
    
    return data

def analyze_data_quality(data_dict):
    """
    分析数据质量：缺失值、重复值、数据类型
    """
    quality_report = {}
    
    for name, df in data_dict.items():
        report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_columns': list(df.select_dtypes(include=['number']).columns)
        }
        quality_report[name] = report
    
    return quality_report

def generate_quality_report():
    """
    生成数据质量报告
    """
    print("开始加载数据...")
    data = load_all_data()
    print(f"成功加载 {len(data)} 个文件")
    
    print("\n开始分析数据质量...")
    quality_report = analyze_data_quality(data)
    
    print("\n=== 数据质量报告 ===")
    for name, report in quality_report.items():
        print(f"\n📊 文件: {name}")
        print(f"数据形状: {report['shape'][0]}行 × {report['shape'][1]}列")
        print(f"缺失值: {report['missing_values']}")
        print(f"重复行: {report['duplicate_rows']}")
        print(f"数值列: {len(report['numeric_columns'])}个")
        if report['numeric_columns']:
            print(f"数值列名称: {', '.join(report['numeric_columns'])}")
    
    return data, quality_report

if __name__ == "__main__":
    data, quality_report = generate_quality_report()