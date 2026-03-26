import pandas as pd
import os
import glob

def analyze_excel_files():
    # 获取所有Excel文件
    excel_files = glob.glob('*.xlsx')
    print(f"找到 {len(excel_files)} 个Excel文件\n")
    
    for file_name in excel_files:
        print(f"=== 分析文件: {file_name} ===\n")
        
        try:
            # 读取Excel文件
            df = pd.read_excel(file_name)
            
            # 打印基本信息
            print(f"数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
            print(f"列名列表: {list(df.columns)}\n")
            
            # 打印前5行数据
            print("前5行数据:")
            print(df.head())
            print()
            
            # 检查数值列的统计信息
            numeric_columns = df.select_dtypes(include=['number']).columns
            if len(numeric_columns) > 0:
                print("数值列的统计信息:")
                print(df[numeric_columns].describe())
            else:
                print("无数值列")
            print()
            
            print("-" * 80)
            print()
            
        except Exception as e:
            print(f"读取文件时出错: {e}")
            print("-" * 80)
            print()

if __name__ == "__main__":
    analyze_excel_files()