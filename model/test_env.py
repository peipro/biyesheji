import sys
print("Python executable:", sys.executable)

# 检查pandas
print("\nChecking pandas...")
try:
    import pandas as pd
    print("pandas version:", pd.__version__)
except ImportError as e:
    print("pandas not available:", e)

# 检查openpyxl
print("\nChecking openpyxl...")
try:
    import openpyxl
    print("openpyxl available")
except ImportError as e:
    print("openpyxl not available:", e)

# 尝试读取Excel
print("\nTrying to read Excel...")
try:
    df = pd.read_excel('c:\\Users\\25771\\Desktop\\毕设\\05data\\A1冷冻泵.xlsx')
    print("Excel read successfully")
    print("Columns:", df.columns.tolist())
except Exception as e:
    print("Error reading Excel:", e)