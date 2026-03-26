import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("data_plus_features.xlsx")

# 1. 检查 COP 的统计分布
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['system_cop'], kde=True)
plt.title(f"COP Distribution (Total: {len(df)})")

# 2. 检查 COP 随时间的变化（看有没有突变点）
plt.subplot(1, 2, 2)
plt.plot(df['system_cop'].values)
plt.axhline(y=1.5, color='r', linestyle='--') # 理论下限
plt.axhline(y=7.0, color='r', linestyle='--') # 理论上限
plt.title("COP Time Series")
plt.show()

# 打印极值
print(f"最大值: {df['system_cop'].max()}, 最小值: {df['system_cop'].min()}")