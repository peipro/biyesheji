"""
v6_flow_reconstructed.py
========================
仅替换数据集中 current_flow 列为流量表 total_flow 的每分钟差分值。
不修改任何已有工程文件。
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs("../pic/v6_analysis", exist_ok=True)

# ==========================
# 第一步：从流量表计算每分钟总流量
# ==========================
print("=" * 60)
print("第一步：从流量表 total_flow 计算每分钟总流量")
print("=" * 60)

df_flow = pd.read_excel("../05data/流量表.xlsx")
for col in ['current_flow', 'total_flow']:
    df_flow[col] = df_flow[col].astype(str).str.replace(',', '', regex=False)
    df_flow[col] = pd.to_numeric(df_flow[col], errors='coerce').fillna(0)
df_flow['date_time'] = pd.to_datetime(df_flow['date_time']).dt.floor('min')
df_flow = df_flow.sort_values(['device_name', 'date_time'])

# 排除冷却塔（不测水流量）
df_flow = df_flow[~df_flow['device_name'].str.contains('冷却塔', na=False)]

# 按设备计算 total_flow 差分
flow_parts = []
for dev in sorted(df_flow['device_name'].unique()):
    sub = df_flow[df_flow['device_name'] == dev].copy()
    sub['dt'] = sub['date_time'].diff().dt.total_seconds() / 60.0
    sub['dflow'] = sub['total_flow'].diff()
    # 跳过复位（total_flow减少或间隔>60min）
    reset = (sub['dflow'] < 0) | (sub['dt'] > 60)
    sub['dflow_clean'] = sub['dflow'].where(~reset)
    sub['flow_rate'] = np.where(sub['dt'] > 0, sub['dflow_clean'] / sub['dt'] * 60, 0).clip(0, 5000)
    flow_parts.append(sub[['date_time', 'flow_rate']])

# 每分钟所有设备流量求和
flow_minute = pd.concat(flow_parts).groupby('date_time')['flow_rate'].sum().reset_index()
flow_minute = flow_minute.rename(columns={'flow_rate': 'new_flow'})
print(f"  流量表总数据: {len(df_flow)} 行")
print(f"  有效分钟数: {len(flow_minute)}")
print(f"  流量 P50={flow_minute['new_flow'].median():.1f}  P99={flow_minute['new_flow'].quantile(0.99):.1f} m³/h")

# ==========================
# 第二步：替换现有数据集中的 current_flow
# ==========================
print("\n" + "=" * 60)
print("第二步：替换数据集中的 current_flow")
print("=" * 60)

INPUT_FILE = "data_feature_engineered_v5.xlsx"
df = pd.read_excel(INPUT_FILE)
df['date_time'] = pd.to_datetime(df['date_time']).dt.floor('min')
print(f"  原始数据集: {len(df)} 行, {len(df.columns)} 列")

# merge_asof 对齐时间
df = pd.merge_asof(df.sort_values('date_time'), flow_minute.sort_values('date_time'),
                   on='date_time', direction='nearest', tolerance=pd.Timedelta('2min'))

# 替换 current_flow，保留原始列用于对比
df.rename(columns={'current_flow': 'current_flow_old'}, inplace=True)
df.rename(columns={'new_flow': 'current_flow'}, inplace=True)

# 丢掉未匹配到的行
before = len(df)
df = df.dropna(subset=['current_flow'])
print(f"  匹配到流量数据: {len(df)} 行 (丢弃 {before - len(df)} 行)")

# ==========================
# 第三步：重算 calc_Q_kw 和 system_cop
# ==========================
print("\n" + "=" * 60)
print("第三步：重算制冷量和 COP")
print("=" * 60)

# v5 数据集已有 temp_diff、total_power_kw，直接用新流量重算 calc_Q_kw 和 COP
df['calc_Q_kw'] = (df['current_flow'] * 4.186 * df['temp_diff']) / 3.6
df['calc_Q_kw'] = df['calc_Q_kw'].clip(lower=0)
df['system_cop'] = np.where(df['total_power_kw'] > 5,
                              df['calc_Q_kw'] / df['total_power_kw'], 0)

# 应用过滤
before = len(df)
df = df[
    (df['total_power_kw'] > 30) &
    (df['calc_Q_kw'] > 1.0) &
    (df['system_cop'] > 0.5) & (df['system_cop'] < 12.0)
].copy()
print(f"  过滤后: {len(df)} 行 (丢弃 {before - len(df)} 行)")

# 保存
OUTPUT = "data_deep_learning_final_v6.xlsx"
df.to_excel(OUTPUT, index=False)
print(f"  已保存: {OUTPUT}")

# ==========================
# 第四步：相关性分析
# ==========================
print("\n" + "=" * 60)
print("第四步：相关性分析")
print("=" * 60)

num_cols = [c for c in df.columns if c not in ['date_time', 'device_compose']
            and 'device_name' not in c and pd.api.types.is_numeric_dtype(df[c])]

results = []
for col in num_cols:
    if col == 'system_cop':
        continue
    s = df[col].dropna()
    if len(s) > 100 and s.nunique() > 2:
        results.append({'feature': col, 'pearson_r': df[col].corr(df['system_cop'])})

rank = pd.DataFrame(results)
rank['abs_r'] = rank['pearson_r'].abs()
rank = rank.sort_values('abs_r', ascending=False).reset_index(drop=True)
rank['rank'] = range(1, len(rank) + 1)

print(f"\n共 {len(rank)} 个有效特征")
print(f"\n{'排名':<6}{'特征':<42}{'Pearson r':<14}{'|r|':<10}")
print("-" * 72)
for _, row in rank.iterrows():
    print(f"{row['rank']:<6}{row['feature']:<42}{row['pearson_r']:<+14.4f}{row['abs_r']:<10.4f}")

cf = rank[rank['feature'] == 'current_flow']
if len(cf) > 0:
    print(f"\n{'=' * 60}")
    print(f"current_flow (流量表 total_flow 差分) 最终排名:")
    print(f"  排名: {cf.iloc[0]['rank']} / {len(rank)}")
    print(f"  Pearson r = {cf.iloc[0]['pearson_r']:.4f}")

print(f"\n{'=' * 60}")
print("关键物理关系验证:")
print(f"{'=' * 60}")
print(f"  current_flow vs calc_Q_kw:      r = {df['current_flow'].corr(df['calc_Q_kw']):.4f}")
print(f"  current_flow vs total_power_kw: r = {df['current_flow'].corr(df['total_power_kw']):.4f}")
print(f"  temp_diff vs calc_Q_kw:         r = {df['temp_diff'].corr(df['calc_Q_kw']):.4f}")
print(f"  temp_diff vs COP:               r = {df['temp_diff'].corr(df['system_cop']):.4f}")
print(f"  calc_Q_kw vs COP:               r = {df['calc_Q_kw'].corr(df['system_cop']):.4f}")
print(f"  total_power_kw vs COP:          r = {df['total_power_kw'].corr(df['system_cop']):.4f}")

# ==========================
# 第五步：图表
# ==========================
print(f"\n{'=' * 60}")
print("第五步：生成图表")
print(f"{'=' * 60}")

# 图1: 相关性排名
fig, ax = plt.subplots(figsize=(12, 14))
colors = ['crimson' if v < 0 else 'steelblue' for v in rank['pearson_r']]
ax.barh(range(len(rank)), rank['pearson_r'], color=colors, height=0.7)
ax.set_yticks(range(len(rank)))
ax.set_yticklabels(rank['feature'], fontsize=7)
ax.axvline(0, color='gray', linewidth=0.5)
cf_idx = rank[rank['feature'] == 'current_flow'].index[0]
ax.barh(cf_idx, rank.iloc[cf_idx]['pearson_r'], color='gold', height=0.8, edgecolor='black', linewidth=2)
ax.set_xlabel('Pearson 相关系数', fontsize=12)
ax.set_title('current_flow(流量表差分) 与 system_cop 相关性', fontsize=14)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../pic/v6_analysis/v6_correlation_bar.png', dpi=150, bbox_inches='tight')
print("  已保存: v6_correlation_bar.png")

# 图2: 散点矩阵
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, ycol, title in [
    (axes[0], 'calc_Q_kw', f'制冷量 (r={df["current_flow"].corr(df["calc_Q_kw"]):.3f})'),
    (axes[1], 'total_power_kw', f'总功率 (r={df["current_flow"].corr(df["total_power_kw"]):.3f})'),
    (axes[2], 'system_cop', f'COP (r={df["current_flow"].corr(df["system_cop"]):.3f})')
]:
    ax.scatter(df['current_flow'], df[ycol], s=3, alpha=0.3, c='steelblue')
    ax.set_xlabel('current_flow (m³/h)')
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.grid(alpha=0.3)
plt.suptitle('流量表差分流量 vs 关键变量', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('../pic/v6_analysis/v6_flow_scatter.png', dpi=150, bbox_inches='tight')
print("  已保存: v6_flow_scatter.png")

# 图3: 流量分布对比
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
try:
    df_old = pd.read_excel('data_deep_learning_final_v3.xlsx')
    axes[0].hist(df_old['current_flow'].dropna(), bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    axes[0].set_title(f'原始 current_flow (均值={df_old["current_flow"].mean():.0f})')
    axes[0].set_xlabel('current_flow (冷量表)')
    axes[0].grid(alpha=0.3)
except:
    pass

axes[1].hist(df['current_flow'], bins=50, color='coral', edgecolor='white', alpha=0.7)
axes[1].set_title(f'流量表差分 current_flow (均值={df["current_flow"].mean():.0f})')
axes[1].set_xlabel('current_flow (m³/h)')
axes[1].grid(alpha=0.3)
plt.suptitle('新旧流量分布对比', fontsize=14)
plt.tight_layout()
plt.savefig('../pic/v6_analysis/v6_flow_distribution.png', dpi=150, bbox_inches='tight')
print("  已保存: v6_flow_distribution.png")

# 图4: 物理解释
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc = axes[0].scatter(df['current_flow'], df['temp_diff'], c=df['system_cop'], cmap='RdYlGn_r', s=4, alpha=0.5)
axes[0].set_xlabel('current_flow (m³/h)')
axes[0].set_ylabel('temp_diff (°C)')
axes[0].set_title('流量 vs 温差 (颜色=COP)')
plt.colorbar(sc, ax=axes[0], label='COP')

sc = axes[1].scatter(df['temp_diff'], df['calc_Q_kw'], c=df['system_cop'], cmap='RdYlGn_r', s=4, alpha=0.5)
axes[1].set_xlabel('temp_diff (°C)')
axes[1].set_ylabel('calc_Q_kw (kW)')
axes[1].set_title(f'温差 vs 制冷量 (r={df["temp_diff"].corr(df["calc_Q_kw"]):.3f})')
plt.colorbar(sc, ax=axes[1], label='COP')
plt.suptitle('物理解释: 温差才是制冷量的主要驱动', fontsize=14)
plt.tight_layout()
plt.savefig('../pic/v6_analysis/v6_physics_explanation.png', dpi=150, bbox_inches='tight')
print("  已保存: v6_physics_explanation.png")

print(f"\n{'=' * 60}")
print("分析完成！")
print(f"{'=' * 60}")

if len(cf) > 0:
    print(f"\n结果摘要:")
    print(f"  current_flow 排名: {cf.iloc[0]['rank']}/{len(rank)} (之前垫底)")
    print(f"  flow vs COP:      r = {df['current_flow'].corr(df['system_cop']):.4f}")
    print(f"  flow vs calc_Q:   r = {df['current_flow'].corr(df['calc_Q_kw']):.4f}")
    print(f"  flow vs power:    r = {df['current_flow'].corr(df['total_power_kw']):.4f}")
    print(f"  temp_diff vs COP: r = {df['temp_diff'].corr(df['system_cop']):.4f}")
