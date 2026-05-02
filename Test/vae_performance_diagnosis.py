import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from scipy.stats import skew, kurtosis

# --- 设置 ---
plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs("../pic/vae_diagnosis", exist_ok=True)

parser = argparse.ArgumentParser(description='VAE性能诊断')
parser.add_argument('--input', type=str, default='data_feature_engineered_v5.xlsx',
                    help='输入数据文件路径')
parser.add_argument('--model', type=str, default='vae_best.pth',
                    help='模型权重文件')
parser.add_argument('--scaler', type=str, default='vae_scaler.pkl',
                    help='Scaler文件')
args = parser.parse_args()

# --- VAE模型定义（与训练一致） ---
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super(VAE, self).__init__()
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

# --- 加载数据 ---
df = pd.read_excel(args.input)
print(f"数据加载: {len(df)} 行")

feature_cols = [
    'temperature', 'humidity', 'temp_diff',
    'lxj_evap_press_avg', 'lxj_cond_press_avg',
    'A4冷冻泵_f', 'A1冷却泵_f', 'A4冷却塔_f',
    'system_cop'
]
cop_idx = 8  # system_cop 在 feature_cols 中的位置

data_raw = df[feature_cols].values.astype(np.float32)

# --- 加载 scaler 和模型 ---
with open(args.scaler, "rb") as f:
    scaler = pickle.load(f)

data_norm = scaler.transform(data_raw).astype(np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(9, 8, [64, 32, 16]).to(device)
model.load_state_dict(torch.load(args.model, map_location=device))
model.eval()
print(f"模型已加载: {args.model}")

# --- 全样本重构 ---
X_tensor = torch.FloatTensor(data_norm)
batch_size = 256
all_recon = []
all_mu = []
all_log_var = []
with torch.no_grad():
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i + batch_size].to(device)
        recon, mu, log_var = model(batch)
        all_recon.append(recon.cpu().numpy())
        all_mu.append(mu.cpu().numpy())
        all_log_var.append(log_var.cpu().numpy())

recon_norm = np.concatenate(all_recon, axis=0)
print("全样本重构完成")

# --- 重构误差计算 ---
# 逐样本MSE: 每个样本在所有11维上的平均误差
sample_errors = np.mean((data_norm - recon_norm) ** 2, axis=1)
# 逐特征MSE: 每个特征在所有样本上的平均误差
feature_errors = np.mean((data_norm - recon_norm) ** 2, axis=0)

# 逆标准化求原始尺度的重构值（用于COP对比）
recon_raw = scaler.inverse_transform(recon_norm)
cop_true = data_raw[:, cop_idx]
cop_recon = recon_raw[:, cop_idx]

# 逐特征 MSE（原始尺度）
feature_errors_raw = np.mean((data_raw - recon_raw) ** 2, axis=0)

print(f"\n{'='*60}")
print(f"          VAE性能诊断报告")
print(f"{'='*60}")

# --- 误差统计 ---
print(f"\n【重构误差统计】")
print(f"  样本数: {len(sample_errors)}")
print(f"  均值: {sample_errors.mean():.6f}")
print(f"  标准差: {sample_errors.std():.6f}")
print(f"  偏度: {skew(sample_errors):.4f}")
print(f"  峰度: {kurtosis(sample_errors):.4f}")
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
p_values = np.percentile(sample_errors, percentiles)
for p, v in zip(percentiles, p_values):
    print(f"  {p:2d}% 分位数: {v:.6f}")

# --- 三种阈值 ---
threshold_95 = np.percentile(sample_errors, 95)
threshold_99 = np.percentile(sample_errors, 99)
threshold_3sigma = sample_errors.mean() + 3 * sample_errors.std()

print(f"\n【三种阈值】")
print(f"  95% 分位阈值: {threshold_95:.6f}")
print(f"  99% 分位阈值: {threshold_99:.6f}")
print(f"  均值+3σ 阈值: {threshold_3sigma:.6f}")

# --- 异常检测结果 ---
anomaly_95 = sample_errors > threshold_95
anomaly_99 = sample_errors > threshold_99
anomaly_3s = sample_errors > threshold_3sigma

print(f"\n【异常检测结果】")
print(f"  95%分位: {anomaly_95.sum()} ({anomaly_95.sum()/len(sample_errors)*100:.2f}%) | 异常样本平均COP: {cop_true[anomaly_95].mean():.4f}")
print(f"  99%分位: {anomaly_99.sum()} ({anomaly_99.sum()/len(sample_errors)*100:.2f}%) | 异常样本平均COP: {cop_true[anomaly_99].mean():.4f}")
print(f"  均值+3σ: {anomaly_3s.sum()} ({anomaly_3s.sum()/len(sample_errors)*100:.2f}%) | 异常样本平均COP: {cop_true[anomaly_3s].mean():.4f}")

# --- COP-误差关联 ---
from scipy.stats import pearsonr
corr, p_corr = pearsonr(cop_true, sample_errors)
print(f"\n【COP-误差关联】")
print(f"  Pearson相关系数: {corr:.4f} (p-value: {p_corr:.2e})")
print(f"  最低5% COP样本被异常标记(95%阈值)比例: {anomaly_95[cop_true <= np.percentile(cop_true, 5)].mean()*100:.1f}%")

# --- 异常时段分析（使用95%阈值） ---
anomaly_mask = anomaly_95.astype(int)
if 'date_time' in df.columns:
    dates = pd.to_datetime(df['date_time'])
else:
    dates = pd.date_range(start='2024-01-01', periods=len(df), freq='T')

# 找出连续异常时段
transitions = np.diff(np.concatenate([[0], anomaly_mask]))
starts = np.where(transitions == 1)[0]
ends = np.where(transitions == -1)[0]

segments = []
for s, e in zip(starts, ends):
    segments.append((s, e, e - s))

# 按持续时长降序排列
segments.sort(key=lambda x: x[2], reverse=True)
n_segments = len(segments)

print(f"\n【异常时段统计 (95%阈值)】")
print(f"  异常时段总数: {n_segments}")
if n_segments > 0:
    avg_duration = np.mean([s[2] for s in segments])
    max_duration = segments[0][2]
    print(f"  平均持续时长: {avg_duration:.1f} 分钟")
    print(f"  最长持续时长: {max_duration} 分钟")

    print(f"\n  Top-3 最长异常时段:")
    for i, (s, e, dur) in enumerate(segments[:3]):
        seg_cop = cop_true[s:e+1]
        print(f"    时段{i+1}: {dates[s]} ~ {dates[e]} ({dur}分钟)")
        print(f"      平均COP: {seg_cop.mean():.4f}, 最小COP: {seg_cop.min():.4f}")
        print(f"      平均重构误差: {sample_errors[s:e+1].mean():.6f}")

# --- 正常 vs 异常特征对比 ---
normal_mask = ~anomaly_95
print(f"\n【正常 vs 异常特征对比 (95%阈值)】")
print(f"  {'特征':<25} {'正常均值':<12} {'异常均值':<12} {'偏差(%)':<12}")
print(f"  {'-'*61}")
for i, col in enumerate(feature_cols):
    normal_mean = data_raw[normal_mask, i].mean()
    anomaly_mean = data_raw[anomaly_mask.astype(bool), i].mean()
    dev = (anomaly_mean - normal_mean) / (normal_mean + 1e-8) * 100
    print(f"  {col:<25} {normal_mean:<12.4f} {anomaly_mean:<12.4f} {dev:<+12.2f}")

print(f"\n{'='*60}")
print(f"          图表输出")
print(f"{'='*60}")

# ============ 图2: 重构误差分布 ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 直方图
ax = axes[0]
ax.hist(sample_errors, bins=80, color='steelblue', edgecolor='white', alpha=0.7, density=True)
ax.axvline(threshold_95, color='red', linestyle='--', linewidth=2, label=f'95%分位 ({threshold_95:.4f})')
ax.axvline(threshold_99, color='darkred', linestyle='--', linewidth=2, label=f'99%分位 ({threshold_99:.4f})')
ax.axvline(threshold_3sigma, color='orange', linestyle='--', linewidth=2, label=f'均值+3σ ({threshold_3sigma:.4f})')
ax.set_xlabel('重构误差 (MSE)', fontsize=12)
ax.set_ylabel('密度', fontsize=12)
ax.set_title('重构误差分布直方图', fontsize=14)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# CDF
ax = axes[1]
sorted_errors = np.sort(sample_errors)
cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
ax.plot(sorted_errors, cdf, color='steelblue', linewidth=1.5)
ax.axhline(0.95, color='red', linestyle='--', linewidth=1.5, label='95%')
ax.axhline(0.99, color='darkred', linestyle='--', linewidth=1.5, label='99%')
ax.set_xlabel('重构误差 (MSE)', fontsize=12)
ax.set_ylabel('累积概率', fontsize=12)
ax.set_title('重构误差累积分布 (CDF)', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../pic/vae_diagnosis/vae_error_distribution.png", dpi=150, bbox_inches='tight')
plt.show()

# ============ 图3: 全时段重构误差时间序列 ============
fig, ax = plt.subplots(figsize=(16, 5))
colors = np.where(anomaly_mask, 'red', 'steelblue')
ax.scatter(range(len(sample_errors)), sample_errors, c=colors, s=2, alpha=0.5)
ax.axhline(threshold_95, color='red', linestyle='--', linewidth=1.5, label=f'95%阈值 ({threshold_95:.4f})')
ax.axhline(threshold_99, color='darkred', linestyle='--', linewidth=1.5, label=f'99%阈值 ({threshold_99:.4f})')
ax.set_xlabel('时间 (分钟)', fontsize=12)
ax.set_ylabel('重构误差 (MSE)', fontsize=12)
ax.set_title('全时段重构误差时间序列', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../pic/vae_diagnosis/vae_error_timeseries.png", dpi=150, bbox_inches='tight')
plt.show()

# ============ 图4: 前2000分钟放大 ============
zoom_end = min(2000, len(sample_errors))
fig, ax = plt.subplots(figsize=(16, 5))
colors_zoom = np.where(anomaly_mask[:zoom_end], 'red', 'steelblue')
ax.scatter(range(zoom_end), sample_errors[:zoom_end], c=colors_zoom, s=5, alpha=0.6)
ax.axhline(threshold_95, color='red', linestyle='--', linewidth=1.5, label=f'95%阈值 ({threshold_95:.4f})')
ax.axhline(threshold_99, color='darkred', linestyle='--', linewidth=1.5, label=f'99%阈值 ({threshold_99:.4f})')
ax.set_xlabel('时间 (分钟)', fontsize=12)
ax.set_ylabel('重构误差 (MSE)', fontsize=12)
ax.set_title(f'前{zoom_end}分钟重构误差时间序列', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../pic/vae_diagnosis/vae_error_timeseries_zoom.png", dpi=150, bbox_inches='tight')
plt.show()

# ============ 图5: COP vs 重构误差散点图 ============
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(cop_true, sample_errors, c=sample_errors, cmap='RdYlGn_r', s=5, alpha=0.5)
ax.axhline(threshold_95, color='red', linestyle='--', linewidth=1.5, label=f'95%阈值')
ax.set_xlabel('真实 COP', fontsize=12)
ax.set_ylabel('重构误差 (MSE)', fontsize=12)
ax.set_title(f'COP vs 重构误差 (Pearson r={corr:.4f})', fontsize=14)
plt.colorbar(sc, ax=ax, label='重构误差')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../pic/vae_diagnosis/vae_cop_vs_error.png", dpi=150, bbox_inches='tight')
plt.show()

# ============ 图6: 逐特征平均重构误差 ============
fig, ax = plt.subplots(figsize=(12, 6))
colors_bar = ['crimson' if i == cop_idx else 'steelblue' for i in range(len(feature_cols))]
bars = ax.barh(range(len(feature_cols)), feature_errors_raw, color=colors_bar, edgecolor='white')
ax.set_yticks(range(len(feature_cols)))
ax.set_yticklabels(feature_cols, fontsize=10)
ax.set_xlabel('平均重构误差 (MSE, 原始尺度)', fontsize=12)
ax.set_title('逐特征平均重构误差', fontsize=14)
ax.grid(axis='x', alpha=0.3)

# 在条上标注数值
for bar, val in zip(bars, feature_errors_raw):
    ax.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig("../pic/vae_diagnosis/vae_feature_errors.png", dpi=150, bbox_inches='tight')
plt.show()

# ============ 图7: 正常 vs 异常特征分布箱线图 ============
n_feat = len(feature_cols)
n_cols = 4
n_rows = (n_feat + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
axes = axes.flatten()

for i in range(n_feat):
    ax = axes[i]
    data_n = data_raw[normal_mask, i]
    data_a = data_raw[anomaly_mask.astype(bool), i]
    bp = ax.boxplot([data_n, data_a], labels=['正常', '异常'], widths=0.5,
                    patch_artist=True,
                    boxprops=dict(linewidth=1.2),
                    medianprops=dict(color='red', linewidth=1.5))
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('crimson')
    ax.set_title(feature_cols[i], fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='y', alpha=0.3)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('正常 vs 异常样本特征分布对比', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("../pic/vae_diagnosis/vae_anomaly_feature_dist.png", dpi=150, bbox_inches='tight')
plt.show()

# ============ 图8: 真实COP vs VAE重构COP ============
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(cop_true, label='真实 COP', color='royalblue', linewidth=1.0, alpha=0.8)
ax.plot(cop_recon, label='VAE重构 COP', color='darkorange', linewidth=0.8, alpha=0.7)
# 标记异常区域
ymin, ymax = ax.get_ylim()
anomaly_regions = np.where(anomaly_mask.astype(bool))[0]
if len(anomaly_regions) > 0:
    # 找到连续区域并填充
    region_breaks = np.where(np.diff(anomaly_regions) > 1)[0]
    region_starts = np.concatenate([[anomaly_regions[0]], anomaly_regions[region_breaks + 1]])
    region_ends = np.concatenate([anomaly_regions[region_breaks], [anomaly_regions[-1]]])
    for rs, re in zip(region_starts, region_ends):
        if re - rs >= 5:  # 只标记持续5个样本以上的异常区域
            ax.axvspan(rs, re, alpha=0.15, color='red')

ax.set_xlabel('时间 (分钟)', fontsize=12)
ax.set_ylabel('system_cop', fontsize=12)
ax.set_title('真实 COP vs VAE重构 COP (红色阴影=异常区域)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../pic/vae_diagnosis/vae_cop_recon_timeseries.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n所有图表已保存到 ../pic/vae_diagnosis/")
print("\nVAE性能诊断完成!")
