"""
双头VAE (Two-Head VAE) 诊断脚本
诊断基于COP预测误差，而非总重构误差
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from scipy.stats import skew, kurtosis, pearsonr

plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs("../pic/vae_dualhead_diagnosis", exist_ok=True)

parser = argparse.ArgumentParser(description='双头VAE性能诊断')
parser.add_argument('--input', type=str, default='data_feature_engineered_v5_kalman_v8.xlsx',
                    help='输入数据文件路径')
parser.add_argument('--model', type=str, default='vae_dualhead_best.pth',
                    help='模型权重文件')
parser.add_argument('--scaler', type=str, default='vae_dualhead_scaler.pkl',
                    help='Scaler文件')
args = parser.parse_args()

# --- 双头VAE模型定义 ---
class TwoHeadVAE(nn.Module):
    def __init__(self, feature_dim, latent_dim, hidden_dims):
        super(TwoHeadVAE, self).__init__()
        encoder_layers = []
        prev_dim = feature_dim
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

        feat_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            feat_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        feat_layers.append(nn.Linear(hidden_dims[0], feature_dim))
        self.decoder_feat = nn.Sequential(*feat_layers)

        self.decoder_cop = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        feat_recon = self.decoder_feat(z)
        cop_recon = self.decoder_cop(z)
        return feat_recon, cop_recon, mu, log_var

# --- 加载数据 ---
df = pd.read_excel(args.input)
print(f"数据加载: {len(df)} 行")

feature_cols = [
    'temperature', 'humidity', 'temp_diff',
    'lxj_evap_press_avg', 'lxj_cond_press_avg',
    'A4冷冻泵_f', 'A1冷却泵_f', 'A4冷却塔_f'
]
target_col = 'system_cop'

X_raw = df[feature_cols].values.astype(np.float32)
y_raw = df[[target_col]].values.astype(np.float32)

# --- 加载 scaler 和模型 ---
with open(args.scaler, "rb") as f:
    scaler = pickle.load(f)

# StandardScaler是在9维上拟合的（8特征+COP），需要分别处理
data_full = np.concatenate([X_raw, y_raw], axis=1)
data_norm = scaler.transform(data_full).astype(np.float32)
X_norm = data_norm[:, :8]
y_norm = data_norm[:, 8:9]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoHeadVAE(8, 8, [64, 32, 16]).to(device)
model.load_state_dict(torch.load(args.model, map_location=device))
model.eval()
print(f"模型已加载: {args.model}")

# --- 全样本预测 ---
X_tensor = torch.FloatTensor(X_norm)
batch_size = 256
all_feat_recon = []
all_cop_recon = []
with torch.no_grad():
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i + batch_size].to(device)
        feat_recon, cop_recon, _, _ = model(batch)
        all_feat_recon.append(feat_recon.cpu().numpy())
        all_cop_recon.append(cop_recon.cpu().numpy())

feat_recon_norm = np.concatenate(all_feat_recon, axis=0)
cop_recon_norm = np.concatenate(all_cop_recon, axis=0)

# --- 计算COP预测误差（诊断核心指标） ---
cop_pred_errors = (y_norm.flatten() - cop_recon_norm.flatten()) ** 2
cop_residuals = (y_norm.flatten() - cop_recon_norm.flatten())  # 有符号残差
feat_recon_errors = np.mean((X_norm - feat_recon_norm) ** 2, axis=1)

# 逆标准化还原到原始尺度
recon_full = np.concatenate([feat_recon_norm, cop_recon_norm], axis=1)
recon_raw = scaler.inverse_transform(recon_full)
cop_true = y_raw.flatten()
cop_pred = recon_raw[:, 8]

print(f"\n{'='*60}")
print(f"          双头VAE性能诊断报告")
print(f"{'='*60}")

# --- COP预测误差统计 ---
print(f"\n【COP预测误差统计】")
print(f"  样本数: {len(cop_pred_errors)}")
print(f"  均值: {cop_pred_errors.mean():.6f}")
print(f"  标准差: {cop_pred_errors.std():.6f}")
print(f"  偏度: {skew(cop_pred_errors):.4f}")
print(f"  峰度: {kurtosis(cop_pred_errors):.4f}")
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
p_values = np.percentile(cop_pred_errors, percentiles)
for p, v in zip(percentiles, p_values):
    print(f"  {p:2d}% 分位数: {v:.6f}")

print(f"\n【COP残差（有符号）统计】")
print(f"  均值: {cop_residuals.mean():.6f}")
print(f"  标准差: {cop_residuals.std():.6f}")
print(f"  COP预测 vs 真实:")
print(f"    真实COP均值: {cop_true.mean():.4f}")
print(f"    预测COP均值: {cop_pred.mean():.4f}")
print(f"  COP预测RMSE: {np.sqrt(np.mean(cop_pred_errors)):.4f} (归一化)")

# --- 阈值设定 ---
# 总误差阈值（双向）
threshold_95 = np.percentile(cop_pred_errors, 95)
threshold_99 = np.percentile(cop_pred_errors, 99)
threshold_3sigma = cop_pred_errors.mean() + 3 * cop_pred_errors.std()

# 退化检测阈值（只关注负残差：实际COP < 预测COP）
res_5th = np.percentile(cop_residuals, 5)   # 5%分位：95%精度
res_1st = np.percentile(cop_residuals, 1)   # 1%分位：99%精度
res_mad = np.median(np.abs(cop_residuals - np.median(cop_residuals)))
res_threshold_z = -3 * 1.4826 * res_mad    # 改进Z-score

print(f"\n【阈值设定】")
print(f"  总误差(双向) 95%分位阈值: {threshold_95:.6f}")
print(f"  总误差(双向) 99%分位阈值: {threshold_99:.6f}")
print(f"  === 退化检测（负残差阈值）===")
print(f"  残差5%分位阈值: {res_5th:.4f} (预期精度95%)")
print(f"  残差1%分位阈值: {res_1st:.4f} (预期精度99%)")
print(f"  改进Z-score阈值: {res_threshold_z:.4f} (MAD={res_mad:.4f})")

# --- 异常检测结果 ---
anomaly_95 = cop_pred_errors > threshold_95
anomaly_99 = cop_pred_errors > threshold_99
anomaly_3s = cop_pred_errors > threshold_3sigma

# 退化检测：残差 < 阈值 = 实际COP低于预期
degradation_95 = cop_residuals < res_5th
degradation_99 = cop_residuals < res_1st
degradation_z = cop_residuals < res_threshold_z

print(f"\n【异常检测结果（总误差双向）】")
print(f"  95%分位: {anomaly_95.sum()} ({anomaly_95.sum()/len(cop_pred_errors)*100:.2f}%) "
      f"| 样本平均COP: {cop_true[anomaly_95].mean():.4f}")
print(f"  99%分位: {anomaly_99.sum()} ({anomaly_99.sum()/len(cop_pred_errors)*100:.2f}%) "
      f"| 样本平均COP: {cop_true[anomaly_99].mean():.4f}")

print(f"\n【退化检测结果（负残差=性能退化）】")
print(f"  残差5%分位: {degradation_95.sum()} ({degradation_95.sum()/len(cop_residuals)*100:.2f}%) "
      f"| 退化样本平均COP: {cop_true[degradation_95].mean():.4f}")
print(f"  残差1%分位: {degradation_99.sum()} ({degradation_99.sum()/len(cop_residuals)*100:.2f}%) "
      f"| 退化样本平均COP: {cop_true[degradation_99].mean():.4f}")
print(f"  改进Z-score: {degradation_z.sum()} ({degradation_z.sum()/len(cop_residuals)*100:.2f}%) "
      f"| 退化样本平均COP: {cop_true[degradation_z].mean():.4f}")

# --- COP-误差关联 ---
corr, p_corr = pearsonr(cop_true, cop_pred_errors)
print(f"\n【COP-预测误差关联】")
print(f"  Pearson相关系数(COP vs 绝对误差): {corr:.4f}")
low_5pct = cop_true <= np.percentile(cop_true, 5)
print(f"  最低5% COP样本被|双向|异常标记比例: {anomaly_95[low_5pct].mean()*100:.1f}%")
print(f"  最低5% COP样本被|退化|检测标记比例: {degradation_95[low_5pct].mean()*100:.1f}%")

corr_res, p_res = pearsonr(cop_true, cop_residuals)
print(f"  COP-残差相关系数: {corr_res:.4f} (正=高COP→正残差, 诊断逻辑正确)")

# --- 异常时段分析（使用退化检测：负残差） ---
anomaly_mask = degradation_95.astype(int)
if 'date_time' in df.columns:
    dates = pd.to_datetime(df['date_time'])
else:
    dates = pd.date_range(start='2024-01-01', periods=len(df), freq='T')

transitions = np.diff(np.concatenate([[0], anomaly_mask]))
starts = np.where(transitions == 1)[0]
ends = np.where(transitions == -1)[0]

segments = []
for s, e in zip(starts, ends):
    segments.append((s, e, e - s))
segments.sort(key=lambda x: x[2], reverse=True)
n_segments = len(segments)

print(f"\n【退化时段统计 (残差5%分位)】")
print(f"  异常时段总数: {n_segments}")
if n_segments > 0:
    avg_duration = np.mean([s[2] for s in segments])
    max_duration = segments[0][2]
    print(f"  平均持续时长: {avg_duration:.1f} 分钟")
    print(f"  最长持续时长: {max_duration} 分钟")
    print(f"\n  Top-3 最长异常时段:")
    for i, (s, e, dur) in enumerate(segments[:3]):
        seg_cop = cop_true[s:e+1]
        seg_res = cop_residuals[s:e+1]
        print(f"    时段{i+1}: {dates[s]} ~ {dates[e]} ({dur}分钟)")
        print(f"      平均COP: {seg_cop.mean():.4f}, 最小COP: {seg_cop.min():.4f}")
        print(f"      平均COP残差: {seg_res.mean():.4f} (负值=性能退化)")

# --- 正常 vs 退化特征对比 ---
normal_mask = ~degradation_95
print(f"\n【正常 vs 退化特征对比 (负残差5%阈值)】")
print(f"  {'特征':<25} {'正常均值':<12} {'异常均值':<12} {'偏差(%)':<12}")
print(f"  {'-'*61}")
all_cols = feature_cols + [target_col]
data_all = np.concatenate([X_raw, y_raw], axis=1)
for i, col in enumerate(all_cols):
    normal_mean = data_all[normal_mask, i].mean()
    anomaly_mean = data_all[anomaly_mask.astype(bool), i].mean()
    dev = (anomaly_mean - normal_mean) / (normal_mean + 1e-8) * 100
    marker = " <--" if col == 'system_cop' and anomaly_mean < normal_mean else ""
    print(f"  {col:<25} {normal_mean:<12.4f} {anomaly_mean:<12.4f} {dev:<+12.2f}{marker}")

# --- 双向异常 vs 退化检测对比 ---
print(f"\n【对比：双向异常 vs 退化检测 (95%阈值)】")
both = anomaly_95 & degradation_95
only_degradation = degradation_95 & ~anomaly_95
only_anomaly = anomaly_95 & ~degradation_95
print(f"  双向异常+退化: {both.sum()}")
print(f"  仅退化检测(双向未标记): {only_degradation.sum()} | 平均COP: {cop_true[only_degradation].mean():.4f}" if only_degradation.sum() > 0 else "  仅退化检测: 0")
print(f"  仅双向异常(退化未标记): {only_anomaly.sum()} | 平均COP: {cop_true[only_anomaly].mean():.4f}" if only_anomaly.sum() > 0 else "  仅双向异常: 0")

print(f"\n{'='*60}")
print(f"          图表输出")
print(f"{'='*60}")

output_dir = "../pic/vae_dualhead_diagnosis"

# ============ 图1: 损失曲线（训练时已生成，这里再生成一份诊断报告） ============

# ============ 图2: COP预测误差分布 ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(cop_pred_errors, bins=80, color='steelblue', edgecolor='white', alpha=0.7, density=True)
ax.axvline(threshold_95, color='red', linestyle='--', linewidth=2, label=f'95%分位 ({threshold_95:.4f})')
ax.axvline(threshold_99, color='darkred', linestyle='--', linewidth=2, label=f'99%分位 ({threshold_99:.4f})')
ax.axvline(threshold_3sigma, color='orange', linestyle='--', linewidth=2, label=f'均值+3σ ({threshold_3sigma:.4f})')
ax.set_xlabel('COP预测误差 (MSE)', fontsize=12)
ax.set_ylabel('密度', fontsize=12)
ax.set_title('双头VAE: COP预测误差分布', fontsize=14)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
sorted_err = np.sort(cop_pred_errors)
cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
ax.plot(sorted_err, cdf, color='steelblue', linewidth=1.5)
ax.axhline(0.95, color='red', linestyle='--', linewidth=1.5, label='95%')
ax.axhline(0.99, color='darkred', linestyle='--', linewidth=1.5, label='99%')
ax.set_xlabel('COP预测误差 (MSE)', fontsize=12)
ax.set_ylabel('累积概率', fontsize=12)
ax.set_title('COP预测误差累积分布', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/error_distribution.png", dpi=150, bbox_inches='tight')
plt.show()

# ============ 图3: 全时段COP预测误差时间序列 ============
fig, ax = plt.subplots(figsize=(16, 5))
colors = np.where(anomaly_mask, 'red', 'steelblue')
ax.scatter(range(len(cop_pred_errors)), cop_pred_errors, c=colors, s=2, alpha=0.5)
ax.axhline(threshold_95, color='red', linestyle='--', linewidth=1.5, label=f'95%阈值')
ax.axhline(threshold_99, color='darkred', linestyle='--', linewidth=1.5, label=f'99%阈值')
ax.set_xlabel('时间 (分钟)', fontsize=12)
ax.set_ylabel('COP预测误差 (MSE)', fontsize=12)
ax.set_title('双头VAE: 全时段COP预测误差 (红色=异常)', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/error_timeseries.png", dpi=150, bbox_inches='tight')
plt.show()

# ============ 图4: 前2000分钟放大 ============
zoom_end = min(2000, len(cop_pred_errors))
fig, ax = plt.subplots(figsize=(16, 5))
colors_zoom = np.where(anomaly_mask[:zoom_end], 'red', 'steelblue')
ax.scatter(range(zoom_end), cop_pred_errors[:zoom_end], c=colors_zoom, s=5, alpha=0.6)
ax.axhline(threshold_95, color='red', linestyle='--', linewidth=1.5, label=f'95%阈值')
ax.axhline(threshold_99, color='darkred', linestyle='--', linewidth=1.5, label=f'99%阈值')
ax.set_xlabel('时间 (分钟)', fontsize=12)
ax.set_ylabel('COP预测误差 (MSE)', fontsize=12)
ax.set_title(f'双头VAE: 前{zoom_end}分钟COP预测误差', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/error_timeseries_zoom.png", dpi=150, bbox_inches='tight')
plt.show()

# ============ 图5: 真实COP vs 预测COP ============
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(cop_true, cop_pred, c=cop_pred_errors, cmap='RdYlGn_r', s=5, alpha=0.5)
ax.plot([cop_true.min(), cop_true.max()], [cop_true.min(), cop_true.max()], 'r--', alpha=0.5, label='y=x')
ax.set_xlabel('真实 COP', fontsize=12)
ax.set_ylabel('双头VAE预测 COP', fontsize=12)
ax.set_title(f'双头VAE: 真实COP vs 预测COP (r={corr:.4f})', fontsize=14)
plt.colorbar(sc, ax=ax, label='COP预测误差')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/cop_prediction_scatter.png", dpi=150, bbox_inches='tight')
plt.show()

# ============ 图6: COP vs 预测误差散点图 ============
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(cop_true, cop_pred_errors, c=cop_pred_errors, cmap='RdYlGn_r', s=5, alpha=0.5)
ax.axhline(threshold_95, color='red', linestyle='--', linewidth=1.5, label=f'95%阈值')
ax.set_xlabel('真实 COP', fontsize=12)
ax.set_ylabel('COP预测误差 (MSE)', fontsize=12)
ax.set_title(f'COP vs 预测误差 (Pearson r={corr:.4f})', fontsize=14)
plt.colorbar(sc, ax=ax, label='预测误差')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/cop_vs_error.png", dpi=150, bbox_inches='tight')
plt.show()

# ============ 图7: 真实COP vs 残差（有符号） ============
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(cop_true, cop_residuals, c=np.abs(cop_residuals), cmap='RdYlGn_r', s=5, alpha=0.5)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax.set_xlabel('真实 COP', fontsize=12)
ax.set_ylabel('COP残差 (真实-预测)', fontsize=12)
ax.set_title(f'COP vs 残差 (Pearson r={corr_res:.4f})', fontsize=14)
plt.colorbar(sc, ax=ax, label='|残差|')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/cop_residual_scatter.png", dpi=150, bbox_inches='tight')
plt.show()

# ============ 图8: 正常 vs 异常 COP时序 + 预测对比 ============
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(cop_true, label='真实 COP', color='royalblue', linewidth=1.0, alpha=0.8)
ax.plot(cop_pred, label='双头VAE预测 COP', color='darkorange', linewidth=0.8, alpha=0.7)
ymin, ymax = ax.get_ylim()
anomaly_regions = np.where(anomaly_mask.astype(bool))[0]
if len(anomaly_regions) > 0:
    region_breaks = np.where(np.diff(anomaly_regions) > 1)[0]
    region_starts = np.concatenate([[anomaly_regions[0]], anomaly_regions[region_breaks + 1]])
    region_ends = np.concatenate([anomaly_regions[region_breaks], [anomaly_regions[-1]]])
    for rs, re in zip(region_starts, region_ends):
        if re - rs >= 5:
            ax.axvspan(rs, re, alpha=0.15, color='red')
ax.set_xlabel('时间 (分钟)', fontsize=12)
ax.set_ylabel('system_cop', fontsize=12)
ax.set_title('真实 COP vs 双头VAE预测 COP (红色阴影=异常)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/cop_timeseries_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# ============ 图9: 正常 vs 异常特征箱线图 ============
n_feat = len(all_cols)
n_cols = 3
n_rows = (n_feat + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
axes = axes.flatten()

for i in range(n_feat):
    ax = axes[i]
    data_n = data_all[normal_mask, i]
    data_a = data_all[anomaly_mask.astype(bool), i]
    bp = ax.boxplot([data_n, data_a], labels=['正常', '异常'], widths=0.5,
                    patch_artist=True, boxprops=dict(linewidth=1.2),
                    medianprops=dict(color='red', linewidth=1.5))
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('crimson')
    ax.set_title(all_cols[i], fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='y', alpha=0.3)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('双头VAE: 正常 vs 异常样本特征分布对比', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{output_dir}/anomaly_feature_dist.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n所有图表已保存到 {output_dir}/")
print("\n双头VAE性能诊断完成!")
