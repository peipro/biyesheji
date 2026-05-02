"""
双头VAE (Two-Head VAE) 训练脚本
架构改动：编码器只编码8个物理特征，解码器拆为特征解码和COP解码两个头
诊断时只用COP重构误差，避免VAE误报运行状态变化
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import copy
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='双头VAE训练')
parser.add_argument('--input', type=str, default='data_feature_engineered_v5_kalman_v8.xlsx',
                    help='输入数据文件路径')
args = parser.parse_args()

# --- 超参数 ---
FEATURE_DIM = 8          # 8个物理特征
LATENT_DIM = 8
HIDDEN_DIMS = [64, 32, 16]
BATCH_SIZE = 64
LR = 1e-3
ALPHA = 5.0              # COP重构损失权重（高于特征权重，迫使z保留COP信息）
BETA_MAX = 0.1           # KL权重
BETA_WARMUP_EPOCHS = 50
MAX_EPOCHS = 200
PATIENCE = 20

# --- 数据加载 ---
df = pd.read_excel(args.input)

feature_cols = [
    'temperature', 'humidity', 'temp_diff',
    'lxj_evap_press_avg', 'lxj_cond_press_avg',
    'A4冷冻泵_f', 'A1冷却泵_f', 'A4冷却塔_f'
]
target_col = 'system_cop'

X = df[feature_cols].values.astype(np.float32)
y = df[[target_col]].values.astype(np.float32)
data = np.concatenate([X, y], axis=1)  # [N, 9]

print(f"数据加载完成: {data.shape[0]} 样本, {data.shape[1]-1}特征+COP")

# --- 80/10/10 随机划分（使用 sklearn train_test_split）---
indices = np.arange(len(data))
train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

train_idx = train_idx.astype(np.int64)
val_idx = val_idx.astype(np.int64)
test_idx = test_idx.astype(np.int64)

# --- 归一化（只在训练集上拟合） ---
scaler = StandardScaler()
scaler.fit(data[train_idx])
data_norm = scaler.transform(data).astype(np.float32)

# 切分特征和COP
X_norm = data_norm[:, :8]
y_norm = data_norm[:, 8:9]

X_train = torch.FloatTensor(X_norm[train_idx])
y_train = torch.FloatTensor(y_norm[train_idx])
X_val = torch.FloatTensor(X_norm[val_idx])
y_val = torch.FloatTensor(y_norm[val_idx])
X_test = torch.FloatTensor(X_norm[test_idx])
y_test = torch.FloatTensor(y_norm[test_idx])

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"划分: 训练 {len(X_train)}, 验证 {len(X_val)}, 测试 {len(X_test)}")

# --- 双头VAE模型 ---
class TwoHeadVAE(nn.Module):
    def __init__(self, feature_dim, latent_dim, hidden_dims):
        super(TwoHeadVAE, self).__init__()

        # 编码器：只编码特征，不编码COP
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

        # 潜在空间
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)

        # 解码器1：特征重构（与特征数相同维度）
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

        # 解码器2：COP预测（1维输出，更小的网络避免过拟合）
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

    def decode_feat(self, z):
        return self.decoder_feat(z)

    def decode_cop(self, z):
        return self.decoder_cop(z)

    def forward(self, x):
        # x: [batch, 8] — 只有特征
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        feat_recon = self.decode_feat(z)
        cop_recon = self.decode_cop(z)
        return feat_recon, cop_recon, mu, log_var


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoHeadVAE(FEATURE_DIM, LATENT_DIM, HIDDEN_DIMS).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
print(f"使用设备: {device}")

# --- 损失函数 ---
def loss_fn(feat_recon, feat, cop_recon, cop, mu, log_var, beta, alpha):
    mse_feat = nn.functional.mse_loss(feat_recon, feat, reduction='sum')
    mse_cop = nn.functional.mse_loss(cop_recon, cop, reduction='sum')
    recon_loss = mse_feat + alpha * mse_cop
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss, mse_feat, mse_cop, kl_loss

# --- 训练循环 ---
best_val_loss = float('inf')
best_model_state = None
early_stop_count = 0
train_loss_history = []
val_loss_history = []
mse_feat_history = []
mse_cop_history = []
kl_loss_history = []

print("开始训练双头VAE...")
for epoch in range(MAX_EPOCHS):
    beta = min(epoch / BETA_WARMUP_EPOCHS, 1.0) * BETA_MAX

    # 训练
    model.train()
    train_loss = 0.0
    train_mse_feat = 0.0
    train_mse_cop = 0.0
    train_kl = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        feat_recon, cop_recon, mu, log_var = model(bx)
        loss, mf, mc, kl = loss_fn(feat_recon, bx, cop_recon, by, mu, log_var, beta, ALPHA)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_mse_feat += mf.item()
        train_mse_cop += mc.item()
        train_kl += kl.item()

    train_loss /= len(X_train)
    train_mse_feat /= len(X_train)
    train_mse_cop /= len(X_train)
    train_kl /= len(X_train)
    train_loss_history.append(train_loss)
    mse_feat_history.append(train_mse_feat)
    mse_cop_history.append(train_mse_cop)
    kl_loss_history.append(train_kl)

    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            feat_recon, cop_recon, mu, log_var = model(bx)
            loss, _, _, _ = loss_fn(feat_recon, bx, cop_recon, by, mu, log_var, beta, ALPHA)
            val_loss += loss.item()
    val_loss /= len(X_val)
    val_loss_history.append(val_loss)

    scheduler.step(val_loss)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} | beta={beta:.3f} | "
              f"Train={train_loss:.6f} | Feat={train_mse_feat:.6f} | COP={train_mse_cop:.6f} | KL={train_kl:.6f} | "
              f"Val={val_loss:.6f}")

    # 早停
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        early_stop_count = 0
    else:
        if epoch >= BETA_WARMUP_EPOCHS:
            early_stop_count += 1
            if early_stop_count >= PATIENCE:
                print(f"\n早停触发! 连续{PATIENCE}轮验证损失未下降")
                break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, "vae_dualhead_best.pth")
print(f"\n训练完成! 最佳验证损失: {best_val_loss:.6f}")

with open("vae_dualhead_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("模型和scaler保存完成: vae_dualhead_best.pth, vae_dualhead_scaler.pkl")

# --- 测试集评估 ---
model.eval()
with torch.no_grad():
    feat_recon_test, cop_recon_test, _, _ = model(X_test.to(device))
    test_mse_feat = nn.functional.mse_loss(feat_recon_test, X_test.to(device), reduction='mean').item()
    test_mse_cop = nn.functional.mse_loss(cop_recon_test, y_test.to(device), reduction='mean').item()
print(f"\n测试集评估:")
print(f"  特征重构MSE: {test_mse_feat:.6f}")
print(f"  COP预测MSE: {test_mse_cop:.6f}")

# --- 损失曲线 ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(train_loss_history, label='训练损失', color='blue', linewidth=1.5)
axes[0].plot(val_loss_history, label='验证损失', color='red', linewidth=1.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('双头VAE 训练/验证损失')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(mse_feat_history, label='特征MSE', color='green', linewidth=1.5)
axes[1].plot(mse_cop_history, label='COP预测MSE', color='orange', linewidth=1.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MSE')
axes[1].set_title(f'特征重构 vs COP预测 (α={ALPHA})')
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(kl_loss_history, label='KL散度', color='purple', linewidth=1.5)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('KL')
axes[2].set_title(f'KL散度 (β_max={BETA_MAX})')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
os.makedirs("../pic/vae_dualhead_diagnosis", exist_ok=True)
plt.savefig("../pic/vae_dualhead_diagnosis/vae_loss.png", dpi=150, bbox_inches='tight')
print("\n损失曲线已保存: ../pic/vae_dualhead_diagnosis/vae_loss.png")
plt.show()
print("双头VAE训练全部完成!")
