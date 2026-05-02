import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import copy
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# --- 设置 ---
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

parser = argparse.ArgumentParser(description='VAE训练')
parser.add_argument('--input', type=str, default='data_feature_engineered_v5.xlsx',
                    help='输入数据文件路径')
args = parser.parse_args()

# --- 超参数 ---
INPUT_DIM = 9           # 8个物理特征 + system_cop
LATENT_DIM = 8
HIDDEN_DIMS = [64, 32, 16]
BATCH_SIZE = 64
LR = 1e-3
BETA_MAX = 0.1           # 降低β权重，优先保障重构质量（KL作为轻度正则化）
BETA_WARMUP_EPOCHS = 50
MAX_EPOCHS = 200
PATIENCE = 20

# --- 数据加载 ---
df = pd.read_excel(args.input)

feature_cols = [
    'temperature', 'humidity', 'temp_diff',
    'lxj_evap_press_avg', 'lxj_cond_press_avg',
    'A4冷冻泵_f', 'A1冷却泵_f', 'A4冷却塔_f',
    'system_cop'
]

data = df[feature_cols].values.astype(np.float32)
print(f"数据加载完成: {data.shape[0]} 样本, {data.shape[1]} 特征")

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

X_train = torch.FloatTensor(data_norm[train_idx])
X_val = torch.FloatTensor(data_norm[val_idx])
X_test = torch.FloatTensor(data_norm[test_idx])

train_loader = DataLoader(TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val), batch_size=BATCH_SIZE, shuffle=False)

print(f"划分: 训练 {len(X_train)}, 验证 {len(X_val)}, 测试 {len(X_test)}")

# --- VAE模型 ---
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super(VAE, self).__init__()

        # 编码器
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

        # 潜在空间
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)

        # 解码器
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(INPUT_DIM, LATENT_DIM, HIDDEN_DIMS).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
print(f"使用设备: {device}")

# --- 损失函数 ---
def vae_loss(recon_x, x, mu, log_var, beta):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# --- 训练循环 ---
best_val_loss = float('inf')
best_model_state = None
early_stop_count = 0
train_loss_history = []
val_loss_history = []
recon_loss_history = []
kl_loss_history = []

print("开始训练VAE...")
for epoch in range(MAX_EPOCHS):
    # β预热
    beta = min(epoch / BETA_WARMUP_EPOCHS, 1.0) * BETA_MAX

    # 训练
    model.train()
    train_loss = 0.0
    train_recon = 0.0
    train_kl = 0.0
    for batch_x, in train_loader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(batch_x)
        loss, recon, kl = vae_loss(recon_batch, batch_x, mu, log_var, beta)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_recon += recon.item()
        train_kl += kl.item()

    train_loss /= len(X_train)
    train_recon /= len(X_train)
    train_kl /= len(X_train)
    train_loss_history.append(train_loss)

    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, in val_loader:
            batch_x = batch_x.to(device)
            recon_batch, mu, log_var = model(batch_x)
            loss, _, _ = vae_loss(recon_batch, batch_x, mu, log_var, beta)
            val_loss += loss.item()
    val_loss /= len(X_val)
    val_loss_history.append(val_loss)

    recon_loss_history.append(train_recon)
    kl_loss_history.append(train_kl)

    scheduler.step(val_loss)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} | β={beta:.3f} | "
              f"Train={train_loss:.6f} (Recon={train_recon:.6f}, KL={train_kl:.6f}) | "
              f"Val={val_loss:.6f}")

    # 早停
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        early_stop_count = 0
    else:
        if epoch >= BETA_WARMUP_EPOCHS:  # β预热完成后才开始早停计数
            early_stop_count += 1
            if early_stop_count >= PATIENCE:
                print(f"\n早停触发! 连续{PATIENCE}轮验证损失未下降")
                break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, "vae_best.pth")
print(f"\n训练完成! 最佳验证损失: {best_val_loss:.6f}")

# 保存 scaler
with open("vae_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("模型和scaler保存完成: vae_best.pth, vae_scaler.pkl")

# --- 损失曲线 ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='训练损失', color='blue', linewidth=1.5)
plt.plot(val_loss_history, label='验证损失', color='red', linewidth=1.5)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('VAE 训练/验证损失曲线', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(recon_loss_history, label='重构损失', color='green', linewidth=1.5)
plt.plot(kl_loss_history, label='KL散度', color='orange', linewidth=1.5)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('VAE 重构损失 vs KL散度', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../pic/vae_diagnosis/vae_loss.png", dpi=150, bbox_inches='tight')
print("损失曲线已保存: ../pic/vae_diagnosis/vae_loss.png")
plt.show()

# --- 测试集评估 ---
model.eval()
with torch.no_grad():
    recon_test, _, _ = model(X_test.to(device))
    test_mse = nn.functional.mse_loss(recon_test, X_test.to(device), reduction='mean').item()
print(f"测试集重构MSE: {test_mse:.6f}")
print("VAE训练全部完成!")
