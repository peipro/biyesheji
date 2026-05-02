import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import copy
import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

# --- 1. 环境配置 ---
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子保证可复现
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 2. 超参数 ---
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 1e-5
DROPOUT = 0.1
EPOCHS = 300
PATIENCE = 25

parser = argparse.ArgumentParser(description='ANN训练')
parser.add_argument('--input', type=str, default='data_feature_engineered_v5.xlsx',
                    help='输入数据文件路径 (默认: data_feature_engineered_v5.xlsx)')
args = parser.parse_args()

# --- 3. 加载数据 ---
df = pd.read_excel(args.input)

# 定义特征列 (所有模型保持一致)
feature_cols = [
    'temperature', 'humidity', 'temp_diff',
    'lxj_evap_press_avg', 'lxj_cond_press_avg',
    'A4冷冻泵_f', 'A1冷却泵_f', 'A4冷却塔_f'
]
target_col = 'system_cop'

X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

# --- 4. 数据预处理 ---
# 随机划分 80%训练, 10%验证, 10%测试
X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 归一化：只在训练集上拟合，防止测试集信息泄漏
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_norm = scaler_x.fit_transform(X_train_raw)
y_train_norm = scaler_y.fit_transform(y_train_raw.reshape(-1, 1))
X_val_norm = scaler_x.transform(X_val_raw)
y_val_norm = scaler_y.transform(y_val_raw.reshape(-1, 1))
X_test_norm = scaler_x.transform(X_test_raw)
y_test_norm = scaler_y.transform(y_test_raw.reshape(-1, 1))

# 转换为 Tensor
X_train = torch.FloatTensor(X_train_norm)
y_train = torch.FloatTensor(y_train_norm)
X_val = torch.FloatTensor(X_val_norm)
y_val = torch.FloatTensor(y_val_norm)
X_test = torch.FloatTensor(X_test_norm)
y_test = torch.FloatTensor(y_test_norm)

# 构建 DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 5. 定义更深的全连接网络 ---
class MLP_Predictor(nn.Module):
    def __init__(self, input_dim, dropout_rate):
        super(MLP_Predictor, self).__init__()
        # 进一步增加模型容量
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 1)  # 输出层
        )
        # Xavier初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP_Predictor(len(feature_cols), DROPOUT).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
criterion = nn.MSELoss()

# --- 6. 训练模型 + 早停 ---
print(f"开始训练改进MLP模型 (设备: {device})...")
print(f"  CUDA可用: {torch.cuda.is_available()}, 设备名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"超参数: batch={BATCH_SIZE}, lr={LR}, weight_decay={WEIGHT_DECAY}, dropout={DROPOUT}")

best_val_loss = float('inf')
best_model_state = None
early_stop_count = 0
train_loss_history = []
val_loss_history = []

for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_x.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    train_loss_history.append(train_loss)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)

    val_loss = val_loss / len(val_loader.dataset)
    val_loss_history.append(val_loss)

    # 更新学习率
    scheduler.step(val_loss)

    # 打印日志
    lr_current = optimizer.param_groups[0]['lr']
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {lr_current:.6f}")

    # 早停检查
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count >= PATIENCE:
            print(f"\n早停触发，Epoch {epoch+1}，连续 {PATIENCE} 轮验证损失不下降")
            break

# 加载最佳模型
if best_model_state is not None:
    model.load_state_dict(best_model_state)
print(f"\n训练完成，最佳验证损失: {best_val_loss:.6f}")

# --- 7. 在测试集评估 ---
model.eval()
with torch.no_grad():
    y_pred_norm = model(X_test.to(device)).cpu().numpy()

# 逆标准化还原
y_pred = scaler_y.inverse_transform(y_pred_norm)
y_true = scaler_y.inverse_transform(y_test.numpy())

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("\n改进MLP模型测试集评估结果:")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# --- 8. 特征重要性分析 (Permutation Importance) ---
print("\n正在计算特征重要性...")
importances = {}

# 计算基准分数：在归一化空间计算R²，与perm R²在同一空间对比
with torch.no_grad():
    y_baseline_norm = model(X_test.to(device)).cpu().numpy()
baseline_r2_norm = r2_score(y_test.numpy(), y_baseline_norm)
print(f"  基线 R2 (归一化空间): {baseline_r2_norm:.4f}")

for i, col in enumerate(feature_cols):
    # 复制一份测试数据
    X_test_permuted = X_test.clone()

    # 随机打乱当前特征列的顺序
    perm_indices = torch.randperm(X_test_permuted.size(0))
    X_test_permuted[:, i] = X_test_permuted[perm_indices, i]

    # 用打乱后的数据预测
    with torch.no_grad():
        y_perm_pred_norm = model(X_test_permuted.to(device)).cpu().numpy()
        perm_r2 = r2_score(y_test.numpy(), y_perm_pred_norm)

    # 重要性 = 基准 R2 - 打乱后的 R2
    importances[col] = baseline_r2_norm - perm_r2

# --- 9. 可视化 ---

# 图1: 损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='训练损失', color='blue', linewidth=1.5)
plt.plot(val_loss_history, label='验证损失', color='red', linewidth=1.5)
plt.title(f"改进MLP: 训练和验证损失曲线", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../pic/baseline/ann_loss.png", dpi=150, bbox_inches='tight')
plt.show()

# 图2: 预测对比图（前150点）
plt.figure(figsize=(10, 6))
plt.plot(y_true[:150], label='真实COP', color='royalblue', linewidth=1.5, alpha=0.8)
plt.plot(y_pred[:150], label='MLP预测', color='red', linestyle='--', linewidth=1.5)
plt.title(f"改进MLP: 真实值vs预测值对比 (R2: {r2:.4f})", fontsize=14)
plt.xlabel("样本索引", fontsize=12)
plt.ylabel("系统COP", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../pic/baseline/ann_pred_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# 图3: 特征重要性条形图
plt.figure(figsize=(10, 6))
importance_series = pd.Series(importances).sort_values()
importance_series.plot(kind='barh', color='salmon')
plt.title("ANN: 特征重要性 (Permutation方法)", fontsize=14)
plt.xlabel("重要性分数 (R2下降幅度)", fontsize=12)
plt.ylabel("特征", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("../pic/baseline/ann_feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()
