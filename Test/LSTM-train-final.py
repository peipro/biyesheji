import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import copy
import random

# 设置随机种子保证可复现
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# 不使用CUDA，移除CUDA种子设置避免初始化CUDA
torch.backends.cudnn.deterministic = True

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 超参数设置 ====================
SEQ_LENGTH = 30        # 增大窗口，让模型看到更多历史
HIDDEN_DIM = 128      # 增大隐藏单元，增加模型容量
DROPOUT = 0.1          # 减小dropout，充分利用容量
BATCH_SIZE = 64        # batch_size
LR = 0.001             # 学习率
WEIGHT_DECAY = 1e-5    # L2正则化
EPOCHS = 100           # 训练更多轮数
PATIENCE = 20          # 早停耐心
CLIP_GRAD_NORM = 5.0   # 梯度裁剪
# =====================================================

# 1. 加载数据
df = pd.read_excel("data_feature_engineered_v4.xlsx")

# 特征选择（与其他模型完全一致，删除B区域全0特征，严禁加入泄露特征）
feature_cols = [
    'temperature', 'humidity', 'temp_diff', 'chiller_running_count',
    'lxj_evap_press_avg', 'lxj_cond_press_avg',
    'A_Chilled_Pump_avg', 'A_Cooling_Pump_avg', 'A_Tower_avg'
]
target_col = 'system_cop'

print(f"数据加载完成，总样本数: {len(df)}, 输入特征数: {len(feature_cols)}")
print(f"使用特征: {feature_cols}")

# 4. 创建滑动窗口序列函数
def create_sequences(x, y, seq_length):
    """
    用过去 seq_length 步预测当前步的COP
    返回形状: (样本数, seq_length, 特征数)
    """
    xi, yi = [], []
    for i in range(len(x) - seq_length):
        xi.append(x[i:i + seq_length])
        yi.append(y[i + seq_length])
    return np.array(xi), np.array(yi)

# 2. 和其他模型（XGBoost/ANN/RF）保持一致：随机划分
# 先对整个数据集归一化，再创建所有序列，然后随机打乱划分训练/验证/测试
# 序列内部仍保持时序顺序，不破坏时间依赖关系
scaler_x = MinMaxScaler(feature_range=(0, 1))
X_all = scaler_x.fit_transform(df[feature_cols])

# y不做归一化，直接用原始值
y_all = df[[target_col]].values

# 创建所有序列
X_all_seq, y_all_seq = create_sequences(X_all, y_all, SEQ_LENGTH)

# 随机打乱序列（保持每个序列内部时序不变）
np.random.seed(42)
perm = np.random.permutation(len(X_all_seq))
X_all_seq = X_all_seq[perm]
y_all_seq = y_all_seq[perm]

# 8:1:1 划分训练/验证/测试
train_end = int(len(X_all_seq) * 0.8)
val_end = int(len(X_all_seq) * 0.9)

X_train_seq, y_train_seq = X_all_seq[:train_end], y_all_seq[:train_end]
X_val_seq, y_val_seq = X_all_seq[train_end:val_end], y_all_seq[train_end:val_end]
X_test_seq, y_test_seq = X_all_seq[val_end:], y_all_seq[val_end:]

print(f"随机划分: 训练 {len(X_train_seq)}, 验证 {len(X_val_seq)}, 测试 {len(X_test_seq)}")
print(f"窗口大小: {SEQ_LENGTH}")

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_seq)
y_train_tensor = torch.FloatTensor(y_train_seq)
X_val_tensor = torch.FloatTensor(X_val_seq)
y_val_tensor = torch.FloatTensor(y_val_seq)
X_test_tensor = torch.FloatTensor(X_test_seq)
y_test_tensor = torch.FloatTensor(y_test_seq)

# 5. 定义LSTM模型（两层LSTM，增加容量）
class COP_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(COP_LSTM, self).__init__()
        # 两层LSTM堆叠，增加模型容量
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        # 全连接预测头
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # x形状: [batch, seq_len, input_dim]
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# 强制使用CPU，避免Windows CUDA虚拟内存不足问题
device = torch.device("cpu")
model = COP_LSTM(
    input_dim=len(feature_cols),
    hidden_dim=HIDDEN_DIM,
    output_dim=1,
    dropout=DROPOUT
).to(device)

# 6. 训练配置
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# 增大patience，减缓学习率衰减
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
criterion = nn.MSELoss()

# 数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"开始训练，设备: {device}")
print(f"超参数: seq_length={SEQ_LENGTH}, hidden_dim={HIDDEN_DIM}, dropout={DROPOUT}, lr={LR}")

# 7. 训练循环 + 早停
best_val_loss = float('inf')
best_model_state = None
early_stop_count = 0
train_loss_history = []
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

        # 梯度裁剪，防止梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
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

    # 打印日志（每轮都打，观察训练过程）
    lr_current = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {lr_current:.6f}")

    # 早停检查，保留最佳模型
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

# 8. 在测试集评估
model.eval()
# 分批预测，避免一次性分配过大内存导致bad allocation
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

y_pred_list = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        y_pred_list.append(outputs.cpu())

y_pred = torch.cat(y_pred_list, dim=0).numpy()
# y不做归一化，直接用原始值
y_true = y_test_tensor.numpy().reshape(-1, 1)

# 统计信息
print(f"\n统计对比:")
print(f"真实值 - 均值: {y_true.mean():.4f}, 标准差: {y_true.std():.4f}, 范围: [{y_true.min():.4f}, {y_true.max():.4f}]")
print(f"预测值 - 均值: {y_pred.mean():.4f}, 标准差: {y_pred.std():.4f}, 范围: [{y_pred.min():.4f}, {y_pred.max():.4f}]")

# 计算评估指标
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"\n最终测试集评估结果:")
print(f"R2 Score: {r2:.4f} (越接近1越好)")
print(f"MAE: {mae:.4f} (越小越好)")
print(f"RMSE: {rmse:.4f} (越小越好)")

# 9. 可视化

# 图1: 损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='训练损失', color='blue', linewidth=1.5)
plt.plot(val_loss_history, label='验证损失', color='red', linewidth=1.5)
plt.title(f"LSTM改进版: 训练和验证损失曲线 (窗口={SEQ_LENGTH})", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MSE Loss (标准化后)", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"../pic/lstm_final_loss_w{SEQ_LENGTH}.png", dpi=150, bbox_inches='tight')
plt.show()

# 图2: 预测对比曲线（前300点）
plt.figure(figsize=(12, 6))
plt.plot(y_true[:300], label='真实COP', color='royalblue', alpha=0.8, linewidth=1.5)
plt.plot(y_pred[:300], label='LSTM预测', color='darkorange', linestyle='--', linewidth=1.5)
plt.title(f"LSTM改进版: 真实值vs预测值对比 (窗口={SEQ_LENGTH}, R²={r2:.4f}, MAE={mae:.4f})", fontsize=14)
plt.xlabel("时间步 (分钟)", fontsize=12)
plt.ylabel("系统COP", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"../pic/lstm_final_pred_w{SEQ_LENGTH}.png", dpi=150, bbox_inches='tight')
plt.show()

# 图3: 预测值vs真实值散点图
plt.figure(figsize=(8, 8))
plt.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.5, s=10)
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x', linewidth=1.5)
plt.xlabel("真实COP", fontsize=12)
plt.ylabel("预测COP", fontsize=12)
plt.title("LSTM改进版: 预测值vs真实值散点图", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig(f"../pic/lstm_final_scatter_w{SEQ_LENGTH}.png", dpi=150, bbox_inches='tight')
plt.show()

# 10. 特征重要性分析 (Permutation Importance)
print("\n正在计算LSTM特征重要性（Permutation方法）...")
importances = {}
baseline_r2 = r2

# 遍历每个特征，计算排列重要性
for i, col in enumerate(feature_cols):
    # 复制一份测试数据，保持原始数据不变
    X_test_permuted = X_test_tensor.clone()

    # 对当前特征，在所有样本和所有时间步上打乱顺序
    # 保持每个序列的结构，只打乱该特征在不同样本间的顺序
    perm_indices = torch.randperm(X_test_permuted.size(0))
    X_test_permuted[:, :, i] = X_test_permuted[perm_indices, :, i]

    # 用打乱后的数据预测
    y_pred_perm_list = []
    with torch.no_grad():
        perm_dataset = TensorDataset(X_test_permuted)
        perm_loader = DataLoader(perm_dataset, batch_size=BATCH_SIZE, shuffle=False)
        for batch_x, in perm_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            y_pred_perm_list.append(outputs.cpu())

    y_pred_perm = torch.cat(y_pred_perm_list, dim=0).numpy()
    perm_r2 = r2_score(y_true, y_pred_perm)

    # 重要性 = 基准 R² - 打乱后的 R²
    importances[col] = baseline_r2 - perm_r2
    print(f"  {col}: {importances[col]:.4f}")

# 可视化特征重要性
plt.figure(figsize=(10, 6))
importance_series = pd.Series(importances).sort_values()
importance_series.plot(kind='barh', color='steelblue')
plt.title(f"LSTM模型: 特征重要性 (Permutation方法, R2={r2:.4f})", fontsize=14)
plt.xlabel("重要性分数 (R2下降幅度)", fontsize=12)
plt.ylabel("特征", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"../pic/lstm_final_feature_importance_w{SEQ_LENGTH}.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n所有分析完成！")
