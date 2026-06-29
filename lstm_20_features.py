import os
import glob
import joblib
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# ==========================================
# 1. 数据集定义与批量处理逻辑
# ==========================================
class load_dataset(Dataset):
    def __init__(self, processed_data_root, scaler=None):
        processed_data_root = os.path.normpath(processed_data_root)
        search_pattern = os.path.join(processed_data_root, "**", "*.csv")
        self.file_paths = glob.glob(search_pattern, recursive=True)
        # 修正了 'rotation' 的拼写错误
        self.feature_cols = [
            'acceleration_x', 'acceleration_y', 'acceleration_z',
            'rotation_rate_alpha', 'rotation_rate_beta', 'rotation_rate_gamma',
            'acc_mag', 'gyro_mag', 'delta_acc_mag', 'delta_gyro_mag',
            'rolling_var_acc', 'rolling_var_gyro'
        ]
        self.scaler = scaler

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        call_id = os.path.splitext(os.path.basename(file_path))[0]
        parent_dir = os.path.basename(os.path.dirname(file_path)).lower()
        label = 1 if parent_dir == 'abnormal' else 0
        
        df = pd.read_csv(file_path)
        features = df[self.feature_cols].values.astype(np.float32)
        
        if self.scaler is not None:
            features = self.scaler.transform(features)
            
        # 转换为 tensor，并转置为 [channels, seq_len]
        features_tensor = torch.tensor(features).transpose(0, 1) 
        return features_tensor, torch.tensor(label, dtype=torch.float32), call_id

def imu_collate_fn(batch):
    sequences, labels, call_ids = zip(*batch)
    
    # 记录每条序列的真实长度，用于后续的 pack_padded_sequence
    lengths = torch.tensor([seq.shape[1] for seq in sequences], dtype=torch.long)
    
    # 转换为 [seq_len, channels] 用于 pad_sequence
    sequences_transposed = [seq.transpose(0, 1) for seq in sequences]
    padded_seqs = pad_sequence(sequences_transposed, batch_first=True, padding_value=0.0)
    
    # 转置回 [batch_size, channels, max_seq_len] 兼容流水线
    padded_seqs = padded_seqs.transpose(1, 2)
    
    # [核心修复] 按序列长度降序排列 batch (pack_padded_sequence 的要求)
    lengths, perm_idx = lengths.sort(0, descending=True)
    padded_seqs = padded_seqs[perm_idx]
    labels = torch.stack(labels)[perm_idx]
    call_ids = [call_ids[i] for i in perm_idx]
    
    return padded_seqs, labels, lengths, call_ids

# ==========================================
# 2. 模型架构定义
# ==========================================
class LSTM(nn.Module):
    """
    基于 LSTM 的多变量时间序列分类网络 (已修复 Padding 与双向切片问题)
    """
    def __init__(self, input_size, hidden_size=64, output_size=1, num_layers=2, bidirectional=True, bias=True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=bidirectional,
            bias=bias,
            dropout=0.5 if num_layers > 1 else 0.0 
        )

        # 全连接层输入维度：双向 LSTM 需乘以 2
        fc_in_features = hidden_size * 2 if bidirectional else hidden_size
        
        # 修正了未定义的 output_dim，改为 output_size
        self.fc = nn.Sequential(
            nn.Linear(fc_in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_size)
        )

    def forward(self, x, lengths):
        # x 原 shape: [batch_size, input_channels, max_seq_len]
        # LSTM 需 shape: [batch_size, max_seq_len, input_channels]
        x = x.transpose(1, 2)

        # [核心修复] 压紧变长序列，防止 0 padding 稀释历史记忆
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        _, (hn, cn) = self.lstm(packed_x)
        
        # [核心修复] 提取双向 LSTM 最后一层的真实最终状态
        # hn shape: [num_layers * num_directions, batch_size, hidden_size]
        if self.bidirectional:
            # hn[-2] 是前向的最后状态，hn[-1] 是反向的最后状态
            out = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        else:
            out = hn[-1, :, :]
            
        out = self.fc(out)
        return out

# ==========================================
# 3. 训练参数设定与数据划分
# ==========================================
batch_size = 64
epochs = 200
learning_rate = 0.001
patience = 20
input_size = 12       # [修正] IMU 特征数修正为 12 
output_size = 1 
hidden_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = 'data'   

# 挂载数据与 sklearn 分层拆分
temp_dataset = load_dataset(processed_data_root=data_path)

print("正在扫描标签分布用于分层抽样...")
all_labels = [1 if 'abnormal' in path.replace('\\', '/').lower() else 0 for path in temp_dataset.file_paths]

indices = list(range(len(temp_dataset)))
train_indices, val_indices = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=all_labels    
)

# 拟合 StandardScaler (仅使用训练集，防止数据泄露)
train_features_flat = []
for idx in train_indices:
    features = temp_dataset[idx][0].numpy()
    train_features_flat.append(features.transpose(1, 0))
train_features_concat = np.concatenate(train_features_flat, axis=0)

scaler = StandardScaler()
scaler.fit(train_features_concat)

scaler_save_path = 'scaler.pkl'
joblib.dump(scaler, scaler_save_path)

# 重新加载带有 StandardScaler 的数据集
dataset = load_dataset(processed_data_root=data_path, scaler=scaler)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=imu_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=imu_collate_fn)

# [修正] 必须先实例化模型，再将参数传入优化器
model = LSTM(
    input_size=input_size, 
    hidden_size=hidden_size, 
    num_layers=2, 
    bidirectional=True, 
    output_size=output_size
)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)        

# ==========================================
# 4. 训练与验证循环 (基于 AUC 的防欺诈早停)
# ==========================================
best_val_auc = 0.0   # [核心修复] 风控场景下改用 AUC 作为最优模型判定标准
epochs_no_improve = 0        
best_model_wts = copy.deepcopy(model.state_dict()) 

for epoch in range(epochs):
    # --- 训练阶段 ---
    model.train()
    train_loss = 0.0
    train_preds, train_targets = [], []

    for batch_x, batch_y, batch_lens, _ in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).unsqueeze(1)  
        
        optimizer.zero_grad()
        outputs = model(batch_x, batch_lens)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        train_loss += loss.item()
        
        # 收集预测概率用于计算全量指标
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        train_preds.extend(probs)
        train_targets.extend(batch_y.cpu().numpy())

    epoch_train_loss = train_loss / len(train_loader)
    
    # 即使存在严重不平衡，AUC 依然能客观反映排序能力
    try:
        train_auc = roc_auc_score(train_targets, train_preds)
    except ValueError:
        train_auc = 0.5 # 极端情况下 batch 只有单一类别

    # --- 验证阶段 ---
    model.eval() 
    val_loss = 0.0
    val_preds, val_targets = [], []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_lens, _ in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            
            outputs = model(batch_x, batch_lens)
            loss = criterion(outputs, batch_y)
            
            val_loss += loss.item()
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            val_preds.extend(probs)
            val_targets.extend(batch_y.cpu().numpy())
            
    epoch_val_loss = val_loss / len(val_loader)
    
    try:
        val_auc = roc_auc_score(val_targets, val_preds)
    except ValueError:
        val_auc = 0.5
        
    # 计算极度不平衡下作为参考的 PR 指标 (默认 0.5 阈值)
    val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
    val_prec = precision_score(val_targets, val_preds_binary, zero_division=0)
    val_rec = recall_score(val_targets, val_preds_binary, zero_division=0)

    # --- 日志打印 ---
    print(f"\nEpoch [{epoch+1:02d}/{epochs}]")
    print(f"Train | Loss: {epoch_train_loss:.4f} | AUC: {train_auc:.4f}")
    print(f"Val   | Loss: {epoch_val_loss:.4f} | AUC: {val_auc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

    # --- 早停判定 (基于 AUC 最大化) ---
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f" --> [更新] 验证集 AUC 达到新高 ({best_val_auc:.4f})，暂存权重。")
    else:
        epochs_no_improve += 1
        print(f" --> [等待] 验证集 AUC 未提升 (累计 {epochs_no_improve}/{patience} 次)。")
        
        if epochs_no_improve >= patience:
            print(f"\n[Early Stopping] 连续 {patience} 个 Epoch 验证集表现未提升，提前终止训练！")
            break

# 循环结束后，加载保存的最佳权重并落盘
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "lstm_imu_model_best.pth")
print(f"\n训练流程全部结束。最佳模型权重 (Val AUC: {best_val_auc:.4f}) 已保存。")