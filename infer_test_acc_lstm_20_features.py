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
from sklearn.metrics import roc_auc_score, precision_score, recall_score,confusion_matrix,accuracy_score
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
input_size = 12       # [修正] IMU 特征数修正为 12 
output_size = 1 
hidden_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler_path="260618.pkl"
data_path = 'data'   
model = LSTM(
    input_size=input_size, 
    hidden_size=hidden_size, 
    num_layers=2, 
    bidirectional=False, 
    output_size=output_size
)
model.load_state_dict(torch.load("lstm.pth", map_location=device))
criterion = nn.BCEWithLogitsLoss()
model.to(device)
if os.path.exists(scaler_path):
    scaler=joblib(scaler_path)
else:
    print("can not find file")

test_dataset = load_dataset(processed_data_root=data_path,scaler=scaler)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=imu_collate_fn)

model.eval() 
all_preds, all_labels = [], []

with torch.no_grad():
    for batch_x, batch_y, _ in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).unsqueeze(1)
        
        outputs = model(batch_x)
                
        preds = (outputs>0).cpu().numpy().astype(int).flatten()
        labels=(batch_y>0.5).cpu().numpy().astype(int).flatten()
        all_preds.extend(preds)
        all_labels.extend(labels)
        
cm=confusion_matrix(all_labels,all_preds,label=[0,1])
tn,fp,fn,tp=cm.ravel()
print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
print(f'Precision: {precision_score(all_labels,all_preds):.4f}')
print(f'Recall: {recall_score(all_labels,all_preds):.4f}')
print(f'Accuracy: {accuracy_score(all_labels,all_preds):.4f}')
