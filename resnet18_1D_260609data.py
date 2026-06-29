import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class ProcessedIMUDataset(Dataset):
    def __init__(self, processed_data_root):
        processed_data_root = os.path.normpath(processed_data_root)
        search_pattern = os.path.join(processed_data_root, "**", "*.csv")
        self.file_paths = glob.glob(search_pattern, recursive=True)
        self.feature_cols = [
            'acceleration_x', 'acceleration_y', 'acceleration_z',
            'rotaion_rate_alpha', 'rotaion_rate_beta', 'rotaion_rate_gamma',
            'acc_mag', 'gyro_mag', 'delta_acc_mag', 'delta_gyro_mag',
            'rolling_var_acc', 'rolling_var_gyro'
        ]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        call_id = os.path.splitext(os.path.basename(file_path))[0]
        parent_dir = os.path.basename(os.path.dirname(file_path)).lower()
        label = 1 if parent_dir == 'abnormal' else 0
        df = pd.read_csv(file_path)
        features = df[self.feature_cols].values.astype(np.float32)
        features_tensor = torch.tensor(features).transpose(0, 1) 
        return features_tensor, torch.tensor(label, dtype=torch.float32), call_id

def imu_collate_fn(batch):
    sequences, labels, call_ids = zip(*batch)
    sequences_transposed = [seq.transpose(0, 1) for seq in sequences]
    padded_seqs = pad_sequence(sequences_transposed, batch_first=True, padding_value=0.0)
    padded_seqs = padded_seqs.transpose(1, 2)
    return padded_seqs, torch.stack(labels), call_ids



class ResidualBlock1D(nn.Module):
    """
    一维残差块 (1D Residual Block)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        
        # 第一层一维卷积
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 第二层一维卷积
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 恒等映射 (Skip Connection)
        # 如果维度或通道数发生变化，需要用 1x1 卷积调整 shape 以便相加
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 将残差加到主路径上
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    一维残差网络 (ResNet-1D)
    """
    def __init__(self, input_channels, num_classes, num_blocks_list=[2, 2, 2, 2]):
        """
        :param input_channels: 序列的特征维度 (例如单变量时间序列为1，多变量为特征数)
        :param num_classes: 预测输出的维度 (回归任务通常为1，分类任务为类别数)
        :param num_blocks_list: 每个网络层中残差块的数量
        """
        super(ResNet1D, self).__init__()
        self.in_channels = 64

        # 初始特征提取层
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, 
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 残差层 (Layers)
        self.layer1 = self._make_layer(64, num_blocks_list[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks_list[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks_list[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks_list[3], stride=2)

        # 自适应平均池化，能够接受任意长度的序列输入
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接预测层
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock1D(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [batch_size, input_channels, sequence_length]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x) # 降维到 [batch_size, channels, 1]
        x = x.view(x.size(0), -1) # 展平为 [batch_size, channels]
        x = self.fc(x)
        
        return x
        
# 测试参数设定
batch_size = 16
seq_length = 1000      # 序列的时间步长度 (例如1000个采样点)
input_channels = 3     # 序列特征数 (例如XYZ三轴传感器数据)
output_dim = 1         # 预测目标 (例如预测1个未来的数值，若是10分类问题则改为10)

# 1. 实例化模型
model = ResNet1D(input_channels=input_channels, num_classes=output_dim)

# 2. 生成随机模拟序列数据
# 注意 PyTorch 1D 卷积的输入格式要求：[Batch大小, 特征通道数, 序列长度]
dummy_input = torch.randn(batch_size, input_channels, seq_length)

# 3. 前向传播预测
predictions = model(dummy_input)

print(f"输入序列 Shape: {dummy_input.shape}")
print(f"预测输出 Shape: {predictions.shape}") 

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset
    import copy

    # --- 1. 基础配置 ---
    PROCESSED_DATA_ROOT = "./train_processed" 
    BATCH_SIZE = 64
    EPOCHS = 50          
    LEARNING_RATE = 0.001
    PATIENCE = 5         
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")

    # --- 2. 挂载数据与 sklearn 分层拆分 ---
    full_dataset = ProcessedIMUDataset(processed_data_root=PROCESSED_DATA_ROOT)
    
    print("正在扫描标签分布用于分层抽样...")
    all_labels = [1 if '/abnormal' in path.replace('\\', '/').lower() else 0 for path in full_dataset.file_paths]
    
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=all_labels    
    )
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=imu_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=imu_collate_fn)
    
    print(f"分层拆分完成: 训练集 {len(train_dataset)} 条，验证集 {len(val_dataset)} 条")

    # --- 3. 实例化模型与优化器 ---
    model = ResNet1D(input_channels=12, num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # --- 4. 包含多指标与早停的完整训练循环 ---
    best_val_loss = float('inf')  
    epochs_no_improve = 0         
    best_model_wts = copy.deepcopy(model.state_dict()) 

    for epoch in range(EPOCHS):
        # ======================= 训练阶段 =======================
        model.train()
        train_loss = 0.0
        train_correct, train_total = 0, 0
        train_tp, train_fp, train_fn = 0, 0, 0 # 新增：混淆矩阵元素累计

        for batch_x, batch_y, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)  
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # --- 指标计算核心逻辑 ---
            preds_bool = (outputs > 0)         # Logits > 0 相当于预测类别 1
            labels_bool = (batch_y > 0.5)      # 真实标签转为布尔值
            
            train_correct += (preds_bool == labels_bool).sum().item()
            train_total += batch_y.size(0)
            
            # 统计 TP, FP, FN
            train_tp += (preds_bool & labels_bool).sum().item()
            train_fp += (preds_bool & ~labels_bool).sum().item()
            train_fn += (~preds_bool & labels_bool).sum().item()

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_correct / train_total
        # 计算 P 和 R，加入 +1e-8 防止除以 0 导致报错
        epoch_train_prec = train_tp / (train_tp + train_fp + 1e-8)
        epoch_train_rec = train_tp / (train_tp + train_fn + 1e-8)

        # ======================= 验证阶段 =======================
        model.eval() 
        val_loss = 0.0
        val_correct, val_total = 0, 0
        val_tp, val_fp, val_fn = 0, 0, 0 # 新增：混淆矩阵元素累计
        
        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).unsqueeze(1)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                
                # --- 指标计算核心逻辑 ---
                preds_bool = (outputs > 0)
                labels_bool = (batch_y > 0.5)
                
                val_correct += (preds_bool == labels_bool).sum().item()
                val_total += batch_y.size(0)
                
                val_tp += (preds_bool & labels_bool).sum().item()
                val_fp += (preds_bool & ~labels_bool).sum().item()
                val_fn += (~preds_bool & labels_bool).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_correct / val_total
        epoch_val_prec = val_tp / (val_tp + val_fp + 1e-8)
        epoch_val_rec = val_tp / (val_tp + val_fn + 1e-8)

        # ======================= 日志打印 =======================
        print(f"\nEpoch [{epoch+1:02d}/{EPOCHS}]")
        print(f"Train | Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f} | Prec: {epoch_train_prec:.4f} | Rec: {epoch_train_rec:.4f}")
        print(f"Val   | Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f} | Prec: {epoch_val_prec:.4f} | Rec: {epoch_val_rec:.4f}")

        # ======================= 早停判定 =======================
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f" --> [更新] 验证集 Loss 达到新低 ({best_val_loss:.4f})，暂存权重。")
        else:
            epochs_no_improve += 1
            print(f" --> [等待] 验证集 Loss 未下降 (累计 {epochs_no_improve}/{PATIENCE} 次)。")
            
            if epochs_no_improve >= PATIENCE:
                print(f"\n[Early Stopping] 连续 {PATIENCE} 个 Epoch 验证集表现未提升，提前终止训练！")
                break

    # 循环结束后，加载保存的最佳权重并落盘
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "resnet1d_imu_model_best.pth")
    print(f"\n训练流程全部结束。最佳模型权重 (Val Loss: {best_val_loss:.4f}) 已保存。")