import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  

class load_dataset(Dataset):
    def __init__(self, processed_data_root,scaler=None):
        processed_data_root = os.path.normpath(processed_data_root)
        search_pattern = os.path.join(processed_data_root, "**", "*.csv")
        self.file_paths = glob.glob(search_pattern, recursive=True)
        self.feature_cols = [
            'acceleration_x', 'acceleration_y', 'acceleration_z',
            'rotaion_rate_alpha', 'rotaion_rate_beta', 'rotaion_rate_gamma',
            'acc_mag', 'gyro_mag', 'delta_acc_mag', 'delta_gyro_mag',
            'rolling_var_acc', 'rolling_var_gyro'
        ]
        self.scaler=scaler

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
            features=self.scaler.transform(features)
        features_tensor = torch.tensor(features).transpose(0, 1) 
        return features_tensor, torch.tensor(label, dtype=torch.float32), call_id

def imu_collate_fn(batch):
    sequences, labels, call_ids = zip(*batch)
    sequences_transposed = [seq.transpose(0, 1) for seq in sequences]
    padded_seqs = pad_sequence(sequences_transposed, batch_first=True, padding_value=0.0)
    padded_seqs = padded_seqs.transpose(1, 2)
    return padded_seqs, torch.stack(labels), call_ids

# ==========================================
# 1. 升级版局部特征提取器 (适应 28 维输入)
# ==========================================
class InstanceExtractor1D_28Feat(nn.Module):
    def __init__(self, in_channels=28, out_features=256):
        super(InstanceExtractor1D_28Feat, self).__init__()
        # 增加卷积通道数以处理 28 维输入带来的高信息量
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, out_features, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # 压缩时间维度，输出定长特征向量
        )

    def forward(self, x):
        features = self.conv_block(x)
        return features.squeeze(-1) # 输出: [Batch * Num_Instances, 256]

# ==========================================
# 2. 匹配高维特征的门控注意力
# ==========================================
class GatedAttention_28Feat(nn.Module):
    def __init__(self, in_features=256, hidden_dim=128):
        super(GatedAttention_28Feat, self).__init__()
        self.attention_V = nn.Linear(in_features, hidden_dim)
        self.attention_U = nn.Linear(in_features, hidden_dim)
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: [Batch, Num_Instances, 256]
        attention_v = torch.tanh(self.attention_V(x))
        attention_u = torch.sigmoid(self.attention_U(x))
        a = self.attention_weights(attention_v * attention_u)
        a = F.softmax(a, dim=1)
        return a

# ==========================================
# 3. 完整 MIL 主网络
# ==========================================
class MobileSensorMIL_28Feat(nn.Module):
    def __init__(self, in_channels=28, window_size=50, step_size=25, num_classes=1):
        super(MobileSensorMIL_28Feat, self).__init__()
        self.window_size = window_size
        self.step_size = step_size
        
        # 实例化升级后的组件
        self.feature_extractor = InstanceExtractor1D_28Feat(in_channels=in_channels, out_features=256)
        self.attention = GatedAttention_28Feat(in_features=256, hidden_dim=128)
        
        # 增强分类器的全连接层
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _create_bags(self, x):
        batch_size, channels, seq_len = x.size()
        
        # 边界保护：如果输入序列比设定的窗口还要短，进行 Padding
        if seq_len < self.window_size:
            pad_size = self.window_size - seq_len
            x = F.pad(x, (0, pad_size), "constant", 0)
            
        x_unfolded = x.unfold(dimension=2, size=self.window_size, step=self.step_size)
        x_bags = x_unfolded.permute(0, 2, 1, 3)
        return x_bags

    def forward(self, x):
        # 1. 自动滑动窗口生成 Bag
        x_bags = self._create_bags(x)
        batch_size, num_instances, channels, window_size = x_bags.size()
        
        # 2. 展平送入 CNN
        x_flat = x_bags.reshape(batch_size * num_instances, channels, window_size)
        inst_features = self.feature_extractor(x_flat)
        
        # 3. 恢复 Bag 结构
        h = inst_features.view(batch_size, num_instances, -1)
        
        # 4. 计算注意力并聚合
        A = self.attention(h)
        bag_representation = torch.bmm(h.transpose(1, 2), A).squeeze(2)
        
        # 5. 输出
        logits = self.classifier(bag_representation)
        return logits, A, num_instances


# --- 1. 测试配置 ---
TEST_DATA_ROOT = "./test_processed" 
MODEL_WEIGHTS_PATH = "mil_imu_model_best.pth" # 替换为你的 MIL 模型权重路径
SCALER_PATH = "feature_scaler.pkl"            # 新增：训练时保存的 scaler 路径
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 正在使用计算设备: {device}")

# --- 2. 加载 Scaler 并挂载测试数据 ---
# 加载在训练集中 fit 好的 scaler
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Scaler [{SCALER_PATH}] 加载成功！")
else:
    raise FileNotFoundError(f"找不到 Scaler 文件: {SCALER_PATH}，请确保训练脚本已正确保存该文件。")

# 修正类名并传入 scaler 
test_dataset = load_dataset(processed_data_root=TEST_DATA_ROOT, scaler=scaler)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=imu_collate_fn)
print(f"✅ 测试集挂载完成! 共计 {len(test_dataset)} 条数据。")

# --- 3. 初始化模型并加载权重 ---
# 使用多示例学习模型，通道数设为 12 
model = MobileSensorMIL_28Feat(in_channels=12, window_size=50, step_size=25, num_classes=1).to(device)

# 加载权重
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))

# 开启评估模式，关闭 Dropout 和 BatchNorm 的动态更新
model.eval() 
print(f"✅ 模型权重 [{MODEL_WEIGHTS_PATH}] 加载成功！")

# --- 4. 开启推理循环 ---
print("\n🚀 开始执行全量推理...")
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y, _ in test_loader:
        batch_x = batch_x.to(device)
        
        # 【核心修改】MIL 模型的 forward 返回三个值：logits, attention_weights, num_instances
        outputs, attn_weights, num_instances = model(batch_x)
        
        # 逻辑判定：Logits > 0 等价于 Sigmoid 概率 > 0.5
        preds_numpy = (outputs > 0).cpu().numpy().astype(int).flatten()
        labels_numpy = (batch_y > 0.5).cpu().numpy().astype(int).flatten()
        
        # 将当前 batch 的结果汇总到全局列表中
        all_preds.extend(preds_numpy)
        all_labels.extend(labels_numpy)

# --- 5. 计算混淆矩阵与核心指标 ---
print("\n" + "="*40)
print("📊 测试集最终评估报告")
print("="*40)

cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

print(f"▶ 准确率 (Accuracy) : {accuracy * 100:.2f}%")
print(f"▶ 精确率 (Precision): {precision * 100:.2f}% (误报控制)")
print(f"▶ 召回率 (Recall)   : {recall * 100:.2f}% (漏报控制)")
print(f"▶ F1-Score         : {f1_score * 100:.2f}%")

print("\n▶ 混淆矩阵 (Confusion Matrix):")
print(f"                预测 正常(0)    预测 异常(1)")
print(f"实际 正常(0)  |  TN: {tn:<8} |  FP: {fp:<8} | (误报)")
print(f"实际 异常(1)  |  FN: {fn:<8} |  TP: {tp:<8} |")
print(f"                (漏报)")

print("\n▶ 详细分类报告 (Classification Report):")
target_names = ['Normal (0)', 'Abnormal (1)']
print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))