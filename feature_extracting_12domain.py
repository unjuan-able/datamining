import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ==========================================
# 1. 特征衍生核心逻辑
# ==========================================
def compute_6dof_features(df, window_size=5):
    """
    针对 6轴 IMU 数据 (加速度 + 陀螺仪) 的特征派生
    """
    # 计算空间总能量 (幅值)
    df['acc_mag'] = np.sqrt(df['acceleration_x']**2 + 
                            df['acceleration_y']**2 + 
                            df['acceleration_z']**2)
    df['gyro_mag'] = np.sqrt(df['rotaion_rate_alpha']**2 + 
                             df['rotaion_rate_beta']**2 + 
                             df['rotaion_rate_gamma']**2)
    
    # 计算一阶时序差分 (捕捉瞬间突变)
    df['delta_acc_mag'] = df['acc_mag'].diff().fillna(0)
    df['delta_gyro_mag'] = df['gyro_mag'].diff().fillna(0)
    
    # 计算滑动窗口局部特征 (捕捉持续的异常剧烈震颤)
    df['rolling_var_acc'] = df['acc_mag'].rolling(window=window_size, min_periods=1).var().fillna(0)
    df['rolling_var_gyro'] = df['gyro_mag'].rolling(window=window_size, min_periods=1).var().fillna(0)

    return df

# ==========================================
# 2. 文件夹遍历与批量处理
# ==========================================
def load_and_process_all_folders(data_root_dir, window_size=5):
    """
    遍历根目录下的所有子文件夹，读取所有的 call_id.csv，提取特征并自动根据文件夹打标签
    """
    search_pattern = os.path.join(data_root_dir, "**", "*.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    print(f"共扫描到 {len(csv_files)} 个 CSV 文件，开始进行自动标注与特征提取...")
    
    all_user_features = {}
    all_user_labels = {}  # 新增：用于存储从路径解析出的标签字典
    output_root_dir = os.path.join(os.path.dirname(data_root_dir), "processed_imu_data")
    
    for file_path in csv_files:
        call_id = os.path.splitext(os.path.basename(file_path))[0]
        
        # --- 核心新增：标签自动解析逻辑 ---
        # 统一路径分隔符为 '/'，并转为小写以防止大小写引发的 bug (如 Normal)
        normalized_path = file_path.replace('\\', '/').lower()
        
        # 严格判断所属文件夹（注意路径斜杠，防止因为外层包含同名文件夹而误判）
        if '/abnormal' in normalized_path:
            label = 1
            category_folder = 'abnormal'
        elif '/normal' in normalized_path:
            label = 0
            category_folder = 'normal'
        else:
            # 如果文件不在指定的两个文件夹结构内，抛出警告并跳过
            print(f"[警告] 跳过无法识别类别的路径: {file_path}")
            continue
        # 增加：构建保存路径并确保文件夹存在
        save_dir = os.path.join(output_root_dir, category_folder)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(file_path))
            
        try:
            df = pd.read_csv(file_path)
            
            if df.empty or len(df) < 2:
                continue
                
            # 执行上一部分的 6轴特征衍生
            df_enriched = compute_6dof_features(df, window_size=window_size)
            
            # 只有当数据成功处理后，才将特征和标签同时入库
            all_user_features[call_id] = df_enriched
            all_user_labels[call_id] = label
            
        except Exception as e:
            print(f"文件处理失败: {file_path}, 报错信息: {e}")
            
    print(f"成功载入 {len(all_user_features)} 个有效数据。其中异常(1): {list(all_user_labels.values()).count(1)} 个，正常(0): {list(all_user_labels.values()).count(0)} 个。")
    return all_user_features, all_user_labels

# ==========================================
# 3. PyTorch Dataset 封装
# ==========================================
class IMU6DofDataset(Dataset):
    def __init__(self, user_data_dict, labels_dict):
        """
        :param user_data_dict: 由 load_and_process_all_folders 返回的字典
        :param labels_dict: 包含 {call_id: 0或1} 的标签字典
        """
        self.call_ids = list(user_data_dict.keys())
        self.user_data = user_data_dict
        self.labels = labels_dict
        
        # 挑选送入模型的最终 12 个通道
        self.feature_cols = [
            'acceleration_x', 'acceleration_y', 'acceleration_z',
            'rotaion_rate_alpha', 'rotaion_rate_beta', 'rotaion_rate_gamma',
            'acc_mag', 'gyro_mag', 
            'delta_acc_mag', 'delta_gyro_mag',
            'rolling_var_acc', 'rolling_var_gyro'
        ]

    def __len__(self):
        return len(self.call_ids)

    def __getitem__(self, idx):
        cid = self.call_ids[idx]
        df = self.user_data[cid]
        
        # 提取特征并转换为 Float32 矩阵
        features = df[self.feature_cols].values.astype(np.float32)
        label = self.labels.get(cid, 0)
        
        # 转换为 ResNet1D 需要的维度顺序: [12, Seq_len]
        features_tensor = torch.tensor(features).transpose(0, 1) 
        
        return features_tensor, torch.tensor(label, dtype=torch.float32), cid

# ==========================================
# 4. 变长序列 Padding (Collate Function)
# ==========================================
def imu_collate_fn(batch):
    """
    处理 100~400 不等长时间步的核心函数，将其打包成规整的 Batch 张量
    """
    # 拆解 batch
    sequences, labels, call_ids = zip(*batch)
    
    # 记录每个序列在 padding 前的真实长度 (如果后续使用 LSTM 会用到，ResNet1D 不强制需要)
    lengths = torch.tensor([seq.shape[1] for seq in sequences])
    
    # 注意：此时每个 seq 的 shape 是 [12, seq_len]
    # pad_sequence 默认是在第一维 (seq_len) 进行填充，所以我们要先转置回来 [seq_len, 12]
    sequences_transposed = [seq.transpose(0, 1) for seq in sequences]
    
    # 自动进行末尾补 0 填充，得到 [Batch_size, Max_seq_len, 12]
    padded_seqs = pad_sequence(sequences_transposed, batch_first=True, padding_value=0.0)
    
    # 再次转置以适应 1D 卷积要求：[Batch_size, 12, Max_seq_len]
    padded_seqs = padded_seqs.transpose(1, 2)
    
    return padded_seqs, torch.stack(labels), lengths, call_ids


# 你的数据根目录，里面应该包含 /normal 和 /abnormal 两个文件夹
DATA_ROOT = "./your_imu_data_folder" 

# 现在的函数会直接返回带有自动标签的两个字典，不再需要你手动 mock 标签了
features_dict, labels_dict = load_and_process_all_folders(data_root_dir=DATA_ROOT)

if len(features_dict) > 0:
    # 将自动提取的字典直接传给 Dataset
    dataset = IMU6DofDataset(user_data_dict=features_dict, labels_dict=labels_dict)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=imu_collate_fn)
    
    for batch_x, batch_y, lengths, cids in dataloader:
        print(f"\n--- Batch 测试抓取 ---")
        print(f"X 张量维度: {batch_x.shape}") 
        print(f"Y 标签维度: {batch_y.shape}")
        print(f"前 5 个真实标签: {batch_y[:5].tolist()}")
        break