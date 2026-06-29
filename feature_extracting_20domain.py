import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ==========================================
# 1. 特征衍生核心逻辑 (已扩充增强)
# ==========================================
def compute_6dof_features(df, window_size=5):
    """
    针对 6轴 IMU 数据 (加速度 + 陀螺仪) 的特征派生增强版
    """
    # 为了避免除零和极小值问题，添加一个极小的 epsilon
    eps = 1e-8

    # --- 基础幅值特征 (物理能量) ---
    df['acc_mag'] = np.sqrt(df['acceleration_x']**2 + 
                            df['acceleration_y']**2 + 
                            df['acceleration_z']**2 + eps)
    df['gyro_mag'] = np.sqrt(df['rotaion_rate_alpha']**2 + 
                             df['rotaion_rate_beta']**2 + 
                             df['rotaion_rate_gamma']**2 + eps)

    # --- 1. Jerk 特征 (加加速度 / 突变度) ---
    # 物理意义：加速度的导数。当设备受到撞击或剧烈改变方向时，Jerk 会激增。对异常检测极度敏感。
    df['jerk_x'] = df['acceleration_x'].diff().fillna(0)
    df['jerk_y'] = df['acceleration_y'].diff().fillna(0)
    df['jerk_z'] = df['acceleration_z'].diff().fillna(0)
    df['jerk_mag'] = np.sqrt(df['jerk_x']**2 + df['jerk_y']**2 + df['jerk_z']**2 + eps)

    # 陀螺仪的导数 (角加速度)
    df['ang_acc_alpha'] = df['rotaion_rate_alpha'].diff().fillna(0)
    df['ang_acc_beta'] = df['rotaion_rate_beta'].diff().fillna(0)
    df['ang_acc_gamma'] = df['rotaion_rate_gamma'].diff().fillna(0)
    df['ang_acc_mag'] = np.sqrt(df['ang_acc_alpha']**2 + df['ang_acc_beta']**2 + df['ang_acc_gamma']**2 + eps)

    # --- 2. SMA 特征 (信号幅值面积 Signal Magnitude Area) ---
    # 物理意义：衡量一段时间内的总体运动强度，是行为识别中区分“静止”与“运动”的核心指标。
    df['acc_sma'] = df['acceleration_x'].abs() + df['acceleration_y'].abs() + df['acceleration_z'].abs()
    df['gyro_sma'] = df['rotaion_rate_alpha'].abs() + df['rotaion_rate_beta'].abs() + df['rotaion_rate_gamma'].abs()

    # --- 3. 姿态角近似特征 (Pitch & Roll) ---
    # 物理意义：利用重力分量估算设备的倾斜状态。对于摔倒、掉落或握持姿势异常的识别非常有帮助。
    # 俯仰角 (Pitch) 和 翻滚角 (Roll)
    df['pitch_approx'] = np.arctan2(df['acceleration_x'], np.sqrt(df['acceleration_y']**2 + df['acceleration_z']**2 + eps))
    df['roll_approx'] = np.arctan2(df['acceleration_y'], df['acceleration_z'] + eps)

    # --- 4. 一阶时序差分 (捕捉相对幅值的瞬间突变) ---
    df['delta_acc_mag'] = df['acc_mag'].diff().fillna(0)
    df['delta_gyro_mag'] = df['gyro_mag'].diff().fillna(0)

    # --- 5. 局部窗口统计特征 (捕捉持续性状态) ---
    acc_roll = df['acc_mag'].rolling(window=window_size, min_periods=1)
    gyro_roll = df['gyro_mag'].rolling(window=window_size, min_periods=1)

    # 方差 (波动程度)
    df['rolling_var_acc'] = acc_roll.var().fillna(0)
    df['rolling_var_gyro'] = gyro_roll.var().fillna(0)
    
    # 标准差 (与原始数据同量纲的波动度)
    df['rolling_std_acc'] = acc_roll.std().fillna(0)
    df['rolling_std_gyro'] = gyro_roll.std().fillna(0)

    # 峰峰值 / 极差 (Peak-to-Peak)
    # 物理意义：窗口内的最大值减去最小值。比方差更能捕捉到窗口内的“极端抖动”或“瞬间冲击”。
    df['rolling_ptp_acc'] = acc_roll.max() - acc_roll.min()
    df['rolling_ptp_gyro'] = gyro_roll.max() - gyro_roll.min()

    # 可以选择清理掉一些中间变量以减少显存占用（视你的模型复杂度而定）
    # df.drop(columns=['jerk_x', 'jerk_y', 'jerk_z', 'ang_acc_alpha', 'ang_acc_beta', 'ang_acc_gamma'], inplace=True)

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