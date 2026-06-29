import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fft import rfft,rfftfreq
from scipy.stats import kurtosis
from scipy.signal import welch
from scipy.stats import entropy
from scipy.signal import find_peaks

# ------------------------------------------------------------
# 归一化函数
# ------------------------------------------------------------
def normalize_data(folder_path,output_folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path=os.path.join(folder_path,file)
            df=pd.read_csv(file_path)
            df=df.iloc[:,2:]
            n_rows=df.shape[0]
            for col in df.columns[0:]:
                norm = np.linalg.norm(df[col],ord=2)
                #避免除以0
                if norm ==0:
                    continue
                df[col]=df[col].multiply(n_rows)/norm
            output_file_path=os.path.join(output_folder_path,file)
            df.to_csv(output_file_path,index=False)

# 初始化归一化代码（如果不需要每次运行都归一化，可注释掉这部分）
base_folder_path="F:\\datamining\\sensor_data260618"
output_folder_path="F:\\datamining\\sensor_data260618\\normalize_feature"

os.makedirs(output_folder_path,exist_ok=True)
subfolders=['normal','abnormal']
for subfolder in subfolders:
    os.makedirs(os.path.join(output_folder_path,subfolder),exist_ok=True)
for subfolder in subfolders:
    folder_path=os.path.join(base_folder_path,subfolder)
    out_folder_path=os.path.join(output_folder_path,subfolder)
    normalize_data(folder_path,out_folder_path)
print("归一化完成")

# ------------------------------------------------------------
# 统计特征
# ------------------------------------------------------------
def compute_statistical_features(series):
    rms = np.sqrt(np.mean(np.square(series))) if not series.empty else 0
    mad = (series - series.mean()).abs().mean() if not series.empty else 0
    waveform_factor = rms / mad if mad != 0 else 0
    impulse_factor = series.max() / mad if mad != 0 else 0
    kurtosis_factor = kurtosis(series) / rms if rms != 0 else 0
    coeff_of_variation = series.std() / series.mean() if series.mean() != 0 else 0
    features= {
        'mean': series.mean(),
        'std': series.std(),
        'max': series.max(),
        'min': series.min(),
        'variance': series.var(),
        'skewness': series.skew(),
        'kurtosis': series.kurtosis(),
        'median': series.median(),
        'range': series.max() - series.min(),
        'energy': np.sum(np.square(series)),
        'rms': rms,
        'mad': mad,
        'waveform_factor': waveform_factor,
        'impulse_factor': impulse_factor,
        'kurtosis_factor': kurtosis_factor,
        'coeff_of_variation': coeff_of_variation
    }
    # 终极防线：遍历字典，把一切形式的 NaN / Null 全部替换为 0.0
    for key, value in features.items():
        if pd.isna(value):
            features[key] = 0.0
    return features

# ------------------------------------------------------------
# 谱熵
# ------------------------------------------------------------
def spectral_entropy(signal, sample_rate=100):
    if signal.empty:
        return 0
    freqs, psd = welch(signal.fillna(0), fs=sample_rate)
    psd_norm = psd / np.sum(psd)
    return entropy(psd_norm) if np.sum(psd) != 0 else 0

def spectral_flatness(psd):
    if len(psd) == 0:
        return 0
    geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
    arithmetic_mean = np.mean(psd)
    return geometric_mean / (arithmetic_mean + 1e-10)

# ------------------------------------------------------------
# 谐波特征
# ------------------------------------------------------------
def harmonic_features(signal, sample_rate=100):
    if signal.empty:
        return 0
    peaks, _ = find_peaks(signal.fillna(0))
    if len(peaks) == 0:
        return 0
    fundamental_freq = np.argmax(signal[peaks])
    harmonics = signal[peaks] / signal[fundamental_freq]
    return np.sum(harmonics)

# ------------------------------------------------------------
# 累积能量
# ------------------------------------------------------------
def cumulative_energy(signal):
    if signal.empty:
        return 0
    cumsum = np.cumsum(np.square(signal.fillna(0)))
    return cumsum[-1]

# ------------------------------------------------------------
# 峰值间隔
# ------------------------------------------------------------
def peak_interval(signal):
    if signal.empty:
        return 0
    peaks, _ = find_peaks(signal.fillna(0))
    intervals = np.diff(peaks)
    return np.mean(intervals) if len(intervals) > 0 else 0

# ------------------------------------------------------------
# FFT 特征
# ------------------------------------------------------------
def compute_fft_features(series, sample_rate=100):
    if series.empty:
        return {'fft_mean': 0}
    series_np = series.fillna(0).to_numpy()
    yf = rfft(series_np)
    xf = rfftfreq(len(series_np), 1 / sample_rate)
    indices = np.where((xf >= 0) & (xf <= 0.1))[0]
    mean_fft = np.abs(yf[indices]).mean()
    return {'fft_mean': mean_fft}

# ------------------------------------------------------------
# 信息熵（已保留正确的分箱计算版本）
# ------------------------------------------------------------
def compute_entropy(series, bins=50):
    if series.empty:
        return {'entropy': 0}
    counts, _ = np.histogram(series.fillna(0), bins=bins)
    probabilities = counts / np.sum(counts)
    probabilities = probabilities[probabilities > 0]
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return {'entropy': entropy_value}

# ------------------------------------------------------------
# 过零率
# ------------------------------------------------------------
def compute_zero_crossing_rate(series):
    if series.empty:
        return {'zero_crossing_rate': 0}
    zero_crossings = np.where(np.diff(np.sign(series.fillna(0))))[0]
    zero_crossings_rate = len(zero_crossings) / len(series)
    return {'zero_crossings_rate': zero_crossings_rate}

# ------------------------------------------------------------
# 自相关
# ------------------------------------------------------------
def compute_autocorrelation(series):
    if series.empty or series.nunique() == 1:
        return {'autocorrelation': 0}
    autocorrelation = series.autocorr()
    if pd.isna(autocorrelation):
        return {'autocorrelation': 0}
    else:
        return {'autocorrelation': autocorrelation}

# ------------------------------------------------------------
# 信噪比
# ------------------------------------------------------------
def compute_signal_to_noise_ratio(series):
    if series.empty:
        return {'snr': 0}
    mean = series.mean()
    std = series.std()
    snr = mean / std if std != 0 else 0
    return {'snr': snr}

# ------------------------------------------------------------
# 频谱特征
# ------------------------------------------------------------
def compute_spectral_features(series, sample_rate=100):
    if series.empty:
        return {'peak_frequency': 0, 'band_energy': 0}
    series_np = series.fillna(0).to_numpy()
    yf = rfft(series_np)
    xf = rfftfreq(len(series_np), 1 / sample_rate)
    peak_frequency = xf[np.argmax(np.abs(yf))]
    band_energy = np.sum(np.square(np.abs(yf)))
    return {'peak_frequency': peak_frequency, 'band_energy': band_energy}

# ------------------------------------------------------------
# 波峰因子
# ------------------------------------------------------------
def compute_crest_factor(series):
    if series.empty:
        return {'crest_factor': 0}
    peak = series.max()
    rms = np.sqrt(np.mean(np.square(series)))
    crest_factor = peak / rms if rms != 0 else 0
    return {'crest_factor': crest_factor}

# ------------------------------------------------------------
# 形状因子
# ------------------------------------------------------------
def compute_shape_factor(series):
    if series.empty:
        return {'shape_factor': 0}
    rms = np.sqrt(np.mean(np.square(series)))
    mean_abs = series.abs().mean()
    shape_factor = rms / mean_abs if mean_abs != 0 else 0
    return {'shape_factor': shape_factor}

# ------------------------------------------------------------
# 附加特征
# ------------------------------------------------------------
def compute_additional_features(series):
    if series.empty:
        return {'cumulative_energy': 0, 'peak_interval': 0}
    cumulative_energy_result = np.cumsum(np.square(series.fillna(0))).iloc[-1]
    peaks, _ = find_peaks(series.fillna(0))
    intervals = np.diff(peaks)
    peak_interval = np.mean(intervals) if len(intervals) > 0 else 0
    additional_features = {
        'cumulative_energy': cumulative_energy_result,
        'peak_interval': peak_interval
    }
    return additional_features

# ------------------------------------------------------------
# Hjorth 参数 (新增：衡量序列复杂度，针对受骗紧张状态)
# ------------------------------------------------------------
def compute_hjorth_parameters(series):
    series_np = series.fillna(0).to_numpy()
    if len(series_np) < 3:
        return {'hjorth_activity': 0, 'hjorth_mobility': 0, 'hjorth_complexity': 0}
        
    activity = np.var(series_np)
    diff1 = np.diff(series_np)
    var_diff1 = np.var(diff1)
    mobility = np.sqrt(var_diff1 / activity) if activity > 0 else 0
    
    diff2 = np.diff(diff1)
    var_diff2 = np.var(diff2)
    mobility_diff = np.sqrt(var_diff2 / var_diff1) if var_diff1 > 0 else 0
    complexity = mobility_diff / mobility if mobility > 0 else 0
    
    return {
        'hjorth_activity': activity,
        'hjorth_mobility': mobility,
        'hjorth_complexity': complexity
    }

# ------------------------------------------------------------
# 主处理循环：特征提取核心
# ------------------------------------------------------------
def process_folder(folder_path, label):
    all_file_features = []
    eps = 1e-10  # 定义极小值，防止在全0的情况下根号或后续除法计算出错
    
    if not os.path.exists(folder_path):
        print(f"路径不存在，跳过: {folder_path}")
        return pd.DataFrame()
        
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            
            # 基础标签初始化
            file_features = {'label': label}
            
            # 【保留用户ID】提取前两列的用户标识信息
            if df.shape[1] >= 2:
                file_features['user_id_1'] = df.iloc[0, 0]
                file_features['user_id_2'] = df.iloc[0, 1]
                
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # ==== 新增 1：构建合向量 (Magnitude) 融合特征 ====
            try:
                # 检查是否存在加速度列，存在则计算合加速度
                if all(c in df.columns for c in ['acc_x', 'acc_y', 'acc_z']):
                    df['acc_mag'] = np.sqrt(df['acc_x']**2 + 
                                            df['acc_y']**2 + 
                                            df['acc_z']**2 + eps)
                # 检查是否存在陀螺仪列，存在则计算合角速度 (注意匹配你的拼写: rotaion_rate)
                if all(c in df.columns for c in ['gyro_alpha', 'gyro_beta', 'gyro_gamma']):
                    df['gyro_mag'] = np.sqrt(df['gyro_alpha']**2 + 
                                             df['gyro_beta']**2 + 
                                             df['gyro_gamma']**2 + eps)
            except Exception as e:
                print(f"构建合向量警告: {e}")

            # ==== 新增 2：跨轴交叉相关性特征 ====
            # 全部使用显式列名，彻底摒弃 iloc，保证万无一失
            if all(c in df.columns for c in ['acc_x', 'acc_y', 'acc_z']):
                file_features['corr_acc_xy'] = df['acc_x'].corr(df['acc_y'])
                file_features['corr_acc_xz'] = df['acc_x'].corr(df['acc_z'])
                file_features['corr_acc_yz'] = df['acc_y'].corr(df['acc_z'])
                
            if all(c in df.columns for c in ['gyro_alpha', 'gyro_beta', 'gyro_gamma']):
                file_features['corr_gyro_xy'] = df['gyro_alpha'].corr(df['gyro_beta'])
                file_features['corr_gyro_xz'] = df['gyro_alpha'].corr(df['gyro_gamma'])
                file_features['corr_gyro_yz'] = df['gyro_beta'].corr(df['gyro_gamma'])
            
            # ==== 新增 3：精准过滤真正的特征列 ====
            # 只对这些明确的传感器列（以及新计算的 mag 列）提取几十种信号学特征，避开 ID 列
            valid_sensor_columns = [
                'acc_x', 'acc_y', 'acc_z', 
                'gyro_alpha', 'gyro_beta', 'gyro_gamma', 
                'acc_mag', 'gyro_mag'
            ]
            target_columns = [col for col in df.columns if col in valid_sensor_columns]
            
            # 循环遍历每一列进行特征提取
            for column in target_columns:
                stats_features = compute_statistical_features(df[column])
                fft_features = compute_fft_features(df[column])
                entropy_features = compute_entropy(df[column])
                zcr_features = compute_zero_crossing_rate(df[column])
                autocorr_features = compute_autocorrelation(df[column])
                snr_features = compute_signal_to_noise_ratio(df[column])
                spectral_features = compute_spectral_features(df[column])
                crest_factor_features = compute_crest_factor(df[column])
                shape_factor_features = compute_shape_factor(df[column])
                additional_features = compute_additional_features(df[column])
                hjorth_features = compute_hjorth_parameters(df[column]) 
                
                # 运动跃度 (Jerk)
                jerk = np.diff(df[column].fillna(0).to_numpy())
                file_features[f"{column}_jerk_mean_abs"] = np.abs(jerk).mean() if len(jerk) > 0 else 0
                file_features[f"{column}_jerk_std"] = np.std(jerk) if len(jerk) > 0 else 0

                # 将所有返回的特征字典平铺合并到 file_features
                feature_dicts = [
                    stats_features, fft_features, entropy_features, zcr_features, 
                    autocorr_features, snr_features, spectral_features, crest_factor_features, 
                    shape_factor_features, additional_features, hjorth_features
                ]
                
                for f_dict in feature_dicts:
                    for feature_name, feature_value in f_dict.items():
                        file_features[f"{column}_{feature_name}"] = feature_value
            
            # 最后防线：确保相关性计算等产生任何 NaN 的地方都被 0 填充
            for key, value in file_features.items():
                if pd.isna(value):
                    file_features[key] = 0.0
                    
            all_file_features.append(file_features)
            
    print(f"文件夹 {folder_path} 特征提取完成")
    return pd.DataFrame(all_file_features)


# ------------------------------------------------------------
# 执行流程
# ------------------------------------------------------------
# 注意填写实际存在的文件夹路径
normal_folder_path='F:\\datamining\\sensor_data260618\\normalize_feature\\normal'
abnormal_folder_path='F:\\datamining\\sensor_data260618\\normalize_feature\\abnormal'
output_folder_path='F:\\datamining\\sensor_data260618\\normalize_feature\\feature'

os.makedirs(output_folder_path, exist_ok=True)

normal_features = process_folder(normal_folder_path, 0)
abnormal_features = process_folder(abnormal_folder_path, 1)

# 防止某一个文件夹为空导致合并报错
frames_to_concat = []
if not normal_features.empty: frames_to_concat.append(normal_features)
if not abnormal_features.empty: frames_to_concat.append(abnormal_features)

if frames_to_concat:
    merged_features = pd.concat(frames_to_concat, ignore_index=True)
    merged_features.dropna(inplace=True)
    
    output_file_path = os.path.join(output_folder_path, "combined_features.csv")
    merged_features.to_csv(output_file_path, index=False)
    print(f"总特征提取合并完成，已保存至: {output_file_path}")
else:
    print("没有提取到任何特征，请检查输入目录是否存在数据文件！")