import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, entropy
from scipy.signal import welch, find_peaks
import pywt  # 新增：用于离散小波变换

# ------------------------------------------------------------
# 归一化函数
# ------------------------------------------------------------
def normalize_data(folder_path, output_folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            df = df.iloc[:, 2:]
            n_rows = df.shape[0]
            for col in df.columns[0:]:
                norm = np.linalg.norm(df[col], ord=2)
                if norm == 0:
                    continue
                df[col] = df[col].multiply(n_rows) / norm
            output_file_path = os.path.join(output_folder_path, file)
            df.to_csv(output_file_path, index=False)

# 初始化归一化代码（如果不需要每次运行都归一化，可注释掉这部分）
base_folder_path = "F:\\datamining\\sensor_data260624"
output_folder_path = "F:\\datamining\\sensor_data260624\\normalize_feature"

os.makedirs(output_folder_path, exist_ok=True)
subfolders = ['normal', 'abnormal']
for subfolder in subfolders:
    os.makedirs(os.path.join(output_folder_path, subfolder), exist_ok=True)
for subfolder in subfolders:
    folder_path = os.path.join(base_folder_path, subfolder)
    out_folder_path = os.path.join(output_folder_path, subfolder)
    normalize_data(folder_path, out_folder_path)
print("归一化完成")

# ------------------------------------------------------------
# 时域统计特征
# ------------------------------------------------------------
def compute_statistical_features(series):
    rms = np.sqrt(np.mean(np.square(series))) if not series.empty else 0
    mad = (series - series.mean()).abs().mean() if not series.empty else 0
    waveform_factor = rms / mad if mad != 0 else 0
    impulse_factor = series.max() / mad if mad != 0 else 0
    kurtosis_factor = kurtosis(series) / rms if rms != 0 else 0
    coeff_of_variation = series.std() / series.mean() if series.mean() != 0 else 0
    features = {
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
    for key, value in features.items():
        if pd.isna(value):
            features[key] = 0.0
    return features

# ------------------------------------------------------------
# 新增时域：四分位数与四分位距 (IQR)
# ------------------------------------------------------------
def compute_percentiles(series):
    if series.empty:
        return {'q1': 0, 'q3': 0, 'iqr': 0}
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return {'q1': q1, 'q3': q3, 'iqr': q3 - q1}

# ------------------------------------------------------------
# 新增时域：均值穿越率 (MCR)
# ------------------------------------------------------------
def compute_mean_crossing_rate(series):
    if series.empty:
        return {'mean_crossing_rate': 0}
    mean_val = series.mean()
    mean_crossings = np.where(np.diff(np.sign(series.fillna(0) - mean_val)))[0]
    return {'mean_crossing_rate': len(mean_crossings) / len(series)}

# ------------------------------------------------------------
# 新增时域：Teager-Kaiser 能量算子 (TKEO)
# ------------------------------------------------------------
def compute_tkeo_features(series):
    series_np = series.fillna(0).to_numpy()
    if len(series_np) < 3:
        return {'tkeo_mean': 0, 'tkeo_std': 0}
    tkeo = series_np[1:-1]**2 - series_np[2:] * series_np[:-2]
    return {'tkeo_mean': np.mean(tkeo), 'tkeo_std': np.std(tkeo)}

# ------------------------------------------------------------
# 基础频域、熵与信号学特征
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

def compute_entropy(series, bins=50):
    if series.empty:
        return {'entropy': 0}
    counts, _ = np.histogram(series.fillna(0), bins=bins)
    probabilities = counts / np.sum(counts)
    probabilities = probabilities[probabilities > 0]
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return {'entropy': entropy_value}

def compute_zero_crossing_rate(series):
    if series.empty:
        return {'zero_crossing_rate': 0}
    zero_crossings = np.where(np.diff(np.sign(series.fillna(0))))[0]
    zero_crossings_rate = len(zero_crossings) / len(series)
    return {'zero_crossings_rate': zero_crossings_rate}

def compute_autocorrelation(series):
    if series.empty or series.nunique() == 1:
        return {'autocorrelation': 0}
    autocorrelation = series.autocorr()
    if pd.isna(autocorrelation):
        return {'autocorrelation': 0}
    else:
        return {'autocorrelation': autocorrelation}

def compute_signal_to_noise_ratio(series):
    if series.empty:
        return {'snr': 0}
    mean = series.mean()
    std = series.std()
    snr = mean / std if std != 0 else 0
    return {'snr': snr}

def compute_spectral_features(series, sample_rate=100):
    if series.empty:
        return {'peak_frequency': 0, 'band_energy': 0}
    series_np = series.fillna(0).to_numpy()
    yf = rfft(series_np)
    xf = rfftfreq(len(series_np), 1 / sample_rate)
    peak_frequency = xf[np.argmax(np.abs(yf))]
    band_energy = np.sum(np.square(np.abs(yf)))
    return {'peak_frequency': peak_frequency, 'band_energy': band_energy}

def compute_crest_factor(series):
    if series.empty:
        return {'crest_factor': 0}
    peak = series.max()
    rms = np.sqrt(np.mean(np.square(series)))
    crest_factor = peak / rms if rms != 0 else 0
    return {'crest_factor': crest_factor}

def compute_shape_factor(series):
    if series.empty:
        return {'shape_factor': 0}
    rms = np.sqrt(np.mean(np.square(series)))
    mean_abs = series.abs().mean()
    shape_factor = rms / mean_abs if mean_abs != 0 else 0
    return {'shape_factor': shape_factor}

def compute_additional_features(series):
    if series.empty:
        return {'cumulative_energy': 0, 'peak_interval': 0}
    cumulative_energy_result = np.cumsum(np.square(series.fillna(0))).iloc[-1]
    peaks, _ = find_peaks(series.fillna(0))
    intervals = np.diff(peaks)
    peak_interval = np.mean(intervals) if len(intervals) > 0 else 0
    return {'cumulative_energy': cumulative_energy_result, 'peak_interval': peak_interval}

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
    return {'hjorth_activity': activity, 'hjorth_mobility': mobility, 'hjorth_complexity': complexity}

# ------------------------------------------------------------
# 新增频域：频谱质心与扩散度
# ------------------------------------------------------------
def compute_advanced_spectral(series, sample_rate=100):
    if series.empty:
        return {'spectral_centroid': 0, 'spectral_spread': 0}
    series_np = series.fillna(0).to_numpy()
    yf = np.abs(rfft(series_np))
    xf = rfftfreq(len(series_np), 1 / sample_rate)
    
    sum_yf = np.sum(yf)
    if sum_yf == 0:
        return {'spectral_centroid': 0, 'spectral_spread': 0}
        
    spectral_centroid = np.sum(xf * yf) / sum_yf
    spectral_spread = np.sqrt(np.sum(((xf - spectral_centroid)**2) * yf) / sum_yf)
    
    return {'spectral_centroid': spectral_centroid, 'spectral_spread': spectral_spread}

# ------------------------------------------------------------
# 新增：离散小波变换能量特征 (DWT)
# ------------------------------------------------------------
def compute_wavelet_energy(series, wavelet='db4', level=3):
    keys = [f'wavelet_energy_{i}' for i in range(level + 1)]
    if series.empty or len(series) < 2**level:
        return {k: 0.0 for k in keys}
    try:
        coeffs = pywt.wavedec(series.fillna(0).to_numpy(), wavelet, level=level)
        return {f'wavelet_energy_{i}': np.sum(np.square(c)) for i, c in enumerate(coeffs)}
    except ValueError:
        return {k: 0.0 for k in keys}

# ------------------------------------------------------------
# 主处理循环：特征提取核心
# ------------------------------------------------------------
def process_folder(folder_path, label):
    all_file_features = []
    eps = 1e-10  # 防止除以零的极小值
    
    if not os.path.exists(folder_path):
        print(f"路径不存在，跳过: {folder_path}")
        return pd.DataFrame()
        
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            
            file_features = {'label': label}
            
            if df.shape[1] >= 2:
                file_features['user_id_1'] = df.iloc[0, 0]
                file_features['user_id_2'] = df.iloc[0, 1]
                
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # ==== 空间/多轴特征 ====
            # 1. 合向量 (Magnitude)
            try:
                if all(c in df.columns for c in ['acc_x', 'acc_y', 'acc_z']):
                    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2 + eps)
                if all(c in df.columns for c in ['gyro_alpha', 'gyro_beta', 'gyro_gamma']):
                    df['gyro_mag'] = np.sqrt(df['gyro_alpha']**2 + df['gyro_beta']**2 + df['gyro_gamma']**2 + eps)
            except Exception as e:
                print(f"构建合向量警告: {e}")

            # 2. 交叉相关性 (Correlation)
            if all(c in df.columns for c in ['acc_x', 'acc_y', 'acc_z']):
                file_features['corr_acc_xy'] = df['acc_x'].corr(df['acc_y'])
                file_features['corr_acc_xz'] = df['acc_x'].corr(df['acc_z'])
                file_features['corr_acc_yz'] = df['acc_y'].corr(df['acc_z'])
                
                # ==== 新增空间：信号幅度面积 (SMA) 与 姿态角 ====
                file_features['acc_sma'] = (df['acc_x'].abs() + df['acc_y'].abs() + df['acc_z'].abs()).mean()
                mean_x, mean_y, mean_z = df['acc_x'].mean(), df['acc_y'].mean(), df['acc_z'].mean()
                file_features['pitch'] = np.arctan2(mean_y, np.sqrt(mean_x**2 + mean_z**2 + eps))
                file_features['roll'] = np.arctan2(-mean_x, mean_z + eps)
                
            if all(c in df.columns for c in ['gyro_alpha', 'gyro_beta', 'gyro_gamma']):
                file_features['corr_gyro_xy'] = df['gyro_alpha'].corr(df['gyro_beta'])
                file_features['corr_gyro_xz'] = df['gyro_alpha'].corr(df['gyro_gamma'])
                file_features['corr_gyro_yz'] = df['gyro_beta'].corr(df['gyro_gamma'])
            
            # ==== 提取各轴单列的深入信号特征 ====
            valid_sensor_columns = [
                'acc_x', 'acc_y', 'acc_z', 
                'gyro_alpha', 'gyro_beta', 'gyro_gamma', 
                'acc_mag', 'gyro_mag'
            ]
            target_columns = [col for col in df.columns if col in valid_sensor_columns]
            
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
                
                # 加入新增的列级别特征
                percentile_features = compute_percentiles(df[column])
                mcr_features = compute_mean_crossing_rate(df[column])
                tkeo_features = compute_tkeo_features(df[column])
                adv_spectral_features = compute_advanced_spectral(df[column])
                wavelet_features = compute_wavelet_energy(df[column])
                
                # 运动跃度 (Jerk)
                jerk = np.diff(df[column].fillna(0).to_numpy())
                file_features[f"{column}_jerk_mean_abs"] = np.abs(jerk).mean() if len(jerk) > 0 else 0
                file_features[f"{column}_jerk_std"] = np.std(jerk) if len(jerk) > 0 else 0

                # 将所有返回的特征字典平铺合并到 file_features
                feature_dicts = [
                    stats_features, fft_features, entropy_features, zcr_features, 
                    autocorr_features, snr_features, spectral_features, crest_factor_features, 
                    shape_factor_features, additional_features, hjorth_features,
                    percentile_features, mcr_features, tkeo_features, adv_spectral_features, wavelet_features
                ]
                
                for f_dict in feature_dicts:
                    for feature_name, feature_value in f_dict.items():
                        file_features[f"{column}_{feature_name}"] = feature_value
            
            # 最后防线：确保产生 NaN 的地方都被 0 填充
            for key, value in file_features.items():
                if pd.isna(value):
                    file_features[key] = 0.0
                    
            all_file_features.append(file_features)
            
    print(f"文件夹 {folder_path} 特征提取完成")
    return pd.DataFrame(all_file_features)


# ------------------------------------------------------------
# 执行流程
# ------------------------------------------------------------
normal_folder_path = 'F:\\datamining\\sensor_data260624\\normalize_feature\\normal'
abnormal_folder_path = 'F:\\datamining\\sensor_data260624\\normalize_feature\\abnormal'
output_folder_path = 'F:\\datamining\\sensor_data260624\\normalize_feature\\feature'

os.makedirs(output_folder_path, exist_ok=True)

normal_features = process_folder(normal_folder_path, 0)
abnormal_features = process_folder(abnormal_folder_path, 1)

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