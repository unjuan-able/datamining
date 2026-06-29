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
# 归一化函数（修正：接受文件路径参数）
# ------------------------------------------------------------
def normalize_data(folder_path,output_subfolder):
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
            output_file_path=os.path.join(output_folder_path,output_subfolder,file)
            df.to_csv(output_file_path,index=False)


base_folder_path="F:\\datamining\\sensor_data"
output_folder_path="F:\\datamining\\sensor_data\\normalize_feature"

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
        'skewness': series.skew(),          # 修正拼写
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
    # 3. 终极防线：遍历字典，把一切形式的 NaN / Null 全部替换为 0.0
    for key, value in features.items():
        if pd.isna(value):
            features[key] = 0.0
    return features


# ------------------------------------------------------------
# 谱熵（修正拼写）
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
# 谐波特征（修正拼写）
# ------------------------------------------------------------
def harmonic_features(signal, sample_rate=100):
    if signal.empty:                     # 修正
        return 0
    peaks, _ = find_peaks(signal.fillna(0))
    if len(peaks) == 0:
        return 0
    fundamental_freq = np.argmax(signal[peaks])
    harmonics = signal[peaks] / signal[fundamental_freq]
    return np.sum(harmonics)


# ------------------------------------------------------------
# 累积能量（修正拼写 + ndarray取值）
# ------------------------------------------------------------
def cumulative_energy(signal):
    if signal.empty:                     # 修正
        return 0
    cumsum = np.cumsum(np.square(signal.fillna(0)))
    return cumsum[-1]                    # 修正（不用 .iloc）


# ------------------------------------------------------------
# 峰值间隔（修正拼写）
# ------------------------------------------------------------
def peak_interval(signal):
    if signal.empty:
        return 0
    peaks, _ = find_peaks(signal.fillna(0))   # 修正 fillna
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
# 信息熵（修正：加负号）
# ------------------------------------------------------------
def compute_entropy(series):
    if series.empty:
        return {'entropy': 0}
    probability_distribution = series.value_counts(normalize=True)
    entropy_value = -np.sum(probability_distribution * np.log2(probability_distribution))   # 加负号
    return {'entropy': entropy_value}

# ------------------------------------------------------------
# 信息熵（修正：分箱）
# ------------------------------------------------------------
def compute_entropy(series, bins=50):
    if series.empty:
        return {'entropy': 0}
    # 使用直方图对连续变量进行分箱计算概率
    counts, _ = np.histogram(series.fillna(0), bins=bins)
    probabilities = counts / np.sum(counts)
    probabilities = probabilities[probabilities > 0] # 避免 log(0)
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return {'entropy': entropy_value}


# ------------------------------------------------------------
# 过零率（修正拼写）
# ------------------------------------------------------------
def compute_zero_crossing_rate(series):
    if series.empty:
        return {'zero_crossing_rate': 0}
    zero_crossings = np.where(np.diff(np.sign(series.fillna(0))))[0]
    zero_crossings_rate = len(zero_crossings) / len(series)
    return {'zero_crossings_rate': zero_crossings_rate}   # 修正 return


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
# 附加特征（修正拼写 + ndarray取值）
# ------------------------------------------------------------
def compute_additional_features(series):
    if series.empty:
        return {'cumulative_energy': 0, 'peak_interval': 0}
    cumulative_energy_result = np.cumsum(np.square(series.fillna(0))).iloc[-1]   # 修正
    peaks, _ = find_peaks(series.fillna(0))                                 # 修正
    intervals = np.diff(peaks)
    peak_interval = np.mean(intervals) if len(intervals) > 0 else 0
    additional_features = {
        'cumulative_energy': cumulative_energy_result,
        'peak_interval': peak_interval
    }
    return additional_features


def process_folder(folder_path,label):
    all_file_features=[]
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path=os.path.join(folder_path,file)
            df=pd.read_csv(file_path)
            file_features={'label':label}
            df=df.loc[:,~df.columns.str.contains('^Unnamed')]
            file_features={'label':label}
            for column in df.columns[0:]:
                stats_features=compute_statistical_features(df[column])
                fft_features=compute_fft_features(df[column])
                entropy_features=compute_entropy(df[column])
                zcr_features=compute_zero_crossing_rate(df[column])
                autocorr_features=compute_autocorrelation(df[column])
                snr_features=compute_signal_to_noise_ratio(df[column])
                spectral_features=compute_spectral_features(df[column])
                crest_factor_features=compute_crest_factor(df[column])
                shape_factor_features=compute_shape_factor(df[column])
                additional_features=compute_additional_features(df[column])

                # ---- 补充：将所有特征依次添加到 file_features ----
                for feature_name, feature_value in stats_features.items():
                    file_features[f"{column}_{feature_name}"] = feature_value
                for feature_name, feature_value in fft_features.items():
                    file_features[f"{column}_{feature_name}"] = feature_value
                for feature_name, feature_value in entropy_features.items():
                    file_features[f"{column}_{feature_name}"] = feature_value
                for feature_name, feature_value in zcr_features.items():
                    file_features[f"{column}_{feature_name}"] = feature_value
                for feature_name, feature_value in autocorr_features.items():
                    file_features[f"{column}_{feature_name}"] = feature_value
                for feature_name, feature_value in snr_features.items():
                    file_features[f"{column}_{feature_name}"] = feature_value
                for feature_name, feature_value in spectral_features.items():
                    file_features[f"{column}_{feature_name}"] = feature_value
                for feature_name, feature_value in crest_factor_features.items():
                    file_features[f"{column}_{feature_name}"] = feature_value
                for feature_name, feature_value in shape_factor_features.items():
                    file_features[f"{column}_{feature_name}"] = feature_value
                for feature_name, feature_value in additional_features.items():
                    file_features[f"{column}_{feature_name}"] = feature_value
        
            all_file_features.append(file_features)
    print("特征提取完成")
    return pd.DataFrame(all_file_features)

normal_folder_path='F:\\datamining\\sensor_data\\normalize_feature\\normal'
abnormal_folder_path='F:\\datamining\\sensor_data\\normalize_feature\\abnormal'
output_folder_path='F:\\datamining\\sensor_data\\normalize_feature\\feature'

os.makedirs(output_folder_path,exist_ok=True)

normal_features=process_folder(normal_folder_path,0)
abnormal_features=process_folder(abnormal_folder_path,1)

merged_features=pd.concat([normal_features,abnormal_features],ignore_index=True)

merged_features.dropna(inplace=True)

output_file_path=os.path.join(output_folder_path,"combined_features.csv")
merged_features.to_csv(output_file_path,index=False)

print("特征提取完成")