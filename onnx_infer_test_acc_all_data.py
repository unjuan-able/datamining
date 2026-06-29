import onnxruntime as ort
import numpy as np
import pandas as pd
import torch
import os
import time
import joblib
from sklearn.preprocessing import StandardScaler
from normalize_feature_infer import normalize_data, process_folder

# ... (前面的 get_image_list 函数和路径配置保持不变) ...

ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
scaler = joblib.load('你的scaler路径.pkl')
csv_list1, csv_dir1 = get_image_list(folder_path)

THRES = 0.5
metric = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
start = time.time()

# ==========================================
# 阶段一：批量特征提取 (只提取，不推理)
# ==========================================
print("⏳ 开始批量提取特征...")
all_features = []      # 用于存放所有文件的特征字典
all_file_paths = []    # 用于同步记录对应的文件路径
all_labels = []        # 用于同步记录真实标签

for ind, file_path in enumerate(csv_list1):
    try:
        data = normalize_data(file_path)
        feature_list = process_folder(data)
        
        # process_folder 返回的是包含一个字典的列表，我们把字典取出来
        feature_dict = feature_list[0] 
        
        # 判定标签
        label = 1 if '/20260319_abnormal_data/' in file_path else 0
        
        all_features.append(feature_dict)
        all_file_paths.append(file_path)
        all_labels.append(label)
        
    except Exception as e:
        print(f"❌ 提取特征失败: {file_path}, 错误: {e}")
        continue

file_count = len(all_file_paths)
if file_count == 0:
    raise ValueError("没有成功提取到任何文件的特征，程序终止。")

print(f"✅ 特征提取完毕，成功处理 {file_count} 个文件。开始批量推理...")

# 1. 将特征字典列表直接转换为 DataFrame
features_df = pd.DataFrame(all_features)

# 2. 将文件路径和标签作为新列，插入到表格的最前面（索引 0 和 1 的位置）
features_df.insert(0, 'file_path', all_file_paths)
features_df.insert(1, 'label', all_labels)

# 3. （安全兜底）将可能因为数学计算产生的 NaN 填补为 0
features_df = features_df.fillna(0)

# 4. 保存为本地 CSV 文件
# 使用 utf-8-sig 编码，防止如果在 Windows 下用 Excel 打开时路径里的中文乱码
save_path = 'extracted_test_features.csv'
features_df.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"💾 所有特征已成功汇总并保存至: {save_path}")



print("⏳ 开始批量推理...")
# ==========================================
# 阶段二：一次性矩阵转换与批量推理
# ==========================================
# 1. 将所有特征字典拼接成一个巨大的 DataFrame，并用 0 填补可能漏网的 NaN
X_df = pd.DataFrame(all_features).fillna(0)
X_numpy = X_df.values

# 2. 一次性归一化所有数据 (严格使用 transform，避免数据泄露)
X_scaled = scaler.transform(X_numpy).astype(np.float32)

# 3. 将整个大矩阵一次性喂给 ONNX 模型
onnx_outputs = ort_session.run(None, {"input": X_scaled})[0]

# 4. 一次性计算所有样本的预测概率
outputs_tensor = torch.from_numpy(onnx_outputs)
y_pred_probs = torch.softmax(outputs_tensor, dim=1)
scores = y_pred_probs[:, 1].numpy()  # 获取所有样本的正类概率
final_preds = (scores > THRES).astype(int) # 一次性得出所有 0/1 预测结果

# ==========================================
# 阶段三：评估与结果写入
# ==========================================
print("⏳ 开始统计指标并写入文件...")
with open(output_txt, 'w', encoding='utf-8') as f:
    for i in range(file_count):
        file_path = all_file_paths[i]
        score = scores[i]
        final_pred = final_preds[i]
        label = all_labels[i]
        
        # 统计 TP, FP, TN, FN
        if label == 0:
            if final_pred != 0:
                metric["fp"] += 1
            else:
                metric["tn"] += 1
        else:
            if final_pred != 0:
                metric["tp"] += 1
            else:
                metric["fn"] += 1
                
        f.write(f"{file_path},{score},{final_pred},{label}\n")

# ... (后面的打印时间和计算 acc/recall 的代码保持不变) ...