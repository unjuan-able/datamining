import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import xgboost as xgb

# ==========================================
# 0. 定义特征抑制兜底函数 (类似 NMS)
# ==========================================
def feature_based_suppression(y_pred, feature_values, threshold, condition='greater'):
    """
    对模型预测为正类(1)的样本进行二次校验。
    """
    y_pred_new = y_pred.copy()
    positive_mask = (y_pred_new == 1)
    
    if condition == 'greater':
        rule_mask = (feature_values > threshold)
    elif condition == 'less':
        rule_mask = (feature_values < threshold)
    else:
        raise ValueError("condition 必须是 'greater' 或 'less'")
        
    suppress_mask = positive_mask & rule_mask
    suppress_count = np.sum(suppress_mask)
    
    if suppress_count > 0:
        y_pred_new[suppress_mask] = 0
        print(f"🛑 [后处理抑制触发] 根据规则 ({condition} {threshold})，共将 {suppress_count} 个疑似误报(1)强制修改为(0)。")
    else:
        print("✅ 没有样本触发抑制规则。")
        
    return y_pred_new

# ==========================================
# 1. 加载保存的预处理组件和模型
# ==========================================
try:
    scaler = joblib.load("scaler.pkl")
    print("✅ 成功加载标准化器: scaler.pkl")
except FileNotFoundError:
    print("⚠️ 未找到 scaler.pkl，请确保路径正确；若训练时未做标准化，可忽略此提示。")
    scaler = None

# 初始化空的 XGBoost 模型结构，并加载权重文件
model = xgb.XGBClassifier()
model.load_model("xgboost_model.json")
print("✅ 成功加载 XGBoost 模型: xgboost_model.json")

# ==========================================
# 2. 加载并预处理真正的测试集 (Test Data)
# ==========================================
test_data_path = "其他目录/your_real_test_data.csv" 
test_data = pd.read_csv(test_data_path)

y_test = test_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
print(f"📊 测试集样本量: {X_test.shape[0]}, 特征数: {X_test.shape[1]}")

if scaler is not None:
    X_test_scaled = scaler.transform(X_test)
    print("⚡ 测试集特征已完成标准化缩放。")
else:
    X_test_scaled = X_test

# ==========================================
# 3. 执行预测与后处理抑制
# ==========================================
# 原始预测结果
y_pred_raw = model.predict(X_test_scaled)

# 🚨 新增拦截逻辑：执行特征抑制
# 【注意】这里请替换为你实际要用来做限制的特征列名，例如 'gyro_gamma_entropy'
feature_col_name = '请替换为实际特征列名' 
threshold_value = 4.97  # 替换为你的阈值
rule_condition = 'greater'  # 'greater' 表示大于阈值抑制，'less' 表示小于阈值抑制

try:
    # 提取原始未缩放的特征用于规则判断
    feature_to_check = test_data[feature_col_name].values 
    
    y_pred = feature_based_suppression(
        y_pred=y_pred_raw, 
        feature_values=feature_to_check, 
        threshold=threshold_value, 
        condition=rule_condition
    )
except KeyError:
    print(f"⚠️ 未在数据集中找到列名 '{feature_col_name}'，已跳过特征抑制步骤，使用原始预测结果。")
    y_pred = y_pred_raw

# ==========================================
# 4. 执行最终评估
# ==========================================
num_classes = len(np.unique(y_test))
avg_method = 'binary' if num_classes == 2 else 'macro'

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average=avg_method)
rec = recall_score(y_test, y_pred, average=avg_method)

print("\n================ 独立测试集最终评估结果 ================")
print(f"测试集准确率 (Accuracy) : {acc:.4f}")
print(f"测试集精确率 (Precision): {prec:.4f} (模式: {avg_method})")
print(f"测试集召回率 (Recall)   : {rec:.4f} (模式: {avg_method})")
print("========================================================")

print("\n详细分类报告 (抑制后):")
print(classification_report(y_test, y_pred))