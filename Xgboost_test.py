import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import xgboost as xgb

# ==========================================
# 1. 加载保存的预处理组件和模型
# ==========================================

# 加载标准化器（如果训练时使用了 StandardScaler 并保存了）
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

# 📝 请将下方路径替换为你实际测试集文件所在的目录
test_data_path = "其他目录/your_real_test_data.csv" 
test_data = pd.read_csv(test_data_path)

# 确保测试集的特征列结构（顺序、数量）与训练集完全一致
# 假设第一列为标签，其余为特征
y_test = test_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
print(f"📊 测试集样本量: {X_test.shape[0]}, 特征数: {X_test.shape[1]}")

# 🚨 关键步骤：使用训练集训练好的 scaler 对测试集进行转换
# 绝对不要使用 fit_transform()
if scaler is not None:
    X_test_scaled = scaler.transform(X_test)
    print("⚡ 测试集特征已完成标准化缩放。")
else:
    X_test_scaled = X_test


# ==========================================
# 3. 执行预测与评估
# ==========================================

# 预测类别结果 (0 或 1)
y_pred = model.predict(X_test_scaled)

# 预测概率结果（可选：如果后续需要画 ROC 曲线或计算 AUC 可以使用）
# y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 自动识别是二分类还是多分类，动态调整评估参数
num_classes = len(np.unique(y_test))
avg_method = 'binary' if num_classes == 2 else 'macro'

# 计算核心指标
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average=avg_method)
rec = recall_score(y_test, y_pred, average=avg_method)

print("\n================ 独立测试集最终评估结果 ================")
print(f"测试集准确率 (Accuracy) : {acc:.4f}")
print(f"测试集精确率 (Precision): {prec:.4f} (模式: {avg_method})")
print(f"测试集召回率 (Recall)   : {rec:.4f} (模式: {avg_method})")
print("========================================================")

# 打印更详细的分类报告（包含各个类别的 F1-score）
print("\n详细分类报告:")
print(classification_report(y_test, y_pred))