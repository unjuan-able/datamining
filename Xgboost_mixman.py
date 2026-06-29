import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb

# 加载数据
data = pd.read_csv("your_data.csv")  # 替换为实际路径

# 第一列为标签，其余为特征
y = data.iloc[:, 0].values
X = data.iloc[:, 1:].values
print(X.shape)

# 可选：标准化（XGBoost 对尺度不敏感，但保留也无妨）
scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler,"000000")

# 检查缺失值
if np.isnan(X).any():
    raise ValueError("数据包含 NaN")

# 划分训练测试集（注意原代码 test_size=0.4）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # 二分类建议分层
)

# 计算类别数（确认是二分类）
num_classes = len(np.unique(y_train))
print(f"类别数: {num_classes}")

# XGBoost 二分类模型
# 若类别不平衡，可设置 scale_pos_weight = (负样本数/正样本数)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if num_classes == 2 else 1

model = xgb.XGBClassifier(
    objective='binary:logistic' if num_classes == 2 else 'multi:softprob',
    eval_metric='logloss' if num_classes == 2 else 'mlogloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight if num_classes == 2 else 1,
    n_estimators=100,
    max_depth=6,
    early_stopping_rounds=15,
    learning_rate=0.1
)

# 训练
model.fit(X_train, y_train,eval_set=[(X_test,y_test)],verbose=True)

# 预测
y_pred = model.predict(X_test)

# 评估
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='binary')
rec = recall_score(y_test, y_pred, average='binary')


print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")

# 保存模型
model.save_model("xgboost_model.json")
print("Model saved as xgboost_model.json")

