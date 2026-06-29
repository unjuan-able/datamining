import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# ==========================================
# 1. 数据加载与预处理
# ==========================================
data = pd.read_csv("your_data.csv")  # 替换为实际路径

y = data.iloc[:, 0].values
X = data.iloc[:, 1:].values
print(f"原始数据形状: {X.shape}")

if np.isnan(X).any():
    raise ValueError("数据包含 NaN")

scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl") # 建议加上后缀名方便识别

# ==========================================
# 2. 构造 1:1 训练集与 1:5 验证集 (核心修改点)
# ==========================================
# 假设当前输入数据 X, y 已经是 1:1 或接近 1:1
mask_pos = (y == 1)
mask_neg = (y == 0)

X_pos, y_pos = X[mask_pos], y[mask_pos]
X_neg, y_neg = X[mask_neg], y[mask_neg]

print(f"原始正样本数: {len(y_pos)}, 原始负样本数: {len(y_neg)}")

# 策略：我们抽取 20% 的负样本用于验证集
X_neg_train, X_neg_val, y_neg_train, y_neg_val = train_test_split(
    X_neg, y_neg, test_size=0.2, random_state=42
)

# 为了让验证集达到 1:5，我们计算验证集所需的正样本数量
# 验证集正样本数 = 验证集负样本数 / 5
num_val_pos = len(y_neg_val) // 5 

# 安全校验：确保切分数量不超过实际拥有的正样本数量
num_val_pos = min(num_val_pos, int(len(y_pos) * 0.4)) 

X_pos_train, X_pos_val, y_pos_train, y_pos_val = train_test_split(
    X_pos, y_pos, test_size=num_val_pos, random_state=42
)

# 合并生成最终的训练集和验证集
X_train = np.vstack((X_pos_train, X_neg_train))
y_train = np.concatenate((y_pos_train, y_neg_train))

X_val = np.vstack((X_pos_val, X_neg_val))
y_val = np.concatenate((y_pos_val, y_neg_val))

print(f"\n构建完毕：")
print(f"训练集形状: {X_train.shape}, 正负比: {sum(y_train==1)}:{sum(y_train==0)}")
print(f"验证集形状: {X_val.shape}, 正负比: {sum(y_val==1)}:{sum(y_val==0)} (目标 1:5)")

# ==========================================
# 3. XGBoost 模型训练
# ==========================================
num_classes = len(np.unique(y_train))

# 因为训练集是 1:1，scale_pos_weight 理论上在 1 左右，保留计算逻辑即可
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if num_classes == 2 else 1

model = xgb.XGBClassifier(
    objective='binary:logistic' if num_classes == 2 else 'multi:softprob',
    # 【修改】风控/极度不平衡场景，用 aucpr (PR AUC) 作为早停指标比 logloss 更能真实反映拦截效果
    eval_metric='aucpr' if num_classes == 2 else 'mlogloss', 
    random_state=42,
    scale_pos_weight=scale_pos_weight if num_classes == 2 else 1,
    n_estimators=200,    # 可以适当放大，由 early_stopping 控制停止
    max_depth=6,
    early_stopping_rounds=20,
    learning_rate=0.1
)

print("\n开始训练...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# ==========================================
# 4. 预测与基于验证集寻找最佳阈值
# ==========================================
# 获取概率值而不是直接获取 0/1 的预测结果
y_pred_proba = model.predict_proba(X_val)[:, 1]

# 因为训练集是 1:1，输出概率天然偏高。我们不能死守 0.5 阈值。
# 在 1:5 的验证集上寻找使得 F1-Score 最大的最佳阈值
best_threshold = 0.5
best_f1 = 0

for thresh in np.arange(0.1, 0.9, 0.05):
    preds = (y_pred_proba >= thresh).astype(int)
    f1 = f1_score(y_val, preds, average='binary', zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"\n在 1:5 验证集上寻找的最佳判定阈值为: {best_threshold:.2f}")

# 使用最佳阈值进行最终评估
y_pred_best = (y_pred_proba >= best_threshold).astype(int)

acc = accuracy_score(y_val, y_pred_best)
prec = precision_score(y_val, y_pred_best, average='binary', zero_division=0)
rec = recall_score(y_val, y_pred_best, average='binary', zero_division=0)

print("-" * 30)
print("验证集 (1:5) 评估结果:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f} (这个指标更能代表线上真实环境)")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {best_f1:.4f}")
print("-" * 30)

# 保存模型
model.save_model("xgboost_model.json")
print("Model saved as xgboost_model.json")