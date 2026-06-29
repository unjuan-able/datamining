import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import xgboost as xgb

# ==========================================
# 阶段一：模型训练、验证与保存 (对应原代码1)
# ==========================================
print(">>> 开始阶段一：模型训练与保存 <<<")

# 1. 加载训练数据
train_data_path = "your_data.csv"  # 替换为实际训练数据路径
data = pd.read_csv(train_data_path)

# 第一列为标签，其余为特征
y = data.iloc[:, 0].values
X = data.iloc[:, 1:].values
print(f"训练基底数据特征形状: {X.shape}")

# 2. 检查缺失值
if np.isnan(X).any():
    raise ValueError("数据包含 NaN，请先进行缺失值填充！")

# 3. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 保存标准化器（统一使用 scaler.pkl）
joblib.dump(scaler, "scaler.pkl")
print("✅ 成功保存标准化器: scaler.pkl")

# 4. 划分训练和验证集 (用于 early_stopping 和初步评估)
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 配置并训练 XGBoost 模型
num_classes = len(np.unique(y_train))
print(f"类别数: {num_classes}")

# 处理类别不平衡
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

# 训练模型
print("开始训练...")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# 在验证集上进行初步评估
y_val_pred = model.predict(X_val)
print("\n--- 验证集评估 (内部 20% 数据) ---")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred, average='binary' if num_classes == 2 else 'macro'):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred, average='binary' if num_classes == 2 else 'macro'):.4f}")

# 6. 保存模型
model.save_model("xgboost_model.json")
print("✅ 成功保存 XGBoost 模型: xgboost_model.json\n")


# ==========================================
# 阶段二：加载组件与独立测试集评估 (对应原代码2)
# ==========================================
print(">>> 开始阶段二：独立测试集评估 <<<")

# 1. 加载保存的预处理组件和模型
try:
    loaded_scaler = joblib.load("scaler.pkl")
    print("✅ 成功加载标准化器: scaler.pkl")
except FileNotFoundError:
    print("⚠️ 未找到 scaler.pkl，请确保路径正确。")
    loaded_scaler = None

loaded_model = xgb.XGBClassifier()
loaded_model.load_model("xgboost_model.json")
print("✅ 成功加载 XGBoost 模型: xgboost_model.json")

# 2. 加载并预处理真正的独立测试集
# 📝 请将下方路径替换为你实际独立测试集文件所在的目录
test_data_path = "其他目录/your_real_test_data.csv" 
try:
    test_data = pd.read_csv(test_data_path)
    
    y_unseen = test_data.iloc[:, 0].values
    X_unseen = test_data.iloc[:, 1:].values
    print(f"📊 独立测试集样本量: {X_unseen.shape[0]}, 特征数: {X_unseen.shape[1]}")

    # 🚨 关键步骤：使用阶段一训练好的 scaler 对测试集进行 transform (千万不用 fit_transform)
    if loaded_scaler is not None:
        X_unseen_scaled = loaded_scaler.transform(X_unseen)
        print("⚡ 独立测试集特征已完成标准化缩放。")
    else:
        X_unseen_scaled = X_unseen

    # 3. 执行预测与评估
    y_unseen_pred = loaded_model.predict(X_unseen_scaled)

    num_classes_unseen = len(np.unique(y_unseen))
    avg_method = 'binary' if num_classes_unseen == 2 else 'macro'

    acc_unseen = accuracy_score(y_unseen, y_unseen_pred)
    prec_unseen = precision_score(y_unseen, y_unseen_pred, average=avg_method)
    rec_unseen = recall_score(y_unseen, y_unseen_pred, average=avg_method)

    print("\n================ 独立测试集最终评估结果 ================")
    print(f"测试集准确率 (Accuracy) : {acc_unseen:.4f}")
    print(f"测试集精确率 (Precision): {prec_unseen:.4f} (模式: {avg_method})")
    print(f"测试集召回率 (Recall)   : {rec_unseen:.4f} (模式: {avg_method})")
    print("========================================================")

    print("\n详细分类报告:")
    print(classification_report(y_unseen, y_unseen_pred))

except FileNotFoundError:
    print(f"\n⚠️ 找不到独立测试集文件: {test_data_path}")
    print("请修改 'test_data_path' 变量以运行最终评估模块。")