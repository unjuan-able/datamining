import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import xgboost as xgb

# ==========================================
# 辅助函数：Bad Case 提取与分析
# ==========================================
def analyze_and_export_badcases(X_raw, y_true, y_pred, y_prob, dataset_name="Dataset"):
    """
    提取预测错误的样本，计算错判类型（FP/FN）和置信度，并导出为CSV。
    """
    # 找到预测错误的索引
    error_indices = np.where(y_true != y_pred)[0]
    
    if len(error_indices) == 0:
        print(f"🎉 {dataset_name} 中没有 Bad Case！")
        return
    
    # 提取错误样本的原始特征、真实标签、预测标签和预测概率
    X_errors = X_raw[error_indices]
    y_true_errors = y_true[error_indices]
    y_pred_errors = y_pred[error_indices]
    
    # 针对二分类获取正类的概率；如果是多分类，获取预测类的概率
    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
        y_prob_errors = np.max(y_prob[error_indices], axis=1) # 多分类取最大概率
    else:
        y_prob_errors = y_prob[error_indices] # 二分类

    # 构建 DataFrame
    df_badcases = pd.DataFrame(X_errors, columns=[f"Feature_{i+1}" for i in range(X_errors.shape[1])])
    df_badcases['True_Label'] = y_true_errors
    df_badcases['Pred_Label'] = y_pred_errors
    df_badcases['Error_Type'] = np.where((y_true_errors == 0) & (y_pred_errors == 1), 'False_Positive (误报)', 
                                np.where((y_true_errors == 1) & (y_pred_errors == 0), 'False_Negative (漏报)', 'Other_Error'))
    df_badcases['Pred_Probability'] = np.round(y_prob_errors, 4)
    
    # 按预测概率排序，看看模型最“自信”却预测错的样本
    df_badcases = df_badcases.sort_values(by='Pred_Probability', ascending=False)
    
    # 导出文件
    file_name = f"{dataset_name}_badcases.csv"
    df_badcases.to_csv(file_name, index=False, encoding='utf-8-sig')
    
    # 打印统计信息
    print(f"\n🔍 [{dataset_name}] Bad Case 统计:")
    print(f"总错误数: {len(error_indices)} / {len(y_true)}")
    print(df_badcases['Error_Type'].value_counts())
    print(f"已将详细 Bad Case 导出至: {file_name}")
    return df_badcases


# ==========================================
# 阶段一：模型训练、验证与 Bad Case 分析
# ==========================================
print(">>> 开始阶段一：模型训练与保存 <<<")

# 1. 加载训练数据 (请替换路径)
data = pd.read_csv("F:\\datamining\\sensor_data260624\\normalize_feature\\feature\\combined_features.csv") 

y = data.iloc[:, 0].values
X = data.iloc[:, 1:].values  # 保留原始 X 用于分析

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# 3. 划分数据集 (注意：这里同时划分了 X_scaled 和原始 X)
X_train_scaled, X_val_scaled, X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_scaled, X, y, test_size=0.2, random_state=42, stratify=y
)

num_classes = len(np.unique(y_train))
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if num_classes == 2 else 1

# 4. 训练模型
model = xgb.XGBClassifier(
    objective='binary:logistic' if num_classes == 2 else 'multi:softprob',
    eval_metric='logloss' if num_classes == 2 else 'mlogloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight if num_classes == 2 else 1,
    n_estimators=100, max_depth=6, early_stopping_rounds=15, learning_rate=0.1
)

model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)

# 5. 验证集评估与 Bad Case 分析
y_val_pred = model.predict(X_val_scaled)
y_val_prob = model.predict_proba(X_val_scaled)[:, 1] if num_classes == 2 else model.predict_proba(X_val_scaled)

print("\n--- 验证集初步评估 ---")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")

# 执行 Bad Case 分析
val_badcases = analyze_and_export_badcases(X_val_raw, y_val, y_val_pred, y_val_prob, dataset_name="Validation")

model.save_model("xgboost_model.json")


# ==========================================
# 阶段二：独立测试集评估与 Bad Case 分析
# ==========================================
print("\n>>> 开始阶段二：独立测试集评估 <<<")

try:
    loaded_scaler = joblib.load("scaler.pkl")
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model("xgboost_model.json")
    
    # 请替换测试集路径
    test_data = pd.read_csv("F:\\datamining \\sensor_data260624\\testdata\\test_combined_features.csv") 
    y_unseen = test_data.iloc[:, 0].values
    X_unseen_raw = test_data.iloc[:, 1:].values  # 原始特征保留
    
    X_unseen_scaled = loaded_scaler.transform(X_unseen_raw) if loaded_scaler else X_unseen_raw
    
    y_unseen_pred = loaded_model.predict(X_unseen_scaled)
    y_unseen_prob = loaded_model.predict_proba(X_unseen_scaled)[:, 1] if num_classes == 2 else loaded_model.predict_proba(X_unseen_scaled)

    print("\n--- 独立测试集详细分类报告 ---")
    print(classification_report(y_unseen, y_unseen_pred))

    # 执行 Bad Case 分析
    test_badcases = analyze_and_export_badcases(X_unseen_raw, y_unseen, y_unseen_pred, y_unseen_prob, dataset_name="Test")

except FileNotFoundError:
    print("⚠️ 找不到相关文件，跳过独立测试集评估。")

from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
import numpy as np

def extract_error_rules_with_tree(X_raw, y_true, y_pred, feature_names=None, max_depth=3):
    """
    使用决策树自动挖掘模型犯错的特征区间规则。
    """
    print("\n" + "="*50)
    print("🌳 自动提取的 Bad Case 错误规则 (Error Tree)")
    print("="*50)
    
    # 1. 构造新的目标变量：预测错误标记为 1，预测正确标记为 0
    is_error = (y_true != y_pred).astype(int)
    
    # 如果全对，直接退出
    if sum(is_error) == 0:
        print("🎉 完美！该数据集中没有任何预测错误，无法提取错误规则。")
        return None
        
    # 2. 处理特征名（如果有真实的列名最好传入，否则用 Feature_X 代替）
    if feature_names is None:
        feature_names = [f"Feature_{i+1}" for i in range(X_raw.shape[1])]
    else:
        # 确保 feature_names 是 list 类型
        feature_names = list(feature_names)
        
    # 3. 训练一棵浅层决策树
    # 设置 max_depth=3 防止规则太长人类看不懂
    # 设置 min_samples_leaf=5 防止树记住极个别的异常点（过拟合）
    # 设置 class_weight='balanced' 因为错误样本(1)通常远少于正确样本(0)
    tree = DecisionTreeClassifier(
        max_depth=max_depth, 
        min_samples_leaf=5, 
        class_weight='balanced',
        random_state=42
    )
    tree.fit(X_raw, is_error)
    
    # 4. 导出为人类可读的规则文本
    tree_rules = export_text(tree, feature_names=feature_names)
    
    print("【阅读指南】:")
    print("👉 class: 1.0 表示该区间内的样本【极易被预测错误】 (Bad Case 高发区)")
    print("👉 class: 0.0 表示该区间内的样本【模型预测很稳】\n")
    print(tree_rules)
    
    # 5. 计算并输出导致错误的“罪魁祸首”特征 Top 3
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Error_Importance': tree.feature_importances_
    }).sort_values(by='Error_Importance', ascending=False)
    
    print("\n⚠️ 导致模型犯错的【高危特征】排行榜 Top 3:")
    for idx, row in importances.head(3).iterrows():
        if row['Error_Importance'] > 0:
            print(f"- {row['Feature']}: 危险权重 {row['Error_Importance']:.4f}")
            
    print("="*50)
    return tree

feature_cols = data.columns[1:].tolist()

# 在计算出 y_val_pred 或 y_unseen_pred 之后调用：
print("\n>>> 开始深挖验证集的错误规律 <<<")
extract_error_rules_with_tree(
    X_raw=X_val_raw,           # 必须使用未经标准化的原始特征
    y_true=y_val,              # 真实的标签
    y_pred=y_val_pred,         # 模型预测的标签
    feature_names=feature_cols # 传入真实的列名（可选）
)