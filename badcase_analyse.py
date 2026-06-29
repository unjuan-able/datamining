# =====================================================================
# 纯原生 Badcase 归因与规则挖掘模块 (No SHAP required)
# =====================================================================
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

print("\n" + "="*40)
print("开始进行纯原生 Badcase 分析与规则挖掘...")
print("="*40)

# 1. 提取预测概率和特征名
y_prob = model.predict_proba(X_test)[:, 1] if num_classes == 2 else np.max(model.predict_proba(X_test), axis=1)
feature_names = data.columns[1:].tolist()

# 2. 将数据逆标准化，还原为真实业务数值，方便找规则
X_test_orig = scaler.inverse_transform(X_test)

# 构建一个完整的分析大表
analysis_df = pd.DataFrame(X_test_orig, columns=feature_names)
analysis_df['True_Label'] = y_test
analysis_df['Pred_Label'] = y_pred
analysis_df['Pred_Prob'] = y_prob

# 3. 标记样本类型：TP(真阳), TN(真阴), FP(误报), FN(漏报)
def get_result_type(row):
    if row['True_Label'] == 1 and row['Pred_Label'] == 1: return 'TP (正确检出)'
    if row['True_Label'] == 0 and row['Pred_Label'] == 0: return 'TN (正确排除)'
    if row['True_Label'] == 0 and row['Pred_Label'] == 1: return 'FP (误报/假阳)'
    if row['True_Label'] == 1 and row['Pred_Label'] == 0: return 'FN (漏报/假阴)'

analysis_df['Result_Type'] = analysis_df.apply(get_result_type, axis=1)

# ==========================================
# 动作一：利用 Pandas 寻找均值差异最大的“异常特征”
# ==========================================
print("\n--- 1. 误报样本 (FP) 异常特征大起底 ---")
fp_df = analysis_df[analysis_df['Result_Type'] == 'FP (误报/假阳)']
tn_df = analysis_df[analysis_df['Result_Type'] == 'TN (正确排除)'] # 拿误报和正确的负样本比

if not fp_df.empty and not tn_df.empty:
    fp_mean = fp_df[feature_names].mean()
    tn_mean = tn_df[feature_names].mean()
    
    # 计算差异倍数（避免除以0，加上极小值）
    diff_ratio = (fp_mean - tn_mean) / (tn_mean.replace(0, 1e-5).abs())
    
    # 找出差异最夸张的 Top 5 特征
    top_diff_features = diff_ratio.abs().sort_values(ascending=False).head(5)
    print("在误报样本中，以下特征的均值严重偏离了正常样本：")
    for feat in top_diff_features.index:
        print(f" -> {feat}: 误报组均值={fp_mean[feat]:.2f} | 正常组均值={tn_mean[feat]:.2f}")
else:
    print("没有误报样本可供分析。")

# 同理可以对漏报 (FN) 和真阳 (TP) 进行对比，代码逻辑相同，这里省略以保持清晰

# ==========================================
# 动作二：训练“代理决策树”自动写出硬编码规则
# ==========================================
print("\n--- 2. 自动挖掘兜底拦截规则 (Rule-based Filter) ---")
# 目标：让决策树学习“什么样的数据会导致模型判错”
# 我们把 Badcase 打标签为 1，Goodcase 打标签为 0
analysis_df['Is_Badcase'] = np.where(analysis_df['Result_Type'].str.contains('FP|FN'), 1, 0)

if analysis_df['Is_Badcase'].sum() > 0:
    # 只拿原始特征去训练这棵浅层树
    X_rule = analysis_df[feature_names]
    y_rule = analysis_df['Is_Badcase']
    
    # 训练一棵最大深度只有 2 或 3 的树（深度越浅，规则越容易被人类写进 if-else）
    rule_tree = DecisionTreeClassifier(max_depth=2, random_state=42, class_weight='balanced')
    rule_tree.fit(X_rule, y_rule)
    
    # 打印树的结构，这就是天然的 IF-THEN 规则！
    tree_rules = export_text(rule_tree, feature_names=feature_names)
    print("发现的 Badcase 潜在触发规则如下 (Class 1 代表判错，Class 0 代表判对)：\n")
    print(tree_rules)
    
    print("\n💡 提示：重点看叶子节点为 'class: 1' 的路径，")
    print("例如：如果 sensor_mag > 500 且 acc_x <= -2.5，模型极大概率会判错。")
    print("你可以直接将这些条件翻译成后置的 if-else 代码。")
else:
    print("没有发现 Badcase，无需挖掘规则！")
    
# 保存全量分析表供人工通过 Excel 透视表查看
analysis_df.to_csv("badcase_pandas_analysis.csv", index=False)
print("\n全量分析数据已保存至: badcase_pandas_analysis.csv (推荐用 Excel 排序筛选查看)")