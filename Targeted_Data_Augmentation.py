import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score
)

from sklearn.inspection import permutation_importance


# ===========================================
# 配置
# ===========================================

TRAIN_PATH="train.csv"

TEST_PATH="real_test.csv"

THRESHOLD=0.35

KNN_K=20

NOISE_STD=0.03

AUG_PER_SAMPLE=5

RANDOM_STATE=42


# ===========================================
# 工具函数
# ===========================================

def clean(df):

    drop=[]

    for c in df.columns:

        if "user_id" in c.lower():

            drop.append(c)

    return df.drop(columns=drop)


# ===========================================
# 读取训练集
# ===========================================

train_df=pd.read_csv(TRAIN_PATH)

train_df=clean(train_df)

feature_names=list(
    train_df.columns[1:]
)

X=train_df.iloc[:,1:]

y=train_df.iloc[:,0]

print(
"训练集:",
X.shape
)


# ===========================================
# train/val
# ===========================================

X_train,X_val,y_train,y_val=(
train_test_split(

X,

y,

test_size=0.2,

stratify=y,

random_state=RANDOM_STATE
)
)

# ===========================================
# scaler
# ===========================================

scaler=StandardScaler()

X_train=scaler.fit_transform(
X_train
)

X_val=scaler.transform(
X_val
)

joblib.dump(
scaler,
"scaler.pkl"
)

joblib.dump(
feature_names,
"feature_names.pkl"
)

# ===========================================
# 类别权重
# ===========================================

scale=(
(y_train==0).sum()
/
(y_train==1).sum()
)

# ===========================================
# 模型
# ===========================================

model=xgb.XGBClassifier(

objective="binary:logistic",

eval_metric=[
"logloss",
"aucpr"
],

n_estimators=1000,

learning_rate=0.03,

max_depth=8,

subsample=0.8,

colsample_bytree=0.8,

scale_pos_weight=scale,

early_stopping_rounds=50,

random_state=RANDOM_STATE
)

# ===========================================
# 训练
# ===========================================

model.fit(

X_train,

y_train,

eval_set=[

(
X_val,
y_val
)

],

verbose=50
)

model.save_model(
"xgb_model.json"
)

print(
"\n训练完成"
)


# ===========================================
# 独立测试集
# ===========================================

test_df=pd.read_csv(
TEST_PATH
)

test_df=clean(
test_df
)

X_test=test_df.iloc[:,1:]

y_test=test_df.iloc[:,0]

X_test=scaler.transform(
X_test
)

print(
"\n独立测试:",
X_test.shape
)


# ===========================================
# 推理
# ===========================================

prob=(
model.predict_proba(
X_test
)
[:,1]
)

pred=(
prob>=THRESHOLD
).astype(
int
)

print(
"\n====== 测试结果 ======"
)

print(
classification_report(

y_test,

pred
)
)

print(
"ACC",
accuracy_score(
y_test,
pred
)
)

print(
"PREC",
precision_score(
y_test,
pred
)
)

print(
"REC",
recall_score(
y_test,
pred
)
)


# ===========================================
# 导出badcase
# ===========================================

bad=test_df.copy()

bad["pred"]=pred

bad["prob"]=prob

bad["error"]=np.where(

bad.iloc[:,0]
!=
bad["pred"],

1,

0
)

bad=bad[
bad.error==1
]

bad["error_type"]=np.where(

(
bad.iloc[:,0]==1
)
&
(
bad["pred"]==0
),

"FN",

"FP"
)

bad.to_csv(

"badcases.csv",

index=False
)

print(
"\nbadcase数量",
len(bad)
)


# ===========================================
# 特征归因
# ===========================================

imp=(
permutation_importance(

model,

X_test,

y_test,

n_repeats=5
)
)

importance=(
pd.DataFrame({

"feature":feature_names,

"score":
imp.importances_mean
})

.sort_values(

"score",

ascending=False
)
)

importance.to_csv(

"badcase_importance.csv",

index=False
)

print(
"\nTop Feature"
)

print(
importance.head(
20
)
)


# ===========================================
# KNN找训练集近邻
# ===========================================

bad_x=scaler.transform(

bad[
feature_names
]
)

knn=NearestNeighbors(

n_neighbors=KNN_K
)

knn.fit(
X_train
)

neighbor_idx=(
knn.kneighbors(

bad_x,

return_distance=False
)
)


# ===========================================
# 定向扩充
# ===========================================

new_x=[]

new_y=[]

for ids in neighbor_idx:

    sample=(
X_train[
ids
]
)

label=(
y_train.iloc[
ids
]

.mode()

.iloc[0]
)

for x in sample:

    for _ in range(

AUG_PER_SAMPLE
):

        noise=(
np.random.normal(

0,

NOISE_STD,

len(
x
)
)
)

new_x.append(

x+noise
)

new_y.append(
label
)

new_x=np.array(
new_x
)

new_x=(
scaler.inverse_transform(
new_x
)
)

origin=(
scaler.inverse_transform(
X_train
)
)

old=pd.DataFrame(

origin,

columns=feature_names
)

old["label"]=(
y_train.values
)

aug=pd.DataFrame(

new_x,

columns=feature_names
)

aug["label"]=new_y

final=pd.concat(

[

old,

aug

]

)

final=final[
["label"]
+
feature_names
]

final.to_csv(

"augmented_train.csv",

index=False
)

print(
"\n完成扩充"
)

print(
"原训练",
len(old)
)

print(
"新增",
len(aug)
)

print(
"总计",
len(final)
)