import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

# ======================
# 参数
# ======================
INPUT = "F:\\datamining\\sensor_data260624\\normalize_feature\\feature\\combined_features.csv"
OUTPUT = "F:\\datamining\\sensor_data260624\\testdata\\test_combined_features.csv"

N_TEST = 100        # 生成测试样本数量
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================
# 读取原始数据
# ============================
df = pd.read_csv(INPUT)

label_col = "label"

generated = []

# ============================
# 按类别生成
# ============================
for label in sorted(df[label_col].unique()):

    src = df[df[label_col] == label]

    n_generate = round(
        N_TEST * len(src) / len(df)
    )

    fake = {}

    for col in df.columns:

        # 保持标签
        if col == label_col:
            fake[col] = [label] * n_generate
            continue

        values = src[col]

        # user id
        if col in ["user_id_1", "user_id_2"]:

            low = int(values.min())
            high = int(values.max())

            fake[col] = np.random.randint(
                low,
                high + 1,
                n_generate
            )

        # 相关系数
        elif col.startswith("corr_"):

            mu = values.mean()
            std = values.std()

            x = np.random.normal(
                mu,
                std,
                n_generate
            )

            fake[col] = np.clip(
                x,
                -1,
                1
            )

        # rate
        elif (
            "rate" in col
            or "autocorrelation" in col
        ):

            mu = values.mean()
            std = values.std()

            x = np.random.normal(
                mu,
                std,
                n_generate
            )

            fake[col] = np.clip(
                x,
                0,
                1
            )

        # 非负统计量
        elif any(
            k in col
            for k in [
                "variance",
                "energy",
                "std",
                "rms",
                "entropy",
                "band_energy",
                "wavelet_energy",
                "range",
                "spread"
            ]
        ):

            mu = values.mean()
            std = values.std()

            x = np.random.normal(
                mu,
                std,
                n_generate
            )

            fake[col] = np.maximum(
                x,
                0
            )

        # 普通连续变量
        else:

            mu = values.mean()
            std = values.std()

            fake[col] = np.random.normal(
                mu,
                std,
                n_generate
            )

    fake_df = pd.DataFrame(fake)

    generated.append(fake_df)

# ============================
# 合并
# ============================
test_df = (
    pd.concat(
        generated,
        ignore_index=True
    )
    .sample(
        frac=1,
        random_state=RANDOM_STATE
    )
    .reset_index(drop=True)
)

# ============================
# 列顺序完全对齐
# ============================
test_df = test_df[df.columns]

# ============================
# 保存
# ============================
test_df.to_csv(
    OUTPUT,
    index=False
)

print("="*50)
print("生成成功")

print("shape:")
print(test_df.shape)

print("\n列是否一致：")
print(
    list(test_df.columns)
    ==
    list(df.columns)
)

print("\n标签分布：")
print(
    test_df["label"]
    .value_counts()
)

print("\n输出：")
print(OUTPUT)