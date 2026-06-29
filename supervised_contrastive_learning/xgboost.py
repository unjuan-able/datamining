import torch
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report
)

from train_repr import (
    Encoder
)

DEVICE="cuda"


def extract(df):

    X=df.drop(
        columns=["label"]
    ).values

    X=torch.tensor(
        X,
        dtype=torch.float32
    ).to(DEVICE)

    model=Encoder(
        X.shape[1]
    )

    model.load_state_dict(
        torch.load(
            "best_repr.pt"
        ),
        strict=False
    )

    model.to(
        DEVICE
    )

    model.eval()

    with torch.no_grad():

        z=model(
            X
        )

    return (
        z.cpu().numpy(),
        df.label.values
    )


train=pd.read_csv(
    "train.csv"
)

test=pd.read_csv(
    "independent_test.csv"
)

Xtr,Ytr=extract(
    train
)

Xte,Yte=extract(
    test
)

clf=XGBClassifier(

    n_estimators=500,

    max_depth=6,

    learning_rate=0.03,

    subsample=0.8,

    random_state=42
)

clf.fit(
    Xtr,
    Ytr
)

pred=clf.predict(
    Xte
)

print(
    classification_report(
        Yte,
        pred
    )
)