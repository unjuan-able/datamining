import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader
)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score
)

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

BATCH=64
EPOCH=200
LR=1e-3
EMB=64
LAMBDA=0.1


# =====================
# Dataset
# =====================
class TableDataset(Dataset):

    def __init__(self,df):

        y=df["label"].values

        X=df.drop(
            columns=["label"]
        ).values

        self.scaler=StandardScaler()

        X=self.scaler.fit_transform(X)

        self.X=torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y=torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.y)

    def __getitem__(self,idx):

        return (
            self.X[idx],
            self.y[idx]
        )


# =====================
# Encoder
# =====================
class Encoder(nn.Module):

    def __init__(
        self,
        input_dim
    ):

        super().__init__()

        self.net=nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.BatchNorm1d(
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                EMB
            )
        )

    def forward(self,x):

        z=self.net(x)

        return nn.functional.normalize(
            z,
            dim=1
        )


# =====================
# classifier
# =====================
class Net(nn.Module):

    def __init__(
        self,
        input_dim
    ):

        super().__init__()

        self.encoder=Encoder(
            input_dim
        )

        self.fc=nn.Linear(
            EMB,
            2
        )

    def forward(self,x):

        z=self.encoder(x)

        y=self.fc(z)

        return z,y


# =====================
# SupCon Loss
# =====================
class SupConLoss(nn.Module):

    def __init__(
        self,
        temp=0.1
    ):

        super().__init__()

        self.temp=temp

    def forward(
        self,
        feat,
        label
    ):

        feat=nn.functional.normalize(
            feat,
            dim=1
        )

        sim=torch.mm(
            feat,
            feat.T
        )

        sim/=self.temp

        mask=(
            label[:,None]
            ==
            label[None,:]
        )

        mask=mask.float()

        logits=(
            sim
            -
            sim.max(
                dim=1,
                keepdim=True
            )[0]
        )

        exp=torch.exp(
            logits
        )

        log_prob=(
            logits
            -
            torch.log(
                exp.sum(
                    1,
                    keepdim=True
                )
            )
        )

        loss=(
            mask
            *
            log_prob
        ).sum(1)

        loss/=(mask.sum(1)+1e-8)

        return -loss.mean()


# =====================
# load
# =====================
train=pd.read_csv(
    "train.csv"
)

val=pd.read_csv(
    "val.csv"
)

train_loader=DataLoader(
    TableDataset(train),
    batch_size=BATCH,
    shuffle=True
)

val_loader=DataLoader(
    TableDataset(val),
    batch_size=BATCH
)

INPUT=train.shape[1]-1

model=Net(
    INPUT
).to(DEVICE)

ce=nn.CrossEntropyLoss()

sup=SupConLoss()

opt=torch.optim.Adam(
    model.parameters(),
    lr=LR
)


# =====================
# train
# =====================
best=0

for epoch in range(EPOCH):

    model.train()

    for x,y in train_loader:

        x=x.to(DEVICE)

        y=y.to(DEVICE)

        z,out=model(x)

        loss_cls=ce(
            out,
            y
        )

        loss_con=sup(
            z,
            y
        )

        loss=(
            loss_cls
            +
            LAMBDA
            *
            loss_con
        )

        opt.zero_grad()

        loss.backward()

        opt.step()

    model.eval()

    pred=[]

    gt=[]

    with torch.no_grad():

        for x,y in val_loader:

            x=x.to(DEVICE)

            _,o=model(x)

            p=o.argmax(1)

            pred.extend(
                p.cpu()
            )

            gt.extend(
                y
            )

    f1=f1_score(
        gt,
        pred
    )

    print(
        epoch,
        loss.item(),
        f1
    )

    if f1>best:

        best=f1

        torch.save(
            model.state_dict(),
            "best_repr.pt"
        )