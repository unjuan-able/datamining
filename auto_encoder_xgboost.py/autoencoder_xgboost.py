import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# =========================
# 读取数据
# =========================
train_df = pd.read_csv("train.csv")
test_df =pd.read_csv("test.csv")

# label
y_train = train_df["label"].values
y_test = test_df["label"].values

# 特征
X_train = train_df.drop(columns=["label"]).values
X_test = test_df.drop(columns=["label"]).values

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

input_dim = X_train.shape[1]

model = AutoEncoder(input_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 100

for epoch in range(EPOCHS):

    recon, z = model(X_train)

    loss = criterion(recon, X_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

model.eval()

with torch.no_grad():

    train_embed = model.encoder(X_train).numpy()
    test_embed = model.encoder(X_test).numpy()

print("Embedding shape:", train_embed.shape)

clf = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

clf.fit(train_embed, y_train)

y_pred = clf.predict(test_embed)
y_prob = clf.predict_proba(test_embed)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("AUC:", roc_auc_score(y_test, y_prob))