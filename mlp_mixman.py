import torch
import torch.nn as nn
import pandas
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score


data=pd.read_csv("")
print(np.array(data).shape)
X=data.iloc[:,1:].values
y=data.iloc[:,0].values
scaler=StandardScaler()
X=scaler.fit_transform(X)
if np.isnan(X).any():
    raise ValueError("数据包含NAN")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)
# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 计算类别数
num_classes = len(torch.unique(y_train))

print(f"类别数: {num_classes}")
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.Dropout(dropout),      # 使用传入的 dropout
            nn.Linear(32, 16),
            #nn.BatchNorm1d(16),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)
Criterion=nn.CrossEntropyLoss()


model.load_state_dict(torch.load(""))
model.eval()
model.to(device)
dummpy_input=torch.randn(1,162)
dummpy_input.to(device)
onnx_path=""
torch.onnx.export(
    model,
    dummpy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":{0:"batch_size"},
        "output":{0:"batch_size"}
    },
    opset_version=11
    
)
print(f"Model export to {onnx_path}")