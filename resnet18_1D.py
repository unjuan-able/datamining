import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    """
    一维残差块 (1D Residual Block)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        
        # 第一层一维卷积
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 第二层一维卷积
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 恒等映射 (Skip Connection)
        # 如果维度或通道数发生变化，需要用 1x1 卷积调整 shape 以便相加
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 将残差加到主路径上
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    一维残差网络 (ResNet-1D)
    """
    def __init__(self, input_channels, num_classes, num_blocks_list=[2, 2, 2, 2]):
        """
        :param input_channels: 序列的特征维度 (例如单变量时间序列为1，多变量为特征数)
        :param num_classes: 预测输出的维度 (回归任务通常为1，分类任务为类别数)
        :param num_blocks_list: 每个网络层中残差块的数量
        """
        super(ResNet1D, self).__init__()
        self.in_channels = 64

        # 初始特征提取层
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, 
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 残差层 (Layers)
        self.layer1 = self._make_layer(64, num_blocks_list[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks_list[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks_list[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks_list[3], stride=2)

        # 自适应平均池化，能够接受任意长度的序列输入
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接预测层
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock1D(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [batch_size, input_channels, sequence_length]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x) # 降维到 [batch_size, channels, 1]
        x = x.view(x.size(0), -1) # 展平为 [batch_size, channels]
        x = self.fc(x)
        
        return x
        
# 测试参数设定
batch_size = 16
seq_length = 1000      # 序列的时间步长度 (例如1000个采样点)
input_channels = 3     # 序列特征数 (例如XYZ三轴传感器数据)
output_dim = 1         # 预测目标 (例如预测1个未来的数值，若是10分类问题则改为10)

# 1. 实例化模型
model = ResNet1D(input_channels=input_channels, num_classes=output_dim)

# 2. 生成随机模拟序列数据
# 注意 PyTorch 1D 卷积的输入格式要求：[Batch大小, 特征通道数, 序列长度]
dummy_input = torch.randn(batch_size, input_channels, seq_length)

# 3. 前向传播预测
predictions = model(dummy_input)

print(f"输入序列 Shape: {dummy_input.shape}")
print(f"预测输出 Shape: {predictions.shape}") 
# 预期输出 Shape: [16, 1]