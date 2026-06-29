import torch
import torch.nn as nn

class Bottleneck1D(nn.Module):
    """
    一维瓶颈残差块 (1D Bottleneck Block) -> 用于 ResNet50/101/152
    """
    expansion = 4  # 每一个 Block 输出的通道数是输入中转通道数的 4 倍

    def __init__(self, in_channels, base_channels, stride=1):
        super(Bottleneck1D, self).__init__()
        out_channels = base_channels * self.expansion

        # 1x1 卷积：压缩通道数
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        
        # 3x3 卷积（1D化）：提取时序特征
        self.conv2 = nn.Conv1d(base_channels, base_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(base_channels)
        
        # 1x1 卷积：展宽、恢复通道数
        self.conv3 = nn.Conv1d(base_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 恒等映射 (Skip Connection)
        self.shortcut = nn.Sequential()
        # 如果步长不为1，或者输入输出通道对不上，需要通过 1x1 卷积升维/降采样
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity  # 残差相加
        out = self.relu(out)
        return out


class ResNet501D(nn.Module):
    """
    一维残差网络 ResNet50-1D
    """
    def __init__(self, input_channels, num_classes, num_blocks_list=[3, 4, 6, 3]):
        """
        ResNet50 的默认块配置是 [3, 4, 6, 3]
        :param input_channels: 输入特征数（陀螺仪 XYZ 为 3）
        :param num_classes: 分类类别数（你的任务是二分类，输出通常设为 1，接 Sigmoid；或者设为 2，接 CrossEntropy）
        """
        super(ResNet501D, self).__init__()
        self.in_channels = 64

        # 1. 初始特征提取层
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 2. 四个残差阶段 (Layers)
        # 注意：Bottleneck 传入的是基本通道数(64, 128...)，实际输出会被乘上 expansion(4)
        self.layer1 = self._make_layer(64, num_blocks_list[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks_list[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks_list[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks_list[3], stride=2)

        # 3. 池化与分类器
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # ResNet50 最后一层的输出通道数是 512 * 4 = 2048
        self.fc = nn.Linear(512 * Bottleneck1D.expansion, num_classes)

    def _make_layer(self, base_channels, num_blocks, stride):
        layers = []
        # 每一层（Stage）的第一个 Block 负责调整步长(stride)和通道数
        layers.append(Bottleneck1D(self.in_channels, base_channels, stride))
        self.in_channels = base_channels * Bottleneck1D.expansion
        
        # 随后的 Block 保持通道数和步长不变
        for _ in range(1, num_blocks):
            layers.append(Bottleneck1D(self.in_channels, base_channels, stride=1))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入格式: [Batch_size, 3, Seq_length]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 展平为 [Batch_size, 2048]
        x = self.fc(x)
        
        return x

# --- 测试运行代码 ---
if __name__ == "__main__":
    batch_size = 16
    seq_length = 250       # 模拟用户的平均时间步长
    input_channels = 3     # 陀螺仪 XYZ 三轴
    output_dim = 1         # 异常检测二分类输出

    # 实例化 ResNet50-1D
    model = ResNet501D(input_channels=input_channels, num_classes=output_dim)

    dummy_input = torch.randn(batch_size, input_channels, seq_length)
    predictions = model(dummy_input)

    print(f"ResNet50-1D 实例化成功！")
    print(f"输入序列形状: {dummy_input.shape}")
    print(f"预测输出形状: {predictions.shape}  -> (特征维度已被成功展宽至 2048 并映射到输出)")