import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 通道注意力模块 (SE Block 1D)
# ==========================================
class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        # 全局平均池化，将时间维度压缩为1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 两层全连接构建门控机制
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        # 将注意力权重乘回原特征图
        return x * y.expand_as(x)

# ==========================================
# 2. 加入了 SE 模块的一维瓶颈残差块
# ==========================================
class Bottleneck1D_SE(nn.Module):
    expansion = 4

    def __init__(self, in_channels, base_channels, stride=1):
        super(Bottleneck1D_SE, self).__init__()
        out_channels = base_channels * self.expansion

        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        
        self.conv2 = nn.Conv1d(base_channels, base_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(base_channels)
        
        self.conv3 = nn.Conv1d(base_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 插入通道注意力模块
        self.se = SELayer1D(out_channels)
        
        self.shortcut = nn.Sequential()
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

        # 在残差相加前，经过 SE 模块重新校准通道权重
        out = self.se(out)
        
        out += identity
        out = self.relu(out)
        return out

# ==========================================
# 3. 门控时序注意力模块 (用于替换全局平均池化)
# ==========================================
class GatedTemporalAttention(nn.Module):
    def __init__(self, in_features, hidden_dim=256):
        super(GatedTemporalAttention, self).__init__()
        self.attention_V = nn.Linear(in_features, hidden_dim)
        self.attention_U = nn.Linear(in_features, hidden_dim)
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: [Batch, Channels(2048), Seq_length]
        # 转换为: [Batch, Seq_length, Channels(2048)] 以适应 Linear 层
        x_trans = x.transpose(1, 2)
        
        # 门控机制计算
        attention_v = torch.tanh(self.attention_V(x_trans))
        attention_u = torch.sigmoid(self.attention_U(x_trans))
        
        # 得到每个时间步的得分: [Batch, Seq_length, 1]
        a = self.attention_weights(attention_v * attention_u) 
        
        # 在时间序列维度上进行 softmax 归一化
        a = F.softmax(a, dim=1) 
        
        # 利用注意力权重进行加权求和，得到定长特征向量
        # a.transpose(1, 2): [Batch, 1, Seq_length]
        # x_trans: [Batch, Seq_length, Channels]
        # 结果: [Batch, 1, Channels] -> [Batch, Channels]
        out = torch.bmm(a.transpose(1, 2), x_trans).squeeze(1)
        
        return out, a

# ==========================================
# 4. 完整的 ResNet50-1D Attention 网络
# ==========================================
class ResNet501D_Attention(nn.Module):
    def __init__(self, input_channels, num_classes, num_blocks_list=[3, 4, 6, 3]):
        super(ResNet501D_Attention, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 使用带有 SE 模块的 Bottleneck
        self.layer1 = self._make_layer(64, num_blocks_list[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks_list[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks_list[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks_list[3], stride=2)

        # 核心修改：使用时序注意力层替换原有的 nn.AdaptiveAvgPool1d(1)
        resnet_out_channels = 512 * Bottleneck1D_SE.expansion # 2048
        self.temporal_attention = GatedTemporalAttention(in_features=resnet_out_channels)
        
        self.fc = nn.Linear(resnet_out_channels, num_classes)

    def _make_layer(self, base_channels, num_blocks, stride):
        layers = []
        layers.append(Bottleneck1D_SE(self.in_channels, base_channels, stride))
        self.in_channels = base_channels * Bottleneck1D_SE.expansion
        
        for _ in range(1, num_blocks):
            layers.append(Bottleneck1D_SE(self.in_channels, base_channels, stride=1))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x 的 shape 在此时通常为: [Batch, 2048, Reduced_Seq_length]
        # 经过时序注意力层，得到加权后的全局特征和注意力分布
        x_attended, attention_weights = self.temporal_attention(x)
        
        # 分类输出
        logits = self.fc(x_attended)
        
        # 返回分类结果以及注意力权重（用于后续的业务可视化与解释）
        return logits, attention_weights

# --- 测试运行代码 ---
if __name__ == "__main__":
    batch_size = 16
    seq_length = 250       # 时间步
    input_channels = 6     # 如果包括 acc xyz 和 gyro xyz，通道数为6
    output_dim = 1         # 二分类

    model = ResNet501D_Attention(input_channels=input_channels, num_classes=output_dim)

    dummy_input = torch.randn(batch_size, input_channels, seq_length)
    predictions, attn_weights = model(dummy_input)

    print(f"输入序列形状: {dummy_input.shape}")
    print(f"预测输出形状: {predictions.shape}")
    print(f"时序注意力权重形状: {attn_weights.shape} -> (Batch, 被池化缩短后的时间步长度, 1)")