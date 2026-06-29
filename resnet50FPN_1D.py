import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ResNet50_1D_FPN(nn.Module):
    """
    带特征金字塔的一维残差网络 ResNet50-1D-FPN
    """
    def __init__(self, input_channels, num_classes, num_blocks_list=[3, 4, 6, 3], fpn_channels=256):
        super(ResNet50_1D_FPN, self).__init__()
        self.in_channels = 64
        self.fpn_channels = fpn_channels

        # --- 1. 初始特征提取层 (Bottom-up) ---
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # --- 2. 四个残差阶段 (C2, C3, C4, C5) ---
        self.layer1 = self._make_layer(64, num_blocks_list[0], stride=1)   # C2: 输出通道 256
        self.layer2 = self._make_layer(128, num_blocks_list[1], stride=2)  # C3: 输出通道 512
        self.layer3 = self._make_layer(256, num_blocks_list[2], stride=2)  # C4: 输出通道 1024
        self.layer4 = self._make_layer(512, num_blocks_list[3], stride=2)  # C5: 输出通道 2048

        # --- 3. FPN 结构 (Top-down & Lateral) ---
        # 横向连接：用 1x1 卷积将 C2~C5 的通道数统一压缩到 fpn_channels (默认256)
        self.latlayer1 = nn.Conv1d(256, fpn_channels, kernel_size=1)
        self.latlayer2 = nn.Conv1d(512, fpn_channels, kernel_size=1)
        self.latlayer3 = nn.Conv1d(1024, fpn_channels, kernel_size=1)
        self.latlayer4 = nn.Conv1d(2048, fpn_channels, kernel_size=1)

        # 平滑层：3x3 卷积，用于消除上采样(插值)带来的混叠效应
        self.smooth1 = nn.Conv1d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv1d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv1d(fpn_channels, fpn_channels, kernel_size=3, padding=1)

        # --- 4. 池化与分类器 ---
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # 因为我们融合了 P2, P3, P4, P5 四个尺度的特征，所以最后全连接层的输入是 fpn_channels * 4
        self.fc = nn.Linear(fpn_channels * 4, num_classes)

    def _make_layer(self, base_channels, num_blocks, stride):
        layers = []
        layers.append(Bottleneck1D(self.in_channels, base_channels, stride))
        self.in_channels = base_channels * Bottleneck1D.expansion
        for _ in range(1, num_blocks):
            layers.append(Bottleneck1D(self.in_channels, base_channels, stride=1))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        """
        对高层特征 x 进行 1D 最近邻上采样，并与浅层特征 y 逐元素相加
        :param x: 高层特征图 (通道数与 y 相同，但时序长度较短)
        :param y: 浅层特征图
        """
        _, _, length = y.size()
        # 1D 上采样匹配时序长度
        return F.interpolate(x, size=length, mode='nearest') + y

    def forward(self, x):
        # --- Bottom-up 过程 ---
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        c2 = self.layer1(x)  # 细节特征
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)  # 抽象语义特征

        # --- Top-down 过程与横向连接 ---
        p5 = self.latlayer4(c5)
        p4 = self._upsample_add(p5, self.latlayer3(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer1(c2))

        # --- 通过平滑层去除混叠效应 ---
        p4 = self.smooth3(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth1(p2)

        # --- 多尺度特征聚合与分类 ---
        # 此时有了 4 个维度的特征图: P2, P3, P4, P5
        # 分别进行全局平均池化展平 [Batch_size, fpn_channels]
        p5_pool = self.avgpool(p5).view(p5.size(0), -1)
        p4_pool = self.avgpool(p4).view(p4.size(0), -1)
        p3_pool = self.avgpool(p3).view(p3.size(0), -1)
        p2_pool = self.avgpool(p2).view(p2.size(0), -1)

        # 将不同尺度的语义向量拼接在特征维度 [Batch_size, fpn_channels * 4]
        multi_scale_features = torch.cat([p5_pool, p4_pool, p3_pool, p2_pool], dim=1)
        
        # 分类输出
        out = self.fc(multi_scale_features)
        
        return out


# 1. 实例化我们之前的模型
model = ResNet50_1D_FPN(input_channels=3, num_classes=1, fpn_channels=256)

# 2. 定义损失函数
# 假设你的异常数据(正样本)占总数据的 10%，正常数据(负样本)占 90%
# 那么正样本的权重可以设置为：负样本数 / 正样本数 = 9.0
pos_weight_value = torch.tensor([9.0]) 
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)

# 3. 定义优化器
# lr=1e-4 是一个非常稳妥的初始学习率，weight_decay 用于防止过拟合
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# 4. 定义学习率调度器 (可选，但强烈推荐)
# 当验证集的 loss 停止下降时，自动将学习率乘以 0.1
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# --- 模拟单步训练过程 ---
model.train()

# 假设取到了一个 batch 的数据和标签
dummy_inputs = torch.randn(16, 3, 250) 
# 标签形状需与输出形状一致：[Batch_size, 1]，0代表正常，1代表异常
dummy_labels = torch.empty(16, 1).random_(2) 

# 前向传播
outputs = model(dummy_inputs)

# 计算损失
loss = criterion(outputs, dummy_labels)

# 反向传播与优化
optimizer.zero_grad()  # 清空梯度
loss.backward()        # 反向传播计算梯度
optimizer.step()       # 更新权重

print(f"当前 Batch 的 Loss: {loss.item():.4f}")