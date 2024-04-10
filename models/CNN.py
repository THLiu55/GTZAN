import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels, height, width, num_classes):
        super(CNN, self).__init__()
        # 为了简化，我们先不计算'padding=same'的等价值，而是选择一个常见的填充值
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=120, kernel_size=(2,2), padding=(1,1))
        
        # 初始化池化层，没有特殊填充值
        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        self.pool_last = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # 初始化批归一化层
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.bn3 = nn.BatchNorm2d(num_features=120)
        
        # 根据输入尺寸和池化层设置动态计算Flatten之后的尺寸
        # 这里仅为示例，实际应用时需要根据实际输入尺寸和池化/卷积操作调整
        flatten_size = 120 * (height // 8) * (width // 8)
        self.fc1 = nn.Linear(in_features=flatten_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)
        
        # 初始化Dropout层
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = F.relu(x)
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool_last(x)
        
        x = torch.flatten(x, 1) # 展平操作，除了批次维度外全部展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
