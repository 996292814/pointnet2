import datetime
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from open3d import io
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from pointnet.model import PointNetfeat
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.pointnet2_utils import PointNetSetAbstraction
import provider
from torchsummary import summary

# 采用根均分误差（RMSE）

# Define PointNet model
class StrawberryPointNet(nn.Module):
    def __init__(self, num_class=1, normal_channel=False):
        super(StrawberryPointNet, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        #x = F.log_softmax(x, -1)
        return x #, l3_points


# Define dataset class
class StrawberryDataset(Dataset):
    def __init__(self, root_dir, csv_file):
        self.data = np.loadtxt(csv_file, dtype=np.str, delimiter=',')
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir, self.data[idx, 0])
        weight = float(self.data[idx, 1])

        # 读取点云数据
        point_cloud = io.read_point_cloud(file_name)
        points = np.asarray(point_cloud.points).T.astype(np.float32)

        # 对每个点的坐标进行标准化
        #points = preprocess_data(points)

        # 这里使用简单的填充，可以根据实际情况选择其他方法
        max_points = 12000  # 你的数据集中最大的点云维度
        if points.shape[1] < max_points:
            # 填充零
            points = np.pad(points, ((0, 0), (0, max_points - points.shape[1])), mode='constant')
        elif points.shape[1] > max_points:
            # 截断
            points = points[:, :max_points]

        return points, weight

class StrawberryDataset2(Dataset):
    def __init__(self, root_dir, csv_file, npoints=3000, data_augmentation=True):
        self.data = np.loadtxt(csv_file, dtype=np.str, delimiter=',')
        self.root_dir = root_dir
        self.npoints = npoints
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir, self.data[idx, 0])
        weight = float(self.data[idx, 1])

        # 读取点云数据
        point_cloud = io.read_point_cloud(file_name)
        points = np.asarray(point_cloud.points).T.astype(np.float32)

        # 随机抽样，动态调整点云数量
        num_points = self.npoints
        choice = np.random.choice(points.shape[1], num_points, replace=True)
        points = points[:, choice]

        # 归一化
        points = points - np.expand_dims(np.mean(points, axis=1), 1)  # center
        dist = np.max(np.sqrt(np.sum(points ** 2, axis=0)), 0)
        points = points / dist  # scale

        # 数据增强
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            points[[0, 2], :] = np.dot(rotation_matrix, points[[0, 2], :])  # random rotation
            points += np.random.normal(0, 0.02, size=points.shape)  # random jitter
            # 数据增强
            # augmented_points = translate_point_cloud(points)
            # augmented_points = rotate_point_cloud(augmented_points)
            # augmented_points = scale_point_cloud(augmented_points)
        return points, weight

# 数据预处理函数的修改
def preprocess_data(data):
    # Reshape data to (batch_size, num_points * num_features)
    batch_size, num_points, num_features = data.shape
    data_reshaped = data.reshape(batch_size, -1)

    # Standardize each point's coordinates
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data_reshaped)

    # Reshape data back to (batch_size, num_points, num_features)
    data_standardized = data_standardized.reshape(batch_size, num_points, num_features)
    return data_standardized

# Train function
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    print(model)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    print(len(train_loader))
    num_batch = len(train_loader)
    blue = lambda x: '\033[94m' + x + '\033[0m'
    all_losses = []  # 用于存储每个训练周期的损失值
    all_test_losses = []  # 存储每个训练周期的测试损失
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (batch_points, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels = labels.float().unsqueeze(1)
            model.train()
            batch_points, labels = batch_points.cuda(), labels.cuda()
            outputs = model(batch_points)
            loss = criterion(outputs, labels)# 使用 MSE 损失函数
            # loss = torch.sqrt(criterion(outputs, labels))  # 使用 RMSE 损失函数
            loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 设置梯度范数的最大值为1.0

            optimizer.step()
            running_loss += loss.item()
            print(f"epoch {epoch + 1}/{num_epochs},  batch{i}/{num_batch} , MSE loss: {loss.item()}")

            if i % 10 == 0:
                j, data = next(enumerate(test_loader))
                points, labels = data
                labels = labels.float().unsqueeze(1)
                model.eval()
                points, labels = points.cuda(), labels.cuda()
                outputs = model(points)
                loss = criterion(outputs, labels)# 使用 MSE 损失函数
                # loss = torch.sqrt(criterion(outputs, labels))  # 使用 RMSE 损失函数
                print("%s" % (blue("test")))
                print(f"epoch {epoch + 1}/{num_epochs},  batch{i}/{num_batch} ,MSE loss: {loss.item()}")
        all_losses.append(running_loss / len(train_loader))
        # 测试模型
        test_loss = test(model, test_loader, criterion)
        all_test_losses.append(test_loss)
        # 更新损失图表
        plt.plot(range(1, epoch + 2), all_losses, marker='o', label='Train Loss', color='blue')
        plt.plot(range(1, epoch + 2), all_test_losses, marker='o', label='Test Loss', color='orange')
        plt.title('Pointnet++ (train-e150-20240301-1.pth)')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xticks(range(1, epoch + 2)[::5])  # 设置 x 轴坐标刻度为整数
        plt.show()  # 非阻塞显示图表
        plt.pause(0.1)  # 等待一小段时间，确保图表有足够的时间显示

        # 调整学习率
        scheduler.step()
        # scheduler.step(running_loss / len(train_loader))
    # 保存模型
    torch.save(model.state_dict(), 'cls/train-e30-20240301-1.pth')

# 定义测试函数
def test(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for points, labels in test_loader:
            labels = labels.float().unsqueeze(1)
            points, labels = points.cuda(), labels.cuda()
            outputs = model(points)
            loss = criterion(outputs, labels)  # 使用 MSE 损失函数
            # loss = torch.sqrt(criterion(outputs, labels)) # 使用 RMSE 损失函数
            running_loss += loss.item()

    test_loss = running_loss / len(test_loader)
    # print(f"Test RMSE Loss: {test_loss}")

    return test_loss

# Main program
if __name__ == "__main__":
    print('开始时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))  # 打印按指定格式排版的时间
    # Load training and testing datasets
    train_dataset = StrawberryDataset( r"D:\3DPointCloud\ProcessedData\train\pcd", r"D:\3DPointCloud\ProcessedData\train\PointCloudWeight.csv")
    test_dataset = StrawberryDataset( r"D:\3DPointCloud\ProcessedData\val\pcd", r"D:\3DPointCloud\ProcessedData\val\PointCloudWeight.csv")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=25, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = StrawberryPointNet().cuda()
    criterion = nn.MSELoss().cuda()
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    summary(model, input_size=(3, 12000))
    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs=100)

    # Evaluate the model on the test set
    # test(model, test_loader)


    print('结束时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))  # 打印按指定格式排版的时间
