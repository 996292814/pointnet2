import torch
import torch.nn as nn
from train_regression5 import StrawberryDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from train_regression5 import StrawberryPointNet
import matplotlib.pyplot as plt
import numpy as np

print(torch.cuda.is_available())
print(torch.__version__)
# 步骤1: 定义相同结构的模型
model = StrawberryPointNet()

# 步骤2: 加载保存的模型参数
saved_model_path = 'cls/train-e30-20240229-3.pth'
checkpoint = torch.load(saved_model_path)

# 步骤3: 将加载的参数应用到模型
model.load_state_dict(checkpoint)

# 步骤4: 将模型设置为评估模式
model.eval()

# 现在你可以使用加载的模型进行推断
# 例如，如果有一个测试数据集 test_loader：
test_dataset = StrawberryDataset( r"D:\3DPointCloud\ProcessedData\val\pcd", r"D:\3DPointCloud\ProcessedData\val\PointCloudWeight.csv")
# Create data loaders
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
all_predictions = []
all_labels = []
with torch.no_grad():
    for i, (points, labels) in enumerate(test_loader):

        labels = labels.float().unsqueeze(1)
        # 模型推断
        outputs = model(points)
        # 获取模型的输出
        predictions = outputs.cpu().numpy()
        # 存储模型的输出和真实标签
        all_predictions.extend(predictions.flatten())  # 调整形状为一维数组
        all_labels.extend(labels.cpu().numpy().flatten())  # 调整形状为一维数组
        # 输出每个点云文件的预测值
        print(f"Point Cloud {i + 1}: Predicted Weight: {predictions}, True Weight: {labels.item()}")

    mse = mean_squared_error(all_labels, all_predictions, squared=False)
    print(f"\nRoot Mean Squared Error on Test Data: {mse}")

# 获取每一个数据点的索引
sample_indices = range(len(all_labels))
# 绘制折线图
plt.plot(sample_indices, all_labels, label='True Weight', color='blue')
plt.plot(sample_indices, all_predictions, label='Predicted Weight', color='orange')

plt.xlabel('Sample Index')
plt.ylabel('Weight')
plt.title('True vs Predicted Weight(Pointnet++)')
plt.legend()
plt.show()