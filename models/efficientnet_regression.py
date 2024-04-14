import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from efficientnet_pytorch import EfficientNet

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        # 加载 EfficientNet-B0 模型
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        # 去掉 EfficientNet 最后的全连接层
        self.efficientnet._fc = nn.Identity()
        # 定义pointnet++提取特征方法
        # self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        # 点云特征和图像特征融合的全连接层
        self.fc1 = nn.Linear(1280, 640)  # 调整全连接层维度以匹配后续网络
        self.fc2 = nn.Linear(640, 320)
        self.fc3 = nn.Linear(320, 1)  # 输出层维度改为 1，因为是回归任务
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(640)
        self.bn2 = nn.BatchNorm1d(320)
        self.relu = nn.ReLU()

    def forward(self, x, img):
        # 提取图片特征
        img_feat = self.efficientnet.extract_features(img)
        img_feat = F.adaptive_avg_pool2d(img_feat, 1).squeeze(-1).squeeze(-1)  # 将特征池化成一个向量
        # 提取点云特征
        # x, trans, trans_feat = self.feat(x)
        # 将提取到的特征向量拼接在一起
        # fused_feat = torch.cat((img_feat, x), dim=1)
        # 全连接层
        # fused_feat = F.relu(self.fc_fusion(fused_feat))
        x = F.relu(self.bn1(self.fc1(img_feat)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)  # 去掉softmax激活函数
        return x, 1

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        # 回归任务使用MSE均方误差
        loss = F.mse_loss(pred, target)
        return loss
