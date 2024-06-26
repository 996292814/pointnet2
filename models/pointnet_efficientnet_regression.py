import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from efficientnet_pytorch import EfficientNet
from efficientnetV2 import EfficientnetV2
class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        # 加载 EfficientNet-B0 模型
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b1')
        # 去掉 EfficientNet 最后的全连接层
        self.efficientnet._fc = nn.Identity()
        # 加载efficientnetV2
        # self.EfficientnetV2 = EfficientnetV2(model_type='s', class_num=1, drop_connect_rate=0.2, se_rate=0.25)

        # 定义pointnet++提取特征方法
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        # 点云特征和图像特征融合的全连接层
        self.fc_fusion = nn.Linear(2304, 1024)  # 增加输出维度以防止信息丢失
        self.fc1 = nn.Linear(1024, 512)  # 调整全连接层维度以匹配后续网络
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # 输出层维度改为 1，因为是回归任务
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x, img):
        # 提取图片特征
        img_feat = self.efficientnet.extract_features(img)
        # img_feat = self.EfficientnetV2(img)
        img_feat = F.adaptive_avg_pool2d(img_feat, 1).squeeze(-1).squeeze(-1)  # 将特征池化成一个向量
        # 提取点云特征
        x, trans, trans_feat = self.feat(x)
        # 特征融合
        # 简单拼接方法
        fused_feat = torch.cat((img_feat, x), dim=1)
        # 计算加权平均特征
        # fused_feat = 0.7 * x + 0.3 * img_feat
        # 全连接层
        fused_feat = F.relu(self.fc_fusion(fused_feat))
        x = F.relu(self.bn1(self.fc1(fused_feat)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)  # 去掉softmax激活函数
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        # 回归任务使用MSE均方误差
        loss = F.mse_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
