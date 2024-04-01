"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import matplotlib.pyplot as plt
from data_utils.StrawberryDataLoader import OwnPointCloudDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--num_category', default=37, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='2024-03-12_19-00', required=False, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--data_path', default=r"D:\3DPointCloud\ProcessedData\ModelNet-filter", help='data path')
    return parser.parse_args()


def test(model, loader, num_class=40):
    # 初始化一个空列表，用于存储每个批次的MAE
    all_mae = []
    all_predictions = []
    all_labels = []
    classifier = model.eval()
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        print('j:', j, target)
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()
        # 将输入数据的维度转置，以符合模型输入的格式要求。
        points = points.transpose(2, 1)
        # 通过模型进行预测，pred 是模型的预测结果。
        pred, _ = classifier(points)
        # 计算每个样本的绝对误差
        abs_error = torch.abs(pred.squeeze() - target.float())
        # 计算每个批次的平均绝对误差（MAE），并添加到列表中
        batch_mae = torch.mean(abs_error).item()
        all_mae.append(batch_mae)
        # 输出每个点云文件的预测值
        if target.item() ==0:
            print('test')
        # print(f"Point Cloud {j + 1}: Predicted Weight: {pred.cpu().numpy()}, True Weight: {target.item()}")
        # 存储模型的输出和真实标签
        all_predictions.extend(pred.cpu().numpy().flatten())  # 调整形状为一维数组
        all_labels.extend(target.cpu().numpy().flatten())  # 调整形状为一维数组
    # 计算所有批次的平均绝对误差（MAE）
    mae = np.mean(all_mae)
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
    return mae


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/regression/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path =  args.data_path

    # test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    test_dataset = OwnPointCloudDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=10)
    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        test_mae = test(classifier.eval(), testDataLoader, num_class=num_class)
        log_string('Test MAE: %f' % test_mae)



if __name__ == '__main__':
    args = parse_args()
    main(args)
