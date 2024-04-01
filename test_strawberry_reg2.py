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
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--num_category', default=37, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='2024-03-21_21-59', required=False, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--data_path', default=r"D:\3DPointCloud\ProcessedData\ModelNet-filter-augmented", help='data path')
    return parser.parse_args()

def evaluate_model(model, data_loader, tolerances):
    """
    Evaluate the model's accuracy based on the percentage of predictions within the specified tolerances.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        tolerances (list of floats): List of tolerance levels for acceptable error in predictions.

    Returns:
        accuracies (list of floats): List of accuracies corresponding to each tolerance level.
    """
    model.eval()
    # 初始化每个容忍度下的正确预测数量
    correct_predictions = [0] * len(tolerances)
    # 初始化样本总数
    total_samples = 0

    with torch.no_grad():
        for points, target in tqdm(data_loader, total=len(data_loader)):
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()
            # 将输入数据的维度转置，以符合模型输入的格式要求。
            points = points.transpose(2, 1)
            pred, _ = model(points)
            # 计算预测值与真实值之间的相对误差
            error = torch.abs(pred.squeeze() - target.float()) / target.float()
            # 更新总样本数
            total_samples += points.size(0)

            # 对于每个容忍度，统计预测误差在容忍范围内的样本数量
            for i, tol in enumerate(tolerances):
                correct_predictions[i] += torch.sum(error <= tol).item()

    # 计算每个容忍度下的准确度，即正确预测数量除以总样本数
    accuracies = [round((correct / total_samples) * 100, 2) for correct in correct_predictions]
    return accuracies

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
    # 设置容忍度
    tolerances = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    best_test_accuracy =0.0
    with torch.no_grad():
        test_accuracy_per_tolerance = evaluate_model(classifier, testDataLoader, tolerances)
        for i, tol in enumerate(tolerances):
            if test_accuracy_per_tolerance[i] >= best_test_accuracy:
                best_test_accuracy = test_accuracy_per_tolerance[i]
        log_string('Test Accuracy for Tolerance {}: {}'.format(tolerances, test_accuracy_per_tolerance))
        log_string('Best Test Accuracy for Tolerance {}'.format(best_test_accuracy))
    # 绘制准确度图表
    plt.plot(tolerances, test_accuracy_per_tolerance, marker='o')
    plt.title('Model Accuracy at Different Tolerances')
    plt.xlabel('Tolerance')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
