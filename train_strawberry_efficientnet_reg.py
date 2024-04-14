"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from data_utils.StrawberryDataLoaderNew import OwnPointCloudDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='pointnet_efficientnet_regression', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=1, type=int, choices=[4, 10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampiling')
    parser.add_argument('--data_path',  default=r"D:\3DPointCloud\ProcessedData\ModelNet-filter-augmented", help='data path')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def test(model, loader, num_class=40):
    # 初始化一个空列表，用于存储每个批次的MAE
    all_mae = []
    classifier = model.eval()
    for j, (points, images, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target, images = points.cuda(), target.cuda(), images.cuda()
        # 将输入数据的维度转置，以符合模型输入的格式要求。
        points = points.transpose(2, 1)
        # 通过模型进行预测，pred 是模型的预测结果。
        pred, _ = classifier(points, images)
        # 计算每个样本的绝对误差
        abs_error = torch.abs(pred.squeeze() - target.float())
        # 计算每个批次的平均绝对误差（MAE），并添加到列表中
        batch_mae = torch.mean(abs_error).item()
        all_mae.append(batch_mae)

        # 计算所有批次的平均绝对误差（MAE）
    mae = np.mean(all_mae)

    return mae

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('regression')  # 模型放到分类的文件夹中
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = args.data_path

    train_dataset = OwnPointCloudDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = OwnPointCloudDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_strawberry_efficientnet_reg.py', str(exp_dir))
    shutil.copy('data_utils/StrawberryDataLoaderNew.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_test_mae = 0.0
    # 在训练循环开始之前定义一个空列表，用于存储每个 epoch 的 MAE
    train_accuracy = []
    test_accuracy = []
    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()
        mean_mae = []
        scheduler.step()
        # print('================batch_id===============')

        for batch_id, (points, images, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader),
                                               smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            # print(points)
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target, images = points.cuda(), target.cuda(), images.cuda()

            pred, trans_feat = classifier(points, images)
            loss = criterion(pred, target.float(), trans_feat)
            loss.backward()
            optimizer.step()
            global_step += 1
            # 计算每个批次的MAE，并添加到列表中
            batch_mae = torch.abs(pred.squeeze() - target.float()).mean().item()
            mean_mae.append(batch_mae)
        # 计算每个epoch的平均MAE
        train_instance_mae = np.mean(mean_mae)
        train_accuracy.append(train_instance_mae)
        log_string('Train Instance MAE: %f' % train_instance_mae)

        with torch.no_grad():
            test_mae = test(classifier.eval(), testDataLoader, num_class=num_class)
            if(epoch==0):
                best_test_mae = test_mae
            if (test_mae <= best_test_mae):
                best_test_mae = test_mae
                best_epoch = epoch + 1

            log_string('Test MAE: %f' % test_mae)
            log_string('Best Test MAE: %f' % best_test_mae)
            test_accuracy.append(test_mae)
            if (test_mae <= best_test_mae):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'test_mae': test_mae,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1
        # 更新损失图表
        plt.plot(range( epoch + 1), train_accuracy, marker='o', label='Train Loss', color='blue')
        plt.plot(range( epoch + 1), test_accuracy, marker='o', label='Test Loss', color='orange')
        plt.title('Pointnet++ ')
        plt.xlabel('Epoch')
        plt.ylabel('MAE Loss')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        # plt.xticks(range( epoch + 1)[::5])  # 设置 x 轴坐标刻度为整数
        plt.show()  # 非阻塞显示图表
        plt.pause(0.1)  # 等待一小段时间，确保图表有足够的时间显示
    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)