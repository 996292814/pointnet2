import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt

# 点云规范化
def PC_NORMLIZE(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


# 随机采样
def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    # N = data.shape[0]
    N = data
    if (N == num_sample):
        return data
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...]
    else:
        # 相当于从N中抽取 num_sample - N个随机数组成一维数组array，成为data的下标索引值
        sample = np.random.choice(N, num_sample - N)
        dup_data = data[sample, ...]  # 取数据
        # 按行拼接
        return np.concatenate([data, dup_data], axis=0)

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

# 定义函数对点云进行随机采样和归一化处理
def sample_and_normalize_point_cloud(pcd, num_points=4096):
    # 随机采样
    pcd = pcd.select_random_points(num_points)

    # 归一化处理
    pcd.normalize_normals()
    pcd.normalize_colors()
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=pcd.get_center())
    return pcd


# 定义函数将点云保存为只有xyz的txt文件
def save_point_cloud_xyz(points, file_path):
    # points = np.asarray(pcd.points)
    np.savetxt(file_path, points, fmt='%f')

def dataLoader(split):
    csv_source_path = r"D:\3DPointCloud\ProcessedData\{}\PointCloudWeight.csv".format(split)
    pcd_source_path = r"D:\3DPointCloud\ProcessedData\{}\pcd".format(split)

    csv_files = np.loadtxt(csv_source_path, dtype=np.str, delimiter=',')
    for csv_file in csv_files:
        file_name = os.path.join(pcd_source_path, csv_file[0])
        weight = round(float(csv_file[1]))  # 四舍五入重量取整
        # 不要小数点的精度，直接以整数形式做标签
        allclass.add(int(weight))
        # 以重量为类别。先检查类别文件夹在不在，不在则创建
        lable = str(int(weight))
        output_folder = os.path.join(target_path, lable)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        # 读取pcd文件
        pcd = o3d.io.read_point_cloud(file_name)
        # 将点云数据转换为 NumPy 数组
        points = np.asarray(pcd.points).astype(np.float32)
        # 对点云进行随机采样和归一化处理
        processed_pcd = farthest_point_sample(points, 2048)
        processed_pcd = PC_NORMLIZE(processed_pcd)
        # processed_pcd = sample_and_normalize_point_cloud(pcd)

        # 将处理后的点云保存为只有xyz的txt文件
        txt_file_path = os.path.join(output_folder, lable + '_' + os.path.splitext(csv_file[0])[0] + ".txt")
        save_point_cloud_xyz(processed_pcd, txt_file_path)
        filelist.append(lable + '_' + os.path.splitext(csv_file[0])[0] + ".txt")
        if 'train' == split or 'filter' == split or 'augmented_filter' == split:
            trainlist.append(lable + '_' + os.path.splitext(csv_file[0])[0])
            train_file_path = target_path+r"\trainlist.txt"
            with open(train_file_path, 'w') as f:
                for item in trainlist:
                    f.write("%s\n" % item)
        if 'val' == split:
            testlist.append(lable + '_' + os.path.splitext(csv_file[0])[0])
            test_file_path = target_path+r"\testlist.txt"
            with open(test_file_path, 'w') as f:
                for item in testlist:
                    f.write("%s\n" % item)
        class_file_path = target_path+r"\allclass.txt"
        with open(class_file_path, 'w') as f:
            for item in allclass:
                f.write("%d\n" % item)

        file_list_path = target_path+r"\filelist.txt"
        with open(file_list_path, 'w') as f:
            for item in filelist:
                f.write("%s\n" % item)
        print(f"{file_name} processed and saved as {os.path.basename(txt_file_path)}")
if __name__ == '__main__':
    allclass = set()
    filelist = list()
    testlist = list()
    trainlist = list()
    target_path = r"D:\3DPointCloud\ProcessedData\ModelNet-filter-augmented"
    # train
    # dataLoader(split='train')
    # dataLoader(split='val')
    # filter 剔除了离群点后的点云
    dataLoader(split='augmented_filter')
    # split = 'train'
    # csv_source_path = r"D:\3DPointCloud\ProcessedData\{}\PointCloudWeight.csv".format(split)
    # pcd_source_path = r"D:\3DPointCloud\ProcessedData\{}\pcd".format(split)
    #
    # csv_files = np.loadtxt(csv_source_path, dtype=np.str, delimiter=',')
    # for csv_file in csv_files:
    #     file_name = os.path.join(pcd_source_path, csv_file[0])
    #     weight = float(csv_file[1])
    #     # 不要小数点的精度，直接以整数形式做标签
    #     allclass.add(int(weight))
    #     # 以重量为类别。先检查类别文件夹在不在，不在则创建
    #     lable = str(int(weight))
    #     output_folder = os.path.join(target_path, lable)
    #     if not os.path.exists(output_folder):
    #         os.mkdir(output_folder)
    #     # 读取pcd文件
    #     pcd = o3d.io.read_point_cloud(file_name)
    #     # 将点云数据转换为 NumPy 数组
    #     points = np.asarray(pcd.points).astype(np.float32)
    #     # 对点云进行随机采样和归一化处理
    #     processed_pcd = farthest_point_sample(points, 4096)
    #     processed_pcd = PC_NORMLIZE(processed_pcd)
    #     # processed_pcd = sample_and_normalize_point_cloud(pcd)
    #
    #     # 将处理后的点云保存为只有xyz的txt文件
    #     txt_file_path = os.path.join(output_folder, os.path.splitext(csv_file[0])[0] + ".txt")
    #     save_point_cloud_xyz(processed_pcd, txt_file_path)
    #     filelist.append(lable+'/'+os.path.splitext(csv_file[0])[0] + ".txt")
    #     if 'train' == split:
    #         trainlist.append(lable+'_'+os.path.splitext(csv_file[0])[0])
    #         train_file_path = r"D:\3DPointCloud\ProcessedData\ModelNet\trainlist.txt"
    #         with open(train_file_path, 'w') as f:
    #             for item in trainlist:
    #                 f.write("%s\n" % item)
    #     if 'test' == split:
    #         testlist.append(lable+'_'+os.path.splitext(csv_file[0])[0])
    #         test_file_path = r"D:\3DPointCloud\ProcessedData\ModelNet\testlist.txt"
    #         with open(test_file_path, 'w') as f:
    #             for item in testlist:
    #                 f.write("%s\n" % item)
    #     class_file_path = r"D:\3DPointCloud\ProcessedData\ModelNet\allclass.txt"
    #     with open(class_file_path, 'w') as f:
    #         for item in allclass:
    #             f.write("%d\n" % item)
    #
    #     file_list_path = r"D:\3DPointCloud\ProcessedData\ModelNet\filelist.txt"
    #     with open(file_list_path, 'w') as f:
    #         for item in filelist:
    #             f.write("%d\n" % item)
    #     print(f"{file_name} processed and saved as {os.path.basename(txt_file_path)}")
    #     # 可视化点云
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(111, projection='3d')
    #     # ax.scatter(processed_pcd[:, 0], processed_pcd[:, 1], processed_pcd[:, 2], c='b', marker='.')
    #     # ax.set_xlabel('X')
    #     # ax.set_ylabel('Y')
    #     # ax.set_zlabel('Z')
    #     # plt.show()