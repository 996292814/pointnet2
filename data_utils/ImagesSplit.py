import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def dataLoader(split):
    csv_source_path = r"D:\3DPointCloud\ProcessedData\{}\PointCloudWeight.csv".format(split)
    image_source_path = r"D:\3DPointCloud\ProcessedData\{}\images".format(split)

    csv_files = np.loadtxt(csv_source_path, dtype=np.str, delimiter=',')
    for csv_file in csv_files:
        file_name = os.path.join(image_source_path, csv_file[0].split(".")[0]+'_rgb.png')
        # 读取彩色图像
        rgb_image = cv2.imread(file_name)
        weight = round(float(csv_file[1]))  # 四舍五入重量取整
        # 不要小数点的精度，直接以整数形式做标签
        allclass.add(int(weight))
        # 以重量为类别。先检查类别文件夹在不在，不在则创建
        lable = str(int(weight))
        output_folder = os.path.join(target_path, lable)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        # 将处理后的点云保存为只有xyz的txt文件
        proc_file_path = os.path.join(output_folder, lable + '_' + csv_file[0].split(".")[0]+'_rgb.png')
        cv2.imwrite(proc_file_path, rgb_image)
        filelist.append(lable + '_' + csv_file[0].split(".")[0]+'_rgb.png')
        if 'train' == split or 'filter' == split or 'augmented_filter' == split or 'new' == split:
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
        print(f"{file_name} processed and saved as {os.path.basename(proc_file_path)}")
if __name__ == '__main__':
    allclass = set()
    filelist = list()
    testlist = list()
    trainlist = list()
    target_path = r"D:\3DPointCloud\ProcessedData\ModelNet-images\original"
    # train
    # dataLoader(split='train')
    # dataLoader(split='val')
    # filter 剔除了离群点后的点云
    dataLoader(split='new')
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