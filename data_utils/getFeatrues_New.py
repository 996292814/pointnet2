import cv2
import numpy as np
import os
import json
import csv
import open3d as o3d
import PCA
import pandas as pd
# 提取图像特征
def extract_features(image, bbox_top_left, bbox_bottom_right, segmentation_mask, depth_image):
    # 计算 Bounding Box 相关特征
    bbox_width = bbox_bottom_right[0] - bbox_top_left[0]
    bbox_height = bbox_bottom_right[1] - bbox_top_left[1]
    bbox_area = bbox_width * bbox_height

    # 计算分割后图像的面积
    object_area = np.count_nonzero(image)/3

    # 计算深度直方图
    depth_histogram = calculate_depth_histogram(depth_image)

    return {
        'bbox_width': bbox_width,
        'bbox_height': bbox_height,
        'bbox_area': bbox_area,
        'object_area': object_area,
        'depth_histogram': depth_histogram
    }

def calculate_depth_histogram( depth_image, num_bins=12):
    # 获取非零像素的深度值
    depth_values = depth_image[depth_image > 0]

    # 计算深度直方图
    histogram, _ = np.histogram(depth_values, bins=num_bins, range=(0, np.max(depth_image)))
    # 归一化直方图
    # normalized_histogram = histogram / np.sum(histogram)

    # 构建特征向量
    # feature_vector = normalized_histogram.reshape(1, -1)  # 将直方图变成一维向量
    return histogram

def is_point_inside_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def generate_point_cloud(rgb_image_path, depth_npy_path, output_pcd_path):
    # 读取RGB图像和深度信息
    rgb_image = cv2.imread(rgb_image_path)
    depth_data = np.load(depth_npy_path)

    # 获取图像的宽度和高度
    height, width, _ = rgb_image.shape

    # 生成点云
    points = []
    colors = []
    for y in range(height):
        for x in range(width):
            # 获取深度信息
            depth = depth_data[y, x]
            # 判断RGB图像中像素值是否为 [0, 0, 0]，如果不是且深度大于 0，则生成点云
            if not np.all(rgb_image[y, x] == [0, 0, 0]) and depth > 0:
                z = depth
                x_coord = (x - width / 2) * z / (width / 2)
                y_coord = (height / 2 - y) * z / (height / 2)
                points.append([x_coord, y_coord, z])
                # 获取RGB颜色信息，指定颜色通道顺序为BGR
                color = rgb_image[y, x][::-1] / 255.0  # 归一化到 [0, 1]，并颜色通道顺序改为RGB
                colors.append(color)

    # 将点云数据转换为NumPy数组
    points = np.array(points)
    colors = np.array(colors)
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # 保存点云文件
    o3d.io.write_point_cloud(output_pcd_path, pcd)

def process_images_in_folder(folder_path, output_folder, annotation_root_folder, pointcloud_folder):
    # 获取文件夹中所有文件名包含"_rgb"的图像
    image_files = [f for f in os.listdir(folder_path) if "_rgb" in f]
    for image_file in image_files:
        # 构建完整的图像文件路径
        image_path = os.path.join(folder_path, image_file)
        # 读取彩色图像
        rgb_image = cv2.imread(image_path)
        # 读取深度图像
        depth_image = cv2.imread(os.path.join(folder_path.replace('images', 'depth'), f'{image_file.split("_rgb")[0]}_pdepth.png'))
        # 构建标注文件夹路径
        subfolder_name = image_file.split('_')[0]
        subfolder_name2 = image_file.split('_')[7]
        annotation_folder = os.path.join(annotation_root_folder, subfolder_name,subfolder_name2)
        an = f'{image_file.split("_rgb")[0]}'
        an2 = f'strawberry{an.split("strawberry")[1]}'
        annotation_file = os.path.join(annotation_folder, f'{an2}_keypoint.json')
        # 读取原始点云文件对应的重量
        weight_df = pd.read_csv(r"D:\3DPointCloud\ProcessedData\filter\PointCloudWeight.csv", header=None, names=['filename', 'weight'])
        original_filename = f'{image_file.split("_rgb")[0]}.pcd'
        print(original_filename)
        row = weight_df[weight_df['filename'] == original_filename]
        if row.empty:
            print('not found:', original_filename)
            continue
        weight = weight_df[weight_df['filename'] == original_filename]['weight'].iloc[0]

        # 使用 json.load() 读取 JSON 文件
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as json_file:
                data = json.load(json_file)
                annotations = data['annotations']

            # 生成只包含标注实例的图像，保留原图颜色和特征
            for idx, annotation in enumerate(annotations):
                in1 = f'{image_file.split("split")[1]}'
                in2 = f'{in1.split("_")[0]}'
                if(int(in2) != idx):
                    continue
                bbox = annotation["bbox"]
                segmentation = annotation["segmentation"]
                x1, y1, x2, y2 = map(int, bbox)
                # 提取图像特征
                features = extract_features(rgb_image, (x1, y1), (x2, y2), segmentation, depth_image)
                # 使用主成分分析点云的朝向
                pca = PCA.PCA(pointcloud_folder+f'\\{image_file.split("_rgb")[0]}.pcd')
                # 构建 CSV 行数据
                csv_data = [f'{image_file.split("_rgb")[0]}.pcd',
                            weight]

                # 将 CSV 行数据写入文件
                csv_file_path = os.path.join(output_folder, "PointCloudWeight.csv")
                with open(csv_file_path, mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(csv_data)

                # 构建 CSV 行数据
                features_data = [f'{image_file.split("_rgb")[0]}.pcd',
                            features['bbox_width'],
                            features['bbox_height'],
                            features['bbox_area'],
                            features['object_area']
                           ]
                # 点云朝向
                features_data.extend(pca.reshape(-1))
                # 深度图直方图
                features_data.extend(features['depth_histogram'].tolist())
                # 将 CSV 行数据写入文件
                csv_file_path = os.path.join(output_folder, "Featrues.csv")
                with open(csv_file_path, mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(features_data)

# 指定待处理数据的文件夹路径
root_folder = r"D:\3DPointCloud\ProcessedData\new\images"
output_folder = r"D:\3DPointCloud\ProcessedData\features_new"
annotation_root_folder = r"D:\3DPointCloud\ICRA2022\annotations\dyson_annotations\dyson_annotations"
pointcloud_folder = r"D:\3DPointCloud\ProcessedData\filter\pcd"
# 处理所有子文件夹中的图像
process_images_in_folder(root_folder, output_folder, annotation_root_folder, pointcloud_folder)
