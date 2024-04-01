import os
import random
import numpy as np
import open3d as o3d
import pandas as pd


def random_translate(point_cloud):
    """在xy平面上随机平移点云，保持z轴固定"""
    translation = np.random.uniform(-10, 10, size=2)
    point_cloud[:, :2] += translation
    return point_cloud, 'translate'

def random_rotate(point_cloud):
    """在所有三个轴上应用随机旋转"""
    angles = np.random.uniform(-10, 10, size=3)
    rotation = o3d.geometry.get_rotation_matrix_from_xyz(np.radians(angles))
    rotated_points = np.dot(point_cloud[:, :3], rotation.T)
    point_cloud[:, :3] = rotated_points
    return point_cloud, 'rotate'


def rotate_point_cloud(point_cloud):
    """随机旋转点云"""
    angles = np.radians(np.random.uniform(-5, 5, size=3))
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]), np.cos(angles[0])]
    ]) @ np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ]) @ np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]), np.cos(angles[2]), 0],
        [0, 0, 1]
    ])

    rotated_points = np.dot(point_cloud[:, :3], rotation_matrix.T)
    rotated_point_cloud = np.column_stack((rotated_points, point_cloud[:, 3:]))
    return rotated_point_cloud, 'rotate'


def add_noise(point_cloud, noise_level=0.01):
    """对点云添加噪声"""
    noise = np.random.normal(0, noise_level, size=point_cloud.shape)
    noisy_point_cloud = point_cloud + noise
    return noisy_point_cloud, 'noise'


def translate_point_cloud(point_cloud):
    """对点云进行随机平移"""
    translation = np.random.uniform(-0.1, 0.1, size=3)
    translated_points = point_cloud[:, :3] + translation
    translated_point_cloud = np.column_stack((translated_points, point_cloud[:, 3:]))
    return translated_point_cloud, 'translate'


def mirror_point_cloud(point_cloud):
    """镜像点云"""
    mirrored_points = point_cloud[:, :3] * np.array([-1, 1, 1])
    mirrored_point_cloud = np.column_stack((mirrored_points, point_cloud[:, 3:]))
    return mirrored_point_cloud, 'mirror'


def augment_point_cloud(input_path, output_folder, original_weight_csv_path, augmented_weight_list):
    # 读取原始点云文件
    original_point_cloud = o3d.io.read_point_cloud(input_path)
    original_points = np.asarray(original_point_cloud.points)
    original_colors = np.asarray(original_point_cloud.colors)
    original_data = np.column_stack((original_points, original_colors))

    # 数据增强
    augmentations = [random_translate, random_rotate]
    # 读取原始点云文件对应的重量
    weight_df = pd.read_csv(original_weight_csv_path, header=None, names=['filename', 'weight'])
    original_filename = os.path.basename(input_path)
    print(original_filename)
    row = weight_df[weight_df['filename'] == original_filename]
    if row.empty:
        return
    original_weight = weight_df[weight_df['filename'] == original_filename]['weight'].iloc[0]

    for augmentation in augmentations:
        augmented_data, method_name = augmentation(original_data.copy())
        # 保存增强后的点云
        augmented_point_cloud = o3d.geometry.PointCloud()
        augmented_point_cloud.points = o3d.utility.Vector3dVector(augmented_data[:, :3])
        augmented_point_cloud.colors = o3d.utility.Vector3dVector(augmented_data[:, 3:])

        output_filename = f'{os.path.splitext(original_filename)[0]}_{method_name}.pcd'
        output_path = os.path.join(output_folder, output_filename)
        o3d.io.write_point_cloud(output_path, augmented_point_cloud)

        # 更新augmented_weights.csv文件
        if original_weight is not None:
            augmented_weight_list.append({'filename': output_filename, 'weight': original_weight})


# 示例调用

if __name__ == "__main__":
    input_folder = r"D:\3DPointCloud\ProcessedData\filter\pcd"
    output_folder = r"D:\3DPointCloud\ProcessedData\augmented_filter\pcd"
    original_weight_csv_path = r"D:\3DPointCloud\ProcessedData\filter\PointCloudWeight.csv"
    augmented_weight_csv_path = r"D:\3DPointCloud\ProcessedData\augmented_filter\PointCloudWeight.csv"
    augmented_weight_list = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.pcd'):
            input_path = os.path.join(input_folder, filename)
            augment_point_cloud(input_path, output_folder, original_weight_csv_path, augmented_weight_list)

    # 生成augmented_weights.csv文件
    augmented_weight_df = pd.DataFrame(augmented_weight_list)
    augmented_weight_df.to_csv(augmented_weight_csv_path, index=False, header=['filename', 'weight'])
