import pandas as pd
import os
import open3d as o3d

# 读取CSV文件
csv_file_path = r'D:\3DPointCloud\ProcessedData\features_new\PointCloudWeight.csv'
df = pd.read_csv(csv_file_path, header=None)
df.columns = ['pcd_file', 'weight']  # 不包括第三列

# 添加第三列
df['point_count'] = ''

# 循环每个PCD文件
for index, row in df.iterrows():
    pcd_file = row['pcd_file']

    # 假设点云文件在同一目录下
    pcd_path = os.path.join(r'D:\3DPointCloud\ProcessedData\filter\pcd', pcd_file)

    # 读取点云文件
    cloud = o3d.io.read_point_cloud(pcd_path)

    # 获取点云的点数量并填充到DataFrame中
    point_count = len(cloud.points)  # 获取点云点数量
    df.at[index, 'point_count'] = point_count

# 将带有点云数量的DataFrame保存回CSV文件
output_csv_file_path = r'D:\3DPointCloud\ProcessedData\features_new\PointCloudWeight_with_point_count.csv'
df.to_csv(output_csv_file_path, index=False)
