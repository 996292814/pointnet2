import pandas as pd
from scipy import stats

# 读取包含点云数量的CSV文件
csv_file_path = r'D:\3DPointCloud\ProcessedData\features_new\PointCloudWeight_with_point_count.csv'
df = pd.read_csv(csv_file_path)

# 找出点云数量低于2048个点的数据
error_point_cloud = df[df['point_count'] < 2048]

# 将异常点云数量的数据保存到errorPointCloud.csv文件中
error_point_cloud.to_csv('errorPointCloud.csv', index=False)

# 读取包含点云数量和重量的CSV文件
csv_file_path = r'D:\3DPointCloud\ProcessedData\features_new\PointCloudWeight_with_point_count.csv'
df = pd.read_csv(csv_file_path)

# 剔除点云数量低于2048的数据
df = df[df['point_count'] >= 2048]

# 四舍五入保留重量值的整数部分
df['rounded_weight'] = df['weight'].round()

# 根据四舍五入后的重量值分组，计算每组的点云数量均值
mean_point_count_by_weight = df.groupby('rounded_weight')['point_count'].mean()

# 使用每组的点云数量均值来填充每个数据点对应的均值
df['mean_point_count_by_weight'] = df['rounded_weight'].map(mean_point_count_by_weight)

# 计算每个数据点与其所在组的均值的相对偏差百分比
df['point_count_percentage_deviation'] = ((df['point_count'] - df['mean_point_count_by_weight']) / df['mean_point_count_by_weight']) * 100

# 设置阈值百分比，根据需求调整
threshold_percentage = -10

# 找出相对偏差小于阈值的数据点
outliers = df[df['point_count_percentage_deviation'] < threshold_percentage]
# 将点云数量低于2048的数据点也加入到异常值中
outliers = pd.concat([outliers, error_point_cloud])
# 输出异常值
print("异常数据：")
print(outliers)
# 输出异常值到error.csv文件
outliers.to_csv('error.csv', index=False)
