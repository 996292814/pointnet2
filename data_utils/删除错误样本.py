import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA


labels_df = pd.read_csv(r"D:\3DPointCloud\ProcessedData\filter\PointCloudWeight.csv", header=None)
labels_df.columns = ["SampleName", "Label"]

labels_df.set_index("SampleName", inplace=True)
# 读取错误数据的文件
# error_data_file = r"D:\3DPointCloud\ProcessedData\filter\ErrorPointCloud.txt"
error_data_file = "error.csv"
error_data = pd.read_csv(error_data_file)

# 将错误数据的样本名转换为列表
error_samples = error_data["pcd_file"].tolist()

# 根据错误数据的样本名，从训练数据集中剔除这些数据
matched_data = labels_df[~labels_df.index.isin(error_samples)]
# 输出异常值到error.csv文件
matched_data.to_csv(r"D:\3DPointCloud\ProcessedData\filter\PointCloudWeight_new.csv", index=True)