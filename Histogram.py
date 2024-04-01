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

matplotlib.use('TkAgg')


# 初始化空列表存储特征和标签
X = []  # 特征
y = []  # 标签

# 加载特征和标签数据，并手动添加列名
features_df = pd.read_csv(r"D:\3DPointCloud\ProcessedData\features_new\Featrues.csv", header=None)
features_df.columns = [f"Feature_{i}" for i in range(features_df.shape[1])]  # 为每列添加一个特征名

labels_df = pd.read_csv(r"D:\3DPointCloud\ProcessedData\features_new\PointCloudWeight.csv", header=None)
labels_df.columns = ["SampleName", "Label"]

# 设置样本名称为索引，方便匹配
features_df.set_index("Feature_0", inplace=True)
labels_df.set_index("SampleName", inplace=True)

# 匹配特征和标签数据
matched_data = features_df.join(labels_df, how="inner")
# # 检查数据中是否存在 NaN 或无穷大
# print("NaN values in X:", np.isnan(X_scaled).any())
# print("Infinity values in X:", np.isinf(X_scaled).any())
# # 找出包含 NaN 值的行
# nan_rows = matched_data[matched_data.isnull().any(axis=1)]
# # 打印包含 NaN 值的行
# print("Rows with NaN values:")
# print(nan_rows)

# 填充 NaN 或无穷大值
matched_data = matched_data.fillna(0)  # 使用 0 填充 NaN 值
matched_data = matched_data.drop(columns=["Feature_1", "Feature_2"])
# 读取错误数据的文件
# error_data_file = r"D:\3DPointCloud\ProcessedData\filter\ErrorPointCloud.txt"
error_data_file = "data_utils/error.csv"
error_data = pd.read_csv(error_data_file)

# 将错误数据的样本名转换为列表
error_samples = error_data["pcd_file"].tolist()

# 根据错误数据的样本名，从训练数据集中剔除这些数据
matched_data = matched_data[~matched_data.index.isin(error_samples)]

# 提取特征和标签
# # 没有PCA,数据标准化
# X = matched_data.drop(columns=["Label", "Feature_5", "Feature_6", "Feature_7"]).values
# y = matched_data["Label"].values

# 有PCA, 标准化除了 "Feature_5", "Feature_6", "Feature_7" 之外的特征
X = matched_data.drop(columns=["Label"]).values
y = matched_data["Label"].values

# 转换为 NumPy 数组
X = np.array(X)
y = np.array(y)

# # 数据标准化,没有PCA
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# 有PCA, 标准化除了 "Feature_5", "Feature_6", "Feature_7" 之外的特征
# features_to_scale = matched_data.drop(columns=["Label", "Feature_5", "Feature_6",
#                                                "Feature_7", "Feature_8", "Feature_9",
#                                                "Feature_10", "Feature_11", "Feature_12",
#                                                "Feature_13"]).values
features_to_scale = matched_data.drop(columns=["Label"]).values
# 创建 PCA 模型，选择要保留的主成分数量
pca = PCA(n_components=4)  # 保证降维后的数据保持90%的信息
# 对数据进行 PCA 变换
X_pca = pca.fit_transform(features_to_scale)
# X_pca = np.hstack((X_pca, matched_data[["Feature_5", "Feature_6", "Feature_7",
#                                               "Feature_8", "Feature_9", "Feature_10",
#                                               "Feature_11", "Feature_12", "Feature_13"]].values))
# 打印解释方差比例
# print("Explained variance ratio:", pca.explained_variance_ratio_)
# 打印主成分特征向量
# print("Principal components:", pca.components_)

# 数据标准化,有PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_to_scale)
# 将  "Feature_5", "Feature_6", "Feature_7", "Feature_8", "Feature_9", "Feature_10",
# "Feature_11", "Feature_12","Feature_13" 的值添加回去
X_scaled = np.hstack((X_scaled, matched_data[["Feature_5", "Feature_6", "Feature_7",
                                              "Feature_8", "Feature_9", "Feature_10",
                                              "Feature_11", "Feature_12", "Feature_13"]].values))



# 标准化标签数据
# scaler_y = StandardScaler()
# y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
# # 初始化最佳模型和最佳精度
# best_model = None
# best_accuracy = -float('inf')
# # 循环训练模型
# for _ in range(100):
#     # 划分训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#
#     # 随机森林回归
#     rf_model = RandomForestRegressor()
#     rf_model.fit(X_train, y_train)
#
#     # 在测试集上进行预测
#     y_pred = rf_model.predict(X_test)
#
#     # 评估模型性能
#     accuracy = r2_score(y_test, y_pred)
#
#     # 如果当前模型的精度更高，则更新最佳模型和最佳精度
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_model = rf_model
# # 在测试集上进行预测
# y_pred = best_model.predict(X_test)

# # 线性回归
# linear_reg_model = LinearRegression()
# linear_reg_model.fit(X_train, y_train)
#
# # 在测试集上进行预测
# y_pred = linear_reg_model.predict(X_test)

# 支持向量回归
# svr_model = SVR()
# svr_model.fit(X_train, y_train)
#
# # 在测试集上进行预测
# y_pred = svr_model.predict(X_test)

# 梯度提升回归
# gb_model = GradientBoostingRegressor()
# gb_model.fit(X_train, y_train)
#
# # 在测试集上进行预测
# y_pred = gb_model.predict(X_test)

# 随机森林回归
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = rf_model.predict(X_test)

# 决策树回归
# dtr = DecisionTreeRegressor()
# dtr.fit(X_train, y_train)
#
# # 在测试集上进行预测
# y_pred = dtr.predict(X_test)


# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# 散点图
# plt.scatter(y_test, y_pred, label='Actual vs Predicted')
# 一条直线表示预测值
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k', label='Perfectly Predicted')
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.legend()
# plt.show()

# 容忍度设置
tolerances = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# 初始化每个容忍度下的正确预测数量
correct_predictions = [0] * len(tolerances)
# 初始化样本总数
total_samples = len(y_test)

# 对于每个样本，计算预测误差是否在容忍范围内
for i in range(len(y_test)):
    error = abs(y_pred[i] - y_test[i]) / y_test[i]  # 计算相对误差
    # 对于每个容忍度，统计预测误差在容忍范围内的样本数量
    for j, tol in enumerate(tolerances):
        if error <= tol:
            correct_predictions[j] += 1

# 计算每个容忍度下的准确率，即正确预测数量除以总样本数
accuracies = [round((correct / total_samples) * 100, 2) for correct in correct_predictions]
print("Accuracies for different tolerances:", accuracies)

# 可视化准确度结果
plt.plot(tolerances, accuracies, marker='o')
plt.title('Model Accuracy with Different Tolerances')
plt.xlabel('Tolerance')
plt.ylabel('Accuracy')
plt.show()