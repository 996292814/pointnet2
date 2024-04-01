import open3d as o3d
import numpy as np
import copy

# 对点云进行颜色着色
def colorize_point_cloud(pcd):
    # 获取点云的 z 坐标
    points = np.asarray(pcd.points)
    z_values = points[:, 2]

    # 计算颜色映射
    height_max = np.max(z_values)
    height_min = np.min(z_values)
    delta_c = abs(height_max - height_min) / (255 * 3)

    colors = np.zeros_like(points)
    for j in range(points.shape[0]):
        color_n = (points[j, 2] - height_min) / delta_c
        if color_n <= 255:
            colors[j, :] = [1, color_n / 255, 0]
        elif color_n <= 255 * 2:
            colors[j, :] = [1 - (color_n - 255) / 255, 1, 0]
        else:
            colors[j, :] = [0, 1, (color_n - 255 * 2) / 255]

    # 创建 Open3D 的颜色对象
    colors = o3d.utility.Vector3dVector(colors)
    return colors

def pca_compute(data, sort=True):
    """
    SVD分解计算点云的特征值与特征向量
    :param data: 输入数据
    :param sort: 是否将特征值特征向量进行排序
    :return: 特征值与特征向量
    """
    average_data = np.mean(data, axis=0)
    decentration_matrix = data - average_data
    H = np.dot(decentration_matrix.T, decentration_matrix)
    eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(H)

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

def PCA(pcd_path):
    # 读取点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    # 获取凸包的顶点
    try:
        hull, idx = pcd.compute_convex_hull()
        hull_cloud = pcd.select_by_index(idx)
    except:
        print('error:', pcd_path)
        return np.asarray([[0, 0, 0]])
    # 对凸包点云应用PCA
    w_hull, v_hull = pca_compute(np.asarray(hull_cloud.points))
    # 打印凸包主方向
    # print('the main orientation of the convex hull is: ', )
    # 使用第一个主成分作为特征
    selected_components = v_hull[:1]
    # 通过将点云数据投影到主成分上，得到点云的主方向
    # features = points.dot(selected_components.T)
    return v_hull

def main():
    # 读取点云
    pcd = o3d.io.read_point_cloud("../1_split1_strawberry_dyson_lincoln_tbd__031_1.pcd")
    points = np.asarray(pcd.points)

    # 获取凸包的顶点
    hull, idx = pcd.compute_convex_hull()
    hull_cloud = pcd.select_by_index(idx)

    # 对凸包点云应用PCA
    w_hull, v_hull = pca_compute(np.asarray(hull_cloud.points))
    # 显示凸包主成分，三个特征向量组成了三个坐标轴
    pcd.colors = colorize_point_cloud(pcd)
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v_hull, center=(0, 0, 0))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=np.mean(points, axis=0)+ [20, 70, 10]).rotate(v_hull, center=(0, 0, 0))  # 设置坐标轴的大小
    o3d.visualization.draw_geometries([pcd, axis])  # 显示原始点云
    # 使用欧拉角创建旋转矩阵
    mesh_r = copy.deepcopy(axis)
    R = axis.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
    # 打印凸包主方向
    print('the main orientation of the convex hull is: ', v_hull[:, 0])

if __name__ == '__main__':
    main()
