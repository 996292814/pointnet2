import cv2
import numpy as np
import os
import json
import csv
import open3d as o3d
from PIL import Image

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

def is_image_black(image_path):
    image = Image.open(image_path)
    width, height = image.size

    for y in range(height):
        for x in range(width):
            r, g, b = image.getpixel((x, y))
            if r != 0 or g != 0 or b != 0:
                return False
    return True

def process_images_in_folder(folder_path, output_folder, annotation_root_folder, weight):
    # 获取文件夹中所有文件名包含"_rgb"的图像
    image_files = [f for f in os.listdir(folder_path) if "_rgb" in f]
    for image_file in image_files:
        # 构建完整的图像文件路径
        image_path = os.path.join(folder_path, image_file)
        # 读取彩色图像
        rgb_image = cv2.imread(image_path)
        # 读取深度图像
        depth_image = cv2.imread(os.path.join(folder_path, f'{image_file.split("_rgb")[0]}_pdepth.png'))
        # 构建标注文件夹路径
        subfolder_name = os.path.basename(folder_path)
        annotation_folder = os.path.join(annotation_root_folder, subfolder_name)
        annotation_file = os.path.join(annotation_folder, f'{image_file.split("_rgb")[0]}_keypoint.json')

        # 使用 json.load() 读取 JSON 文件
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as json_file:
                data = json.load(json_file)
                annotations = data['annotations']

            # 生成只包含标注实例的图像，保留原图颜色和特征
            for idx, annotation in enumerate(annotations):
                bbox = annotation["bbox"]
                segmentation = annotation["segmentation"]
                x1, y1, x2, y2 = map(int, bbox)

                # 创建一个空白图像，与原图像相同大小
                blank_image = np.zeros_like(rgb_image)

                # 创建一个空的掩码图像
                mask = np.zeros_like(rgb_image, dtype=np.uint8)

                # 使用多边形绘制标注实例的分割区域
                poly = np.array(segmentation[0], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [poly], (1, 1, 1))

                # 从原图像复制标注实例部分
                instance_image = rgb_image[y1:y2, x1:x2, :]

                # 将标注实例部分覆盖到空白图像上，保留原图颜色和特征
                blank_image[y1:y2, x1:x2, :] = instance_image * mask[y1:y2, x1:x2, :]
                # 构建保存路径，并保存只包含标注实例的图像，同时保留原图颜色和特征
                output_path = os.path.join(output_folder+"\images", f"4_split{idx}_{image_file}")
                cv2.imwrite(output_path, blank_image)

                # 创建一个空白图像，与原图像相同大小
                blank_depth_image = np.zeros_like(depth_image)

                # 创建一个空的掩码图像
                mask = np.zeros_like(depth_image, dtype=np.uint8)

                # 使用多边形绘制标注实例的分割区域
                poly = np.array(segmentation[0], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [poly], (1, 1, 1))

                # 从原图像复制标注实例部分
                instance_depth_image = depth_image[y1:y2, x1:x2, :]

                # 将标注实例部分覆盖到空白图像上，保留原图颜色和特征
                blank_depth_image[y1:y2, x1:x2, :] = instance_depth_image * mask[y1:y2, x1:x2, :]
                # 构建保存路径，并保存只包含标注实例的图像，同时保留原图颜色和特征
                depth_output_path = os.path.join(output_folder + "\depth",
                                           f'4_split{idx}_{image_file.split("_rgb")[0]}_pdepth.png')
                cv2.imwrite(depth_output_path, blank_depth_image)
                if is_image_black(depth_output_path):
                    os.remove(depth_output_path)
                    os.remove(output_path)
                    print(f"Deleted {depth_output_path}")
                    continue
                # 获取 npy 文件中的坐标数据
                npy_file_path = os.path.join(folder_path, f'{image_file.split("_rgb")[0]}_label.npy')

                # 如果 .npy 文件存在，处理坐标信息
                if os.path.exists(npy_file_path):
                    # 加载 npy 文件中的坐标数据
                    npy_data = np.load(npy_file_path)
                    # 遍历坐标，检查是否在矩形框内
                    for coord in npy_data:
                        if len(coord) == 7:
                            x, y = coord[5:7]
                            if weight == 0:
                                weight = coord[1]
                        if len(coord) == 3:
                            x, y = coord[1:3]
                        if is_point_inside_bbox((x, y), bbox):
                            # 构建 CSV 行数据
                            csv_data = [f'4_split{idx}_{image_file.split("_rgb")[0]}.pcd', weight]

                            # 将 CSV 行数据写入文件
                            csv_file_path = os.path.join(output_folder, "PointCloudWeight.csv")
                            with open(csv_file_path, mode='a', newline='') as csv_file:
                                csv_writer = csv.writer(csv_file)
                                csv_writer.writerow(csv_data)

                            # 获取 npy 文件中的深度信息
                            depth_npy_file_path = os.path.join(folder_path,
                                                               f'{image_file.split("_rgb")[0]}_rdepth.npy')
                            # 构建点云保存路径
                            pcd_output_path = os.path.join(output_folder + "\pcd", f'4_split{idx}_{image_file.split("_rgb")[0]}.pcd')
                            # 生成点云并保存
                            generate_point_cloud(output_path, depth_npy_file_path, pcd_output_path)
                            print('saved:', pcd_output_path)

def process_images_in_folders(root_folder, output_folder, annotation_root_folder):
    # 获取根文件夹下的所有子文件夹
    subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

    for subfolder in subfolders:
        # 构建完整的子文件夹路径
        folder_path = os.path.join(root_folder, subfolder)
        # 获取文件夹中所有文件名包含"_label"的文件
        label_files = [f for f in os.listdir(folder_path) if "_label" in f]
        # 缺省重量值
        weight = 0
        for label_file in label_files:
            # 获取 npy 文件中的坐标数据
            npy_file_path = os.path.join(folder_path, label_file)
            # 如果 .npy 文件存在，处理坐标信息
            if os.path.exists(npy_file_path):
                # 加载 npy 文件中的坐标数据
                npy_data = np.load(npy_file_path)
                # 遍历坐标，检查是否是7列的数据，如果是7列则代表有重量标注
                for coord in npy_data:
                    if len(coord) == 7:
                        if(weight > 0):
                            print('不一样的重量：', folder_path, label_file)
                        weight = coord[1]
                        break
        # 处理子文件夹中的图像
        process_images_in_folder(folder_path, output_folder, annotation_root_folder, weight)

# 指定待处理数据的文件夹路径
root_folder = r"D:\3DPointCloud\ICRA2022\dataset-1\4\4"
output_folder = r"D:\3DPointCloud\ProcessedData\new"
annotation_root_folder = r"D:\3DPointCloud\ICRA2022\annotations\dyson_annotations\dyson_annotations\4"

# 处理所有子文件夹中的图像
process_images_in_folders(root_folder, output_folder, annotation_root_folder)
