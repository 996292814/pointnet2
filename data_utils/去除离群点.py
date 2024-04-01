import open3d as o3d
import os
import numpy as np

def process_pcd(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.pcd'):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)

            # Load PCD file
            print(f"Processing: {input_file}")
            pcd = o3d.io.read_point_cloud(input_file)

            # Apply rotation
            R = pcd.get_rotation_matrix_from_xyz(rotation=(np.pi, 0, 0))
            pcd.rotate(R=R, center=(0, 0, 0))

            # Downsample
            voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)

            # Remove outlier
            print("Statistical outlier removal")
            cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)

            # Save processed PCD
            o3d.io.write_point_cloud(output_file, voxel_down_pcd.select_by_index(ind))
            print(f"Processed file saved as: {output_file}")

if __name__ == "__main__":
    input_folder = r"D:\3DPointCloud\ProcessedData\new\pcd"
    output_folder = r"D:\3DPointCloud\ProcessedData\filter\pcd"
    process_pcd(input_folder, output_folder)
