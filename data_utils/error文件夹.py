import os
import csv

def list_image_files(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
    return image_files

def save_to_csv(image_files, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['pcd_file'])
        for image_file in image_files:
            writer.writerow([image_file])

if __name__ == "__main__":
    folder_path =  r'D:\3DPointCloud\ProcessedData\new\error'  # 替换成你的文件夹路径
    csv_file = 'errorFolder.csv'  # CSV文件名
    image_files = list_image_files(folder_path)
    save_to_csv(image_files, csv_file)
    print(f"Image filenames saved to {csv_file}")
