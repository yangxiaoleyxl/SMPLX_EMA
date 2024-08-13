import numpy as np
import os 

# def load_npz(file_path):
#     """
#     加载 .npz 文件并返回包含数据的字典。
#     """
#     with np.load(file_path) as data:
#         return {key: data[key] for key in data}

# def merge_npz_files(file1_path, file2_path, output_file_path):
#     """
#     合并两个 .npz 文件，并将结果保存到一个新的 .npz 文件中。

#     参数：
#     file1_path (str): 第一个 .npz 文件的路径
#     file2_path (str): 第二个 .npz 文件的路径
#     output_file_path (str): 合并后的 .npz 文件的保存路径
#     """
#     # 加载两个 .npz 文件的数据
#     file1_data = load_npz(file1_path)
#     file2_data = load_npz(file2_path)
    
#     # 合并数据
#     merged_data = {**file1_data, **file2_data}
    
#     # 保存合并后的数据到新的 .npz 文件
#     np.savez(output_file_path, **merged_data)
#     print(f"Merged data saved to: {output_file_path}")


# video_num = 'demo00365'  
# # base_path = '/home/kqk/SMPLX/results' 
# base_path = '/Users/lxy/Gitlab/smplx/results' 

# # 定义两个 .npz 文件的路径
# data_path_3d = f'{base_path}/{video_num}/smplx_ema/ema_result_tensor_3d.npz' 
# data_path_6d = f'{base_path}/{video_num}/smplx_ema/ema_result_tensor_6d.npz'

# # 定义合并后的 .npz 文件保存路径
# output_file_path = f'{base_path}/{video_num}/smplx_ema/ema_result_tensor.npz'  # 替换为你想要保存的新文件路径

# # 合并 .npz 文件并保存结果
# merge_npz_files(data_path_3d, data_path_6d, output_file_path) 


def merge_npz_files_6d_3d(ema_data_6d_dir, ema_data_3d_dir, output_dir):
    """
    合并两组 .npz 文件（6D 和 3D 数据），并将结果保存到一个新的 .npz 文件中。

    参数：
    ema_data_6d_dir (str): 6D 数据的 .npz 文件目录
    ema_data_3d_dir (str): 3D 数据的 .npz 文件目录
    output_dir (str): 合并后 .npz 文件的保存目录
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取文件列表并排序
    files_6d = sorted(os.listdir(ema_data_6d_dir))
    files_3d = sorted(os.listdir(ema_data_3d_dir))
    
    # 遍历所有文件，合并并保存
    for file_6d, file_3d in zip(files_6d, files_3d):
        # 加载6D和3D数据
        file_6d_path = os.path.join(ema_data_6d_dir, file_6d)
        file_3d_path = os.path.join(ema_data_3d_dir, file_3d)
        
        data_6d = np.load(file_6d_path)
        data_3d = np.load(file_3d_path)
        
        # 合并两个字典
        merged_data = {**data_6d, **data_3d}
        
        # 保存合并后的数据
        output_file_path = os.path.join(output_dir, file_6d)
        np.savez(output_file_path, **merged_data)
    
    print(f"All files merged and saved to: {output_dir}")

# 定义保存 6D 和 3D 数据的目录
video_num = 'demo00009'  
base_path = '/Users/lxy/Desktop/yxl/results'

ema_data_6d_dir = f'{base_path}/{video_num}/smplx_ema_splits_6d'
ema_data_3d_dir = f'{base_path}/{video_num}/smplx_ema_splits_3d'
output_dir = f'{base_path}/{video_num}/smplx_ema_splits_merged'

# 合并并保存结果
merge_npz_files_6d_3d(ema_data_6d_dir, ema_data_3d_dir, output_dir)

print(f"Processing completed. Files saved to {output_dir}")

