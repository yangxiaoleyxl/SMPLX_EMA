# 创建 tensor   
import numpy as np
import os 
import cv2    

def batch_rodrigues(vectors):
    """
    将一批Rodrigues向量转换为旋转矩阵

    参数：
    vectors (ndarray): 形状为 (N, 3) 的 Rodrigues 向量

    返回：
    ndarray: 形状为 (N, 3, 3) 的旋转矩阵
    """
    # 动态初始化旋转矩阵数组
    rotation_matrices = np.zeros((vectors.shape[0], 3, 3))
    
    for i in range(vectors.shape[0]):
        rotation_matrices[i] = cv2.Rodrigues(vectors[i])[0]
    
    return rotation_matrices


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    # Drop the last row and reshape to 6D representation
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


video_num = 'demo00048'  
base_path = '/Users/lxy/Desktop/yxl/results'

# 设置路径 
path = f'{base_path}/{video_num}/smplx'

# 初始化存储合成数据的字典
combined_data_6d = {
    'global_orient': [],
    'body_pose': [],
    'left_hand_pose': [],
    'right_hand_pose': [],
    'jaw_pose': [],
    'leye_pose': [],
    'reye_pose': []
}

# 遍历文件名从 00001_0.npz 到 00100_0.npz
for i in range(1, 101):
    file_name = f'{i:05d}_0.npz'
    file_path = os.path.join(path, file_name)
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在，跳过...")
        continue
    # 读取 .npz 文件
    data = np.load(file_path)
    
    # 将数据添加到相应的列表中
    for key in combined_data_6d.keys():
        if key in data:      
            rotation_matrix = batch_rodrigues(data[key]) # N * 3 -> N * 3 * 3   
            rotation_matrix_6d = matrix_to_rotation_6d(rotation_matrix) # 3 * 3 -> 6  
            combined_data_6d[key].append(rotation_matrix_6d)
        else:
            print(f"文件 {file_path} 中不包含键 {key}") 

# 转换为所需的张量形式
combined_data = {
    key: np.stack(value, axis=0) for key, value in combined_data_6d.items()
}

# 打印张量的形状以确认 
for key, tensor in combined_data.items():
    print(f"{key}: {tensor.shape}")

# 将合成后的数据保存为一个新的 .npz 文件   
np.savez(f'{base_path}/{video_num}/combined_data_6d.npz', **combined_data)    
print(f' "combine_data_6d.npz" is saved !')   

# EMA 6D data  
def exponential_moving_average(data, alpha):
    """
    计算一维数据的指数平滑平均（EMA）。

    参数：
    data (ndarray): 一维时间序列数据
    alpha (float): 平滑因子，取值范围在0到1之间

    返回：
    ndarray: 一维 EMA 数据
    """
    ema = np.zeros_like(data)
    ema[0] = data[0]  # 初始值为第一个数据点
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]
    return ema

def exponential_moving_average_tensor(tensor, alpha, body_pose_flag=None):
    """
    对三维张量进行指数平滑平均（EMA）。

    参数：
    tensor (ndarray): 三维时间序列数据，形状为 (100, 15, 3)
    alpha (float): 平滑因子，取值范围在0到1之间

    返回：
    ndarray: 形状为 (100, 15, 3) 的 EMA 数据
    """
    ema_tensor = np.zeros_like(tensor)
    T, D1, D2 = tensor.shape 
    print(f"T is {T}, D1 is {D1}, D2 is {D2}")

    # 对每个位置的二维向量进行 EMA 计算 
    if body_pose_flag: 
        for i in range(D1):  
            for j in range(D2):   
                if i in [2]:
                    ema_tensor[:, i, j] = exponential_moving_average(tensor[:, i, j], 0.1)  
                else:  
                    ema_tensor[:, i, j] = exponential_moving_average(tensor[:, i, j], alpha)  
    else:  
        for i in range(D1):  
            for j in range(D2): 
                ema_tensor[:, i, j] = exponential_moving_average(tensor[:, i, j], alpha) 

    return ema_tensor 

def normalize(vectors, axis=-1):
    """
    Normalizes the input vectors along the specified axis.
    
    Args:
        vectors: Input vectors to be normalized
        axis: Axis along which to normalize
    
    Returns:
        Normalized vectors
    """
    norm = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / norm

def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram-Schmidt orthogonalization per Section B of [1].
    
    Args:
        d6: 6D rotation representation, of size (*, 6)
    
    Returns:
        Batch of rotation matrices of size (*, 3, 3)
    
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    
    # Normalize the first vector
    b1 = normalize(a1, axis=-1)
    
    # Compute the second vector and normalize it
    dot_product = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot_product * b1
    b2 = normalize(b2, axis=-1)
    
    # Compute the third vector using cross product
    b3 = np.cross(b1, b2, axis=-1)
    
    # Stack the vectors to form rotation matrices
    rotation_matrices = np.stack((b1, b2, b3), axis=-2)
    
    return rotation_matrices 


def batch_rodrigues_from_matrix(rotation_matrices: np.ndarray) -> np.ndarray:
    """
    Converts a batch of rotation matrices to Rodrigues vectors.
    
    Args:
        rotation_matrices: Batch of rotation matrices, of size (N, 3, 3)
    
    Returns:
        Rodrigues vectors, of size (N, 3)
    """
    rodrigues_vectors = np.zeros((rotation_matrices.shape[0], 3))
    for i in range(rotation_matrices.shape[0]):
        rodrigues_vector, _ = cv2.Rodrigues(rotation_matrices[i])
        rodrigues_vectors[i] = rodrigues_vector.squeeze()
    return rodrigues_vectors 


# 加载 npz 文件
file_path = f'{base_path}/{video_num}/combined_data_6d.npz' 
data = np.load(file_path)

# 平滑因子
alpha = 0.8 
alpha_global_orient = 0.1  
alpha_body_pose = 0.1

# 初始化存储 EMA 结果的字典
ema_data = {}

# 对每个键的数据进行 EMA 计算
for key in data: 
    # print(key) 
    if key in ['global_orient']:   # 如果是 ‘global_orient’, 使用 0.1 的平滑因子  
        # 获取三维数据
        three_dimensional_data = data[key] 
        print(f"{key} data shape is {three_dimensional_data.shape}")  
        # 计算 EMA,
        ema_result = exponential_moving_average_tensor(three_dimensional_data, alpha_global_orient)   
    elif key in ['body_pose']:   # 如果是 ‘body_pose’, 对 2 5 8 使用 0.1 的平滑因子  
        # 获取三维数据
        three_dimensional_data = data[key] 
        print(f"{key} data shape is {three_dimensional_data.shape}")  
        # 计算 EMA, 
        ema_result = exponential_moving_average_tensor(three_dimensional_data, alpha, body_pose_flag=True)  
    else: 
        # 获取三维数据 
        three_dimensional_data = data[key] 
        print(f"{key} data shape is {three_dimensional_data.shape}")  
        # 计算 EMA 
        ema_result = exponential_moving_average_tensor(three_dimensional_data, alpha)  
        # print(ema_result.shape)   

    # 将 EMA 结果从 6D 转换为旋转矩阵
    rotation_matrices = rotation_6d_to_matrix(ema_result) 
    # print(rotation_matrices.shape)
    
    # 将旋转矩阵转换为轴角数据
    rodrigues_vectors = np.zeros((rotation_matrices.shape[0], rotation_matrices.shape[1], 3))
    for i in range(rotation_matrices.shape[0]):
        rodrigues_vectors[i] = batch_rodrigues_from_matrix(rotation_matrices[i])  
         
    print(rodrigues_vectors.shape)
    # 存储结果
    ema_data[key] = rodrigues_vectors

# 定义新的保存目录
new_directory = f'{base_path}/{video_num}/smplx_ema'

# 确保新目录存在
if not os.path.exists(new_directory):
    os.makedirs(new_directory)

# 新的 npz 文件路径
new_output_file_path = os.path.join(new_directory, 'ema_result_tensor_6d.npz')

# 保存处理后的数据到新的目录下
np.savez(new_output_file_path, **ema_data)

print(f"Processed EMA data saved to: {new_output_file_path}")   


# 定义处理后的 npz 文件路径
# processed_file_path = f'{base_path}/{video_num}/smplx_ema/ema_result_tensor_6d.npz'

# # 加载处理后的数据
# ema_data = np.load(processed_file_path)

# # 定义新的保存目录
# output_directory = f'{base_path}/{video_num}/smplx_ema_splits'

# # 遍历 EMA 结果中的每个键（假设我们只处理一个键）
# for key in ema_data: 
#     print(key)
#     tensor = ema_data[key]
#     print(f"Processing {key} with shape {tensor.shape}")
    
#     # 保存每个时间步
#     save_each_time_step(tensor, output_directory) 

# print(f"Processing completed. Files saved to {output_directory}") 


def save_each_time_step(ema_data, output_dir):
    """
    将每个时间步的所有键对应的二维张量保存为独立的 .npz 文件。

    参数：
    ema_data (dict): 包含多个键的三维张量字典，每个键的值形状为 (100, D1, D2)
    output_dir (str): 保存每个时间步文件的目录
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取总的时间步数 (T)
    keys = list(ema_data.keys())
    T = ema_data[keys[0]].shape[0]
    
    # 遍历每个时间步
    for t in range(T):
        time_step_data = {}
        
        # 提取当前时间步的所有键的数据
        for key in ema_data:
            time_step_data[key] = ema_data[key][t].astype(np.float32)
        
        # 生成保存文件的路径
        file_name = f'time_step_{t + 1:03d}.npz'
        file_path = os.path.join(output_dir, file_name)
        
        # 保存为 .npz 文件，将所有键的数据保存进去
        np.savez(file_path, **time_step_data)
    
    print(f"All time steps saved to: {output_dir}")

# 定义处理后的 npz 文件路径
processed_file_path = f'{base_path}/{video_num}/smplx_ema/ema_result_tensor_6d.npz'

# 加载处理后的数据
ema_data = np.load(processed_file_path)

# 定义新的保存目录
output_directory = f'{base_path}/{video_num}/smplx_ema_splits_6d' 

# 保存每个时间步的数据
save_each_time_step(ema_data, output_directory) 

print(f"Processing completed. Files saved to {output_directory}")
