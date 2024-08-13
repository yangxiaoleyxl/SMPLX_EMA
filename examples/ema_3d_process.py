import numpy as np
import os 
import cv2  

# 创建 tensor (非6D) 直接一次指数平滑平均 
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

def exponential_moving_average_tensor(tensor, alpha):
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

    # 对每个位置的二维向量进行 EMA 计算
    for i in range(D1):
        for j in range(D2):
            ema_tensor[:, i, j] = exponential_moving_average(tensor[:, i, j], alpha)

    return ema_tensor 

video_num = 'demo00009'  
base_path = '/Users/lxy/Desktop/yxl/results'

# 设置路径
path = f'{base_path}/{video_num}/smplx'

# 初始化存储合成数据的字典
combined_data_3d = {
    'betas': [],
    'expression': [],
    'transl': []   
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
    for key in combined_data_3d.keys():
        if key in data:    
            print(key)
            if key == 'transl': 
                combined_data_3d[key].append(np.array([[0, 0, 0]]))  
                # combined_data_3d[key].append(data[key]) 
            else:  
                combined_data_3d[key].append(data[key])
        else: 
            print(f"文件 {file_path} 中不包含键 {key}") 

# 转换为所需的张量形式 
combined_data = {
    key: np.stack(value, axis=0) for key, value in combined_data_3d.items()
}

# 打印张量的形状以确认 
for key, tensor in combined_data.items():
    print(f"{key}: {tensor.shape}")

# 将合成后的数据保存为一个新的 .npz 文件  
np.savez(f'{base_path}/{video_num}/combined_data_3d.npz', **combined_data)


# EMA - 3d 
# 加载 npz 文件
file_path = f'{base_path}/{video_num}/combined_data_3d.npz' 
data = np.load(file_path) 

# 平滑因子
alpha = 0.8

# 初始化存储 EMA 结果的字典
ema_data = {}

# 对每个键的数据进行 EMA 计算
for key in data:
    # 获取三维数据
    three_dimensional_data = data[key]
    # 计算 EMA
    ema_result = exponential_moving_average_tensor(three_dimensional_data, alpha)
    # 存储结果
    ema_data[key] = ema_result

# 定义新的保存目录 
new_directory = f'{base_path}/{video_num}/smplx_ema'

# 确保新目录存在
if not os.path.exists(new_directory):
    os.makedirs(new_directory)

# 新的 npz 文件路径
new_output_file_path = os.path.join(new_directory, 'ema_result_tensor_3d.npz')

# 保存处理后的数据到新的目录下
np.savez(new_output_file_path, **ema_data)

print(f"Processed EMA data saved to: {new_output_file_path}")   



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
processed_file_path = f'{base_path}/{video_num}/smplx_ema/ema_result_tensor_3d.npz'

# 加载处理后的数据 
ema_data = np.load(processed_file_path) 

# 定义新的保存目录, 3d的键，分时间步 
output_directory = f'{base_path}/{video_num}/smplx_ema_splits_3d' 

# 保存每个时间步的数据
save_each_time_step(ema_data, output_directory)

print(f"Processing completed. Files saved to {output_directory}")