import numpy as np
import torch
import os


def normalize(data, col_min, col_max):
    return (data - col_min) / (col_max - col_min)


def standardize(x):
    x = np.asarray(x, dtype=np.float64)  # 确保使用双精度浮点数
    mean = np.mean(x)  # 计算平均值
    std_dev = np.std(x, ddof=0)  # 计算标准差，使用与PyTorch相同的自由度调整
    y = (x - mean) / std_dev
    return y, mean, std_dev


def denormalize(normalized_data, original_col_min, original_col_max):
    return normalized_data * (original_col_max - original_col_min) + original_col_min


def inverse_standardize(y, mean, std_dev):
    x = y * std_dev + mean
    return x


def find_min_max(array):
    return np.min(array), np.max(array)


def add_gaussian_noise(data, mean, std, snr_db):
    standardized_data = (data - mean) / std
    snr_linear = 10 ** (snr_db / 10)
    noise_std = 1 / np.sqrt(snr_linear)
    noise = np.random.normal(0, noise_std, standardized_data.shape)
    noisy_standardized_data = standardized_data + noise
    noisy_data = noisy_standardized_data * std + mean
    return noisy_data


def add_awgn_noise(signal, snr_db):
    """
    在信号上添加加性高斯白噪声（AWGN）

    参数：
        signal: torch.Tensor 或 np.ndarray，原始信号
        snr_db: float，信噪比（SNR）dB

    返回：
        带噪声的信号（与输入类型一致）
    """
    # 如果输入是 PyTorch Tensor，先转换为 NumPy
    is_tensor = isinstance(signal, torch.Tensor)
    if is_tensor:
        signal_np = signal.detach().cpu().numpy()  # 转换为 NumPy
    else:
        signal_np = np.array(signal)

    # 计算信号功率
    signal_power = np.mean(signal_np ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal_np.shape)

    # 添加噪声
    noisy_signal = signal_np + noise

    # 如果原始输入是 PyTorch Tensor，转换回 Tensor
    if is_tensor:
        return torch.tensor(noisy_signal, dtype=signal.dtype, device=signal.device)
    return noisy_signal



# 计算给定极坐标的笛卡尔坐标（x、y）
def PolarToCartesian(A39, index):
    distance = A39.iloc[index]['dis']  # 第一个点的距离
    angle = A39.iloc[index]['ang']  # 第一个点的方位（度数）
    # 计算两个点在二维平面上的坐标
    x = distance * np.cos(np.radians(angle))  # 第一个点的 x 坐标
    y = distance * np.sin(np.radians(angle))  # 第一个点的 y 坐标
    return x, y


def euclidean_distance_between_rows(A39, index):
    x1, y1 = PolarToCartesian(A39, index - 1)
    x2, y2 = PolarToCartesian(A39, index)
    # 计算欧式距离
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def calculate_velocity_and_acceleration(A39, index):
    # 获取两个相邻点之间的笛卡尔坐标和时间戳
    distance1 = euclidean_distance_between_rows(A39, index - 1)
    distance2 = euclidean_distance_between_rows(A39, index)

    # 检查 A39 是否有 time 列，如果没有，尝试使用 time_position 列
    if 'timeAtServer' in A39.columns:
        time_stamp_range = A39.loc[index - 2:index, 'timeAtServer']
    elif 'Time' in A39.columns:
        time_stamp_range = A39.loc[index - 2:index, 'Time']
    elif 'time' in A39.columns:
        time_stamp_range = A39.loc[index - 2:index, 'time']
    elif 'time_position' in A39.columns:
        time_stamp_range = A39.loc[index - 2:index, 'time_position']
    else:
        raise ValueError("No valid time column found in test data")

    time_stamp_diff = np.diff(time_stamp_range)

    # 计算速度
    velocity1 = distance1 / time_stamp_diff[0]
    velocity2 = distance2 / time_stamp_diff[1]

    # 计算加速度
    acceleration = (velocity2 - velocity1) / ((time_stamp_diff[1] + time_stamp_diff[0]) / 2)

    return velocity1, velocity2, abs(acceleration)


def DetectFalseTar(A39):
    """
       检测虚假目标
       参数:
       A39 (pandas.DataFrame): 包含点迹信息的DataFrame，列包括时间戳、距离、方位等
       返回值:
       tuple: (int, bool)，返回当前处理的行索引和是否为虚假目标的布尔值
           - int: 当前处理的行索引-接下来是kalman初始化，所以就是过虚假目标的当前行
           - bool: 是否为虚假目标，True表示为虚假目标，False表示为真实目标
       """

    point_row_num = A39.shape[0]  # 获取点迹行数
    start_idx = 2
    for idx, A39_line in A39.iterrows():
        if idx >= point_row_num - 1:
            return idx, False,
        elif idx >= start_idx:
            velocity1, velocity2, acceleration = calculate_velocity_and_acceleration(A39, idx)
            if velocity1 <= 350 and velocity2 <= 350 and acceleration <= 30:
                return idx, True, start_idx


