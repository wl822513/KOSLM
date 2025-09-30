import torch
import numpy as np
from .process_data import DetectFalseTar, add_awgn_noise  # 注意 `.` 代表当前目录
from functions import polar_to_cartesian, standardize
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from random import shuffle
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler


def to_supervised(data, n_input, n_out):
    X, y = [], []
    for i in range(len(data)):
        in_end = i + n_input
        out_end = in_end + n_out
        if out_end <= len(data):
            X.append(data[i:in_end])
            y.append(data[in_end:out_end])
    return np.array(X), np.array(y)


def to_supervised1(data, n_input, n_out):
    X, y = [], []
    for i in range(len(data)):
        in_end = i + n_input
        out_end = in_end + n_out - 1
        if out_end <= len(data):
            X.append(data[i:in_end])
            y.append(data[in_end - 1:out_end])
    return np.array(X), np.array(y)


def train_data_process(data, snr_db=33):
    data = data.dropna()
    time_col = data.get('timeAtServer') or data.get('Time') or data.get('time_position')
    if time_col is None:
        raise ValueError("No valid time column found in data")
    # 将时间列转为NumPy数组
    time_col = time_col.to_numpy()
    time_col -= time_col.min()

    #  将极坐标系转换为笛卡尔坐标系
    x_coords, y_coords = polar_to_cartesian(data['dis'], data['ang'])   # 采用kf_net标准化里的极坐标转笛卡尔坐标系
    col1_norm, _, _ = standardize(x_coords)
    col2_norm, _, _ = standardize(y_coords)
    velocity_col1 = np.diff(col1_norm) / np.diff(time_col)
    velocity_col2 = np.diff(col2_norm) / np.diff(time_col)

    #  训练集加噪
    # mean1 = np.mean(x_coords.detach().cpu().numpy())
    # std1 = np.std(x_coords.detach().cpu().numpy())
    # mean2 = np.mean(y_coords.detach().cpu().numpy())
    # std2 = np.std(y_coords.detach().cpu().numpy())
    x_coords_noisy = add_awgn_noise(x_coords, snr_db)
    y_coords_noisy = add_awgn_noise(y_coords, snr_db)
    # x_coords_noisy = add_gaussian_noise(x_coords, mean1, std1, snr_db)
    # y_coords_noisy = add_gaussian_noise(y_coords, mean2, std2, snr_db)
    col1_norm_noisy, _, _ = standardize(x_coords_noisy)
    col2_norm_noisy, _, _ = standardize(y_coords_noisy)
    velocity_col1_noisy = np.diff(col1_norm_noisy) / np.diff(time_col)
    velocity_col2_noisy = np.diff(col2_norm_noisy) / np.diff(time_col)

    flag_col = np.zeros(velocity_col1.shape)
    flag_col[0] = 1
    time_norm, _, _ = standardize(time_col)
    data_norm = np.column_stack((time_col[1:], flag_col, time_norm[1:], col1_norm[1:], col2_norm[1:], velocity_col1, velocity_col2))
    data_norm_noisy = np.column_stack((time_col[1:], flag_col, time_norm[1:], col1_norm_noisy[1:], col2_norm_noisy[1:],
                                       velocity_col1_noisy, velocity_col2_noisy))

    data_norm = data_norm[~np.isnan(data_norm).any(axis=1)]
    data_norm_noisy = data_norm_noisy[~np.isnan(data_norm_noisy).any(axis=1)]

    # 调试图显对比
    # data_norm_tensor = torch.tensor(data_norm, dtype=torch.float32)
    # data_norm_noisy_tensor = torch.tensor(data_norm_noisy, dtype=torch.float32)
    # plot_kf_output_comparison(data_norm_tensor, data_norm_noisy_tensor)

    return data_norm, data_norm_noisy  # t flag x y vx vy
# def train_data_process(data, snr_db=33):
#     data = data.dropna()
#     time_col = data.get('timeAtServer') or data.get('Time') or data.get('time_position')
#     if time_col is None:
#         raise ValueError("No valid time column found in data")
#     # 将时间列转为NumPy数组
#     time_col = time_col.to_numpy()
#     time_col -= time_col.min()
#
#     #  将极坐标系转换为笛卡尔坐标系
#     x_coords, y_coords = polar_to_cartesian(data['dis'], data['ang'])   # 采用kf_net标准化里的极坐标转笛卡尔坐标系
#     col1_norm, _, _ = standardize(x_coords)
#     col2_norm, _, _ = standardize(y_coords)
#     velocity_col1 = np.diff(col1_norm) / np.diff(time_col)
#     velocity_col2 = np.diff(col2_norm) / np.diff(time_col)
#
#     #  训练集加噪
#     # mean1 = np.mean(x_coords.detach().cpu().numpy())
#     # std1 = np.std(x_coords.detach().cpu().numpy())
#     # mean2 = np.mean(y_coords.detach().cpu().numpy())
#     # std2 = np.std(y_coords.detach().cpu().numpy())
#     x_coords_noisy = add_awgn_noise(x_coords, snr_db)
#     y_coords_noisy = add_awgn_noise(y_coords, snr_db)
#     # x_coords_noisy = add_gaussian_noise(x_coords, mean1, std1, snr_db)
#     # y_coords_noisy = add_gaussian_noise(y_coords, mean2, std2, snr_db)
#     col1_norm_noisy, _, _ = standardize(x_coords_noisy)
#     col2_norm_noisy, _, _ = standardize(y_coords_noisy)
#     velocity_col1_noisy = np.diff(col1_norm_noisy) / np.diff(time_col)
#     velocity_col2_noisy = np.diff(col2_norm_noisy) / np.diff(time_col)
#
#     flag_col = np.zeros(velocity_col1.shape)
#     flag_col[0] = 1
#     data_norm = np.column_stack((time_col[1:], flag_col, col1_norm[1:], col2_norm[1:], velocity_col1, velocity_col2))
#     data_norm_noisy = np.column_stack((time_col[1:], flag_col, col1_norm_noisy[1:], col2_norm_noisy[1:],
#                                        velocity_col1_noisy, velocity_col2_noisy))
#
#     data_norm = data_norm[~np.isnan(data_norm).any(axis=1)]
#     data_norm_noisy = data_norm_noisy[~np.isnan(data_norm_noisy).any(axis=1)]
#
#     # 调试图显对比
#     # data_norm_tensor = torch.tensor(data_norm, dtype=torch.float32)
#     # data_norm_noisy_tensor = torch.tensor(data_norm_noisy, dtype=torch.float32)
#     # plot_kf_output_comparison(data_norm_tensor, data_norm_noisy_tensor)
#
#     return data_norm, data_norm_noisy  # t flag x y vx vy


def load_data(train_directory, input_time_steps, output_time_steps, snr_db=33):
    train_files = read_csv_files(train_directory)
    X_trains, y_trains = [], []
    for file in train_files:
        data_file = pd.read_csv(file)
        train_norm, data_norm_noisy = train_data_process(data_file, snr_db)
        _, y_train = to_supervised1(train_norm, input_time_steps, output_time_steps)
        X_train_noisy, _ = to_supervised1(data_norm_noisy, input_time_steps, output_time_steps)
        # 加噪的x作为X_trains， 没加噪的作为目标
        for sample in X_train_noisy:
            X_trains.append(sample.tolist())
        for sample in y_train:
            y_trains.append(sample.tolist())

    X_trains = np.array(X_trains)
    y_trains = np.array(y_trains)
    return X_trains, y_trains


def batch_data_loaders(X_trains, y_trains, batch_size):
    X_trains = np.array(X_trains)
    y_trains = np.array(y_trains)

    # 使用 X_trains 第一个时间步的第二个特征来标记每个文件的数据起始位置
    indices = np.where(X_trains[:, 0, 1] == 1)[0]
    datasets = []

    for i in range(len(indices)):
        start_idx = indices[i]
        end_idx = indices[i + 1] if i + 1 < len(indices) else len(X_trains)

        # 计算保留的数据量（丢弃不足 batch_size 的部分）
        num_batches = (end_idx - start_idx) // batch_size
        X_file = X_trains[start_idx:start_idx + num_batches * batch_size]
        y_file = y_trains[start_idx:start_idx + num_batches * batch_size]

        # 将训练集转换为张量
        X_train_tensor = torch.tensor(X_file, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_file, dtype=torch.float32)
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        datasets.append(dataset)

    # 打乱 datasets 的顺序，确保打乱的是不同航迹之间的顺序
    shuffle(datasets)

    # 将打乱后的 datasets 合并为一个 ConcatDataset
    concat_dataset = ConcatDataset(datasets)

    # 创建 DataLoader，不打乱航迹内的顺序
    data_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=False)

    return data_loader


def prepare_test_data(test_file_path, input_features):
    test_data = pd.read_csv(test_file_path)
    test_data = test_data.dropna()
    # 使用 DetectFalseTar 检测
    ret_idx, is_false, start_idx = DetectFalseTar(test_data)
    if is_false:
        # 如果 is_false 为 True，则去掉前 ret_idx - start_idx 行数据
        test_data = test_data[ret_idx-start_idx:]
    test_data_col1, test_data_col2 = polar_to_cartesian(test_data['dis'], test_data['ang'])
    if 'timeAtServer' in test_data.columns:
        time_col = test_data['timeAtServer']
    elif 'Time' in test_data.columns:
        time_col = test_data['Time']
    elif 'time' in test_data.columns:
        time_col = test_data['time']
    elif 'time_position' in test_data.columns:
        time_col = test_data['time_position']
    else:
        raise ValueError("No valid time column found in data")

    col1_norm_test, col1_mean, col1_std_dev = standardize(test_data_col1)
    col2_norm_test, col2_mean, col2_std_dev = standardize(test_data_col2)
    min_max_test = np.array([[col1_mean, col1_std_dev], [col2_mean, col2_std_dev]])

    velocity_col1 = np.diff(col1_norm_test) / np.diff(time_col)
    velocity_col2 = np.diff(col2_norm_test) / np.diff(time_col)

    flag_col = np.zeros(velocity_col1.shape)
    flag_col[0] = 1
    time_norm, time_mean, time_std = standardize(np.array(time_col))
    test_data_norm = np.column_stack((time_col[1:], time_norm[1:], flag_col, col1_norm_test[1:],
                                      col2_norm_test[1:], velocity_col1, velocity_col2))
    non_nan_rows = ~np.isnan(test_data_norm).any(axis=1)
    test_data_norm = test_data_norm[non_nan_rows]

    return test_data_norm[:, 0:input_features+2], min_max_test, test_data, time_mean, time_std


def read_csv_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]



