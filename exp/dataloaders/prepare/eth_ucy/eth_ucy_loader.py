import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ETH/UCY 数据集路径
DATA_PATHS = {
    "ETH_Hotel": "../Trajectory Prediction/eth-ucy/ETH Processed data + Code/ETH_Hotel/Final_data_eth_hotel.csv",
    "ETH_Univ": "../Trajectory Prediction/eth-ucy/ETH Processed data + Code/ETH_Univ/Final_data_eth_univ.csv",
    "UCY_Univ": "../Trajectory Prediction/eth-ucy/UCY Processed data + Code/Univ/uni_examples/Final_data_uni_examples.csv",
    "UCY_Zara1": "../Trajectory Prediction/eth-ucy/UCY Processed data + Code/Zara 01/Final_data_zara01.csv",
    "UCY_Zara2": "../Trajectory Prediction/eth-ucy/UCY Processed data + Code/Zara 02/Final_data_zara02.csv"
}


def load_eth_ucy_data(file_path):
    """ 加载 ETH/UCY 数据集，保留 FrameID 作为时间步 """
    df = pd.read_csv(file_path, header=None, names=['FrameID', 'PedID', 'PosX', 'PosY'])
    return df[['FrameID', 'PedID', 'PosX', 'PosY']]


def preprocess_trajectories(df, input_time_steps, output_time_steps, batch_size, normalize=True):
    """
    处理行人轨迹数据，计算速度特征，并进行归一化/标准化，最后划分批次。
    """
    grouped = df.groupby('PedID')
    all_positions = []  # 存储所有 (PosX, PosY) 位置数据，用于计算标准化/归一化参数
    processed_trajectories = []

    # 遍历每个行人 ID
    for ped_id, group in grouped:
        group = group.sort_values(by='FrameID')
        frames = group['FrameID'].values
        traj_x = group['PosX'].values
        traj_y = group['PosY'].values

        all_positions.append(np.column_stack((traj_x, traj_y)))  # 记录所有位置数据

        # 计算速度特征
        velocity_x = np.diff(traj_x) / 0.4  # 设定帧间间隔 0.4s
        velocity_y = np.diff(traj_y) / 0.4
        velocity_x = np.insert(velocity_x, 0, 0)  # 第一帧速度设为 0
        velocity_y = np.insert(velocity_y, 0, 0)

        # 归一化/标准化
        all_positions_arr = np.concatenate(all_positions, axis=0)
        if normalize:
            min_pos = all_positions_arr.min(axis=0)
            max_pos = all_positions_arr.max(axis=0)
            traj_x = (traj_x - min_pos[0]) / (max_pos[0] - min_pos[0])
            traj_y = (traj_y - min_pos[1]) / (max_pos[1] - min_pos[1])
            velocity_x = velocity_x / np.max(np.abs(velocity_x))  # 归一化速度
            velocity_y = velocity_y / np.max(np.abs(velocity_y))
            norm_params = (min_pos, max_pos)
        else:
            mean_pos = all_positions_arr.mean(axis=0)
            std_pos = all_positions_arr.std(axis=0)
            traj_x = (traj_x - mean_pos[0]) / std_pos[0]
            traj_y = (traj_y - mean_pos[1]) / std_pos[1]
            velocity_x = (velocity_x - np.mean(velocity_x)) / np.std(velocity_x)  # 标准化速度
            velocity_y = (velocity_y - np.mean(velocity_y)) / np.std(velocity_y)
            norm_params = (mean_pos, std_pos)

        # 滑动窗口方式切片
        for i in range(0, len(frames) - input_time_steps - output_time_steps, 1):
            input_seq = np.stack([traj_x[i:i + input_time_steps],
                                  traj_y[i:i + input_time_steps],
                                  velocity_x[i:i + input_time_steps],
                                  velocity_y[i:i + input_time_steps],
                                  frames[i:i + input_time_steps]], axis=1)

            target_seq = np.stack([traj_x[i + input_time_steps:i + input_time_steps + output_time_steps],
                                   traj_y[i + input_time_steps:i + input_time_steps + output_time_steps],
                                   frames[i + input_time_steps:i + input_time_steps + output_time_steps]], axis=1)

            processed_trajectories.append((input_seq, target_seq))

    # 转换为 PyTorch 张量，并划分批次
    input_data = torch.tensor([traj[0] for traj in processed_trajectories], dtype=torch.float32)
    target_data = torch.tensor([traj[1] for traj in processed_trajectories], dtype=torch.float32)

    dataset = TensorDataset(input_data, target_data)

    # 这里不进行打乱数据，确保时序性
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return dataloader, norm_params


def process_eth_ucy(input_time_steps, output_time_steps, batch_size, normalize=True):
    """ 处理所有 ETH/UCY 数据集 """
    data_processed = {}
    normalization_params = {}
    for scene, path in DATA_PATHS.items():
        trajectories, params = preprocess_trajectories(load_eth_ucy_data(path), input_time_steps, output_time_steps,
                                                       batch_size, normalize)
        data_processed[scene] = trajectories
        normalization_params[scene] = params

    return data_processed, normalization_params


def split_leave_one_out(data_processed):
    """ 按 leave-one-out 方式划分训练/测试集 """
    scenes = list(DATA_PATHS.keys())
    split_data = {}

    for test_scene in scenes:
        train_scenes = [s for s in scenes if s != test_scene]
        train_data = sum([list(data_processed[s]) for s in train_scenes], [])
        test_data = data_processed[test_scene]

        split_data[test_scene] = {"train": train_data, "test": test_data}

    return split_data


def save_data(split_data, normalization_params):
    """ 保存训练和测试数据到 npy 文件 """
    for test_scene, data in split_data.items():
        np.save(f'train_{test_scene}.npy', data["train"])
        np.save(f'test_{test_scene}.npy', data["test"])

        # 保存归一化/标准化的参数
        np.save(f'normalization_params_{test_scene}.npy', normalization_params[test_scene])


def load_data(input_time_steps, output_time_steps, batch_size, normalize=True):
    """ 加载数据、处理、划分和保存，返回数据和归一化/标准化参数 """
    data_processed, normalization_params = process_eth_ucy(input_time_steps, output_time_steps, batch_size, normalize)
    split_data = split_leave_one_out(data_processed)
    # save_data(split_data, normalization_params)

    return split_data, normalization_params


# 反标准化/反归一化函数
def reverse_normalization(data, params, normalize=True):
    """ 对数据进行反归一化/反标准化 """
    if normalize:
        min_pos, max_pos = params
        # 反归一化
        data[:, :2] = data[:, :2] * (max_pos - min_pos) + min_pos
        return data
    else:
        mean_pos, std_pos = params
        # 反标准化
        data[:, :2] = data[:, :2] * std_pos + mean_pos
        return data


if __name__ == "__main__":
    input_time_steps = 8
    output_time_steps = 12
    batch_size = 32
    split_data, normalization_params = load_data(input_time_steps, output_time_steps, batch_size, normalize=True)
    print("数据加载和处理完成。")

    # 假设我们要反归一化/反标准化某个数据：
    example_data = split_data['ETH_Hotel']['test'][0][0]  # 获取一个测试样本的输入数据
    example_params = normalization_params['ETH_Hotel']  # 获取ETH_Hotel的归一化/标准化参数

    # 反归一化或反标准化
    reversed_data = reverse_normalization(example_data, example_params, normalize=True)
    print("反归一化数据：", reversed_data)
