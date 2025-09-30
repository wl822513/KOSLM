import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler



DATA_PATHS = {
    "electricity": "six_dataset/electricity/electricity.csv",
    "ETT-small": "six_dataset/ETT-small/ETTm2.csv",
    "exchange_rate": "six_dataset/exchange_rate/exchange_rate.csv",
    "national_illness": "six_dataset/illness/national_illness.csv",
    "traffic": "six_dataset/traffic/traffic.csv",
    "weather": "six_dataset/weather/weather.csv",
}


def load_data(data_path, input_time_steps, output_time_steps, batch_size, train_ratio=0.7, val_ratio=0.2):
    all_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]

    all_x, all_y = [], []

    print("Loading dataset...")

    for file in tqdm(all_files, desc="Processing files"):
        df = pd.read_csv(file)
        # **单独标准化每个数据集**
        scaler = StandardScaler()
        data_values = scaler.fit_transform(df.iloc[:, 1:].values.astype(np.float32))

        # **创建滑动窗口**
        def create_windows_torch(data, input_time_steps, output_time_steps, stride):
            data = torch.tensor(data, dtype=torch.float32)
            num_samples, num_features = data.shape
            print(f"num_samples: {num_samples}")
            print(f"input_time_steps: {input_time_steps}")
            print(f"output_time_steps: {output_time_steps}")
            print(f"stride: {stride}")
            print(f"end: {num_samples - input_time_steps - output_time_steps + 1}")

            indices = torch.arange(0, num_samples - input_time_steps - output_time_steps + 1, step=stride)

            x = torch.stack([data[i:i + input_time_steps] for i in indices])
            y = torch.stack([data[i + input_time_steps:i + input_time_steps + output_time_steps] for i in indices])

            return x.numpy(), y.numpy()

        x, y = create_windows_torch(data_values, input_time_steps, output_time_steps, stride=2)
        all_x.append(x)
        all_y.append(y)

    # **拼接所有文件的数据**
    all_x = np.concatenate(all_x, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    num_samples = len(all_x)

    # **按照时间顺序划分训练、验证、测试集**
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    train_x, train_y = all_x[:train_end], all_y[:train_end]
    val_x, val_y = all_x[train_end:val_end], all_y[train_end:val_end]
    test_x, test_y = all_x[val_end:], all_y[val_end:]

    # **转换为 PyTorch 数据集**
    train_loader = DataLoader(TensorDataset(torch.tensor(train_x), torch.tensor(train_y)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_x), torch.tensor(val_y)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_x), torch.tensor(test_y)), batch_size=batch_size, shuffle=False)

    print("✅ 数据集加载完成（符合 LSTF 标准）")

    return train_loader, val_loader, test_loader, scaler


def load_small_inference_set(data_path, input_time_steps, output_time_steps, batch_size):
    all_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]

    print("Loading small inference dataset (first 5 samples)...")

    for file in tqdm(all_files, desc="Processing files"):
        df = pd.read_csv(file)
        scaler = StandardScaler()
        data_values = scaler.fit_transform(df.iloc[:, 1:].values.astype(np.float32))

        def create_windows_torch(data, input_time_steps, output_time_steps, max_samples=1, stride=1):
            data = torch.tensor(data, dtype=torch.float32)
            num_samples = data.shape[0]
            max_start = min(num_samples - input_time_steps - output_time_steps + 1, max_samples)
            indices = torch.arange(0, max_start, step=stride)
            x = torch.stack([data[i:i + input_time_steps] for i in indices])
            y = torch.stack([data[i + input_time_steps:i + input_time_steps + output_time_steps] for i in indices])
            return x.numpy(), y.numpy()

        x, y = create_windows_torch(data_values, input_time_steps, output_time_steps)

        inference_loader = DataLoader(
            TensorDataset(torch.tensor(x), torch.tensor(y)),
            batch_size=batch_size, shuffle=False
        )

        break  # 只处理第一个文件

    print("✅ 推理样本加载完成（仅用于性能测试）")
    return inference_loader




