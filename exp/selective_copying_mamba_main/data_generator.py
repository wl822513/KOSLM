import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset


def torch_copying_data_Context_Aware(L, M, A, intensity=0.1, variable=False, variable_length=False, batch_shape=(),
                                     one_hot=False, reverse=False):
    """
    基于torch_copying_data改造，添加context-aware污染token
    - 污染token插入在data tokens后面（紧邻）
    - 污染token值为对应data tokens的intensity比例，模拟污染影响（用float实现）

    参数：
    intensity (float): 污染程度比例，0~1之间。0表示无污染，1表示污染和data tokens相同。

    返回：
    x: 输入序列，float类型（含污染token）
    y: 目标tokens，long类型
    """
    if variable_length:
        M = int(random.random() * M) + 1

    tokens = torch.randint(low=1, high=A - 1, size=batch_shape + (M,))

    if variable:
        total_batch = int(np.prod(batch_shape))
        inds = torch.stack([
            torch.randperm(L + M)[:M]
            for _ in range(total_batch)
        ], 0)
        inds = inds.reshape(batch_shape + (M,))
        inds, _ = inds.sort()
    else:
        inds = torch.arange(M).repeat(batch_shape + (1,))

    # 先生成长度L+M+M(污染token数量同data tokens数量)的零tensor，float类型
    zeros_x = torch.zeros(batch_shape + (L + M * 2,), dtype=torch.float)

    # 把tokens放到L开始的位置
    zeros_x[..., L:L + M] = tokens.float()

    # 污染token放在 data tokens 后面，污染强度为 intensity * data tokens（模拟污染）
    pollution_tokens = tokens.float() * intensity
    noise = torch.rand_like(pollution_tokens) * 0.1  # 可调噪声
    pollution_tokens = pollution_tokens * (1.0 + noise)
    zeros_x[..., L + M:L + M * 2] = pollution_tokens

    # 目标y仍为tokens原始整数序列
    y_ = tokens
    if reverse:
        y_ = y_.flip(-1)

    # 在输入序列尾部添加M个分隔符（A-1）
    markers = (A - 1) * torch.ones(batch_shape + (M,), dtype=torch.float)
    x_ = torch.cat([zeros_x, markers], dim=-1)

    if one_hot:
        # one-hot编码需要整数，污染token是float，不能one-hot，提示用户
        raise ValueError("one_hot encoding不支持含污染token的float输入")
    else:
        x = x_
    y = y_
    return x, y


def torch_copying_data(L, M, A, variable=False, variable_length=False, batch_shape=(), one_hot=False, reverse=False):
    """
    Generate a dataset for a sequence copying task.
    This code is adopted from the copying.py script in the S4 repository. The original code can be found at:
    https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/src/dataloaders/datasets/copying.py

    Parameters:
    L (int): Number of padding tokens
    M (int): Number of tokens to memorize
    A (int): Alphabet size
    variable (bool): If True, selective copying task
    variable_length (bool): If True, randomize number of tokens to memorize
    batch_shape (tuple): Shape of the batch
    one_hot (bool): If True, convert the input sequence into a one-hot encoded tensor
    reverse (bool): If True, reverse the order of the target sequence

    Returns:
    tuple: Generated input sequence and target sequence
    """
    if variable_length:
        M = int(random.random() * M) + 1
    tokens = torch.randint(low=1, high=A-1, size=batch_shape+(M,))
    if variable:
        total_batch = int(np.prod(batch_shape))
        inds = torch.stack([
            torch.randperm(L+M)[:M]
            for _ in range(total_batch)
            ], 0)
        inds = inds.reshape(batch_shape+(M,))
        inds, _ = inds.sort()
    else:
        inds = torch.arange(M).repeat(batch_shape+(1,))
    zeros_x = torch.zeros(batch_shape+(M+L,), dtype=torch.long)
    zeros_x.scatter_(-1, inds, tokens)
    markers = (A-1) * torch.ones(batch_shape+(M,), dtype=torch.long)

    x_ = torch.cat([zeros_x, markers], dim=-1)
    y_ = torch.cat([tokens], dim=-1)
    if reverse: y_ = y_.flip(-1)
    if one_hot: x = F.one_hot(x_, A).float()
    else: x = x_
    y = y_
    return x, y

"""
Examples:
print(torch_copying_data(10, 5, 10, variable=False, variable_length=False, batch_shape=(), one_hot=False, reverse=False))
print(torch_copying_data(10, 5, 10, variable=True, variable_length=False, batch_shape=(), one_hot=False, reverse=False))
Outputs:
(tensor([2, 2, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9]), tensor([2, 2, 2, 4, 6])) # copying memory task
(tensor([0, 6, 0, 0, 0, 0, 0, 6, 7, 0, 7, 5, 0, 0, 0, 9, 9, 9, 9, 9]), tensor([6, 6, 7, 7, 5])) # selective copying task
(tensor([0, 6, 0.6, 0, 0, 0, 0, 6, 7, 0.7, 7, 5, 0.5, 0, 0, 9, 9, 9, 9, 9]), tensor([6, 6, 7, 7, 5])) # Context_Aware selective copying task
"""
def generate_dataset(dataset_config, training_config):
    """
    Generate a dataset based on the provided configuration.

    Parameters:
    dataset_config (dict): Configuration for the dataset
    training_config (dict): Configuration for the training

    Returns:
    tuple: Generated inputs and targets
    """
    if dataset_config["intensity"]:
        x, y = torch_copying_data_Context_Aware(dataset_config["l_noise"], dataset_config["l_memorize"],
                                                dataset_config["n_tokens"], dataset_config["intensity"],
                                                batch_shape=(training_config["batch_size"],),variable=dataset_config["variable"],
                                                variable_length=dataset_config["variable_length"], one_hot=dataset_config["one_hot"],
                                                reverse=dataset_config["reverse"])
    else:
        x, y = torch_copying_data(dataset_config["l_noise"], dataset_config["l_memorize"], dataset_config["n_tokens"],
                                  batch_shape=(training_config["batch_size"],),variable=dataset_config["variable"],
                                  variable_length=dataset_config["variable_length"], one_hot=dataset_config["one_hot"],
                                  reverse=dataset_config["reverse"])
    return x, y


class CopyingDataset(Dataset):
    def __init__(self, dataset_config, training_config):
        self.inputs = []
        self.targets = []
        for _ in range(training_config["num_steps"] // training_config["num_page"]):
            x, y = torch_copying_data(dataset_config["l_noise"], dataset_config["l_memorize"],
                                      dataset_config["n_tokens"],
                                      batch_shape=(training_config["batch_size"],), variable=dataset_config["variable"],
                                      variable_length=dataset_config["variable_length"],
                                      one_hot=dataset_config["one_hot"],
                                      reverse=dataset_config["reverse"])
            self.inputs.append(x)
            self.targets.append(y)
        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]




