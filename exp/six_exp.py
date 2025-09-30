"""
日志：
20240530: 用标准的建模架构
20240530：要先将经纬度这种绝对数据转换为dis-ang这种相对数据，但训练要用经纬度数据里避免0-360°翻转的问题
20240531: 引入加性注意力机制，效果非常好
20240606: LSTM+transformer+KF+自注意力 效果非常好
20240615: 改成固定周期,重写lstm_model_test
20240615: 改写了各种注意力模型
20240615: 改写成informer(效果很好)
20240618: test里融合预测值和观测值
20240618: test里融合预测值和观测值dis和ang分开融合
"""
import torch
import numpy as np
import csv
import sys
import torch.nn as nn
import time
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from functions import inverse_standardize, cartesian_to_polar, haversine_distances
from matplot_func import polar_plot_actual_pred1, plot_training_losses
# 现在可以导入 src 里的模块
from dataloaders.prepare.six.six_datasets_loader import load_data
from sklearn.metrics import mean_squared_error, mean_absolute_error
import csv
from datetime import datetime  # 用于获取当前时间
from Efficiency_Benchmark import benchmark_efficiency


import pandas as pd
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from datetime import datetime


import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def visualize_best_samples(best_samples, base_samples_dir='./samples', plot_feature_idx=6):
    inputs = best_samples['inputs']   # (batch_size, seq_len_in, feature_dim)
    outputs = best_samples['outputs']  # (batch_size, seq_len_out, feature_dim)
    targets = best_samples['targets']  # (batch_size, seq_len_out, feature_dim)

    input_seq = inputs[0].cpu().numpy()   # (seq_len_in, feature_dim)
    target_seq = targets[0].cpu().numpy() # (seq_len_out, feature_dim)
    output_seq = outputs[0].cpu().numpy() # (seq_len_out, feature_dim)

    seq_len_in, feature_dim = input_seq.shape
    seq_len_out = target_seq.shape[0]

    # ================== 可视化（只画一个特征） ==================
    real_seq_plot = np.concatenate([input_seq[:, plot_feature_idx], target_seq[:, plot_feature_idx]])
    pred_seq_plot = np.concatenate([np.full(seq_len_in, np.nan), output_seq[:, plot_feature_idx]])

    plt.figure(figsize=(12, 6))
    plt.plot(real_seq_plot, label='Real (Input + Target)', color='blue')
    plt.plot(pred_seq_plot, label='Prediction (Output)', color='orange')
    plt.legend()
    plt.title(f'Best batch example: Feature {plot_feature_idx}')
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()

    # ================== 保存为 CSV（保存所有特征） ==================
    os.makedirs(base_samples_dir, exist_ok=True)
    csv_path = os.path.join(base_samples_dir, f'best_sequence_{inputs.shape[1]}.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写表头
        header = ['TimeStep']
        for i in range(feature_dim):
            header += [f'real_f{i}', f'pred_f{i}']
        writer.writerow(header)

        # 写数据行
        for t in range(seq_len_in + seq_len_out):
            row = [t]
            for i in range(feature_dim):
                # 真实值
                if t < seq_len_in:
                    real_val = input_seq[t, i]
                else:
                    real_val = target_seq[t - seq_len_in, i]
                # 预测值
                if t < seq_len_in:
                    pred_val = np.nan
                else:
                    pred_val = output_seq[t - seq_len_in, i]
                row += [real_val, pred_val]
            writer.writerow(row)

    print(f"✅ 所有特征的时间序列已保存为 CSV: {csv_path}")




class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        """
        早停类，用于监控验证损失
        Args:
            patience (int): 没有改善的验证损失允许的最大次数
            min_delta (float): 最小改善值，如果变化小于这个值，则不认为有改善
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


import torch
import torch.nn as nn


class MSEHuberLoss(nn.Module):
    def __init__(self, delta=1.0, alpha=0.5):
        """
        组合 MSE 和 Huber 作为损失函数，适用于 LSTF 任务
        :param delta: Huber 损失的阈值
        :param alpha: MSE 和 Huber 的加权系数 (0~1)，默认为 0.5
        """
        super(MSEHuberLoss, self).__init__()
        self.delta = delta
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def huber_loss(self, y_pred, y_true):
        abs_error = torch.abs(y_true - y_pred)
        quadratic = 0.5 * abs_error ** 2
        linear = self.delta * (abs_error - 0.5 * self.delta)
        return torch.where(abs_error < self.delta, quadratic, linear).mean()

    def forward(self, y_pred, y_true):
        mse = self.mse_loss(y_pred, y_true)
        huber = self.huber_loss(y_pred, y_true)
        return self.alpha * mse + (1 - self.alpha) * huber

    def evaluate(self, y_pred, y_true):
        """
        评估模型时计算 MSE 和 MAE
        :param y_pred: 预测值
        :param y_true: 真实值
        :return: MSE 和 MAE
        """
        mse = self.mse_loss(y_pred, y_true)
        mae = self.mae_loss(y_pred, y_true)
        return mse, mae
# class TimeSeriesLoss(nn.Module):
#     """
#     适用于时间序列数据的损失函数
#     - 训练时使用 MSE 作为优化目标
#     - 评估时计算 MSE 和 MAE 进行比较
#     """
#     def __init__(self):
#         super(TimeSeriesLoss, self).__init__()
#         self.mse_loss = nn.MSELoss()
#         self.mae_loss = nn.L1Loss()
#
#     def forward(self, y_pred, y_true):
#         """
#         :param y_pred: 预测值 (batch_size, output_time_steps, num_features)
#         :param y_true: 真实值 (batch_size, output_time_steps, num_features)
#         :return: MSE 损失（用于训练）
#         """
#         return self.mse_loss(y_pred, y_true)
#
#     def evaluate(self, y_pred, y_true):
#         """
#         评估模型时计算 MSE 和 MAE
#         :param y_pred: 预测值
#         :param y_true: 真实值
#         :return: MSE 和 MAE
#         """
#         mse = self.mse_loss(y_pred, y_true)
#         mae = self.mae_loss(y_pred, y_true)
#         return mse, mae


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, min_lr=0.000001, patience=2):
    """
    训练模型并添加验证逻辑。

    :param model: 训练的 PyTorch 模型
    :param train_loader: 训练数据集的 DataLoader
    :param val_loader: 验证数据集的 DataLoader
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param scheduler: 学习率调度器
    :param input_features: 输入特征数
    :param output_features: 输出特征数
    :param num_epochs: 训练轮数
    :param device: 训练设备 (CPU / GPU)
    :param min_lr: 学习率的最小值
    :param patience: 早停的耐心值（多少个 epoch 验证损失未提升则停止）
    :return: 训练损失、验证损失
    """

    early_stopping = EarlyStopping(patience=patience, min_delta=min_lr)
    train_losses, val_losses = [], []

    model.to(device)
    count_parameters(model)  # ✅ 放在这里只统计/打印一次

    for epoch in range(num_epochs):
        torch.cuda.reset_peak_memory_stats(device)  # ✅ 插在模型送入设备之后
        # ====== 训练阶段 ======
        model.train()
        epoch_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            pbar.set_postfix({'Train Loss': f'{loss.item():.6e}'})

        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"🚀 Peak GPU memory usage (epoch {epoch + 1}): {peak_memory:.2f} MB")
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        pbar.close()

        # ====== 验证阶段 ======
        model.eval()
        epoch_val_loss = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                epoch_val_loss += val_loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # 学习率调度器更新
        scheduler.step(avg_val_loss)  # 这里使用验证损失调整学习率
        for param_group in optimizer.param_groups:
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr

        # 检查早停（基于验证损失）
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"⏹️ Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
            break

    return train_losses, val_losses


def save_training_results(model, optimizer, num_epochs, train_losses, mse, mae, best_samples, model_name, time_step,
                          base_model_dir='./models', base_loss_dir='./loss', base_metric_dir='./metrics',
                          base_samples_dir='./samples'):
    """
    仅当当前 MSE 低于历史最小 MSE 时，保存训练损失、模型参数和测试指标。

    :param model: PyTorch 训练模型
    :param optimizer: 优化器
    :param num_epochs: 训练轮数
    :param train_losses: 训练损失列表
    :param mse: 评估指标 MSE
    :param mae: 评估指标 MAE
    :param model_name: 模型名称
    :param time_step: 时间步长 (如 96, 128)
    :param base_model_dir: 保存模型的基础路径
    :param base_loss_dir: 保存损失的基础路径
    :param base_metric_dir: 保存评价指标的基础路径
    """

    # 获取当前时间字符串，格式化为 'YYYYMMDD_HHMMSS'，用于版本管理
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ================== 1. 目录管理 ==================
    loss_dir = os.path.join(base_loss_dir, 'train_loss')   # 训练损失文件夹
    model_dir = os.path.join(base_model_dir, 'modelSavePth')  # 模型参数文件夹
    metric_dir = os.path.join(base_metric_dir, 'metrics')  # 评价指标文件夹

    # 确保所有目录存在
    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

    # ================== 2. 文件路径 ==================
    loss_file_path = os.path.join(loss_dir, f'loss_{model_name}_T{time_step}.csv')  # 训练损失文件
    metric_file_path = os.path.join(metric_dir, f'metrics_{model_name}_T{time_step}.csv')  # 评价指标文件
    model_path = os.path.join(model_dir, f'{model_name}_T{time_step}.pth')  # 模型参数文件

    # ================== 3. 读取历史最小 MSE ==================
    best_mse = float('inf')  # 初始设为无穷大
    if os.path.exists(metric_file_path):
        with open(metric_file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                best_mse = float(row[0])  # 读取历史最小 MSE

    # ================== 4. 仅当当前 MSE 低于历史最小 MSE 时，才保存数据 ==================
    plot_feature_idx = 5  # exchange_rate = 6、 electricity = 5, ETT-small, national_illness, weather=38 , traffic = 5
    if mse < best_mse:
        print(f"🔹 当前 MSE ({mse:.6f}) 低于历史最小 MSE ({best_mse:.6f})，更新数据...")

        # ========== (1) 保存新的训练损失 ==========
        with open(loss_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Loss', 'TimeStep'])  # 添加时间步信息
            for epoch, loss in enumerate(train_losses, 1):
                writer.writerow([epoch, loss, time_step])

        print(f"✅ 训练损失已保存到 {loss_file_path}")

        # ========== (2) 保存模型参数 ==========
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_losses,
            'mse': mse,  # 存储最优 MSE
            'mae': mae,  # 存储最优 MAE
            'time_step': time_step,  # 额外存储时间步信息
            'timestamp': current_time  # 额外存储时间戳
        }, model_path)

        print(f"✅ 模型已保存到 {model_path}")

        # ========== (3) 保存新的 MSE 和 MAE ==========
        with open(metric_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['MSE', 'MAE', 'TimeStep'])  # 确保第一行是 MSE、MAE 和时间步长
            writer.writerow([mse, mae, time_step])

        print(f"✅ 评价指标已保存到 {metric_file_path}")

        # visualize_best_samples(best_samples, base_samples_dir, plot_feature_idx=plot_feature_idx)

    else:
        # visualize_best_samples(best_samples, base_samples_dir, plot_feature_idx=plot_feature_idx)
        print(f"❌ 当前 MSE ({mse:.6f}) 大于或等于历史最小 MSE ({best_mse:.6f})，跳过保存。")


def dataset_selection(datasetSelection):
    """根据 datasetSelection 选择并加载相应的数据集"""

    # 数据集路径字典
    dataset_dict = {
        "electricity": {
            "dataset_path": os.path.join("..", "six_dataset", "electricity"),
            "vis_path": os.path.join("..", "six_dataset", "electricity", "result", "FDU"),
            "num_features": 321  # 这里是特征个数，示例值
        },
        "ETT-small": {
            "dataset_path": os.path.join("..", "six_dataset", "ETT-small"),
            "vis_path": os.path.join("..", "six_dataset", "ETT-small", "result", "IDS", "Attention"),
            "num_features": 7
        },
        "exchange_rate": {
            "dataset_path": os.path.join("..", "six_dataset", "exchange_rate"),
            "vis_path": os.path.join("..", "six_dataset", "exchange_rate", "result", "IDS", "S4"),
            "num_features": 8
        },
        "national_illness": {
            "dataset_path": os.path.join("..", "six_dataset", "illness"),
            "vis_path": os.path.join("..", "six_dataset", "illness", "result", "FDU"),
            "num_features": 7
        },
        "traffic": {
            "dataset_path": os.path.join("..", "six_dataset", "traffic"),
            "vis_path": os.path.join("..", "six_dataset", "traffic", "result", "FDU"),
            "num_features": 862
        },
        "weather": {
            "dataset_path": os.path.join("..", "six_dataset", "weather"),
            "vis_path": os.path.join("..", "six_dataset", "weather", "result", "FDU", "LSTM"),
            "num_features": 21
        }
    }

    # 检查数据集是否有效
    if datasetSelection not in dataset_dict:
        raise ValueError(f"❌ 未知的数据集选择: {datasetSelection}")

    # 获取数据集路径
    dataset_info = dataset_dict[datasetSelection]
    dataset_path, vis_path, num_features = dataset_info["dataset_path"], dataset_info["vis_path"], dataset_info["num_features"]

    print(f"✅ 已加载数据集: {datasetSelection}")
    return vis_path, dataset_path, num_features  #


import torch
import torch.nn as nn


def validate_model(model, dataloader, device):
    """
    在测试集上验证模型，并计算 MSE 和 MAE 评价指标（使用 PyTorch 内置函数）。

    :param model: 训练好的深度学习模型
    :param dataloader: 测试数据集的 DataLoader
    :param device: 设备 (CPU or CUDA)
    :return: (MSE, MAE, best_samples)
             best_samples 是一个 dict，包含 'inputs', 'outputs', 'targets'，对应 MSE 最小的批次
    """
    model.eval()  # 进入评估模式
    mse_loss_fn = nn.MSELoss(reduction='mean')  # 计算总 MSE，稍后归一化
    mae_loss_fn = nn.L1Loss(reduction='mean')  # 计算总 MAE，稍后归一化

    total_mse, total_mae = 0, 0
    count = len(dataloader)

    best_mse = float('inf')
    best_item = {'inputs': None, 'outputs': None, 'targets': None}

    with torch.no_grad():  # 关闭梯度计算
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # 进行预测

            batch_mse = mse_loss_fn(outputs, targets).item()
            batch_mae = mae_loss_fn(outputs, targets).item()

            total_mse += batch_mse
            total_mae += batch_mae

            # 更新最优批次
            if batch_mse < best_mse:
                best_mse = batch_mse
                # 为了后续使用，保存在 CPU 上的 tensor
                best_item['inputs'] = inputs.cpu()
                best_item['outputs'] = outputs.cpu()
                best_item['targets'] = targets.cpu()

    avg_mse = total_mse / count
    avg_mae = total_mae / count

    return avg_mse, avg_mae, best_item


def benchmark_throughput(model, dataloader, device='cuda', warmup=10, trials=100):
    """
    用测试集中的真实数据测量模型推理吞吐量（samples/s）

    参数:
        model: 已加载参数的模型
        dataloader: 用于推理的 DataLoader（建议用 test_loader）
        device: 设备 (默认 'cuda')
        warmup: 预热次数（默认 10）
        trials: 正式测量的批次数（默认 100）
    """
    model.to(device)
    model.eval()

    it = iter(dataloader)
    timings = []

    with torch.no_grad():
        # 预热阶段（不计时）
        for _ in range(warmup):
            try:
                inputs, _ = next(it)
            except StopIteration:
                it = iter(dataloader)
                inputs, _ = next(it)
            inputs = inputs.to(device)
            _ = model(inputs)

        torch.cuda.synchronize()

        # 正式计时
        for _ in range(trials):
            try:
                inputs, _ = next(it)
            except StopIteration:
                it = iter(dataloader)
                inputs, _ = next(it)
            inputs = inputs.to(device)

            torch.cuda.synchronize()
            start = time.time()
            _ = model(inputs)
            torch.cuda.synchronize()
            end = time.time()

            timings.append(end - start)

    avg_latency = sum(timings) / len(timings)
    batch_size = inputs.size(0)
    throughput = batch_size / avg_latency

    print(f"📈 Avg Latency per batch: {avg_latency * 1000:.2f} ms")
    print(f"⚡ Throughput: {throughput:.2f} samples/s")
    return avg_latency, throughput


# **添加 models 目录到 sys.path**
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前
project_root = os.path.dirname(current_dir)  # 获取 KF-O3S1 目录
models_path = os.path.join(project_root, "models")  # 确保 models 目录在 sys.path 里
if models_path not in sys.path:
    sys.path.append(models_path)


def model_selective(modelSelection, num_features):
    """根据 modelSelection 选择并实例化不同的模型"""
    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # **模型字典：映射模型名称到 (模块名, 参数变量, 模型类)**
    model_dict = {
        "LSTM": ("LSTMDefinite", "paramsLSTM", "LSTMModel"),
        "Mamba": ("MambaDefinite", "paramsMamba", "MambaModel"),
        "Transformer": ("TransformerDefine", "paramsTransformer", "TransformerModel"),
        "KOSS": ("KOSSDefiniteSix", "paramsKOSS", "KOSSModel"),
        "KOSLM": ("KOSLMDefiniteSix", "paramsConfig", "KOSLMModel"),
    }

    # 检查是否是有效模型
    if modelSelection not in model_dict:
        raise ValueError(f"❌ 未知的模型选择: {modelSelection}")

    # 获取模块、参数、模型名称
    module_name, params_name, model_name = model_dict[modelSelection]

    # 动态导入模块
    module = __import__(module_name, fromlist=[params_name, model_name])

    # 获取参数和模型
    params = getattr(module, params_name)
    ModelClass = getattr(module, model_name)
    params["input_features"] = num_features
    params["output_features"] = num_features

    # 通用参数解析
    model_kwargs = {
        "input_features": params["input_features"],
        "output_time_steps": params["output_time_steps"],
        "output_features": params["output_features"],
    }

    # 各模型的额外参数
    if modelSelection == "LSTM":
        model_kwargs.update({
            "lstm_hidden_size": params["lstm_hidden_size"],
            "lstm_num_layers": params["lstm_num_layers"],
            "input_time_steps": params["input_time_steps"],
            "period": params["period"],
        })
    elif modelSelection == "Mamba":
        model_kwargs.update({
            "mamba_hidden_size": params["mamba_hidden_size"],
            "mamba_num_layers": params["mamba_num_layers"],
            "period": params["period"],
        })
    elif modelSelection == "Transformer":
        model_kwargs.update({
            "transformer_hidden_size": params["transformer_hidden_size"],
            "num_transformer_layers": params["num_transformer_layers"],
            "num_heads": params["num_heads"],
            "input_time_steps": params["input_time_steps"],
        })
    elif modelSelection == "KOSS":
        model_kwargs.update({
            "KOSS_hidden_size": params["KOSS_hidden_size"],
            "KOSS_num_layers": params["KOSS_num_layers"],
        })

    elif modelSelection == "KOSLM":
        model_kwargs.update({
            "KOSLM_hidden_size": params["KOSLM_hidden_size"],
            "KOSLM_num_layers": params["KOSLM_num_layers"],
            "observation_level_size": params["observation_level_size"],
        })

    # 实例化模型并移动到设备
    model = ModelClass(**model_kwargs).to(device)

    print(f"✅ 已选择模型: {modelSelection}")
    return params, model, device


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = total_params / 1e6
    trainable_params = trainable_params / 1e6

    print(f"Total parameters: {total_params:.2f}M")
    print(f"Trainable parameters: {trainable_params:.2f}M")

    return total_params, trainable_params


# 选择模型
# modelSelection = "LSTM"  # 可更改为 "Mamba" 或 "Transformer"或"Mamba"或"S6KF"
# modelSelection = "Mamba"  # 可更改为 "Mamba" 或 "Transformer"或"Mamba"或"S6KF"
# modelSelection = "Transformer"  # 可更改为 "Mamba" 或 "Transformer"或"Mamba"或"S6KF"
# modelSelection = "KOSS"  # 可更改为 "Mamba" 或 "Transformer"或"Mamba"或"S6KF"
modelSelection = "KOSLM"  # 可更改为 "Mamba" 或 "Transformer"或"Mamba"或"S6KF"
datasetSelection = "exchange_rate"  # exchange_rate、 electricity, ETT-small, national_illness, weather, traffic


def main():
    vis_path, dataset_path, num_features = dataset_selection(datasetSelection)
    criterion = MSEHuberLoss()

    #############################benchmark_efficiency################################
    # _, model_transformer, device = model_selective("Transformer", num_features)
    # _, model_mamba, device = model_selective("Mamba", num_features)
    # _, model_lstm, device = model_selective("LSTM", num_features)
    # _, model_koslm, device = model_selective("KOSLM", num_features)
    #
    # models = {
    #     "Transformer": model_transformer.to(device),
    #     "Mamba": model_mamba.to(device),
    #     "LSTM": model_lstm.to(device),
    #     "KOSLM": model_koslm.to(device),
    # }
    #
    # benchmark_efficiency(models, dataset_path, L_list=[512, 1024, 2048, 4096, 8192, 16384], B=8, D=num_features, amp=True)
    #############################benchmark_efficiency################################

    params, model, device = model_selective(modelSelection, num_features)
    train_loader, val_loader, test_loader, scaler = load_data(
        dataset_path, params["input_time_steps"], params["output_time_steps"], params["batch_size"]
    )

    best_mae = float("inf")
    best_mse = None
    best_train_losses = None
    best_samples = None
    total_mse, total_mae = 0, 0
    total_latency, total_throughput = 0, 0

    for run in range(3):  # 进行4次训练
        print(f"\n===== 第 {run + 1} 次训练 =====\n")
        model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        # 训练
        train_losses, _ = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                      params["num_epochs"], device)

        avg_latency, throughput = benchmark_throughput(model, test_loader, device='cuda')

        # 测试
        mse, mae, best_item = validate_model(model, test_loader, device)
        print(f"Run {run + 1} - Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")
        total_mse += mse
        total_mae += mae
        total_latency += avg_latency
        total_throughput += throughput

        # # 更新最佳模型
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_train_losses = train_losses
            best_samples = best_item

    print(f"\n✅ 训练完成！最优 MAE: {best_mae:.4f}，对应 MSE: {best_mse:.4f}")
    # 画最优损失函数
    plot_training_losses(best_train_losses)

    # best_mse = total_mse/3
    # best_mae = total_mae/3
    print(f"\n✅ 训练完成！平均 MAE: {best_mae:.4f}，对应 MSE: {best_mse:.4f}")

    print(f"📈 Avg Latency per batch: {total_latency / 3 * 1000:.2f} ms")
    print(f"⚡ Avg Throughput: {total_throughput / 3:.2f} samples/s")
    # 保存最优模型
    save_training_results(
        model=model,
        optimizer=optimizer,
        num_epochs=params["num_epochs"],
        train_losses=best_train_losses,
        mse=best_mse,
        mae=best_mae,
        best_samples=best_samples,
        model_name=modelSelection,
        time_step=params["input_time_steps"],
        base_model_dir=vis_path,
        base_loss_dir=vis_path,
        base_metric_dir=vis_path,
        base_samples_dir=vis_path
    )


if __name__ == "__main__":
    main()
