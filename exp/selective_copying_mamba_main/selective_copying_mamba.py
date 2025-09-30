# 在导入 Triton 或其他库之前先加载 VS 编译环境
from init_env import load_visual_studio_env

# 加载 VS 环境
load_visual_studio_env()

import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from config import training_config, dataset_config, MambaConfig
from data_generator import generate_dataset, CopyingDataset
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# Define model
mambaconfig = MambaConfig()
model = MambaLMHeadModel(mambaconfig, device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])
# Training function
def train():
    """
    Train the model.
    """
    model.train()
    start_time = time.time()

    for Page in range(training_config["num_page"]):
        logger.info(f'Page:{Page + 1} ')
        # 预生成所有训练数据并构建 DataLoader
        train_dataset = CopyingDataset(dataset_config, training_config)
        num_workers = min(8, os.cpu_count())  # 自动设置为最多20核心
        train_loader = DataLoader(train_dataset, batch_size=1,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)

        for step, (inputs, targets) in enumerate(train_loader):
            if step >= training_config["num_steps"] // training_config["num_page"]:
                break
            step_loss = 0
            correct = 0
            total = 0
            inputs = inputs.to(device).squeeze(0)  # 现在 inputs shape: (batch_size, seq_len)
            targets = targets.to(device).squeeze(0)
            outputs = model(inputs, num_last_tokens=dataset_config['l_memorize']).logits
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_loss += loss.item()
            total += targets.size(0) * targets.size(1)
            correct += (outputs.argmax(1) == targets).sum().item()
            accuracy = 100 * correct / total
            logger.info(f'Step [{step+1}/{training_config["num_steps"]}], Loss: {step_loss/training_config["batch_size"]:.4f}, Accuracy: {accuracy:.2f}%')

    end_time = time.time()
    logger.info(f'Training completed in: {(end_time - start_time)/60:.2f} minutes')

# Validation function
def validate():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        inputs, targets = generate_dataset(dataset_config, training_config)
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs, num_last_tokens=dataset_config['l_memorize']).logits
        total += targets.size(0) * targets.size(1)
        correct += (outputs.argmax(1) == targets).sum().item()
        accuracy = 100 * correct / total
        logger.info(f'Validation Accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    train()
    validate()

