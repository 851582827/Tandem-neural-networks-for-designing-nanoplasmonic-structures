import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 配置管理
config = {
    'device': "cuda:0" if torch.cuda.is_available() else "cpu",
    'seed': 2024,
    'batch_size': 128,
    'lr': 0.001,
    'epochs': 3000,
    'validation_interval': 10,
    'model_save_dir': "Model",
    'data_path': 'Data/train/',
    'train_files': ('train_thickness.csv', 'train_spectra.csv'),
    'valid_files': ('valid_thickness.csv', 'valid_spectra.csv')
}


def setup_environment(config):
    """初始化环境"""
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    os.makedirs(config['model_save_dir'], exist_ok=True)


def load_data(config, files):
    """加载数据集"""
    thickness_data = pd.read_csv(os.path.join(config['data_path'], files[0]))
    spectra_data = pd.read_csv(os.path.join(config['data_path'], files[1]))
    return torch.tensor(thickness_data.values, dtype=torch.float32), torch.tensor(spectra_data.values,
                                                                                  dtype=torch.float32)


def Mre(predictions, targets):
    """计算相对误差均值 (Mean Relative Error)"""
    relative_errors = torch.abs(predictions - targets) / torch.abs(targets)
    mre = torch.mean(relative_errors)
    return mre


def plot_results(train_loss, val_loss, val_mre, config):
    """绘制结果"""
    plt.figure(figsize=(3.54, 3.54))

    # 绘制训练和验证损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss', linestyle='-')
    plt.plot(range(config['validation_interval'], len(val_loss) * config['validation_interval'] + 1,
                   config['validation_interval']), val_loss, label='Validation Loss', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # 绘制验证MRE曲线
    plt.subplot(2, 1, 2)
    plt.plot(range(config['validation_interval'], len(val_mre) * config['validation_interval'] + 1,
                   config['validation_interval']), [m.cpu().numpy() for m in val_mre], label='Validation MRE',
             linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('MRE')
    plt.legend()
    plt.title('Validation Mean Relative Error')

    plt.tight_layout()
    plt.savefig(f'Pic/predict_loss.png')
    plt.show()


def train_model(model, config, train_dataloader, val_dataloader):
    optimizer = torch.optim.NAdam(model.parameters(), lr=config['lr'], weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=15, verbose=True)
    loss_p = nn.SmoothL1Loss()

    best_val_mre = float('inf')
    train_loss, val_loss, val_mre = [], [], []

    start_time = time.time()
    print("Starting training on device:", config['device'])
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            thickness, spectra = (tensor.to(config['device']) for tensor in batch)
            optimizer.zero_grad()
            outputs = model(thickness)
            loss = loss_p(outputs, spectra)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_dataloader)
        train_loss.append(average_loss)
        print(f'Epoch [{epoch + 1}/{config["epochs"]}], Average Loss: {average_loss}')

        if (epoch + 1) % config['validation_interval'] == 0:
            model.eval()
            with torch.no_grad():
                val_loss_total, val_mre_total = 0.0, 0.0
                for val_batch in val_dataloader:
                    vals, labels = (tensor.to(config['device']) for tensor in val_batch)
                    val_outputs = model(vals)
                    loss = loss_p(val_outputs, labels)
                    val_loss_total += loss.item()
                    val_mre_total += Mre(val_outputs, labels)

                average_val_loss = val_loss_total / len(val_dataloader)
                average_val_mre = val_mre_total / len(val_dataloader)
                val_loss.append(average_val_loss)
                val_mre.append(average_val_mre)

                print(f'Epoch [{epoch + 1}/{config["epochs"]}], Average Validation Loss: {average_val_loss}')
                print(f'Validation MRE: {average_val_mre}')

                if average_val_mre < best_val_mre:
                    best_val_mre = average_val_mre
                    torch.save(model.state_dict(),
                               os.path.join(config['model_save_dir'], f"predict_best.pth"))
                    print(f"Best model saved at epoch {epoch + 1} with validation MRE {best_val_mre}")

        scheduler.step(average_loss)

    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("time-consuming: {:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds)))
    plot_results(train_loss, val_loss, val_mre, config)


class Predict(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 400)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    setup_environment(config)
    thickness_tensor, spectra_tensor = load_data(config, config['train_files'])
    val_thickness_tensor, val_spectra_tensor = load_data(config, config['valid_files'])

    train_dataset = TensorDataset(thickness_tensor, spectra_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataset = TensorDataset(val_thickness_tensor, val_spectra_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    model = Predict().to(config['device'])
    train_model(model, config, train_dataloader, val_dataloader)