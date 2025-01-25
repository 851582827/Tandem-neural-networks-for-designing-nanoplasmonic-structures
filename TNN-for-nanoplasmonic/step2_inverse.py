import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cuda:0" if (torch.cuda.is_available() > 0) else "cpu")
print(device)

seed = 2024
random.seed(seed)
torch.manual_seed(seed)

# 层数
batch_size = 512
lr = 0.0001

# 加载训练数据
thickness_data = pd.read_csv(f'Data/train/train_thickness.csv')
spectra_data = pd.read_csv(f'Data/train/train_spectra.csv')
thickness_tensor = torch.tensor(thickness_data.values, dtype=torch.float32)
spectra_tensor = torch.tensor(spectra_data.values, dtype=torch.float32)

val_thickness_data = pd.read_csv(f'Data/train/valid_thickness.csv')
val_spectra_data = pd.read_csv(f'Data/train/valid_spectra.csv')
val_thickness_tensor = torch.tensor(val_thickness_data.values, dtype=torch.float32)
val_spectra_tensor = torch.tensor(val_spectra_data.values, dtype=torch.float32)

# 创建数据集
train_dataset = TensorDataset(thickness_tensor, spectra_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(val_thickness_tensor, val_spectra_tensor)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 模型保存位置
model_save_dir = "Model"
os.makedirs(model_save_dir, exist_ok=True)


# 预测网络
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
        output = self.model(x)
        return output


# 设计网络
class Inverse(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(400, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 4)
        )

    def forward(self, x):
        output = self.model(x)
        return output


# 加载预训练的前向网络参数
P_net = Predict().to(device)
P_net.load_state_dict(torch.load(f'Model/predict_best.pth'))
P_net.eval()

# 冻结预测网络参数
for param in P_net.parameters():
    param.requires_grad = False

# 初始化逆向设计网络
I_net = Inverse().to(device)

# 设置优化器和损失函数
# 损失函数和优化器
loss_sml1 = nn.SmoothL1Loss()
loss_ce = nn.CrossEntropyLoss()

# 初始化学习率调度器
optimizer = optim.NAdam(I_net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=15, verbose=True)


# 定义厚度边界损失
class BoundLoss(nn.Module):
    def __init__(self):
        super(BoundLoss, self).__init__()

    def forward(self, outputs):
        # 定义每个输出的边界
        bounds = [
            (1, 3),  # 距离的边界
            (1, 5),  # Ag的边界
            (1, 30),  # 介质的边界
        ]
        total_loss = 0.0
        # 逐一计算每个输出的边界损失
        for i in range(outputs.size(1)):
            lower_bound, upper_bound = bounds[i]
            mean = (upper_bound + lower_bound) / 2
            range_ = (upper_bound - lower_bound)
            # 计算边界损失
            boundary_loss = torch.relu(torch.abs(outputs[:, i] - mean) - range_ / 2)
            total_loss += boundary_loss.mean()
        # 返回批次中每个样本的平均边界损失
        return total_loss



# 定义类别损失
class ClassLoss(nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()

    def forward(self, predictions, targets=None):
        # 计算每个预测值与0、1、2的距离的最小值
        total_loss = torch.min(torch.abs(predictions - 0), torch.abs(predictions - 1))
        total_loss = torch.min(total_loss, torch.abs(predictions - 2))
        # 对每个样本的最小距离取平均
        loss = torch.mean(total_loss)
        return loss


loss_Bound = BoundLoss()
loss_Class = ClassLoss()


# MRE计算
def Mre(predictions, targets):
    relative_errors = torch.abs(predictions - targets) / torch.abs(targets)
    mre = torch.mean(relative_errors)
    return mre


start_time = time.time()
# 开始训练
train_loss = []
val_loss = []
val_mre = []
best_val_mre = 1.0
best_model_state = None
best_epoch = 0
epochs = 5000
validation_interval = 10
A = 0.4
B = 0.3
for epoch in range(epochs):
    I_net.train()
    total_loss = 0.0

    for batch in train_dataloader:
        thickness, spectra = batch
        thickness, spectra = thickness.to(device), spectra.to(device)

        optimizer.zero_grad()

        design_output = I_net(spectra)
        response_output = P_net(design_output)

        # 光谱损失
        loss_spectra = loss_sml1(spectra, response_output)
        # # 类别损失
        loss_class = loss_Class(design_output[:, 0])
        # 厚度损失
        loss_bound = loss_Bound(design_output[:, 1:])
        # 总损失
        loss = A * loss_spectra + (1 - A) * (B * loss_bound + (1 - B) * loss_class)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    train_loss.append(average_loss)
    print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {average_loss}')

    # 每10轮进行一次验证
    if (epoch+1) % validation_interval == 0:
        I_net.eval()
        val_loss_total = 0.0
        val_mre_total = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                thickness, spectra = val_batch
                thickness, spectra = thickness.to(device), spectra.to(device)
                design_output = I_net(spectra)
                response_output = P_net(design_output)

                # 光谱损失
                loss_spectra = loss_sml1(response_output, spectra)
                # 类别损失
                loss_class = loss_Class(design_output[:, 0])
                # 边界损失
                loss_bound = loss_Bound(design_output[:, 1:])

                # 总损失
                loss = A * loss_spectra + (1 - A) * (B * loss_bound + (1 - B) * loss_class)

                val_loss_total += loss.item()
                # 计算平均相对误差 (MRE)
                val_mre_total += Mre(response_output, spectra)

        average_val_loss = val_loss_total / len(val_dataloader)
        average_val_mre = val_mre_total / len(val_dataloader)
        val_loss.append(average_val_loss)
        val_mre.append(average_val_mre)

        print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {average_val_loss}')
        print(f'Validation MRE: {average_val_mre}')

        # 如果当前模型的准确性优于之前保存的最佳模型，则保存当前模型参数
        if average_val_mre < best_val_mre:
            best_model_state = I_net.state_dict()
            best_val_mre = average_val_mre  # 同步更新 best_val_mre
            torch.save(best_model_state, os.path.join(model_save_dir, f"design_best.pth"))
            best_epoch = epoch + 1

    # 更新学习率调度器
    scheduler.step(average_loss)

    # 保存最终模型
print(f"Best model saved at epoch {best_epoch} with validation mre {best_val_mre} as 'design_best.pth'")
end_time = time.time()
execution_time = end_time - start_time
hours, remainder = divmod(execution_time, 3600)
minutes, seconds = divmod(remainder, 60)
print("time-consuming: {:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds)))


plt.figure(figsize=(3.54, 3.54))
# plt.subplot(2, 1, 1)
plt.subplot(1, 1, 1)
# 绘制训练损失曲线
plt.plot(range(1, epochs + 1), train_loss, label='Train Loss', linestyle='-')
# 绘制验证损失曲线，x轴是每个验证间隔的epoch
plt.plot(range(validation_interval, epochs + 1, validation_interval), val_loss, label='Validation Loss', linestyle='-')
plt.xlabel('Epoch', fontsize=11, fontname='Arial')
plt.ylabel('Loss', fontsize=11, fontname='Arial')
# 添加图例，显示曲线的标签
plt.legend()
# 设置y轴范围和刻度
plt.ylim(0, 1000)
plt.yticks([0, 500, 1000], fontsize=11, fontname='Arial')
plt.xlim(0, 5000)
plt.xticks([0, 2500, 5000], fontsize=11, fontname='Arial')
# 启用网格线
# plt.grid(True)
plt.tick_params(axis='both', direction='in', labelsize=11)
# plt.title('Training and Validation Loss')

# 绘制 R² 曲线
# plt.subplot(2, 1, 2)
#
# # 将验证准确率从张量转换为numpy数组
# plt_accuracy = [accuracy.cpu().numpy() for accuracy in val_accuracy]
# plt.plot(range(validation_interval, epochs + 1, validation_interval), plt_accuracy, label='Validation Accuracy',
#          linestyle='-')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.title('Validation Accuracy')

plt.tight_layout()
plt.savefig(f'Pic/design_loss.png')
plt.show()
