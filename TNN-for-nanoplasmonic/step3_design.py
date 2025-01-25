import torch
import torch.nn as nn
import pandas as pd

device = torch.device("cuda:0" if (torch.cuda.is_available() > 0) else "cpu")

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
            nn.Linear(1000, 400),
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
            nn.Linear(1000, 4),
        )

    def forward(self, x):
        output = self.model(x)
        return output


P_net = Predict().to(device)
P_net.load_state_dict(torch.load(f"Model/predict_best.pth", map_location=device))
P_net.eval()
for param in P_net.parameters():
    param.requires_grad = False


I_net = Inverse().to(device)
I_net.load_state_dict(torch.load(f"Model/design_best.pth", map_location=device))
I_net.eval()
for param in I_net.parameters():
    param.requires_grad = False

# 加载理想光谱数据
# Au2S  SiO2  Si
design_data = pd.read_csv(f'Data/design/design_1.csv')
design_tensor = torch.tensor(design_data.values, dtype=torch.float32).to(device)

# 获取设计和响应光谱
with torch.no_grad():
    design_out = I_net(design_tensor)
    print(design_out)
    design_out[:, 0] = torch.round(design_out[:, 0])
    design_out[:, 1:] = torch.round(design_out[:, 1:] * 100) / 100
    response_out = P_net(design_out)
    # 将负值置为0
    response_out = torch.clamp(response_out, min=0)

# 将结果保存到result.csv
print("Design:")
print(design_out)

# 计算平均相对误差 (Mean Relative Error)
relative_error = torch.abs((response_out - design_tensor) / design_tensor)
mean_relative_error = torch.mean(relative_error).item() * 100
print(f"Mean Relative Error (MRE): {mean_relative_error:.2f}%")

# 将response保存到CSV文件
design_df = pd.DataFrame(design_out.cpu().numpy())
design_df.to_csv(f'Data/design/structure_out_1.csv', index=False, header=False)
# 将response保存到CSV文件
response_df = pd.DataFrame(response_out.cpu().numpy())
response_df.to_csv(f'Data/design/response_out_1.csv', index=False, header=False)


