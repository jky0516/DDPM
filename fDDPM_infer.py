import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import seaborn as sns
from sklearn.model_selection import train_test_split
from DDPM_unet import ImprovedSelfUNet
import matplotlib.gridspec as gridspec

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")

steps = 1000  # 常见设置
beta_start = 1e-4
beta_end = 0.02

betas = torch.linspace(beta_start, beta_end, steps, device=device)  # (steps,)
alphas = 1.0 - betas  # (steps,)
alpha_bars = torch.cumprod(alphas, dim=0)  # (steps,)

# 加载numpy数据文件
X = np.load('X_scaled_zc.npy')
y = np.load('target_scaled_zc.npy')  

# 划分训练集和验证集 (80%训练, 20%验证)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_val_tensor = torch.FloatTensor(X_val)      # (N, 4)
    
# ---- 输出 y reshape 成图像 ----
y_val_tensor = torch.FloatTensor(y_val).view(-1, 1, 35 ,31)

# 创建DataLoader用于批量加载
batch_size = 128
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 模型定义

net = ImprovedSelfUNet().to(device)
net.load_state_dict(torch.load('self_cUnet.pth', map_location=torch.device('cpu')))

net.eval()

# --------------sampling------------------

num = 3000
inputs = X_val_tensor[:num].to(device)  # 选一些条件向量
recovered_pdf = torch.randn_like(y_val_tensor[:num].to(device))
with torch.no_grad():
    for t_step in tqdm(reversed(range(steps))):
        
        t = torch.full((num,), t_step, device=device, dtype=torch.long).view(-1, 1)
        beta_t = betas[t].view(-1, 1, 1, 1)  # β_t
        alpha_t = alphas[t].view(-1, 1, 1, 1)  # α_t
        alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)  # ᾱ_t
        
        coef1 = 1 / alpha_t.sqrt()
        coef2 = (1 - alpha_t) / (1 - alpha_bar_t).sqrt()
        
        pred_noise = net(recovered_pdf, t, inputs)
        
        mean = coef1 * (recovered_pdf - coef2 * pred_noise)

        if t_step > 0:
            noise = torch.randn_like(recovered_pdf)
            sigma_t = beta_t.sqrt()
            recovered_pdf = mean + sigma_t * noise
        else:
            recovered_pdf = mean  # 最后一轮不加噪声

gen_flat = recovered_pdf.detach().cpu().numpy().reshape(num, -1)
gt_flat = y_val_tensor[:num].detach().cpu().numpy().reshape(num, -1)
pairs_array = np.vstack([gt_flat.flatten(), gen_flat.flatten()]).T

# 绘制散点图
plt.figure(figsize=(8, 8))
plt.scatter(pairs_array[:, 0], pairs_array[:, 1], alpha=0.3, s=10)
plt.plot([pairs_array.min(), pairs_array.max()], [pairs_array.min(), pairs_array.max()], 'r--', label='Ideal Match')
plt.xlabel("True Value")
plt.ylabel("Predicted Value")
plt.title("Scatter Plot: Predicted vs True (All Dimensions)")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig('predicted_vs_true_scatter.png')

# ------------------ 展示前5个样本 ------------------------
num_show = 5
for i in range(num_show):
    # 提取数据，转为numpy
    gt_img = y_val_tensor[0:num][i, 0].detach().cpu().numpy()
    pred_img = recovered_pdf[i, 0].detach().cpu().numpy()

    vmin = min(gt_img.min(), pred_img.min())
    vmax = max(gt_img.max(), pred_img.max())

    # 创建子图
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])  # 两张图 + 一个colorbar

    ax0 = plt.subplot(gs[0])
    sns.heatmap(gt_img, cmap='viridis', vmin=vmin, vmax=vmax, cbar=False, ax=ax0)
    ax0.set_title(f"Ground Truth #{i}")
    ax0.set_xlabel("X")
    ax0.set_ylabel("Y")

    ax1 = plt.subplot(gs[1])
    sns.heatmap(pred_img, cmap='viridis', vmin=vmin, vmax=vmax, cbar=False, ax=ax1)
    ax1.set_title(f"Predicted #{i}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # 公共colorbar
    cax = plt.subplot(gs[2])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax)

    plt.tight_layout()
    plt.savefig(f'compare_true_vs_pred_heatmap_{i}.png')
    plt.show()
# -----------------------------------------