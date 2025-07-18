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
from DDPM_unet import cDDPM_UNet
import matplotlib.gridspec as gridspec
import time
import math

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda")

##############################
#       Beta schedule        #
##############################

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas).to(device)

steps = 1000
"Linear"
betas = torch.linspace(1e-4, 0.02, steps, device=device)
"Cosine"
# betas = betas_for_alpha_bar(steps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)

alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

##############################
#       Read the Data       #
##############################

X = np.load('X_scaled_zc.npy')
y = np.load('target_scaled_zc.npy')  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val).view(-1, 1, 35 ,31)

batch_size = 128
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

##############################
#    Read the DDPM model    #
##############################

net = cDDPM_UNet().to(device)
net.load_state_dict(torch.load('DDPM_Unet.pth', map_location=torch.device('cuda')))

net.eval()

##############################
#          Sampling          #
##############################

def ddpm_sampler(model, cond_input, steps=1000, device='cuda'):

    num = cond_input.size(0)
    recovered_pdf = torch.randn(num, 1, 35, 31, device=device)

    with torch.no_grad():

        for t_step in tqdm(reversed(range(steps))):
            
            t = torch.full((num,), t_step, device=device, dtype=torch.long).view(-1, 1)
            beta_t = betas[t].view(-1, 1, 1, 1)  # β_t
            alpha_t = alphas[t].view(-1, 1, 1, 1)  # α_t
            alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)  # ᾱ_t
            
            coef1 = 1 / alpha_t.sqrt()
            coef2 = (1 - alpha_t) / (1 - alpha_bar_t).sqrt()
            
            pred_noise = model(recovered_pdf, t, inputs)
            
            mean = coef1 * (recovered_pdf - coef2 * pred_noise)

            if t_step > 0:
                noise = torch.randn_like(recovered_pdf)
                sigma_t = beta_t.sqrt()
                recovered_pdf = mean + sigma_t * noise
            else:
                recovered_pdf = mean  # 最后一轮不加噪声

    return recovered_pdf

def ddim_sampler(model, cond_input, ddim_steps=10, ddpm_steps=1000, device='cuda', eta=0.0):
    """
        eta: 控制随机性的参数 (η=0 为确定性采样)
    """
    num = cond_input.size(0)
    recovered_pdf = torch.randn(num, 1, 35, 31, device=device)
    
    # 创建时间步序列 (DDIM可以使用更少的步数)
    timesteps = np.linspace(0, ddpm_steps-1, num=ddim_steps, dtype=int)
    timesteps = list(reversed(timesteps))
    
    with torch.no_grad():
        for i, t_step in enumerate(tqdm(timesteps)):
            t = torch.full((num,), t_step, device=device, dtype=torch.long)            
            # 获取当前和上一个时间步的alpha_bar
            alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
            if i < len(timesteps) - 1:
                alpha_bar_t_prev = alpha_bars[timesteps[i+1]].view(-1, 1, 1, 1)
            else:
                alpha_bar_t_prev = torch.ones_like(alpha_bar_t)            
            # 预测噪声
            t = t.view(-1, 1)
            pred_noise = model(recovered_pdf, t, cond_input)            
            # 计算预测的x0
            pred_x0 = (recovered_pdf - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()            
            # DDIM更新公式
            sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)            
            # 方向项
            dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * pred_noise            
            # 均值预测
            mean_pred = alpha_bar_t_prev.sqrt() * pred_x0 + dir_xt            
            if t_step > 0:
                noise = torch.randn_like(recovered_pdf)
                recovered_pdf = mean_pred + sigma_t * noise
            else:
                recovered_pdf = mean_pred    
    return recovered_pdf

num = 5000
inputs = X_val_tensor[:num].to(device)  # 选一些条件向量

t1 = time.time()
recovered_pdf = ddim_sampler(net, inputs)
recovered_pdf = torch.exp(recovered_pdf) - 1e-6
recovered_pdf = recovered_pdf.clamp(min=0.0)
recovered_pdf = recovered_pdf/recovered_pdf.sum(dim=(1,2,3), keepdim=True)
t2 = time.time()

print(f"sampling time: {t2 - t1}s")

##############################
#          Plotting          #
##############################

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

# 展示前5个样本
num_show = 5
for i in range(num_show):
    # 提取数据，转为numpy
    gt_img = y_val_tensor[0:num][i, 0].detach().cpu().numpy()
    pred_img = recovered_pdf[i, 0].detach().cpu().numpy()

    vmin = min(gt_img.min(), pred_img.min())
    vmax = max(gt_img.max(), pred_img.max())

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

    cax = plt.subplot(gs[2])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax)

    plt.tight_layout()
    plt.savefig(f'compare_true_vs_pred_heatmap_{i}.png')
    plt.show()