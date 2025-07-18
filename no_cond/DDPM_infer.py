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
from DDPM_unet import DDPM_UNet
import matplotlib.gridspec as gridspec
import time
import math
from torchvision.utils import make_grid

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
#        Read data           #
##############################

X = np.load('X_scaled_zc.npy')
y = np.load('target_scaled_zc.npy')  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_val_tensor = torch.FloatTensor(X_val)      # (N, 4)
y_val_tensor = torch.FloatTensor(y_val).view(-1, 1, 35 ,31)

batch_size = 128
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

##############################
#    Read the DDPM model    #
##############################

net = DDPM_UNet().to(device)
net.load_state_dict(torch.load('DDPM_Unet.pth', map_location=torch.device('cuda')))
net.eval()

##############################
#          Sampling          #
##############################

def ddpm_sampler(model, batch_size=16, steps=1000, device='cuda'):

    num = batch_size
    recovered_pdf = torch.randn(num, 1, 35, 31, device=device)

    with torch.no_grad():

        for t_step in tqdm(reversed(range(steps))):
            
            t = torch.full((num,), t_step, device=device, dtype=torch.long).view(-1, 1)
            beta_t = betas[t].view(-1, 1, 1, 1)  # β_t
            alpha_t = alphas[t].view(-1, 1, 1, 1)  # α_t
            alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)  # ᾱ_t
            
            coef1 = 1 / alpha_t.sqrt()
            coef2 = (1 - alpha_t) / (1 - alpha_bar_t).sqrt()
            
            pred_noise = model(recovered_pdf, t)
            
            mean = coef1 * (recovered_pdf - coef2 * pred_noise)

            if t_step > 0:
                noise = torch.randn_like(recovered_pdf)
                sigma_t = beta_t.sqrt()
                recovered_pdf = mean + sigma_t * noise
            else:
                recovered_pdf = mean  # 最后一轮不加噪声

    return recovered_pdf

def ddim_sampler(model, batch_size=16, ddim_steps=10, ddpm_steps=1000, device='cuda', eta=0.0):
    """
        eta: 控制随机性的参数 (η=0 为确定性采样)
    """
    num = batch_size
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
            pred_noise = model(recovered_pdf, t)            
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

num = 16

t1 = time.time()
recovered_pdf = ddim_sampler(net)
recovered_pdf = torch.exp(recovered_pdf) - 1e-6
recovered_pdf = recovered_pdf.clamp(min=0.0)
recovered_pdf = recovered_pdf/recovered_pdf.sum(dim=(1,2,3), keepdim=True)
t2 = time.time()

print(f"sampling time: {t2 - t1}s")

##############################
#          Plotting          #
##############################

sample_grid = make_grid(recovered_pdf, nrow=int(np.sqrt(num)))
grayscale = sample_grid.mean(dim=0)
plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(grayscale.cpu().squeeze(), cmap='viridis')
plt.colorbar()
plt.title("ddim sampler")
plt.savefig('ddim_unconditioned_samples.png')
plt.show()