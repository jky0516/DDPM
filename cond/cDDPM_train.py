import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from DDPM_unet import cDDPM_UNet
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

X = np.load("X_scaled_zc.npy")             # shape: (N, 4)
y = np.load("target_scaled_zc.npy")        # shape: (N, 35*31)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1, 1, 35, 31)
y_train = torch.log(y_train + 1e-6)

X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val).view(-1, 1, 35, 31)
y_val = torch.log(y_val + 1e-6)

dataset_train = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(dataset_train, batch_size=128, shuffle=True)

dataset_val = TensorDataset(X_val, y_val)
val_dataloader = DataLoader(dataset_val, batch_size=128, shuffle=True)

##############################
#           Traing           #
##############################

DDPM_epochs = 100

net = cDDPM_UNet().to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

loss_fn = nn.MSELoss()

train_loss_history = []
val_loss_history = []


for epoch in range(DDPM_epochs):
    net.train()
    train_loss = 0
    for cond_input, img in tqdm(train_dataloader):

        cond_input = cond_input.to(device)      
        img = img.to(device)

        t = torch.randint(0, steps, (cond_input.size(0),), device=device, dtype=torch.long).view(-1, 1)
        alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
        
        noise = torch.randn_like(img).to(device)
        noisy_img = alpha_bar_t.sqrt() * img + (1 - alpha_bar_t).sqrt() * noise

        pred = net(noisy_img, t, cond_input)
        loss = loss_fn(pred, noise)

        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item()
        

    net.eval()
    val_loss = 0
    with torch.no_grad():
        for cond_input, img in tqdm(val_dataloader):

            cond_input = cond_input.to(device)          
            img = img.to(device)
            
            t = torch.randint(0, steps, (cond_input.size(0),), device=device, dtype=torch.long).view(-1, 1)
            alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
            
            noise = torch.randn_like(img).to(device)
            noisy_img = alpha_bar_t.sqrt() * img + (1 - alpha_bar_t).sqrt() * noise

            pred = net(noisy_img, t, cond_input)
            loss = loss_fn(pred, noise)

            val_loss += loss.item()  
              
    train_loss /= len(train_dataloader)
    val_loss /= len(val_dataloader)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)    
    
    print(f"[Epoch {epoch}] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")  
    
    torch.save(net.state_dict(), 'DDPM_Unet.pth')
    
plt.figure(figsize=(6, 5))
plt.rcParams['axes.unicode_minus'] = False
plt.plot(train_loss_history, label='train_loss')
plt.plot(val_loss_history, label='vali_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Test Loss Curve')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()            