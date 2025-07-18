import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 时间步 Sinusoidal Embedding + MLP
# 时间步嵌入
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim*  4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)            
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -(np.log(10000) / (half_dim - 1)))
        emb = t * emb  # t: [B,1] emb: [half_dim] ➔ broadcast
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)  # [B, dim]

# FiLM 模块
class FiLM(nn.Module):
    def __init__(self, emb_dim, feature_channels, num_groups=8):
        super().__init__()
        self.scale = nn.Linear(emb_dim, feature_channels)
        self.shift = nn.Linear(emb_dim, feature_channels)

    def forward(self, x, emb):
        scale = self.scale(emb).unsqueeze(-1).unsqueeze(-1)
        shift = self.shift(emb).unsqueeze(-1).unsqueeze(-1)
        x = x * (1 + scale) + shift
        return x

# 主网络
class DDPM_UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, time_emb_dim=64, latent_dim=512):
        super().__init__()

        self.time_mlp = SinusoidalTimeEmbedding(time_emb_dim)

        # Down Blocks
        self.down1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)   # [B, 32, 35, 31]
        self.norm1 = nn.GroupNorm(8, base_channels)
        self.film1 = FiLM(time_emb_dim, base_channels)

        self.down2 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)   # [B, 64, 18, 16]
        self.norm2 = nn.GroupNorm(8, base_channels * 2)
        self.film2 = FiLM(time_emb_dim, base_channels * 2)

        self.down3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)   # [B, 128, 9, 8]
        self.norm3 = nn.GroupNorm(8, base_channels * 4)
        self.film3 = FiLM(time_emb_dim, base_channels * 4)

        self.down4 = nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)   # [B, 256, 4, 4]
        self.norm4 = nn.GroupNorm(8, base_channels * 8)
        self.film4 = FiLM(time_emb_dim, base_channels * 8)        

        self.middle = nn.Conv2d(base_channels * 8, latent_dim, 3, padding=1)   # [B, 512, 4, 4]

        # Up Blocks
        self.up1 = nn.Conv2d(latent_dim, base_channels * 8, 3, padding=1)   # [B, 256, 4, 4]
        self.norm5 = nn.GroupNorm(8, base_channels * 8)
        self.film5 = FiLM(time_emb_dim, base_channels * 8) 

        # up4: 固定输出尺寸为 (9, 8)   
        self.up2 = nn.Sequential(                                           
            nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1),     # [B, 128, 9, 8]
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.Upsample(size=(9, 8), mode='bilinear', align_corners=False)
        )
        self.norm6 = nn.GroupNorm(8, base_channels * 4)
        self.film6 = FiLM(time_emb_dim, base_channels * 4)        
        
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, stride=2, padding=1, output_padding=1)   # [B, 64, 18, 16]
        self.norm7 = nn.GroupNorm(8, base_channels * 2)
        self.film7 = FiLM(time_emb_dim, base_channels * 2)

        # up4: 固定输出尺寸为 (35,31)
        self.up4 = nn.Sequential(                                           # [B, 32, 35, 31]
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Upsample(size=(35, 31), mode='bilinear', align_corners=False)
        )
        self.film8 = FiLM(time_emb_dim, base_channels)

        self.final = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x, t):
        B, C, H, W = x.shape
        t_emb = self.time_mlp(t)

        # Down
        d1 = F.silu(self.film1(self.norm1(self.down1(x)), t_emb))        # [B, 32, 35, 31]
        d2 = F.silu(self.film2(self.norm2(self.down2(d1)), t_emb))       # [B, 64, 18, 16]
        d3 = F.silu(self.film3(self.norm3(self.down3(d2)), t_emb))       # [B, 128, 9, 8]
        d4 = F.silu(self.film4(self.norm4(self.down4(d3)), t_emb))       # [B, 256, 4, 4]

        mid = F.silu(self.middle(d4))                                          # [B, latent_dim, 4, 4]

        # Up 1：对齐 d4
        u1_ = self.film5(self.norm5(self.up1(mid)), t_emb)                # [B, 256, 4, 4]
        u1 = F.silu(u1_ + d4)

        # Up 2：上采样后对齐 d3
        u2_ = self.film6(self.up2(u1), t_emb)                          # [B, 128, 9, 8]                                         
        u2 = F.silu(u2_ + d3)

        # Up 3：上采样后对齐 d2
        u3_ = self.film7(self.norm7(self.up3(u2)), t_emb)                 # [B, 64, 18, 16]                                         
        u3 = F.silu(u3_ + d2)

        # Up 4：上采样后对齐 d1
        u4_ = self.film8(self.up4(u3), t_emb)                 # [B, 32, 35, 31]                                                
        u4 = F.silu(u4_ + d1)        

        out = self.final(u4)                                                   # [B,1,35,31]
        
        return out
    