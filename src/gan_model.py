# src/gan_model.py
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    def __init__(self, num_classes, latent_dim=100):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = 128 // 4 
        
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh() # Output: [-1, 1]
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        
        # ==========================================================
        # ✅ [核心修改] 谱归一化架构 (Report Section 2.2)
        # ❌ [删除] 所有 BatchNorm2d (会导致模式崩塌)
        # ==========================================================
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(2, 64, 3, 2, 1)), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            spectral_norm(nn.Conv2d(64, 128, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            spectral_norm(nn.Conv2d(128, 256, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            spectral_norm(nn.Conv2d(256, 512, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        
        ds_size = 128 // 2 ** 4
        self.adv_layer = nn.Sequential(
            spectral_norm(nn.Linear(512 * ds_size ** 2, 1)),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # 空间标签拼接
        label_map = labels.view(-1, 1, 1, 1).float()
        label_map = label_map.expand(img.size(0), 1, img.size(2), img.size(3))
        label_map = label_map / 10.0 
        d_in = torch.cat((img, label_map), 1)
        
        out = self.model(d_in)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity