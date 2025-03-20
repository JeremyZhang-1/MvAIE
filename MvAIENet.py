# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:54:23 2025

@author: Jeremy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MvAIENet(nn.Module):
    def __init__(self, in_channels=3, base_channels=24):
        super(MvAIENet, self).__init__()
        self.mns = MainNetworkStructure(in_channels, base_channels)
        self.brightness_module = BrightnessAdjustmentModule(in_channels)         
    def forward(self, x, xg, xh):
        Fout = self.mns(x, xg, xh)
        return self.brightness_module(Fout + x)

class MainNetworkStructure(nn.Module):
    def __init__(self, in_channels, channel):
        super(MainNetworkStructure, self).__init__()
        
        # Input processing
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, channel//3, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channel//3),
            nn.GELU()
        )
        
        # Gamma and HSV branch processing
        self.process_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, channel//3, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(channel//3),
                nn.GELU()
            ) for _ in range(5)
        ])
        
        # Main processing modules
        self.mgla = MGLAttentionModel(channel)
        self.encoder = Encoder(channel)
        self.decoder = Decoder(channel)
        
        # Output processing
        self.conv_out = nn.Sequential(
            nn.Conv2d(channel, in_channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        
        # Learnable weights
        self.alpha_params = nn.Parameter(torch.ones(2) * 0.5)
        
    def forward(self, x, xg, xh):
        alphas = torch.sigmoid(self.alpha_params)
        
        x_rgb = self.process_branches[0](x)
        xg_1 = self.process_branches[1](xg[:,:3,:,:])
        xg_2 = self.process_branches[2](xg[:,3:,:,:])
        xh_1 = self.process_branches[3](xh[:,:3,:,:])
        xh_2 = self.process_branches[4](xh[:,3:,:,:])
                
        xg_fused = alphas[0] * xg_1 + (1-alphas[0]) * xg_2
        xh_fused = alphas[1] * xh_1 + (1-alphas[1]) * xh_2
        
        fused_features = torch.cat([x_rgb, xg_fused, xh_fused], 1)       
        mgla_out = self.mgla(fused_features)
        e3, e2, e1 = self.encoder(mgla_out)
        
        decoded = self.decoder(e3, e2, e1)
        
        out = self.conv_out(decoded)
        
        return out

class GlobalLocalAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super(GlobalLocalAttention, self).__init__()
        
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
        
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 3, padding=1, groups=channels//reduction),
            nn.GELU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        global_attn = self.global_branch(x)
        local_attn = self.local_branch(x)
        return x * (global_attn + local_attn) + x

class MGLAttentionModel(nn.Module):
    def __init__(self, channel):
        super(MGLAttentionModel, self).__init__()
        
        self.attention = GlobalLocalAttention(channel)
        
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv2d(channel, channel, 3, padding=d, dilation=d, bias=False)
            for d in [1, 3, 5, 7]
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channel*4, channel, 1),
            nn.InstanceNorm2d(channel),
            nn.GELU()
        )
        
    def forward(self, x):
        features = [conv(x) for conv in self.multi_scale_convs]
        fused = self.fusion(torch.cat(features, dim=1))
        return self.attention(fused)


class Encoder(nn.Module):
    def __init__(self, channel):
        super(Encoder, self).__init__()
        
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel * (2**i), channel * (2**(i+1)), 3, 2, 1),
                nn.InstanceNorm2d(channel * (2**(i+1))),
                nn.GELU()
            ) for i in range(3)
        ])
        
    def forward(self, x):
        features = []
        current = x
        
        for stage in self.stages:
            current = stage(current)
            features.append(current)
        
        return features[::-1]  # Return in reverse order for decoder

class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(channel * (2**(i+1)), channel * (2**i), 4, 2, 1),
                nn.InstanceNorm2d(channel * (2**i)),
                nn.GELU()
            ) for i in range(2, -1, -1)
        ])
        
    def forward(self, e3, e2, e1):
        encoded_features = [e3, e2, e1]
        current = encoded_features[0]
        
        for i, stage in enumerate(self.stages):
            current = stage(current)
            if i < len(encoded_features) - 1:
                current = current + encoded_features[i + 1]
        
        return current

class BrightnessAdjustmentModule(nn.Module):
    def __init__(self, channels):
        super(BrightnessAdjustmentModule, self).__init__()
        
        # Global brightness estimation
        self.brightness_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 16, 1),  # Fixed intermediate channel size
            nn.GELU(),
            nn.Conv2d(16, channels, 1),
            nn.Sigmoid()
        )
        
        # Adaptive adjustment parameters
        self.adjustment_params = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x):
        # Estimate current brightness
        brightness_weight = self.brightness_estimator(x)
        
        # Calculate adaptive adjustment factor
        adjustment_factor = torch.sigmoid(self.adjustment_params)
        
        # Apply brightness adjustment
        adjusted = x * (1 + adjustment_factor * brightness_weight)
        
        # Ensure values stay in valid range
        return torch.clamp(adjusted, -1, 1)
