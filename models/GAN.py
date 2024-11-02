#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 01:28:09 2021
@author: qiwang
"""
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch
from torchvision.models import vgg19
import math
import sys
sys.path.append("../")

from models import Res 

def count_params(model:nn.Module):
    if isinstance(model, torch.nn.Module):
        count = sum(para.data.nelement() for para in model.parameters())
        count /= 1024**2
        print(f"Num of params: {count=:.2f} M")

class FeatureExtractor(nn.Module):
    def __init__(self, FE_type:str=None):
        super(FeatureExtractor, self).__init__()
        if not FE_type:
            vgg19_model = vgg19(pretrained=False)
            self.model = nn.Sequential(*list(vgg19_model.features.children())[:37])
        else:
            self.model = Res()
            self.model.load_model='results/pretrained_resnet50_veronika/v1_epoch=372-step=5287275.ckpt'

    def forward(self, img):
        return self.model(img)

class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x
    
class Generator(nn.Module): # interpolation scheme happens
    # A Generator inheritating SN for saving singular value of conv2d
    def __init__(self, channels=1, filters=64, num_res_blocks=32, num_upsample=0):
        super(Generator, self).__init__()
        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3,stride=1,padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Upsample(scale_factor=2,mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.LeakyReLU(),
                nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0)
            ]
            
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )
        
        #self.apply(self.init_weights)
        count_params(self)
    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_shape:tuple = (1,64,64)):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        self.features = [64, 128, 256, 512]
        patch_h, patch_w = int(in_height / 2 ** len(self.features)), int(in_width / 2 ** len(self.features))
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))#NOTE: x0.5 FOV
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

        count_params(self)

    def forward(self, img):
        return self.model(img)


class Discriminator_Unet(nn.Module):
    def __init__(self, input_shape:tuple=(3,64,64), num_feature=64,skip_connection=True):
        super(Discriminator_Unet, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        self.in_channel = input_shape[0]
        self.input_shape = input_shape
        self.output_shape = (1, int(input_shape[1]/2**2), int(input_shape[2]/2**2))
        layers_down = []
        self.conv_in = nn.Conv2d(self.in_channel, num_feature, 3, 1, 1)
        
        # down module
        [layers_down.extend([norm(nn.Conv2d(num_feature*2**i,
            num_feature*2*2**i, 4, 2, 1, bias=False)),nn.LeakyReLU(0.2, inplace=True)]) for i in range(3)]
        self.down = nn.Sequential(*layers_down)

        layers_up = []
        # up module
        [layers_up.extend([norm(nn.Conv2d(num_feature*2*2**i,
            num_feature*2**i, 3, 1, 1, bias=False)),nn.LeakyReLU(0.2, inplace=True)]) for i in range(2,-1,-1)]
        self.up = nn.Sequential(*layers_up)

        self.conv_out = nn.Conv2d(num_feature, 1, 3, 1, 1)
    def forward(self,x):
        x = self.conv_in(x)
        x_low = self.down(x)
        x_high = F.interpolate(x_low, scale_factor=2, mode='bilinear', align_corners = False)
        x_high = self.up(x_high)
        return self.conv_out(x_high)

