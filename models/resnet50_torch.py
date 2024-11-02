"""
changed to in_channel=1
- [x] add property to self.m(nn.Module) to actively update weight loading
- [x] use load functions in the parent folder
"""
from typing import List
from torchvision.models import resnet50
import torch.nn as nn
from functools import partial
import torch
import torch.nn.functional as F

from utils import load_ckpt, update_dict_keys, init_weights# utils is visible from the main.py scope

"""
class Res50(nn.Module):
    # USAGE: m = Res50.load_state_dict('./results/pretrained_resnet50/v1_epoch=372-step=5287275.ckpt')
    def __init__(self):
        super().__init__()
        self.m = resnet50(weights=False)
        self.ckpt=None

        if hasattr(self.m, 'conv1'): self.m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self,x):
        return self.m(x)

    @classmethod
    def load_state_dict(cls, ckpt_path:str=''):
        ckpt = load_ckpt(ckpt_path) 
        model = cls()
        new_dict = update_dict_keys(ckpt, model.m.state_dict(), prefix_model= '') # TODO: instance m not found
        model.m.load_state_dict(new_dict)
        return model
"""
class Upsample(nn.Module):
    def __init__(self, ratio:int=2):
        super().__init__()
        self.interp_fn = partial(nn.functional.interpolate, scale_factor=ratio, mode='bilinear', align_corners=False)
    def forward(self, x):
        return self.interp_fn(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_channels = 1_000 # from ResNet50 latents
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)
        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
                layers.append(self.up)
            layers.append(nn.LeakyReLU(0.2, inplace=True)) 
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1))
            layers.append(self.up)
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = self.input_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1)) # 4 Conv layers in block

        self.model = nn.Sequential(*layers)

    def hook_fn(self, m, i, o):
        block_res = f"\n\033[96mlayer_res_{str(i[0].shape[-1])}_\033[0m"
        print(block_res)
        print(f"\033[93minput tensor shape: {i[0].shape}\033[0m")
        print(f"\033[0mID of layer: {id(m)}\033[0m")
        print(f"\033[91moutput tensor shape: {o.shape}\033[0m")

    @staticmethod
    def add_hooks(m, fn):
        for _,v in m.named_modules():
            if isinstance(v, nn.Upsample): v.register_forward_hook(fn)
        return m

    def forward(self, x):
        #self.model = self.add_hooks(self.model, self.hook_fn)
        return self.model(x)

class Res(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet50(weights=False)
        if hasattr(self.encoder, 'conv1'): self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.decoder = Decoder()

        #[m.apply(init_weights) for m in [self.encoder, self.decoder]]
        self.encoder.apply(init_weights)

    @property
    def load_model(self):
        return self.encoder
    @load_model.setter
    def load_model(self, ckpt_path:str=None):
        if not ckpt_path: raise ValueError("ckpt path is empty.")
        ckpt = load_ckpt(ckpt_path)
        new_dict = update_dict_keys(ckpt['state_dict'], self.state_dict(), prefix_local='', prefix_model= '') #FIXME: unmatched keys
        self.load_state_dict(new_dict)
        print('\033[93mEncoder loaded from  ckpt: {}\033[0m'.format(ckpt_path))

    @staticmethod
    def loss_fn(tensors:List[torch.Tensor]):
        pred, inputs = tensors
        return {'loss':F.mse_loss(pred, inputs)}

    def forward(self,x):
        z = self.encoder(x)
        #z = z[...,None,None] # 2D latent to 4D tensor [B, C, H, W]
        #o = self.decoder(z)
        return z

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = 1_000

        self.encoder = resnet50(weights=False) # output: [B, 1_000]
        if hasattr(self.encoder, 'conv1'): self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.decoder = Decoder() 

        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim, self.latent_dim)

        self.decoder_input_layer = nn.Linear(self.latent_dim, self.latent_dim)

        #[m.apply(init_weights) for m in [self.encoder, self.decoder, self.fc_mu, self.fc_var, self.decoder_input_layer]]

    def reparameterization(self, mu:torch.Tensor, logvar:torch.Tensor):
        # input, output: [B, latent_dim]
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode_fn(self, x:torch.Tensor):
        z = self.encoder(x)
        mu = self.fc_mu(z)
        logvar = self.fc_var(z)
        return [mu, logvar]

    def decode_fn(self, z:torch.Tensor):
        # input: [B, 1_000, 1, 1], output: [B, 1, 128, 128]
        rst = self.decoder_input_layer(z)[...,None,None]
        rst = self.decoder(rst)
        return rst

    @property
    def load_model(self):
        return self.encoder
    @load_model.setter
    def load_model(self, ckpt_path:str=None):
        if not ckpt_path: raise ValueError("ckpt path is empty.")
        ckpt = load_ckpt(ckpt_path)
        new_dict = update_dict_keys(ckpt, self.encoder.state_dict(), prefix_model= '')
        self.encoder.load_state_dict(new_dict)
        print('\033[93mEncoder loaded from  ckpt: {}\033[0m'.format(ckpt_path))

    @staticmethod
    def loss_fn(tensors:List[torch.Tensor], **kwargs):
        pred, inputs, mu, logvar = tensors

        kld_weight = kwargs['kl_weight'] if 'kl_weight' in kwargs else 0.00025
        recon_loss = F.mse_loss(pred, inputs)
        kld_loss = torch.mean(-.5 * torch.sum(1 + logvar - mu **2 - logvar.exp(), dim=1), dim=0)
        loss = recon_loss + kld_weight*kld_loss
        return {'loss': loss, 'recon_loss': recon_loss, 'kld':kld_loss}

    def forward(self,x):
        mu, logvar = self.encode_fn(x)
        z = self.reparameterization(mu ,logvar)
        o = self.decode_fn(z)
        return [o, x, mu, logvar]
