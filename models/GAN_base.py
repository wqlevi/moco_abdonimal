"""
TODO:
    - [x] noise design
    - [x] update sigma for noise design
    - [x] test forward back ward
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from models import Generator, Discriminator, FeatureExtractor
from utils import load_ckpt, load_ckpt_and_update_dict

class GAN_class:
    def __init__(self, rank, opt):
        super().__init__()
        arch_config = {'num_res_blocks': 16,
                       'FE_type': 'resnet50'}
        #opt = vars(opt)
        self.Gnet = Generator(num_res_blocks=arch_config['num_res_blocks']).to(rank)
        self.Dnet = Discriminator(input_shape=(1,opt['image_size'],opt['image_size'])).to(rank)
        if opt['FE']: self.FE = FeatureExtractor(arch_config['FE_type']).to(rank)
        self.use_FE = True if opt['FE'] else False

        if not opt['ckpt_epoch'] == 0:
            ckpt = load_ckpt(Path(opt['ckpt_dir'], opt['ckpt_name']))
            self.Gnet = load_ckpt_and_update_dict(self.Gnet, ckpt, key='Gnet')
            self.Dnet = load_ckpt_and_update_dict(self.Dnet, ckpt, key='Dnet')


        if opt['distributed']:
            self.Gnet = DDP(self.Gnet, device_ids=[rank])
            self.Dnet = DDP(self.Dnet, device_ids=[rank])
            self.FE = DDP(self.FE, device_ids=[rank])
            self.d_input_shape = (opt['microbatch'], *self.Dnet.module.input_shape)
            self.d_output_shape = (opt['microbatch'], *self.Dnet.module.output_shape)
            process_group = torch.distributed.new_group()
            self.Gnet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.Gnet, process_group)
            self.Dnet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.Dnet, process_group)
            self.FE = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.FE, process_group)
        else:
            self.d_input_shape = (opt['microbatch'], *self.Dnet.input_shape)
            self.d_output_shape = (opt['microbatch'], *self.Dnet.output_shape)

        self.rank = rank

        self.instance_noise = None 

        self.valid = torch.ones(self.d_output_shape, dtype= torch.float32, device=rank)
        self.fake = torch.zeros(self.d_output_shape, dtype= torch.float32, device=rank)

        self.GAN_loss_g = nn.BCEWithLogitsLoss()
        self.GAN_loss_d = nn.BCEWithLogitsLoss()

    @property
    def update_noise(self):
        return self.instance_noise

    @update_noise.setter
    def update_noise(self, sigma:float):
        # need updating sigma input in external train_step
        self.instance_noise = torch.normal(mean = torch.zeros(self.d_input_shape, dtype=torch.float32, device=self.rank),
                            std = torch.full(self.d_input_shape,
                                              sigma, dtype=torch.float32, device=self.rank)
                            ).detach()

    def G_loss(self, img, gt):
        pred = self.Gnet(img)
        with torch.no_grad():
            pred_fake = self.Dnet(pred + self.instance_noise)
            pred_real = self.Dnet(gt + self.instance_noise).detach()
        if not self.use_FE:
            loss = F.l1_loss(pred, gt) + 5e-3*self.GAN_loss_g(pred_fake - pred_real.mean(0, keepdim=True), self.valid)
        else:
            loss = F.l1_loss(pred, gt) + 1e-2*F.l1_loss(self.FE(pred), self.FE(gt)) + 5e-3*self.GAN_loss_g(pred_fake - pred_real.mean(0, keepdim=True), self.valid) #FIXME: 'boo obj hsa no attr 'requires_grad'
        return loss, pred

    def D_loss(self, img, gt):
        with torch.no_grad():
            pred = self.Gnet(img)
        pred_real = self.Dnet(gt + self.instance_noise)
        pred_fake = self.Dnet(pred.detach()+ self.instance_noise)
        loss_real = self.GAN_loss_d(pred_real - pred_fake.mean(0, keepdim=True), self.valid)
        loss_fake = self.GAN_loss_d(pred_fake - pred_real.mean(0, keepdim=True), self.fake)
        return (loss_real + loss_fake) /2 


