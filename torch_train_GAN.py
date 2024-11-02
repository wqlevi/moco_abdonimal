"""
-[ ] Prediction still look un normalized, with extra bright contrast.  
"""
import torch.nn as nn
import torch
import os, argparse, sys
from pathlib import Path
import random
from functools import partial
from typing import Tuple, List

from torchvision.models import resnet50
from torch.utils.data import random_split
from torch.multiprocessing import spawn

from models import Res, VAE, GAN_class
from logger import Logger, build_log_writer, get_date_time
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloader import MyDset, MyLoader, Cast_jnp
from utils import ssim, psnr
from dist_utils import init_process, all_gather, clean_up, all_cat_cpu
from torchvision.utils import make_grid

from pdb import set_trace as pb

def create_training_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test_GAN")
    parser.add_argument("--log-dir", type=Path, default="./logs")
    parser.add_argument("--ckpt-dir", type=Path, default="./results/unet_torch")
    parser.add_argument("--ckpt-name", type=Path, default="test_GAN_epoch_0.ckpt")
    parser.add_argument("--ckpt-epoch", type=int, default=0)
    parser.add_argument("--log-writer", type=str, default=None)
    parser.add_argument("--datapath", type=str, default="/mnt/qdata/share/rawangq1/ukbdata_70k/abdominal_MRI/2d")

    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--pretrained", action='store_true')
    parser.add_argument("--FE", action='store_true')
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--noise-max-epochs", type=int, default=20)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    opt = parser.parse_args()

    opt.ckpt_dir = opt.ckpt_dir / opt.name 
    assert opt.batch_size % opt.microbatch == 0, f"{opt.batch_size=} is not dividable by {opt.microbatch}!"
    os.makedirs(opt.ckpt_dir, exist_ok=True)
    return opt

def seed_everything(seed:int=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def build_loader(rank, world_size, dataset, batch_size):
    data_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = MyLoader(dataset, batch_size=batch_size, shuffle=False, sampler=data_sampler if world_size>1 else None, drop_last=True, to_numpy=False)
    return dataloader

def train_step(rank, world_size, epoch, model, dataloader, optimizers:Tuple, writer, log, opt)->int:

    current_sigma = (opt.noise_max_epochs - epoch)/ opt.noise_max_epochs
    current_sigma = max(current_sigma, 0)
    model.update_noise = current_sigma # update sigma in instance noise
    log.info("[Noise]: sigma = {}".format(current_sigma))
    writer.add_scalar(epoch, 'noise', current_sigma)

    train_dataloader, val_dataloader= dataloader
    optimizer_g, optimizer_d = optimizers

    n_inner_iter = opt.batch_size // (world_size * opt.microbatch)
    loss_epoch_g= 0
    loss_epoch_d= 0

    for i, batch in enumerate(train_dataloader):
        #======G update======#
        model.Gnet.train()
        model.Dnet.eval()

        img, gt = batch['noisy'].to(rank), batch['gt'].to(rank)

        loss_inner_iter_g = 0
        optimizer_g.zero_grad()
        for _ in range(n_inner_iter):
            loss, pred = model.G_loss(img, gt)
            loss.backward()
            loss_inner_iter_g += loss.item()
        optimizer_g.step()
        loss_inner_iter_g /= n_inner_iter
        loss_epoch_g += loss_inner_iter_g
        writer.add_scalar(epoch*len(train_dataloader)+i, 'train/loss_g', loss_inner_iter_g)
        log.info("[ITER]: G: {} / {}".format(i, len(train_dataloader)))

        model.Gnet.eval()
        model.Dnet.train()

        #======D update======#
        loss_inner_iter_d = 0
        optimizer_d.zero_grad()
        for _ in range(n_inner_iter):
            loss = model.D_loss(img, gt)
            loss.backward()
            loss_inner_iter_d += loss.item()
        loss_inner_iter_d /= n_inner_iter
        loss_epoch_d += loss_inner_iter_d
        optimizer_d.step()
        writer.add_scalar(epoch*len(train_dataloader)+i, 'train/loss_d', loss_inner_iter_d)
        log.info("[ITER]: D: {} / {}".format(i, len(train_dataloader)))
        
        if (i+1) % 100 == 0:
            gt_total = all_cat_cpu(opt, log, batch['gt'].to(rank))
            img_total = all_cat_cpu(opt, log, pred)

            grid_gt = make_grid(gt_total, nrow=16)
            grid = make_grid(img_total, nrow=16)
            writer.add_image(epoch*len(train_dataloader)+i, 'train/gt', grid_gt)
            writer.add_image(epoch*len(train_dataloader)+i, 'train/pred', grid)

            val_step(rank, world_size, epoch*len(train_dataloader)+i, model, val_dataloader, writer, log, opt)

    log.info("[epoch]: {} | [G loss]: {} | [D loss]: {}".format(epoch, loss_epoch_g/i, loss_epoch_d/i))
    return epoch*len(train_dataloader)+i

@torch.no_grad()
def val_step(rank, world_size, step, model, dataloader, writer, log, opt):
    #metrics_fn = lambda dst, src: (psnr(dst, src), ssim(dst, src))
    metrics_fn = lambda dst, src: psnr(dst, src)
    model.Gnet.eval()
    model.Dnet.eval()
    batch = next(iter(dataloader))
    img, gt = batch['noisy'].to(rank), batch['gt'].to(rank)
    loss_v, pred = model.G_loss(img, gt)

    #psnr_v, ssim_v = metrics_fn(pred.cpu().numpy(), batch['gt'].numpy())
    psnr_v = metrics_fn(pred.cpu().numpy(), batch['gt'].numpy())
    writer.add_scalar(step, 'val/loss', loss_v.item())
    writer.add_scalar(step, 'val/psnr', psnr_v)
    #writer.add_scalar(step, 'val/ssim', ssim_v)
    img_total = all_cat_cpu(opt, log, pred)
    img_total_gt = all_cat_cpu(opt, log, gt)
    img_total_noisy = all_cat_cpu(opt, log, img)
    grid = make_grid(img_total, nrow=16)
    grid_gt = make_grid(img_total_gt, nrow=16)
    grid_noisy = make_grid(img_total_noisy, nrow=16)
    writer.add_image(step, 'val/gt', grid_gt)
    writer.add_image(step, 'val/pred', grid)
    writer.add_image(step, 'val/noisy', grid_noisy)

def main(rank, world_size, *args, **kwargs):
    seed_everything(2024)
    init_process(rank, world_size)
    
    opt = create_training_options()
    opt.rank = rank
    log = Logger(opt)
    log.info("=======================================================")
    log.info("         ResNet50 for UKB50K MoCo")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    writer = build_log_writer(opt)

    torch.autograd.set_detect_anomaly(True)

    #==================MODEL   INIT==================#
    model = GAN_class(rank, vars(opt))
    #m = Res().to(rank)
    if opt.pretrained: model.load_model = './results/pretrained_resnet50_veronika/v1_epoch=372-step=5287275.ckpt'  


    #model = DDP(m, device_ids=[rank])
    optimizer_g = torch.optim.AdamW(model.Gnet.parameters(), lr=opt.lr)
    optimizer_d = torch.optim.AdamW(model.Dnet.parameters(), lr=opt.lr)
    
    #==================DATA SET INIT==================#
    dst_train, dst_val = random_split(MyDset(opt.datapath,(opt.image_size, opt.image_size), transform=Cast_jnp(torch_tensor=True)),[.1, .9]) # TODO: transorm Callback for Dset
    build_loader_fn = partial(build_loader, rank=rank, world_size=world_size, batch_size=opt.microbatch)
    train_dataloader, val_dataloader = [build_loader_fn(dataset=x) for x in [dst_train, dst_val]]

    ## FIXME: not correctly ckpt for torch.compiled model with DDP
    #if os.listdir(opt.ckpt_dir):
    #    last_ckpt = sorted(os.listdir(opt.ckpt_dir))[-1]
    #    last_ckpt = opt.ckpt_dir / last_ckpt
    #    checkpoint = torch.load(last_ckpt,  map_location="cpu")
    #    model.load_state_dict(checkpoint)
    #    log.info(f"[Net] Loaded network ckpt: {last_ckpt}!")
    ckpt_epoch = opt.ckpt_epoch
    for epoch in range(ckpt_epoch, opt.epochs): # TODO: add ckpt['epoch'] as start
        step=train_step(rank, world_size, epoch, model, (train_dataloader, val_dataloader), (optimizer_g, optimizer_d), writer, log, opt)
        #val_step(rank, world_size, step, model, val_dataloader, writer, log, opt)

        torch.save({'Gnet':model.Gnet.state_dict(),
                    'Dnet':model.Dnet.state_dict()},
                   os.path.join(opt.ckpt_dir, "{}_epoch_{}.ckpt".format(opt.name, epoch)))

    clean_up()

def spawn_fn(fn):
    try: # for SLURM 
        world_size = int(os.environ.get('WORLD_SIZE'))
        print("\033[93musing WORLD_SIZE from os.environ\033[0m")
    except:
        world_size = torch.cuda.device_count()
        print("\033[93musing WORLD_SIZE from torch.cuda.device_count\033[0m")
    spawn(fn,
          args=(world_size,),
          nprocs=world_size,
          join = True)

if __name__ == '__main__':
    spawn_fn(main)
