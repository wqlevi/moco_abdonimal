"""
-[ ] Prediction still look un normalized, with extra bright contrast.  
"""
import torch.nn as nn
import torch
import os, argparse, sys
from pathlib import Path
import random
from functools import partial

from torchvision.models import resnet50
from torch.utils.data import random_split
from torch.multiprocessing import spawn

from models.resnet50_torch import Res, VAE
from logger import Logger, build_log_writer, get_date_time
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloader import MyDset, MyLoader, Cast_jnp
from utils import ssim, psnr
from dist_utils import init_process, all_gather, clean_up, all_cat_cpu
from torchvision.utils import make_grid

from pdb import set_trace as pb

def create_training_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test_unet")
    parser.add_argument("--log-dir", type=Path, default="./logs")
    parser.add_argument("--ckp-dir", type=Path, default="./results/unet_torch")
    parser.add_argument("--log-writer", type=str, default=None)
    parser.add_argument("--datapath", type=str, default="/mnt/qdata/share/rawangq1/ukbdata_70k/abdominal_MRI/2d")

    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--pretrained", action='store_true')

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    opt = parser.parse_args()

    opt.ckp_dir = opt.ckp_dir / opt.name 
    assert opt.batch_size % opt.microbatch == 0, f"{opt.batch_size=} is not dividable by {opt.microbatch}!"
    os.makedirs(opt.ckp_dir, exist_ok=True)
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

def train_step(rank, world_size, epoch, model, dataloader, optimizer, writer, log, opt)->int:
    def backward_step(batch)->float:
        img, gt = batch['noisy'].to(rank), batch['gt'].to(rank)
        out_tensors = model(img)
        loss = model.module.loss_fn(out_tensors)['loss']
        loss.backward()
        return loss.item(), out_tensors[0]

    model.train()
    n_inner_iter = opt.batch_size // (world_size * opt.microbatch)
    loss_epoch= 0
    for i, batch in enumerate(dataloader):
        loss_inner_iter = 0
        optimizer.zero_grad()
        for _ in range(n_inner_iter):
            loss_v, pred = backward_step(batch)
            loss_inner_iter += loss_v
        loss_inner_iter /= n_inner_iter
        loss_epoch += loss_inner_iter
        optimizer.step()
        writer.add_scalar(epoch*len(dataloader)+i, 'train/loss', loss_inner_iter)

    gt_total = all_cat_cpu(opt, log, batch['gt'].to(rank))
    img_total = all_cat_cpu(opt, log, pred)

    grid_gt = make_grid(gt_total, nrow=16)
    grid = make_grid(img_total, nrow=16)
    writer.add_image(epoch*len(dataloader), 'train/gt', grid_gt)
    writer.add_image(epoch*len(dataloader), 'train/pred', grid)
    log.info("[epoch]: {} | [loss]: {}".format(epoch, loss_epoch/i))
    return epoch*len(dataloader)+i

@torch.no_grad()
def val_step(rank, world_size, step, model, dataloader, writer, log, opt):
    metrics_fn = lambda dst, src: (psnr(dst, src), ssim(dst, src))
    model.eval()
    batch = next(iter(dataloader))
    img, gt = batch['noisy'].to(rank), batch['gt'].to(rank)
    img.to(rank)
    gt.to(rank)
    out_tensors = model(img)
    loss_v = model.module.loss_fn(out_tensors,kl_weight=1)['loss'].item() 

    pred = out_tensors[0]
    psnr_v, ssim_v = metrics_fn(pred.cpu().numpy(), gt.cpu().numpy())
    writer.add_scalar(step, 'val/loss', loss_v)
    writer.add_scalar(step, 'val/psnr', psnr_v)
    writer.add_scalar(step, 'val/ssim', ssim_v)
    img_total = all_cat_cpu(opt, log, pred)
    grid = make_grid(img_total, nrow=16)
    writer.add_image(step, 'val/pred', grid)

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

    #==================MODEL   INIT==================#
    m = VAE().to(rank)
    #m = Res().to(rank)
    if opt.pretrained: m.load_model = './results/pretrained_resnet50_veronika/v1_epoch=372-step=5287275.ckpt'  


    model = DDP(m, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)
    
    #==================DATA SET INIT==================#
    dst_train, dst_val = random_split(MyDset(opt.datapath,(opt.image_size, opt.image_size), transform=Cast_jnp(torch_tensor=True)),[.7, .3]) # TODO: transorm Callback for Dset
    build_loader_fn = partial(build_loader, rank=rank, world_size=world_size, batch_size=opt.microbatch)
    train_dataloader, val_dataloader = [build_loader_fn(dataset=x) for x in [dst_train, dst_val]]

    model = torch.compile(model)
    ## FIXME: not correctly ckpt for torch.compiled model with DDP
    #if os.listdir(opt.ckp_dir):
    #    last_ckpt = sorted(os.listdir(opt.ckp_dir))[-1]
    #    last_ckpt = opt.ckp_dir / last_ckpt
    #    checkpoint = torch.load(last_ckpt,  map_location="cpu")
    #    model.load_state_dict(checkpoint)
    #    log.info(f"[Net] Loaded network ckpt: {last_ckpt}!")
    for epoch in range(opt.epochs): # TODO: add ckpt['epoch'] as start
        step=train_step(rank, world_size, epoch, model, train_dataloader, optimizer, writer, log, opt)
        val_step(rank, world_size, step, model, val_dataloader, writer, log, opt)

    torch.save(model.state_dict(), os.path.join(opt.ckp_dir, "{}_epoch_{}.ckpt".format(opt.name, epoch)))

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
    
    #main(0, 1)
