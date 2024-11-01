from models import Generator, Discriminator, GAN_class
import os
import torch

from dist_utils import init_process, all_gather, clean_up, all_cat_cpu
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn

def run(rank, world_size, model, opt):
    x = torch.ones(opt['microbatch'],1,128,128, dtype=torch.float32, device=rank)

    #opt = {'image_size': 128,
    #       'microbatch': 4,
    #       'FE':1,
    #       'distributed':0}

    #m = GAN_class(rank, opt)

    model.update_noise = .5

    opt_g = torch.optim.AdamW(model.Gnet.parameters(), lr=1e-3)
    opt_d = torch.optim.AdamW(model.Dnet.parameters(), lr=1e-3)
    opt_g.zero_grad()
    for _ in range(4):
        loss, pred = model.G_loss(x, x+torch.randn_like(x))
        loss.backward()
    opt_g.step()

    opt_d.zero_grad()
    for _ in range(4):
        loss = model.D_loss(torch.ones_like(pred), pred.detach())
        loss.backward()
    opt_d.step()
    print(model.instance_noise.std())

def main(rank, world_size):
    opt = {'image_size': 128,
           'microbatch': 8,
           'FE':1,
           'distributed':0,
           'ckpt_dir':'results/unet_torch',
           'name':'test_GAN',
           'ckpt_name':'test_GAN_epoch_9.ckpt',
           'ckpt_epoch':9}
    if opt['distributed']:
        init_process(rank, world_size)
        torch.autograd.set_detect_anomaly(True)
    model = GAN_class(rank, opt)
    print(model.Gnet.state_dict())
    run(rank, world_size, model, opt)

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
    #spawn_fn(main)
    main(0, 1)
