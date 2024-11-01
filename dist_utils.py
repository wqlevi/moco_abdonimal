import os

import numpy as np
import torch

import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group, barrier

def init_process(rank, world_size):
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='1235'
    init_process_group("nccl", rank=rank, world_size=world_size)

def clean_up():
    barrier()
    destroy_process_group()

def all_gather(tensor, log=None):
    """
    gather tensors from different groups into a list
    """
    if log: log.info("Gathering tensor across {} devices... ".format(dist.get_world_size()))
    gathered_tensors = [
        torch.zeros_like(tensor).contiguous() for _ in range(dist.get_world_size())
    ]
    tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
    with torch.no_grad():
        dist.all_gather(gathered_tensors, tensor)
    return gathered_tensors

def all_cat_cpu(opt, log, t:torch.Tensor):
    #if not opt.distributed: return t.detach().cpu()
    gathered_t = all_gather(t.to(opt.rank), log=log)# return tensor gathered from all devices
    return torch.cat(gathered_t).detach().cpu()
