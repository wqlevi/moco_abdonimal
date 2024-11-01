"""
TODO:
    - [x] why batch is not on deivce("gpu")
        * only moved during loss function call
    - [x] finish train step
    - [x] finish loss function setup
    - [x] finish paired image load for motion removal  
    - [x] eval and image log to correct scale
    - [ ] multi-device training implementation
        TODO: - [ ] sharding input batch across devices
        refer: https://www.machinelearningnuggets.com/distributed-training-with-jax-and-flax/
    - [ ] more variety of sim motion 
    - [x] random split of dataloader too train and val
    - [x] train on patches to keep consistent image size for all planes (torchvision.transforms.RandomCrop)
    FIXME: loged image scale not correct:
    - [x] [0, 255] for grayscale in uint8 in tensorboard?
    - [x] out tensor in different scale (mean value shift)
    - [x] BatchNorm not working properly in eval() [ref:](https://github.com/google/flax/blob/cb6843f29d3400d7dab6751d4b693e4862f57d98/docs/guides/training_techniques/batch_norm.rst)
    - [ ] training hang at first iteration, not sure if the device-wise batching is correctly sharded.
"""
from flax.training import train_state, checkpoints
from functools import partial
import flax
from jax import lax
import optax
import jax
import os, sys
import argparse
from torch import Tensor
from pathlib import Path

from pdb import set_trace as pb

import numpy as np
import jax.numpy as jnp
from dataloader import MyDset, MyLoader, Cast_jnp
from models.unet_jax import UNet
from torch.utils.data import random_split

from logger import Logger, build_log_writer
from utils.metrics import ssim, psnr

class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict

# --- init --- #
def create_training_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="mp_unet")
    parser.add_argument("--log-dir", type=Path, default="./logs")
    parser.add_argument("--ckp-dir", type=Path, default="./results/unet_jax")
    parser.add_argument("--log-writer", type=str, default=None)
    parser.add_argument("--datapath", type=str, default="/mnt/qdata/share/rawangq1/ukbdata_70k/abdominal_MRI/2d")

    parser.add_argument("--rank", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    opt = parser.parse_args()
    
    opt.ckp_dir /=  opt.name  # make subdir

    os.makedirs(opt.ckp_dir, exist_ok=True)
    return opt

@jax.pmap# FIXME: opt not hashable
def create_state(rng, opt):
    model = UNet()
    params = model.init(rng, 
                        jnp.ones([opt.batch_size, opt.image_size, opt.image_size, NUM_CHANNEL]))['params']
    tx = opt.adamw(learning_rate=opt.lr)
    return TrainState.create(
            apply_fn=model.apply,
            params=params['params'],
            batch_stats=params['batch_stats'],
            tx=tx
            )

@partial(jax.pmap, in_axes=(None, None, None, 0), axis_name='batch')
def calculate_loss(params, state, rng, batch, train=True):
    labels, imgs = batch['gt'], batch['noisy'] # clean target, noisey image
    rng, dropout_apply_rng = jax.random.split(rng)
    logits = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x = imgs,
            train=True,
            mutable=['batch_stats'],
            rngs={'dropout':dropout_apply_rng}
            )# logits[0]: output, logits[1]: params['batch_stats']
    out, updates = logits if train else (logits, None)
    loss = optax.l2_loss(out, labels).mean()
    return loss, (rng, out, updates)

def eval_model(state, batch, rng):
    rng, dropout_apply_rng = jax.random.split(rng)
    logits = state.apply_fn(
            {'params': state.params, 'batch_stats': state.batch_stats},
            x = batch['noisy'],
            train=False,
            mutable=False,
            rngs={'dropout':dropout_apply_rng}
            )
    loss = optax.l2_loss(logits, batch['gt']).mean()
    return logits, loss

def get_metrics(src:np.ndarray, gt:np.ndarray)->float:
    return psnr(src, gt), ssim(src, gt)

def make_grid(arr:Tensor, nrow:int)->Tensor:
    """
    arr: [B, H, W, C]
    out: [C, n*H, B*W/n]
    """
    assert arr.ndim == 4, "input Tensor ndim should == 4!"
    grid = arr.reshape(nrow*arr.shape[1], int(arr.shape[0]*arr.shape[2]/nrow), arr.shape[3]).permute(2,0,1)
    return grid 

@jax.jit
def train_step(state, rng, batch):
    loss_fn = lambda params: calculate_loss(params, state, rng, batch) # functionalize loss
    (loss, ret), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params) # backprop
    rng, out, updates = ret

    grads = lax.pmean(grads, axis_name='batch')
    state = jax.pmap(state.apply_gradients(grads=grads)) # update params 
    state = state.replace(batch_stats= updates['batch_stats'] ) # NOTE: update `batch_stats`, otherwise it's not used
    return state, rng, loss, out

def val_step(state, rng, batch):
    #loss_v, (_, out) = calculate_loss(state.params, rng, batch, train=False)
    out, loss_v = eval_model(state, batch, rng) 
    psnr_v, ssim_v = get_metrics(np.array(out)[:,0], np.array(batch['gt'])[:,0])
    return out, (loss_v, psnr_v, ssim_v)

def train_epoch(state, rng, dataloader, epoch, log, writer, opt):
    losses = 0
    train_dataloader, val_dataloader = dataloader
    for i, batch in enumerate(train_dataloader):
        state, rng, loss, out = train_step(state, rng, batch) # out: [B, H, W, C]
        writer.add_image(i+epoch*len(dataloader), 'train/pred', make_grid(Tensor(np.array(out)), 16))
        writer.add_image(i+epoch*len(dataloader), 'train/gt', make_grid(Tensor(np.array(batch['gt'])), 16))
        losses += loss
        
        if i%100 == 0:
            val_batch = next(iter(val_dataloader))
            val_epoch(state, rng, val_batch, i+epoch*len(train_dataloader), writer)

        if i%500 == 0:
            checkpoints.save_checkpoint(
                    ckpt_dir = opt.ckp_dir.resolve(),
                    target={'params': state.params,
                            'batch_stats':state.batch_stats},
                    step=i*epoch*len(train_dataloader),
                    overwrite=True
                    )
            log.info(f"[ckpt]: {opt.ckp_dir}")
    acc = losses/i
    log.info("epoch mean:{}".format(acc))
    return state, rng, acc

def val_epoch(state, rng, batch, step, writer):
    out, (loss_v, psnr_v, ssim_v) = val_step(state, rng, batch)
    # log scalar, imgs
    [writer.add_scalar(step, title, scalar) for title, scalar in zip(['loss','psnr', 'ssim'],
                                                                     [np.array(loss_v), np.array(psnr_v), np.array(ssim_v)])]
    writer.add_image(step, 'val/pred', make_grid(Tensor(np.array(out)), 16)) # [C, H*n, W]
    writer.add_image(step, 'val/gt', make_grid(Tensor(np.array(batch['gt'])), 16))
    writer.add_image(step, 'val/noisy', make_grid(Tensor(np.array(batch['noisy'])), 16))


if __name__ == '__main__':
    opt = create_training_options()
    rng = jax.random.PRNGKey(42)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    global NUM_CHANNEL
    NUM_CHANNEL:int = 1
    #input_shape = (opt.batch_size, opt.image_size, opt.image_size, NUM_CHANNEL)
    #dtype_model = jnp.float32

    #model = UNet()
    #params = model.init(init_rng, jnp.ones(input_shape, dtype_model), train=True)
    #optimizer = optax.adamw(learning_rate=opt.lr)

    #model_state = TrainState.create(apply_fn=model.apply,
    #                                            params=params['params'],
    #                                            batch_stats = params['batch_stats'],
    #                                            tx=optimizer) #init: step=0, opt_state
    model_state = create_state(rng, opt)

    dst = MyDset(opt.datapath, crop_size=(opt.image_size, opt.image_size), transform=Cast_jnp())
    train_dst, val_dst = random_split(dst, [.7,.3])
    train_dataloader = MyLoader(train_dst, batch_size=opt.batch_size, drop_last=True, shuffle=True )
    val_dataloader = MyLoader(val_dst, batch_size=opt.batch_size, drop_last=True, shuffle=True )

    log = Logger(opt)
    log.info("=======================================================")
    log.info("         UNet for UKB50K MoCo")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))

    log.info("[DEVICE] count:{}".format(jax.device_count))

    writer = build_log_writer(opt)
    for epoch in range(opt.epochs):
        log.info(f"[epoch]: {epoch=}")
        model_state, rng, acc = train_epoch(model_state, rng, (train_dataloader, val_dataloader), epoch, log, writer, opt)

