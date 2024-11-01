# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import time
import logging
import torch

from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.logging import RichHandler
import datetime

norm = lambda x: (x-x.min())*255/(x.max() - x.min())
def get_time(sec):
    h = int(sec//3600)
    m = int((sec//60)%60)
    s = int(sec%60)
    return h,m,s

class TimeFilter(logging.Filter):

    def filter(self, record):
        try:
          start = self.start
        except AttributeError:
          start = self.start = time.time()

        time_elapsed = get_time(time.time() - start)

        record.relative = "{0}:{1:02d}:{2:02d}".format(*time_elapsed)

        # self.last = record.relativeCreated/1000.0
        return True

def get_date_time():
    return datetime.datetime.now().strftime("%m-%d-%Y_%H-%M")

class Logger(object):
    def __init__(self, opt):
        # other libraries may set logging before arriving at this line.
        # by reloading logging, we can get rid of previous configs set by other libraries.
        from importlib import reload
        reload(logging)
        self.rank = opt.rank
        if self.rank == 0:
            os.makedirs(opt.log_dir, exist_ok=True)
            
            datestamp = get_date_time()
            log_file = open(os.path.join(opt.log_dir, "log_{}_{}.txt".format(opt.name, datestamp)), "w")
            file_console = Console(file=log_file, width=150)
            logging.basicConfig(
                level=logging.INFO,
                format="(%(relative)s) %(message)s",
                datefmt="[%X]",
                force=True,
                handlers=[
                    RichHandler(show_path=False),
                    RichHandler(console=file_console, show_path=False)
                ],
            )
            # https://stackoverflow.com/questions/31521859/python-logging-module-time-since-last-log
            log = logging.getLogger()
            [hndl.addFilter(TimeFilter()) for hndl in log.handlers]

    def info(self, string, *args):
        if self.rank == 0:
            logging.info(string, *args)

    def warning(self, string, *args):
        if self.rank == 0:
            logging.warning(string, *args)

    def error(self, string, *args):
        if self.rank == 0:
            logging.error(string, *args)

class BaseWriter(object):
    def __init__(self, opt):
        self.rank = opt.rank
    def add_scalar(self, step, key, val):
        pass # do nothing
    def add_image(self, step, key, image):
        pass # do nothing
    def close(self): pass

class TensorBoardWriter(BaseWriter):
    def __init__(self, opt):
        super(TensorBoardWriter,self).__init__(opt)
        if self.rank == 0:
            run_dir = str(opt.log_dir / opt.name)
            os.makedirs(run_dir, exist_ok=True)
            self.writer=SummaryWriter(log_dir=run_dir, flush_secs=20)

    def add_scalar(self, global_step, key, val):
        if self.rank == 0: self.writer.add_scalar(key, val, global_step=global_step)

    def add_image(self, global_step, key, image):
        if self.rank == 0:
            image = norm(image.mul(21.1146).add_(13.1823)).to("cpu", torch.uint8) # add mean of dataset, mul scale of std
            self.writer.add_image(key, image, global_step=global_step)

    def close(self):
        if self.rank == 0: self.writer.close()

def build_log_writer(opt):
    if opt.log_writer == 'tensorboard': return TensorBoardWriter(opt)
    else: return BaseWriter(opt) # do nothing

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def space_indices(num_steps, count):
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps

def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]

# NOTE: my implementation
def time_wrap(name:str=None):
    def time_wrap(fn):
        def inner(*args, **kwargs):
            start_time = time.time()
            out = fn(*args, **kwargs)
            #print(f"Function {name}\tTime spent : {time.time() - start_time} sec")
            time_used = time.time() - start_time
            return time_used, out
        return inner
    return time_wrap
