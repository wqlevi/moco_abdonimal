import torch
import sys, os
from PIL import Image
from pathlib import Path
import numpy as np
from models import Generator
from utils import update_dict_keys

MEAN, STD = 13.1823, 21.1146
norm = lambda x: x-MEAN / STD
denorm = lambda x: (x - x.min())*255 / (x.max() - x.min())

def infer_array(x:np.ndarray, m, device)-> np.ndarray:
    ts = torch.Tensor(norm(x))[None,None].to(device)
    arr_o = m(ts)[0,0].detach().cpu().numpy()
    arr_o_denorm = denorm(arr_o).astype('uint8')
    return arr_o_denorm

def load_ckpt_and_update_dict(m, ckpt):
    new_dict = update_dict_keys(ckpt['Gnet'], m.state_dict(), prefix_local='module.')
    m.load_state_dict(new_dict)
    return m

def im_io(img_path:str, save_path:str, func, *args):
    im = Image.open(img_path)
    arr = np.array(im)
    out = func(arr, *args)
    im_o = Image.fromarray(out)
    im_o.save(save_path)

if __name__ == '__main__':
    input_path = sys.argv[1] 
    ckpt_path = 'results/unet_torch/test_GAN/test_GAN_epoch_9.ckpt'
    ckpt = torch.load(ckpt_path)

    device = torch.device("cuda:0")
    model = Generator(num_res_blocks=16).to(device)

    model = load_ckpt_and_update_dict(model, ckpt)

    save_name = input_path.rsplit('/',1)[-1]
    save_path = Path(ckpt_path.rsplit('/',1)[0], 'img')

    os.makedirs(save_path, exist_ok=True)
    save_path = Path(save_path, save_name)
    im_io(input_path, save_path, infer_array, model, device)



    

