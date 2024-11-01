"""
How blobs passed through:
    PIL -> Tensor -> numpy
"""
from glob import glob
from pathlib import Path
from typing import List, Tuple

import jax.numpy as jnp
from torch.utils import data
from torch import Tensor
from jax.tree_util import tree_map
from torchvision.io import read_image
from torchvision.transforms import Resize, Normalize, RandomCrop
import torchvision.transforms.functional as TF
import numpy as np

def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))

class Cast_jnp(object):
    """
    dataloader transformation fn:
        1. permute to JAX: [B,H,W,C] or torch.Tensor: [B,C,H,W];
        2. cast img to numpy.NdArray or torch.Tensor
    """
    def __init__(self, torch_tensor:bool=False):
        self.torch_tensor = torch_tensor
        mean_data:float=13.1823
        std_data:float=21.1146
        self.normalize = lambda x: (x - mean_data) / std_data
    def __call__(self, img:Tensor):
        img = img.permute(1,2,0) if not self.torch_tensor else img
        self.arr = np.array(img, dtype=jnp.float32)
        self.arr = self.normalize(self.arr)
        return self.arr

class MyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler = None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, to_numpy=True
                 ):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate if to_numpy else None,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)
        
class MyDset(data.Dataset):
    def __init__(self, img_dir, crop_size:Tuple[int, int]=True, transform=None):
            self.img_dir = img_dir
            self.transform = transform
            self.files = sorted(Path(img_dir).glob("*.png"))
            self.files_sim = sorted(Path(img_dir,"sim").glob("*.png"))

            self.crop_size_h, self.crop_size_w = crop_size

    def _randomcrop(self, img, sim):
        i,j,h,w = RandomCrop.get_params(
                img, output_size=(self.crop_size_h,
                                  self.crop_size_w)
                )
        img, sim = [TF.crop(x, i, j, h, w) for x in [img, sim]]
        return img, sim

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = read_image(self.files[idx])
        image_sim = read_image(self.files_sim[idx])
        image, image_sim = self._randomcrop(image, image_sim)
        image = self.transform(image)
        image_sim = self.transform(image_sim)
        return {'gt':image, 'noisy':image_sim}

