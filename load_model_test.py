from utils import update_dict_keys
import torch
from models import Res


ckpt_path = 'results/pretrained_resnet50_veronika/v1_epoch=372-step=5287275.ckpt'
ckpt = torch.load(ckpt_path)

m = Res()
print('raw weights:')
print(m.encoder.conv1.weight.data.mean())

print('ckpt weights:')
new_dict= update_dict_keys(ckpt['state_dict'], m.state_dict(), prefix_local='')

m.load_state_dict(new_dict)
print(m.encoder.conv1.weight.data.mean())
