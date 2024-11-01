import torch.nn as nn

def _init_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0., std=1.)
        if module.bias is not None: module.bias.data.zero_()
