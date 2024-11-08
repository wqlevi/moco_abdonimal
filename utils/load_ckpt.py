import torch
from collections import OrderedDict

def load_ckpt(ckpt_name:str)->dict:
    # output: 
    #   dict(layers: weights)
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device("cuda:0")
    ckpt_file = torch.load(ckpt_name, map_location=device)
    return ckpt_file

def update_dict_keys(dict_local:OrderedDict, dict_model:OrderedDict, prefix_local:str='encoder.', prefix_model:str='', clean_proj_layers:bool=True)-> OrderedDict:
    # input:
    #   dict_local: ckpt model dict [state_dict]
    #   dict_model: defined model dict [state_dict]
    #   prefix_local: ckpt model prefix [str]
    #   prefix_model: defined model prefix [str]
    # output:
    #   state_dict(keys: weight)
    # USAGE:
    #   model.load_state_dict(update_dict_keys(*args, **kwargs))
    if clean_proj_layers: dict_local = {k:v for k,v in dict_local.items() if not k.startswith('projection')}
    new_dict = { k.replace(prefix_local,prefix_model):v for k,v in dict_local.items() if prefix_local in k}
    assert new_dict.keys() == dict_model.keys(), "keys of dicts are not matching!"
    dict_model.update(new_dict)
    return dict_model

def load_ckpt_and_update_dict(m, ckpt, key:str=None):
    new_dict = update_dict_keys(ckpt[key], m.state_dict(), prefix_local='module.')
    m.load_state_dict(new_dict)
    return m
