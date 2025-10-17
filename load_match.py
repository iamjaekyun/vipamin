import torch

import argparse
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
import numpy as np
from src.data import loader as data_loader
from src.models.build_model import build_model
from src.configs.config import get_cfg
import torch.nn as nn
import argparse
import random
from pathlib import Path

parser = argparse.ArgumentParser()


parser.add_argument('--data-set', type=str)
args = parser.parse_args()

T = []
cfg = get_cfg()
mn = 'mae'
cfg.merge_from_file(f'cfg_for_vpt/{mn}_{args.data_set}.yaml')

Path(f'match_{mn}').mkdir(exist_ok=True)

block_num = 12
prompt_len = 100
return_layers = {}

return_layers['enc.pos_drop'] = -1
return_layers[f'enc.blocks.0.attn.qkv']= 0

# res = np.zeros(block_num)
train_loader = data_loader.construct_train_loader(cfg)

from collections import defaultdict

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

model, cur_device = build_model(cfg)
model = model.to('cuda')
melo = MidGetter(model, return_layers=return_layers, keep_output=False)


for batch_idx, (data) in enumerate(train_loader):
    if not isinstance(data["image"], torch.Tensor):
        for k, v in data.items():
            data[k] = torch.from_numpy(v)

    inputs = data["image"].float()
    labels = data["label"]
    inputs, labels = inputs.to('cuda'), labels.to('cuda')
    with torch.no_grad():
        mid_outputs, _ = melo(inputs)
    B,N,C = 256, 1 + prompt_len + 196, 768
    ret = defaultdict(list)
    output = mid_outputs[0]
    qkv = output.reshape(B, N, 3, 12, C // 12).permute(2, 0, 3, 1, 4)
    k = qkv[1].mean(1).mean(0)[1:, :]
    adj = nn.functional.normalize(k, p=2, dim=-1)
    cos = adj @ adj.transpose(-1,-2)
    cos.diagonal(dim1=-1, dim2=-2).fill_(-torch.inf)
    cos[:prompt_len, :prompt_len].fill_(-torch.inf)
    I = torch.topk(cos,128)[1][:prompt_len, :]
    
    X = mid_outputs[-1].mean(0)
    for k in [2,8,32,128]:
        match_cand = torch.stack([X[idxs+1-prompt_len].mean(dim=0) for idxs in I[:, :k]]).unsqueeze(0)
        torch.save(match_cand, f'match_{mn}/{args.data_set}_{k}')
    quit()