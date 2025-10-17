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
parser = argparse.ArgumentParser()


parser.add_argument('--data-set', type=str)
args = parser.parse_args()

mn = 'mae'
T = []
cfg = get_cfg()
cfg.merge_from_file(f'cfg_for_vpt_deep/{mn}_{args.data_set}.yaml')

block_num = 12
prompt_len = 20
return_layers = {}
return_layers['enc.pos_drop'] = -1
# return_layers['enc.pos_drop'] = -1
for i in range(block_num):
    return_layers[f'enc.blocks.{i}.attn.qkv']= 2*i 
    return_layers[f'enc.blocks.{i}'] = 2*i + 1

# res = np.zeros(block_num)
train_loader = data_loader.construct_train_loader(cfg)

from collections import defaultdict

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

model, cur_device = build_model(cfg)
model = model.to('cuda')
model.eval()
melo = MidGetter(model, return_layers=return_layers, keep_output=False)

from pathlib import Path
Path(f'match_{mn}_deep').mkdir(exist_ok=True)

for batch_idx, (data) in enumerate(train_loader):
    if not isinstance(data["image"], torch.Tensor):
        for k, v in data.items():
            data[k] = torch.from_numpy(v)

    inputs = data["image"].float()
    labels = data["label"]
    inputs, labels = inputs.to('cuda'), labels.to('cuda')
    # num_data += targets.shape[0]
    with torch.no_grad():
        mid_outputs, _ = melo(inputs)
    ret = defaultdict(list)
    B, N, C = 256, 1+prompt_len+196, 768
    for block in range(block_num):
        output = mid_outputs[2*block]
        qkv = output.reshape(B,N,3,12,C//12).permute(2,0,3,1,4)
        k = qkv[1].mean(1).mean(0)[1:, :]
        # kk = qkv[1].mean(1)[:, 1:, :]
        adj = nn.functional.normalize(k, p=2, dim=-1)
        cos = adj @ adj.transpose(-1,-2)
        cos.diagonal(dim1=-1, dim2=-2).fill_(-torch.inf)
        cos[:prompt_len, :prompt_len].fill_(-torch.inf)
        # print(cos.size()); quit()
        I = torch.topk(cos,128)[1][:prompt_len, :]
        if block == 0:
            X = mid_outputs[2*block -1].mean(0)
        else:
            X = mid_outputs[2*block-1].mean(0)
        # print(block, I.size(), I.min().item(),I.max().item(), X.size())
        for k in [2,8,32,128]:
            if block == 0:
                oracle_cand = torch.stack([X[idxs+1-prompt_len].mean(dim=0) for idxs in I[:, :k]])
            else:
                oracle_cand = torch.stack([X[idxs+1].mean(dim=0) for idxs in I[:, :k]])
            ret[k].append(oracle_cand)
    for xx in ret:
        torch.save(torch.stack(ret[xx]), f'match_{mn}_deep/{args.data_set}_{xx}')
    quit()