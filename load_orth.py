import torch
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from src.data import loader as data_loader
from src.models.build_model import build_model
from src.configs.config import get_cfg

import argparse
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, reduce
from operator import mul
from pathlib import Path


mn = 'mae'

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


Path(f'orth_{mn}').mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--data-set', type=str)
args = parser.parse_args()

T = []
cfg = get_cfg()

cfg.merge_from_file(f'cfg_for_full/{mn}_{args.data_set}.yaml')

model, cur_device = build_model(cfg)
model = model.to('cuda')
model.eval()

model.enc.blocks[0].attn.dummy = nn.Identity()

original_forward = model.enc.blocks[0].attn.forward

def new_forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)

    # Save intermediate output
    x_attn = self.dummy(x_attn)

    x = self.proj(x_attn)
    x = self.proj_drop(x)
    return x

model.enc.blocks[0].attn.forward = new_forward.__get__(model.enc.blocks[0].attn)


for n, p in model.named_parameters():
    if 'qkv.weight' in n and 'enc.blocks.0' in n:
        d = p.size(1)
        W_V = p[-d:, :]
    if 'qkv.bias' in n and 'enc.blocks.0' in n:
        # qkv
        d = p.size(0)//3
        V_b = p[-d:]

block_num = 12
return_layers = {}
# B X N X D
return_layers[f'enc.blocks.0.attn.dummy'] = 0
melo = MidGetter(model, return_layers=return_layers, keep_output=False)
# res = np.zeros(block_num)
train_loader = data_loader.construct_train_loader(cfg)

import math
prompt_len = 100
orthogonal_vectors = []

for batch_idx, (data) in enumerate(train_loader):
    if not isinstance(data["image"], torch.Tensor):
        for k, v in data.items():
            data[k] = torch.from_numpy(v)

    inputs = data["image"].float()
    labels = data["label"]
    inputs, labels = inputs.to('cuda'), labels.to('cuda')
    with torch.no_grad():
        mid_outputs, _ = melo(inputs)
        O = mid_outputs[0].mean(0)
        n,d = O.shape
        Q, _ = torch.linalg.qr(O.T, mode='reduced')
        P = torch.eye(d).cuda() - Q @ Q.T
        for _ in range(prompt_len):
            # Normalize if desired
            val = math.sqrt(6. / float(3 * 16**2 + d))
            rand_vec = torch.randn(d).uniform_(-val, val).cuda()
            rand_vec = rand_vec @ W_V.T + V_b
            orth_vec = P @ rand_vec
            orth_vec = (orth_vec - V_b) @ torch.linalg.pinv(W_V)
            orthogonal_vectors.append(orth_vec)

        ret = torch.stack(orthogonal_vectors).unsqueeze(0)
        torch.save(ret, f'orth_{mn}/{args.data_set}')
        quit()