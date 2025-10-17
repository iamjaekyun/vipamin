import torch
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from src.data import loader as data_loader
from src.models.build_model import build_model
from src.configs.config import get_cfg

import argparse
import numpy as np
import random
from pathlib import Path

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--data-set', type=str)
args = parser.parse_args()

T = []
cfg = get_cfg()
mn = 'mae'
cfg.merge_from_file(f'cfg_for_vpt_deep/{mn}_{args.data_set}.yaml')

Path(f'spts_{mn}_deep').mkdir(exist_ok=True)


model, cur_device = build_model(cfg)
model = model.to('cuda')
model.eval()
block_num = 12
return_layers = {}
for i in range(block_num-1):
    return_layers[f'enc.blocks.{i}']= i+1
return_layers[f'enc.pos_drop'] = 0
melo = MidGetter(model, return_layers=return_layers, keep_output=False)
# res = np.zeros(block_num)
train_loader = data_loader.construct_val_loader(cfg)

import random

reservoir = [[] for _ in range(block_num)]
num_seen = [0 for _ in range(block_num)]
k = 20

# kmeans = KMeans(n_clusters=20)
for batch_idx, (data) in enumerate(train_loader):
    if not isinstance(data["image"], torch.Tensor):
        for k, v in data.items():
            data[k] = torch.from_numpy(v)

    inputs = data["image"].float()
    labels = data["label"]
    inputs, labels = inputs.to('cuda'), labels.to('cuda')
    with torch.no_grad():
        mid_outputs, _ = melo(inputs)
    for key_ in mid_outputs:
        out_array = mid_outputs[key_].view(-1,768)
        
        for row in out_array:
            if num_seen[key_] < k:
                # Fill the reservoir until we have k items
                reservoir[key_].append(row)
            else:
                # Randomly replace elements with decreasing probability
                r = random.randrange(num_seen[key_] + 1)
                if r < k:
                    reservoir[key_][r] = row
            num_seen[key_] += 1

ret = []
for k_ in range(len(reservoir)):
    tmp =  torch.stack(reservoir[k_])
    ret.append(tmp)
    # print(tmp.size())
hehe = torch.stack(ret)
# print(hehe.size()); quit()
torch.save(hehe, f'spts_{mn}_deep/{args.data_set}')