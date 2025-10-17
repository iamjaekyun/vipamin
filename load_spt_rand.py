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
cfg.merge_from_file(f'cfg_for_full/{mn}_{args.data_set}.yaml')

Path(f'spts_{mn}').mkdir(exist_ok=True)


model, cur_device = build_model(cfg)
model = model.to('cuda')
model.eval()
block_num = 12
return_layers = {}
return_layers[f'enc.pos_drop'] = 0
melo = MidGetter(model, return_layers=return_layers, keep_output=False)
# res = np.zeros(block_num)
train_loader = data_loader.construct_train_loader(cfg)


reservoir = []
num_seen = 0
k = 100

for batch_idx, (data) in enumerate(train_loader):
    if not isinstance(data["image"], torch.Tensor):
        for k, v in data.items():
            data[k] = torch.from_numpy(v)

    inputs = data["image"].float()
    labels = data["label"]
    inputs, labels = inputs.to('cuda'), labels.to('cuda')
    with torch.no_grad():
        mid_outputs, _ = melo(inputs)
        out_array = mid_outputs[0].view(-1,768)        
        for row in out_array:
            if num_seen < k:
                # Fill the reservoir until we have k items
                reservoir.append(row)
            else:
                # Randomly replace elements with decreasing probability
                r = random.randrange(num_seen + 1)
                if r < k:
                    reservoir[r] = row
            num_seen += 1
reservoir_torch = torch.stack(reservoir).unsqueeze(0)
torch.save(reservoir_torch, f'spts_{mn}/{args.data_set}')
