#!/usr/bin/env python3
import numpy as np
import torch
import os

from .vit_backbones.vit_moco import vit_base
from .vit_backbones.vit_mae import build_model as mae_vit_model
from .vit_backbones.vit_dino import vit_small_patch16 as vit_dino_small_patch16

from .vit_prompt.vit_moco import vit_base as prompt_vit_base
from .vit_prompt.vit_mae import build_model as prompt_mae_vit_model
from .vit_prompt.vit_dino import vit_small as prompt_dino_vit_small

# from .vit_prompt.vit_ablations import PromptedAblationVisionTransformer 通过这行可以加入vpt的ablations
# 先暂时不加 先弄一个vpt的原始版本来进行测试

from .vit_adapter.vit_mae import build_model as adapter_mae_vit_model
from .vit_adapter.vit_moco import vit_base as adapter_vit_base

from .vit_adapter.vit import ADPT_VisionTransformer
MODEL_ZOO = {
    "mae_vitb16": "mae_pretrain_vit_base.pth",
    "mocov3_vitb": "mocov3-ViT-B.pth.tar",
    "dino_vits16": "dino_deitsmall16_pretrain.pth",
    "mae_vitt16": "mae_tiny_400e.pth.tar",
}

def build_mae_model(
    model_type, crop_size, prompt_cfg, p_vk_cfg, model_root, adapter_cfg=None
): # new added p_vk_cfg
    if prompt_cfg is not None:
        model = prompt_mae_vit_model(model_type, prompt_cfg)
    elif adapter_cfg is not None:
        model = adapter_mae_vit_model(model_type, adapter_cfg)
    else:
        model = mae_vit_model(model_type)
    out_dim = model.embed_dim

    ckpt = os.path.join(model_root, MODEL_ZOO[model_type])

    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['model']
    
    model.load_state_dict(state_dict, strict=False)
    model.head = torch.nn.Identity()
    return model, out_dim


def build_mocov3_model(
    model_type, crop_size, prompt_cfg, p_vk_cfg, model_root, adapter_cfg=None
):
    if model_type != "mocov3_vitb":
        raise ValueError("Does not support other arch")
    if prompt_cfg is not None:
        model = prompt_vit_base(prompt_cfg)
    elif adapter_cfg is not None:
        model = adapter_vit_base(adapter_cfg)
    else:
        model = vit_base()
    out_dim = 768
    # print('load_ckpt mocov3_linear-vit-b-300ep.pth.tar')
    ckpt = os.path.join(model_root,MODEL_ZOO[model_type])

    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.'):
            # remove prefix
            state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.head = torch.nn.Identity()
    return model, out_dim


def build_dino_model(
    model_type, crop_size, prompt_cfg, p_vk_cfg, model_root, adapter_cfg=None
):
    if model_type != "dino_vits16":
        raise ValueError("Does not support other arch")
    if prompt_cfg is not None:
        model = prompt_dino_vit_small(prompt_cfg)
    else:
        model = vit_dino_small_patch16()
    out_dim = 384
    ckpt = os.path.join(model_root, MODEL_ZOO[model_type])

    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    model.head = torch.nn.Identity()
    return model, out_dim

