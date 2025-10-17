#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
borrowed from https://github.com/facebookresearch/mae/blob/main/models_vit.py
"""
from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformerDino(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, **kwargs):
        super(VisionTransformerDino, self).__init__(**kwargs)
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        outcome = x[:, 0]

        return outcome


def build_model(model_type):
    if "vitb" in model_type:
        return vit_base_patch16()
    elif "vits" in model_type:
        return vit_small_patch16()


def vit_base_patch16(**kwargs):
    model = VisionTransformerDino(
        drop_path_rate=0.1,
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_patch16(**kwargs):
    model = VisionTransformerDino(
        drop_path_rate=0.1,
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
