#!/usr/bin/env python3
"""
vit-moco-v3 with prompt
"""
import math
import torch
import torch.nn as nn
import torchvision as tv

from functools import partial, reduce
from operator import mul
from torch.nn import Conv2d, Dropout
from timm.models.vision_transformer import _cfg

from ..vit_backbones.vit_dino import VisionTransformerDino
from ...utils import logging
logger = logging.get_logger("visual_prompt")


class PromptedVisionTransformerDino(VisionTransformerDino):
    def __init__(self, prompt_config, **kwargs):
        super().__init__(**kwargs)
        self.prompt_config = prompt_config

        if self.prompt_config.DEEP and self.prompt_config.LOCATION not in ["prepend", ]:
            raise ValueError("Deep-{} is not supported".format(self.prompt_config.LOCATION))

        num_tokens = self.prompt_config.NUM_TOKENS

        self.num_tokens = num_tokens
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, self.embed_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            if self.prompt_config.DEEP:
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    len(self.blocks) - 1,
                    num_tokens, self.embed_dim
                ))
                # xavier_uniform initialization
                nn.init.uniform_(
                    self.deep_prompt_embeddings.data, -val, val)
        elif self.prompt_config.INITIATION == "vipamin":
            if self.prompt_config.DEEP:  # noqa
                vip = torch.load(f'match_mae_deep/{self.prompt_config.ID}_{self.prompt_config.K}')
                anti = torch.load(f'orth_mae_deep/{self.prompt_config.ID}')
                self.prompt_embeddings = nn.Parameter(((1-self.prompt_config.LAMB) * vip[0] + (self.prompt_config.LAMB) * anti[0]).unsqueeze(0), requires_grad=True)
                self.deep_prompt_embeddings = nn.Parameter((1-self.prompt_config.LAMB) * vip[1:] + (self.prompt_config.LAMB) * anti[1:], requires_grad=True)
            else:
                vip = torch.load(f'match_mae/{self.prompt_config.ID}_{self.prompt_config.K}', weights_only=True)
                anti = torch.load(f'orth_mae/{self.prompt_config.ID}', weights_only=True)
                self.prompt_embeddings = nn.Parameter((1-self.prompt_config.LAMB) * vip + (self.prompt_config.LAMB) * anti, requires_grad=True)
        elif self.prompt_config.INITIATION == "spt":
            if self.prompt_config.DEEP:
                spt = torch.load(f'spts_mae_deep/{self.prompt_config.ID}')
                self.prompt_embeddings = nn.Parameter(spt[0].unsqueeze(0), requires_grad=True)
                self.deep_prompt_embeddings = nn.Parameter(spt[1:], requires_grad=True)
            else:
                spt = torch.load(f'spts_mae/{self.prompt_config.ID}', weights_only=True)
                self.prompt_embeddings = nn.Parameter(vip, requires_grad=True)
        else:
            raise ValueError("Other initiation scheme is not supported")

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        if self.prompt_config.LOCATION == "prepend":
            # after CLS token, all before image patches
            x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
            x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(
                        self.prompt_embeddings.expand(B, -1, -1)),
                    x[:, 1:, :]
                ), dim=1)
            # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        else:
            raise ValueError("Other prompt locations are not supported")

        return x

    def embeddings(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((
                cls_token, self.dist_token.expand(x.shape[0], -1, -1), x),
            dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.blocks.eval()
            self.patch_embed.eval()
            self.pos_drop.eval()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_features(self, x):
        x = self.incorporate_prompt(x)

        # deep
        if self.prompt_config.DEEP:
            B = x.shape[0]
            num_layers = len(self.blocks)

            for i in range(num_layers):
                if i == 0:
                    x = self.blocks[i](x)
                else:
                    # prepend
                    x = torch.cat((
                        x[:, :1, :],
                        self.prompt_dropout(
                            self.deep_prompt_embeddings[i-1].expand(B, -1, -1)
                        ),
                        x[:, (1 + self.num_tokens):, :]
                    ), dim=1)
                    x = self.blocks[i](x)
        else:
            # not deep:
            x = self.blocks(x)

        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]


def vit_base(prompt_cfg, **kwargs):
    model = PromptedVisionTransformerDino(
        prompt_cfg,
        patch_size=16, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_small(prompt_cfg, **kwargs):
    model = PromptedVisionTransformerDino(
        prompt_cfg,
        patch_size=16, embed_dim=384, depth=12,
        num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

