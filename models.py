# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from functools import partial
import torchvision

from convit import VisionTransformer, VisionTransformerbase
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from collections import OrderedDict

class block_resnet(nn.Module):
    def __init__(self, pretrained = False):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.backbone = nn.Sequential(*list(torchvision.models.resnet50(pretrained=pretrained).children())[:-2])
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.backbone(x) # input = 1024
        
        return x

@register_model
def convit_tiny(pretrained=False, **kwargs):
    num_heads = 4
    kwargs['embed_dim'] *= num_heads
    model = VisionTransformer(
        num_heads=num_heads, hybrid_backbone=block_resnet(),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/convit/convit_tiny.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint)
    return model

@register_model
def convit_small(pretrained=False, **kwargs):
    num_heads = 9
    kwargs['embed_dim'] *= num_heads
    model = VisionTransformer(
        num_heads=num_heads, hybrid_backbone=block_resnet(),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/convit/convit_small.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint)
    return model


@register_model
def convit_base(pretrained=False, **kwargs):
    num_heads = 16
    kwargs['embed_dim'] *= num_heads
    
    model = VisionTransformerbase(
        num_heads=num_heads, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/convit/convit_base.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint)
    return model

def load_state_dict(model, ckpt, in_fc=15501):
    out_fc = 64500
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, out_fc)
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint["model"]
    # removing module if saved in lightning mode:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'model' in k:
            name = k[6:] # remove `model.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    # import pdb; pdb.set_trace()
    # del new_state_dict['patch_embed.proj.weight'], new_state_dict['patch_embed.proj.bias']
    model.load_state_dict(new_state_dict, strict = True)
    print('loading checkpoints weights \n')
    model.head = nn.Linear(num_ftrs, in_fc)
    del checkpoint, state_dict, new_state_dict

    return model

@register_model
def convit_base_patch(pretrained=False, **kwargs):
    num_heads = 16
    kwargs['embed_dim'] *= num_heads
    
    model = VisionTransformer(
        num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/convit/convit_base.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint)
    return model