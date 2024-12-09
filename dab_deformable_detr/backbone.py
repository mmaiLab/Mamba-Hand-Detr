# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules with the option to use ResNet or Swin Transformer.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from utils.miscdetr import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from .swin_transformer import SwinTransformer
from .mamba_vision import MambaVision
from .spatial_mamba import Backbone_SpatialMamba

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rsqrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class SwinBackbone(nn.Module):
    """Swin Transformer backbone."""
    def __init__(self, cfg, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        # Build the Swin Transformer model
        self.swin = SwinTransformer(
            img_size=cfg.input_size,
            patch_size=cfg.patch_size,
            in_chans=3,
            embed_dim=cfg.embed_dim,
            depths=cfg.depths,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            qk_scale=cfg.qk_scale,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            norm_layer=nn.LayerNorm,
            ape=cfg.ape,
            patch_norm=cfg.patch_norm,
            use_checkpoint=cfg.use_checkpoint,
        )
        # Freeze parameters if needed
        for name, parameter in self.swin.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)
        # Set return layers
        if return_interm_layers:
            self.return_layers = ["0", "1", "2", "3"]
        else:
            self.return_layers = ["3"]
        # Get output channels and strides
        self.num_channels = [cfg.embed_dim*2, cfg.embed_dim*4, cfg.embed_dim*8, cfg.embed_dim*8]
        #self.num_channels = [self.swin.embed_dim * 2 ** i for i in range(len(self.return_layers))]
        self.strides = [8, 16, 32, 32]

        #self.strides = [cfg.patch_size * (2 ** i) for i in range(len(self.return_layers))]

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors  # x is of shape [B, C, H, W]
        xs = self.swin.forward_features(x)  # xs is a dict of feature maps
        out: Dict[str, NestedTensor] = {}
        m = tensor_list.mask
        assert m is not None

        for name, x in xs.items():
            if name in self.return_layers:
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                out[name] = NestedTensor(x, mask)
        return out

class MambaVisionBackbone(nn.Module):
    """Mamba Vision backbone."""
    def __init__(self, cfg, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        # Build the MambaVision model
        self.mamba = MambaVision(
            dim=cfg.dim,
            in_dim=cfg.in_dim,
            depths=cfg.depths,
            window_size=cfg.window_size,
            mlp_ratio=cfg.mlp_ratio,
            num_heads=cfg.num_heads,
            drop_path_rate=cfg.drop_path_rate,
            in_chans=cfg.in_chans,
            num_classes=cfg.num_classes,
            qkv_bias=cfg.qkv_bias,
            qk_scale=cfg.qk_scale,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            layer_scale=cfg.layer_scale,
            layer_scale_conv=cfg.layer_scale_conv,
        )
        # freeze parameters(if needed)
        for name, parameter in self.mamba.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)
        # set return layers
        if return_interm_layers:
            self.return_layers = ["0", "1", "2", "3"]
        else:
            self.return_layers = ["3"]
        # get channels and strides of outputs
        self.num_channels = [cfg.dim*2, cfg.dim*4, cfg.dim*8, cfg.dim*8]
        self.strides = [4, 8, 16, 32]  

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors  # x's shape is [b, c, w, h]
        xs = {}  # save the output of each layer
        x = self.mamba.patch_embed(x)
        for idx, level in enumerate(self.mamba.levels):
            x = level(x)
            if str(idx) in self.return_layers:
                xs[str(idx)] = x
        out: Dict[str, NestedTensor] = {}
        m = tensor_list.mask
        assert m is not None

        for name, x in xs.items():
            if name in self.return_layers:
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                out[name] = NestedTensor(x, mask)
        return out

class SpatialMambaBackbone(nn.Module):
    """Spatial Mamba backbone."""
    def __init__(self, cfg, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        # Build the SpatialMamba model
        self.spatial_mamba = Backbone_SpatialMamba(
            img_size=cfg.input_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            dims=cfg.dims,
            depths=cfg.depths,
            num_classes=cfg.num_classes,
            mlp_ratio=cfg.mlp_ratio,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            use_checkpoint=cfg.use_checkpoint,
        )
        # Freeze parameters if needed
        for name, parameter in self.spatial_mamba.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)
        # Set return layers
        if return_interm_layers:
            self.return_layers = ["0", "1", "2", "3"]
        else:
            self.return_layers = ["3"]
        # Get output channels and strides
        self.num_channels = [cfg.dims*2, cfg.dims*4, cfg.dims*8, cfg.dims*8]
        self.strides = [4, 8, 16, 32]  # Assuming the same as in SwinBackbone

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors  # x's shape is [B, C, H, W]
        xs = self.spatial_mamba(x)  # xs is a list of feature maps
        out: Dict[str, NestedTensor] = {}
        m = tensor_list.mask
        assert m is not None

        for idx, x in enumerate(xs):
            name = str(idx)
            if name in self.return_layers:
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                out[name] = NestedTensor(x, mask)
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(cfg):
    position_embedding = build_position_encoding(cfg)
    train_backbone = cfg.lr_backbone > 0
    return_interm_layers = cfg.masks or (cfg.num_feature_levels > 1)
    backbone_type = cfg.backbone_type  # type of backbone

    if backbone_type == 'resnet':
        backbone = Backbone(cfg.backbone, train_backbone, return_interm_layers, cfg.dilation)
    elif backbone_type == 'swin_transformer':
        backbone = SwinBackbone(cfg, train_backbone, return_interm_layers)
    elif backbone_type == 'mamba_vision':
        backbone = MambaVisionBackbone(cfg, train_backbone, return_interm_layers)
    elif backbone_type == 'spatial_mamba':
        backbone = SpatialMambaBackbone(cfg, train_backbone, return_interm_layers)
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")

    model = Joiner(backbone, position_embedding)
    return model