import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath

class GatedCNNBlock(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args: 
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve practical efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and 
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(self, dim, 
                 expansion_ratio=8/3, 
                 kernel_size=7, 
                 conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(
            conv_channels, conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=conv_channels
        )
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        x: [B, H, W, C]
        returns: [B, H, W, C]
        """
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2)
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1)

        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        return x + shortcut

class StemLayer(nn.Module):
    def __init__(self, in_chans=3, stem_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, stem_channels // 2, kernel_size=3, stride=2, padding=1)
        self.ln1 = nn.LayerNorm(stem_channels // 2, elementwise_affine=True)
        self.conv2 = nn.Conv2d(stem_channels // 2, stem_channels, kernel_size=3, stride=2, padding=1)
        self.ln2 = nn.LayerNorm(stem_channels, elementwise_affine=True)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, 3, H, W]
        # -- conv1 --
        x = self.conv1(x)                       # => [B, stem_channels/2, H/2, W/2]
        x = x.permute(0, 2, 3, 1)               # => [B, H/2, W/2, stem_channels/2]
        x = self.ln1(x)
        x = x.permute(0, 3, 1, 2)               # => [B, stem_channels/2, H/2, W/2]
        x = self.act(x)

        # -- conv2 --
        x = self.conv2(x)                       # => [B, stem_channels, H/4, W/4]
        x = x.permute(0, 2, 3, 1)               # => [B, H/4, W/4, stem_channels]
        x = self.ln2(x)
        x = x.permute(0, 3, 1, 2)               # => [B, stem_channels, H/4, W/4]
        x = self.act(x)
        return x


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.ln = nn.LayerNorm(out_channels, elementwise_affine=True)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, C_in, H, W]
        x = self.conv(x)                        # => [B, C_out, H/2, W/2]
        x = x.permute(0, 2, 3, 1)               # => [B, H/2, W/2, C_out]
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)               # => [B, C_out, H/2, W/2]
        x = self.act(x)
        return x


class GatedCNNBlockAdaptor(nn.Module):
    def __init__(self, 
                 in_channels, 
                 expansion_ratio=8/3, 
                 kernel_size=7, 
                 conv_ratio=1.0,
                 drop_path=0.0):
        super().__init__()
        self.dim = in_channels
        self.block = GatedCNNBlock(
            dim=self.dim,
            expansion_ratio=expansion_ratio,
            kernel_size=kernel_size,
            conv_ratio=conv_ratio,
            drop_path=drop_path,
        )

    def forward(self, x):
        # x: [B, C, H, W]
        # => [B, H, W, C]
        x_in = x.permute(0, 2, 3, 1).contiguous()
        x_out = self.block(x_in)
        # => [B, C, H, W]
        x_out = x_out.permute(0, 3, 1, 2).contiguous()
        return x_out


class MambaOut(nn.Module):
    def __init__(self,
                 in_chans=3,
                 dim=64,
                 depths=[2, 2, 2, 2],
                 drop_path_rate=0.0,
                 expansion_ratio=8/3,
                 kernel_size=7,
                 conv_ratio=1.0):
       
        super().__init__()
        dims = [dim*2, dim*4, dim*8, dim*8]
        assert len(dims) == 4 and len(depths) == 4
        # --- 1) Stem ---
        self.stem = StemLayer(in_chans, dim)

        total_depth = sum(depths)
        dpr_list = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        cur = 0
        block_list = []
        for _ in range(depths[0]):
            block_list.append(
                GatedCNNBlockAdaptor(
                    in_channels=dim,
                    expansion_ratio=expansion_ratio,
                    kernel_size=kernel_size,
                    conv_ratio=conv_ratio,
                    drop_path=dpr_list[cur]
                )
            )
            cur += 1
        self.stage1_blocks = nn.Sequential(*block_list)
        self.stage1_downsample = DownsampleLayer(dim, dims[0])

        block_list = []
        for _ in range(depths[1]):
            block_list.append(
                GatedCNNBlockAdaptor(
                    in_channels=dims[0],
                    expansion_ratio=expansion_ratio,
                    kernel_size=kernel_size,
                    conv_ratio=conv_ratio,
                    drop_path=dpr_list[cur]
                )
            )
            cur += 1
        self.stage2_blocks = nn.Sequential(*block_list)
        self.stage2_downsample = DownsampleLayer(dims[0], dims[1])

        block_list = []
        for _ in range(depths[2]):
            block_list.append(
                GatedCNNBlockAdaptor(
                    in_channels=dims[1],
                    expansion_ratio=expansion_ratio,
                    kernel_size=kernel_size,
                    conv_ratio=conv_ratio,
                    drop_path=dpr_list[cur]
                )
            )
            cur += 1
        self.stage3_blocks = nn.Sequential(*block_list)
        self.stage3_downsample = DownsampleLayer(dims[1], dims[2])

        block_list = []
        for _ in range(depths[3]):
            block_list.append(
                GatedCNNBlockAdaptor(
                    in_channels=dims[2],
                    expansion_ratio=expansion_ratio,
                    kernel_size=kernel_size,
                    conv_ratio=conv_ratio,
                    drop_path=dpr_list[cur]
                )
            )
            cur += 1
        self.stage4_blocks = nn.Sequential(*block_list)
        self.stage4_downsample = nn.Identity()

    def forward(self, x):
        """
        Returns:
          dict with 4 keys: "0","1","2","3"
            - "0": [B, dims[0], H/8,  W/8 ]
            - "1": [B, dims[1], H/16, W/16]
            - "2": [B, dims[2], H/32, W/32]
            - "3": [B, dims[2], H/32, W/32]
        """
        # -- stem --
        x = self.stem(x)  # => [B, stem_channels, H/4, W/4]

        # -- stage1 --
        x = self.stage1_blocks(x)        # => [B, dim, H/4, W/4]
        x = self.stage1_downsample(x)    # => [B, dims[0], H/8, W/8]
        out1 = x

        # -- stage2 --
        x = self.stage2_blocks(x)        # => [B, dims[0], H/8, W/8]
        x = self.stage2_downsample(x)    # => [B, dims[1], H/16, W/16]
        out2 = x

        # -- stage3 --
        x = self.stage3_blocks(x)        # => [B, dims[1], H/16, W/16]
        x = self.stage3_downsample(x)    # => [B, dims[2], H/32, W/32]
        out3 = x

        # -- stage4 (no downsample) --
        x = self.stage4_blocks(x)        # => [B, dims[2], H/32, W/32]
        x = self.stage4_downsample(x)    # => Identity => [B, dims[2], H/32, W/32]
        out4 = x

        return {
            "0": out1,
            "1": out2,
            "2": out3,
            "3": out4,
        }


# ------------------
#  简单测试 (可选)
# ------------------
if __name__ == "__main__":
    model = MambaOut(in_chans=3, dim=64, depths=[2,2,2,2])
    dummy_input = torch.randn(1, 3, 224, 224)
    outputs = model(dummy_input)
    for k, v in outputs.items():
        print(f"Output {k}: shape = {v.shape}")
    
    # 预期输出 (示例):
    # Output 0: shape = [1, 128, 28, 28]  (H/8,  W/8 )
    # Output 1: shape = [1, 256, 14, 14]  (H/16, W/16)
    # Output 2: shape = [1, 512, 7,  7 ]  (H/32, W/32)
    # Output 3: shape = [1, 512, 7,  7 ]  (H/32, W/32)
