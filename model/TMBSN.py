import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class super_shift(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, k: int = 1) -> torch.Tensor:
        x = F.pad(x, (0, k, k, 0)) # left right top bottom
        x = x[:, :, :-k, k:]
        return x

class shift_u(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, k: int = 1) -> torch.Tensor:
        x = F.pad(x, (0, 0, k, 0)) # left right top bottom
        x = x[:, :, :-k, :]
        return x

class shift_r(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, k: int = 1) -> torch.Tensor:
        x = F.pad(x, (0, k, 0, 0)) # left right top bottom
        x = x[:, :, :, k:]
        return x

class Conv(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, k: int):
        super().__init__(in_channels, out_channels, kernel_size=k,
                         stride=1, padding=k//2, dilation=1, groups=1, bias=True)

        mask = torch.ones_like(self.weight)
        tri2d = torch.triu(torch.ones(k, k, dtype=mask.dtype, device=mask.device))
        with torch.no_grad():
            mask *= tri2d
        self.register_buffer("mask", mask, persistent=True)

        self.weight.register_hook(lambda g: g * self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.mask
        return F.conv2d(x, w, self.bias, stride=1, padding=self.padding)

class rotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x90 = x.transpose(2,3).flip(3)
        x180 = x.flip(2).flip(3)
        x270 = x.transpose(2,3).flip(2)
        x = torch.cat((x,x90,x180,x270), dim=0)
        return x

class unrotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x0, x90, x180, x270 = torch.chunk(x, 4, dim=0)
        x90 = x90.transpose(2,3).flip(2)
        x180 = x180.flip(2).flip(3)
        x270 = x270.transpose(2,3).flip(3)
        x = torch.cat((x0,x90,x180,x270), dim=1)
        return x


class ResBlock(nn.Module):
    def __init__(self, feat_channels):
        super(ResBlock, self).__init__()
        self.conv1 = Conv(feat_channels, feat_channels, k=3)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv(feat_channels, feat_channels, k=3)

    def forward(self, x):
        net1 = self.relu(self.conv1(x))
        out = self.conv2(net1)
        return out + x

class RIRBlock(nn.Module):
    def __init__(self, num, feat_channels):
        super(RIRBlock, self).__init__()
        self.res_blocks = nn.ModuleList([ResBlock(feat_channels) for _ in range(num)])
        self.conv_out = Conv(feat_channels, feat_channels, k=3)

    def forward(self, x):
        head = x
        for res_block in self.res_blocks:
            head = res_block(head)
        out = self.conv_out(head)
        return out + x



class Blind_Net(nn.Module):
    def __init__(self, in_channels=3, feat_channels=48, block_nums=5):
        super().__init__()

        self.conv1 = Conv(in_channels, feat_channels, k=3)
        self.conv2 = Conv(feat_channels, feat_channels, k=3)
        self.RIR_blocks = nn.ModuleList([RIRBlock(block_nums, feat_channels) for _ in range(block_nums)])

    def forward(self, input):

        x1 = self.conv1(input)

        head = x1

        for rir_block in self.RIR_blocks:
            head = rir_block(head)

        x2 = self.conv2(head)

        x3 = x1 + x2

        return x3


class TMBSN(nn.Module):
    def __init__(self, in_channels=3, feat_channels=48, block_nums=5):
        super().__init__()
        self.rotate = rotate()
        self.unet = Blind_Net(in_channels=in_channels, feat_channels=feat_channels, block_nums=block_nums)
        self.shift_u = shift_u()
        self.shift_r = shift_r()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(feat_channels*8, feat_channels*8, 1)
        self.nin_B = nn.Conv2d(feat_channels*8, feat_channels*2, 1)
        self.nin_C = nn.Conv2d(feat_channels*2, in_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, hole_size=1):
        x = self.rotate(x)

        x = self.unet(x)
        if hole_size > 0:
            x_u = self.shift_u(x, hole_size)
            x_r = self.shift_r(x, hole_size)
            x = torch.cat((x_u,x_r), dim=1)
        if hole_size == 0:
            x_u = x
            x_r = x
            x = torch.cat((x_u,x_r), dim=1)
        x = self.unrotate(x)

        x = self.relu(self.nin_A(x))
        x = self.relu(self.nin_B(x))
        x = self.nin_C(x)

        return x

    @torch.no_grad()
    def forward_outs(self, x: torch.Tensor, h_set) -> torch.Tensor:
        x_rot = self.rotate(x)            # [4B, C, H, W]
        feats = self.unet(x_rot)          # [4B, F, H, W]

        outs = []
        for hs in h_set:
            if hs > 0:
                x_u = self.shift_u(feats, hs)
                x_r = self.shift_r(feats, hs)
                x_cat = torch.cat((x_u, x_r), dim=1)
            else:  # hs == 0
                x_cat = torch.cat((feats, feats), dim=1)

            x_unrot = self.unrotate(x_cat)          # [B, 8F, H, W]
            z = self.relu(self.nin_A(x_unrot))
            z = self.relu(self.nin_B(z))
            z = self.nin_C(z)                       # [B, C, H, W]
            outs.append(z)

        return outs
