import numpy as np

import torch
from torch import nn

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .homography import IPM

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, C):
        super(CamEncode, self).__init__()
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, self.C)

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        return self.get_eff_depth(x)


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class HDMapNet(nn.Module):
    def __init__(self, xbound, ybound, outC, camC=64):
        super(HDMapNet, self).__init__()
        self.xbound = xbound
        self.ybound = ybound
        self.camC = camC
        self.downsample = 16
        self.ipm = IPM(xbound, ybound, N=6, C=camC)

        self.camencode = CamEncode(camC)
        self.bevencode = BevEncode(inC=camC, outC=outC)

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, imH//self.downsample, imW//self.downsample)
        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans, z, yaw, pitch, roll):
        # x = self.get_cam_feats(x)
        with open('master_input.npy', 'wb') as f:
            np.save(f, x.detach().cpu().numpy())

        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device, dtype=torch.double).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        # Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ks[:, :, :3, :3] = intrins

        Rs = torch.eye(4, device=rots.device, dtype=torch.double).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        # Rs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Rs[:, :, :3, :3] = rots.transpose(-1, -2)
        Ts = torch.eye(4, device=trans.device, dtype=torch.double).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        # Ts = torch.eye(4, device=trans.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts

        post_RTs = torch.eye(4, device=post_rots.device, dtype=torch.double).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        # post_RTs = torch.eye(4, device=post_rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        post_RTs[:, :, :3, :3] = post_rots
        post_RTs[:, :, :3, 3] = post_trans

        # scale = torch.Tensor([
        #     [1/self.downsample, 0, 0, 0],
        #     [0, 1/self.downsample, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]
        # ]).cuda()
        # post_RTs = scale @ post_RTs

        topdown = self.ipm(x, Ks, RTs, z, yaw, pitch, roll, post_RTs)

        with open('master_cam_topdown.npy', 'wb') as f:
            np.save(f, topdown.detach().cpu().numpy())

        import ipdb; ipdb.set_trace()

        return self.bevencode(topdown)
