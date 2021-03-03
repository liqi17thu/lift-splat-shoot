import torch
from torch import nn

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .homography import IPM, bilinear_sampler
from .tools import plane_grid_2d, get_rot_2d, cam_to_pixel


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


class ViewFusionModule(nn.Module):
    def __init__(self, fv_size, bv_size, n_views=6):
        super(ViewFusionModule, self).__init__()
        self.n_views = n_views
        self.hw_mat = []
        self.bv_size = bv_size
        fv_dim = fv_size[0] * fv_size[1]
        bv_dim = bv_size[0] * bv_size[1]
        for i in range(self.n_views):
            fc_transform = nn.Sequential(
                nn.Linear(fv_dim, bv_dim),
                nn.ReLU(),
                nn.Linear(bv_dim, bv_dim),
                nn.ReLU()
            )
            self.hw_mat.append(fc_transform)
        self.hw_mat = nn.ModuleList(self.hw_mat)

    def forward(self, feat):
        B, N, C, H, W = feat.shape
        feat = feat.view(B, N, C, H*W)
        output = self.hw_mat[0](feat[:, 0])
        for i in range(1, N):
            output += self.hw_mat[i](feat[:, i])
        output = output.view(B, C, self.bv_size[0], self.bv_size[1])
        return output


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
    def __init__(self, inC, outC, instance_seg=True, embedded_dim=16):
        super(BevEncode, self).__init__()
        self.instance_seg = instance_seg

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

        if instance_seg:
            self.up1_embedded = Up(64 + 256, 256, scale_factor=4)
            self.up2_embedded = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, embedded_dim, kernel_size=1, padding=0),
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x2 = self.layer3(x)

        x = self.up1(x2, x1)
        x = self.up2(x)

        if self.instance_seg:
            x_embedded = self.up1_embedded(x2, x1)
            x_embedded = self.up2_embedded(x_embedded)
            return x, x_embedded
        else:
            return x


class HDMapNet(nn.Module):
    def __init__(self, xbound, ybound, zbound, outC, camC=64, instance_seg=True, embedded_dim=16):
        super(HDMapNet, self).__init__()
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.d = int((zbound[1] - zbound[0]) / zbound[2]) + 1
        self.camC = camC
        self.downsample = 16
        self.ipm = IPM(xbound, ybound, zbound)
        # self.ipm = IPM(xbound, ybound, zbound, visual=True)

        fv_size = (8, 22)
        bv_size = (20, 40)
        self.view_fusion = ViewFusionModule(fv_size, bv_size, n_views=6)
        self.depth_esti = nn.Sequential(
            nn.Upsample(scale_factor=10, mode='bilinear', align_corners=True),
            nn.Conv2d(camC, self.d, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.d),
            nn.Sigmoid(),
        )
        self.prior = torch.tensor([0, 0.0, 0.4, 0.0, 0]).cuda().reshape(1, 5, 1, 1)

        self.camencode = CamEncode(camC)
        self.bevencode = BevEncode(inC=camC, outC=outC, instance_seg=instance_seg, embedded_dim=embedded_dim)

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, imH//self.downsample, imW//self.downsample)
        return x

    def get_Ks_RTs_and_post_RTs(self, intrins, rots, trans, post_rots, post_trans):
        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ks[:, :, :3, :3] = intrins

        Rs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Rs[:, :, :3, :3] = rots.transpose(-1, -2)
        Ts = torch.eye(4, device=trans.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts

        post_RTs = torch.eye(4, device=post_rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        post_RTs[:, :, :3, :3] = post_rots
        post_RTs[:, :, :3, 3] = post_trans

        scale = torch.Tensor([
            [1/self.downsample, 0, 0, 0],
            [0, 1/self.downsample, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]).cuda()
        post_RTs = scale @ post_RTs

        return Ks, RTs, post_RTs

    def forward(self, x, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll):
        x = self.get_cam_feats(x)

        z = self.view_fusion(x)
        z = self.depth_esti(z)  # B, D, H, W
        z = 10 * (z + self.prior)
        z = z.softmax(1)

        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)
        topdown = self.ipm(x, Ks, RTs, post_RTs)  # [B, C, D, H, W]
        # import ipdb; ipdb.set_trace()
        B, C, D, H, W = topdown.shape
        z = z.unsqueeze(1).repeat(1, C, 1, 1, 1)
        topdown = (topdown * z).sum(2)

        return self.bevencode(topdown)


class TemporalHDMapNet(HDMapNet):
    def __init__(self, xbound, ybound, zbound, outC, camC=64, instance_seg=True, embedded_dim=16):
        super(TemporalHDMapNet, self).__init__(xbound, ybound, zbound, outC, camC, instance_seg, embedded_dim)

    def get_cam_feats(self, x):
        """Return B x T x N x H/downsample x W/downsample x C
        """
        B, T, N, C, imH, imW = x.shape

        x = x.view(B*T*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, T, N, self.camC, imH//self.downsample, imW//self.downsample)
        return x

    def temporal_fusion(self, topdown, translation, yaw):
        B, T, C, H, W = topdown.shape

        if T == 1:
            return topdown[:, 0]

        grid = plane_grid_2d(self.xbound, self.ybound).view(1, 1, 2, H*W).repeat(B, T-1, 1, 1)
        rot0 = get_rot_2d(yaw[:, 1:])
        trans0 = translation[:, 1:, :2].view(B, T-1, 2, 1)
        rot1 = get_rot_2d(yaw[:, 0].view(B, 1).repeat(1, T-1))
        trans1 = translation[:, 0, :2].view(B, 1, 2, 1).repeat(1, T-1, 1, 1)
        grid = rot1.transpose(2, 3) @ grid
        grid = grid + trans1
        grid = grid - trans0
        grid = rot0 @ grid
        grid = grid.view(B*(T-1), 2, H, W).permute(0, 2, 3, 1)
        grid = cam_to_pixel(grid, self.xbound, self.ybound)
        topdown = topdown.permute(0, 1, 3, 4, 2)
        prev_topdown = topdown[:, 1:]
        warped_prev_topdown = bilinear_sampler(prev_topdown.reshape(B*(T-1), H, W, C), grid).view(B, T-1, H, W, C)
        topdown = torch.cat([topdown[:, 0].unsqueeze(1), warped_prev_topdown], axis=1)
        topdown = topdown.view(B, T, H, W, C)
        topdown = topdown.max(1)[0]
        topdown = topdown.permute(0, 3, 1, 2)
        return topdown

    def forward(self, x, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll):
        x = self.get_cam_feats(x)
        B, T, N, C, h, w = x.shape

        x = x.view(B*T, N, C, h, w)
        intrins = intrins.view(B*T, N, 3, 3)
        rots = rots.view(B*T, N, 3, 3)
        trans = trans.view(B*T, N, 3)
        post_rots = post_rots.view(B*T, N, 3, 3)
        post_trans = post_trans.view(B*T, N, 3)
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)
        topdown = self.ipm(x, Ks, RTs, post_RTs)
        _, C, H, W = topdown.shape
        topdown = topdown.view(B, T, C, H, W)
        topdown = self.temporal_fusion(topdown, translation, yaw_pitch_roll[..., 0])
        return self.bevencode(topdown)
