import torch
from torch import nn

from .homography import bilinear_sampler
from .tools import plane_grid_2d, get_rot_2d, cam_to_pixel
from .spatial_gate import SpatialGate
from .homography import IPM
from .pointpillar import PointPillarEncoder

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18


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

        # self.sp_idx = [1, 3, 5, 11]
        # self.sp_idx = [6, 8, 10, 12, 14]
        self.sp_idx = []
        self.spatial_gates = []
        for i in range(len(self.sp_idx)):
            self.spatial_gates.append(SpatialGate())
        self.spatial_gates = nn.ModuleList(self.spatial_gates)

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
            if idx in self.sp_idx:
                x = self.spatial_gates[self.sp_idx.index(idx)](x)
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


class ViewFusionModule(nn.Module):
    def __init__(self, fv_size, bv_size, n_views=6):
        super(ViewFusionModule, self).__init__()
        self.n_views = n_views
        self.hw_mat = []
        # self.spatial_gates = []
        self.bv_size = bv_size
        fv_dim = fv_size[0] * fv_size[1]
        bv_dim = bv_size[0] * bv_size[1]
        for i in range(self.n_views):
            fc_transform = nn.Sequential(
                nn.Linear(fv_dim, bv_dim),
                # nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(bv_dim, bv_dim),
                # nn.BatchNorm1d(64),
                nn.ReLU()
            )
            self.hw_mat.append(fc_transform)
            # self.spatial_gates.append(SpatialGate())
        self.hw_mat = nn.ModuleList(self.hw_mat)
        # self.spatial_gates = nn.ModuleList(self.spatial_gates)

    def forward(self, feat):
        B, N, C, H, W = feat.shape
        feat = feat.view(B, N, C, H*W)
        outputs = []
        for i in range(N):
            output = self.hw_mat[i](feat[:, i]).view(B, C, self.bv_size[0], self.bv_size[1])
            # output = self.spatial_gates[i](output)
            outputs.append(output)
        outputs = torch.stack(outputs, 1)
        return outputs


class VPNModel(nn.Module):
    def __init__(self, outC, camC=64, instance_seg=True, embedded_dim=16, extrinsic=False, lidar=False, xbound=None, ybound=None, zbound=None):
        super(VPNModel, self).__init__()
        self.camC = camC
        self.extrinsic = extrinsic
        self.downsample = 16

        ipm_xbound = [-60, 60, 0.6]
        ipm_ybound = [-30, 30, 0.6]
        self.ipm = IPM(ipm_xbound, ipm_ybound, N=6, C=camC)

        self.camencode = CamEncode(camC)
        self.view_fusion = ViewFusionModule(fv_size=(8, 22), bv_size=(8, 22))
        self.up_sampler = nn.Upsample(size=(200, 400), mode='bilinear', align_corners=True)
        self.lidar = lidar
        if lidar:
            self.pp = PointPillarEncoder(128, xbound, ybound, zbound)
            self.bevencode = BevEncode(inC=camC+128, outC=outC, instance_seg=instance_seg, embedded_dim=embedded_dim)
        else:
            self.bevencode = BevEncode(inC=camC, outC=outC, instance_seg=instance_seg, embedded_dim=embedded_dim)


    def get_Ks_RTs_and_post_RTs(self, intrins, rots, trans, post_rots, post_trans):
        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        # Ks[:, :, :3, :3] = intrins

        Rs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Rs[:, :, :3, :3] = rots.transpose(-1, -2).contiguous()
        Ts = torch.eye(4, device=trans.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts

        post_RTs = None
        # post_RTs = torch.eye(4, device=post_rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        # post_RTs[:, :, :3, :3] = post_rots
        # post_RTs[:, :, :3, 3] = post_trans

        # if self.cam_encoding:
        #     scale = torch.Tensor([
        #         [1/self.downsample, 0, 0, 0],
        #         [0, 1/self.downsample, 0, 0],
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 1]
        #     ]).cuda()
        #     post_RTs = scale @ post_RTs

        return Ks, RTs, post_RTs

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, imH//self.downsample, imW//self.downsample)
        return x

    def forward(self, points, points_mask, x, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll):
        x = self.get_cam_feats(x)
        x = self.view_fusion(x)
        if self.extrinsic:
            Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)
            topdown = self.ipm(x, Ks, RTs, translation, yaw_pitch_roll, post_RTs)
        else:
            topdown = x.mean(1)
        topdown = self.up_sampler(topdown)
        if self.lidar:
            lidar_feature = self.pp(points, points_mask)
            topdown = torch.cat([topdown, lidar_feature], dim=1)
        return self.bevencode(topdown)


class TemporalVPNet(nn.Module):
    def __init__(self, xbound, ybound, outC, camC=64, instance_seg=True, embedded_dim=16):
        super(TemporalVPNet, self).__init__()
        self.xbound = xbound
        self.ybound = ybound
        self.camC = camC
        self.downsample = 16

        self.camencode = CamEncode(camC)
        self.view_fusion = ViewFusionModule(fv_size=(8, 22), bv_size=(8, 22))
        self.up_sampler = nn.Upsample(size=(200, 400), mode='bilinear', align_corners=True)
        self.bevencode = BevEncode(inC=camC, outC=outC, instance_seg=instance_seg, embedded_dim=embedded_dim)

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
        grid = grid.view(B*(T-1), 2, H, W).permute(0, 2, 3, 1).contiguous()
        grid = cam_to_pixel(grid, self.xbound, self.ybound)
        topdown = topdown.permute(0, 1, 3, 4, 2).contiguous()
        prev_topdown = topdown[:, 1:]
        warped_prev_topdown = bilinear_sampler(prev_topdown.reshape(B*(T-1), H, W, C), grid).view(B, T-1, H, W, C)
        topdown = torch.cat([topdown[:, 0].unsqueeze(1), warped_prev_topdown], axis=1)
        topdown = topdown.view(B, T, H, W, C)
        topdown = topdown.max(1)[0]
        topdown = topdown.permute(0, 3, 1, 2).contiguous()
        return topdown

    def forward(self, points, points_mask, x, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll):
        x = self.get_cam_feats(x)
        B, T, N, C, h, w = x.shape
        x = x.view(B*T, N, C, h, w)
        topdown = self.view_fusion(x)
        topdown = self.up_sampler(topdown)
        _, C, H, W = topdown.shape
        topdown = topdown.view(B, T, C, H, W)
        topdown = self.temporal_fusion(topdown, translation, yaw_pitch_roll[..., 0])
        return self.bevencode(topdown)
