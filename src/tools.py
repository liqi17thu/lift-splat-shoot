"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from pyquaternion import Quaternion
from PIL import Image
from functools import reduce

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from .metric import LaneSegMetric


# import matplotlib as mpl
# mpl.use('Agg')

import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion.map_api import NuScenesMap


def plane_grid_2d(xbound, ybound):
    xmin, xmax = xbound[0], xbound[1]
    num_x = int((xbound[1] - xbound[0]) / xbound[2])
    ymin, ymax = ybound[0], ybound[1]
    num_y = int((ybound[1] - ybound[0]) / ybound[2])

    y = torch.linspace(xmin, xmax, num_x).cuda()
    x = torch.linspace(ymin, ymax, num_y).cuda()
    y, x = torch.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()

    coords = torch.stack([x, y], axis=0)
    return coords


def cam_to_pixel(points, xbound, ybound):
    new_points = torch.zeros_like(points)
    new_points[..., 0] = (points[..., 0] - xbound[0]) / xbound[2]
    new_points[..., 1] = (points[..., 1] - ybound[0]) / ybound[2]
    return new_points


def get_rot_2d(yaw):
    sin_yaw = torch.sin(yaw)
    cos_yaw = torch.cos(yaw)
    rot = torch.zeros(list(yaw.shape) + [2, 2]).cuda()
    rot[..., 0, 0] = cos_yaw
    rot[..., 0, 1] = sin_yaw
    rot[..., 1, 0] = -sin_yaw
    rot[..., 1, 1] = cos_yaw
    return rot


def get_lidar_data(nusc, sample_rec, nsweeps, min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((5, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                       inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                           Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
        times = time_lag * np.ones((1, current_pc.nbr_points()))

        new_points = np.concatenate((current_pc.points, times), 0)
        points = np.concatenate((points, new_points), 1)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points


def ego_to_cam(points, rot, trans, intrins):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    points = points - trans.unsqueeze(1)
    points = rot.permute(1, 0).matmul(points)

    points = intrins.matmul(points)
    points[:2] /= points[2:3]

    return points


def cam_to_ego(points, rot, trans, intrins):
    """Transform points (3 x N) from pinhole camera with depth
    to the ego frame
    """
    points = torch.cat((points[:2] * points[2:3], points[2:3]))
    points = intrins.inverse().matmul(points)

    points = rot.matmul(points)
    points += trans.unsqueeze(1)

    return points


def get_only_in_img_mask(pts, H, W):
    """pts should be 3 x N
    """
    return (pts[2] > 0) & \
           (pts[0] > 1) & (pts[0] < W - 1) & \
           (pts[1] > 1) & (pts[1] < H - 1)


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):

    # img = img.resize((352, 128))
    # post_rot[0, 0] *= 352/1600
    # post_rot[1, 1] *= 128/900

    # return img, post_rot, post_tran

    # adjust image
    img = img.resize(resize_dims)
    # img = img.crop(crop)
    # if flip:
    #     img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    # img = img.rotate(rotate)

    rot_resize = torch.Tensor([[resize[0], 0],
                               [0, resize[1]]])
    post_rot = rot_resize @ post_rot
    post_tran = rot_resize @ post_tran

    # if flip:
    #     rot_flip = torch.Tensor([[-1, 0],
    #                                 [0, 1]])
    #     tran_flip = torch.Tensor([resize_dims[0], 0])
    #     post_rot = rot_flip @ post_rot
    #     post_tran = rot_flip @ post_tran + tran_flip

    # rot_rot = get_rot(rotate / 180 * np.pi)
    # tran_flip = torch.Tensor(resize_dims) / 2
    # post_rot = rot_rot @ post_rot
    # post_tran = rot_rot @ post_tran + tran_flip

    return img, post_rot, post_tran

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0],
                          [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


random_erasing = torchvision.transforms.RandomErasing()

denormalize_img = torchvision.transforms.Compose((
    NormalizeInverse(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    torchvision.transforms.ToPILImage(),
))

normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce == 'mean':
            return torch.mean(F_loss)
        elif self.reduce == 'sum':
            return torch.sum(F_loss)
        else:
            raise NotImplementedError


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss


class DiscriminativeLoss(nn.Module):
    def __init__(self, embed_dim, delta_v, delta_d):
        super(DiscriminativeLoss, self).__init__()
        self.embed_dim = embed_dim
        self.delta_v = delta_v
        self.delta_d = delta_d

    def forward(self, embedding, seg_gt):
        bs = embedding.shape[0]

        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        for b in range(bs):
            embedding_b = embedding[b]  # (embed_dim, H, W)
            seg_gt_b = seg_gt[b]

            labels = torch.unique(seg_gt_b)
            labels = labels[labels != 0]
            num_lanes = len(labels)
            if num_lanes == 0:
                # please refer to issue here: https://github.com/harryhan618/LaneNet/issues/12
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = (seg_gt_b == lane_idx)
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i]

                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)

                # ---------- var_loss -------------
                var_loss = var_loss + torch.mean(F.relu(torch.norm(embedding_i-mean_i.reshape(self.embed_dim, 1), dim=0) - self.delta_v) ** 2) / num_lanes
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, self.embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, self.embed_dim)
                dist = torch.norm(centroid_mean1-centroid_mean2, dim=2)  # shape (num_lanes, num_lanes)
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype, device=dist.device) * self.delta_d  # diagonal elements are 0, now mask above delta_d

                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_d)**2) / (num_lanes * (num_lanes-1)) / 2

            # reg_loss is not used in original paper
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / bs
        dist_loss = dist_loss / bs
        reg_loss = reg_loss / bs
        return var_loss, dist_loss, reg_loss


def get_batch_iou(preds, binimgs):
    """Assumes preds has NOT been sigmoided yet
    """
    with torch.no_grad():
        pred = (preds > 0)
        tgt = binimgs.bool()
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()
    return intersect, union, intersect / union if (union > 0) else 1.0


def get_accuracy_precision_recall(preds, binimgs):
    with torch.no_grad():
        preds = (preds > 0)
        tgt = binimgs.bool()
        tot = preds.shape.numel()
        cor = (preds == tgt).sum().float().item()
        tp = (preds & tgt).sum().float().item()
        fp = (preds & ~tgt).sum().float().item()
        fn = (~preds & tgt).sum().float().item()
    return tot, cor, tp, fp, fn


def label_onehot_decoding(onehot):
    return torch.argmax(onehot, axis=0)


def label_onehot_encoding(label, num_classes=4):
    H, W = label.shape
    onehot = torch.zeros((num_classes, H, W))
    onehot.scatter_(0, label[None].long(), 1)
    return onehot


def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def get_batch_iou_multi_class(preds, binimgs):
    intersects = []
    unions = []
    with torch.no_grad():
        preds = onehot_encoding(preds).bool()
        tgts = binimgs.bool()
        for i in range(preds.shape[1]):
            pred = preds[:, i]
            tgt = tgts[:, i]
            intersect = (pred & tgt).sum().float().item()
            union = (pred | tgt).sum().float().item()
            intersects.append(intersect)
            unions.append(union)
    intersects = np.array(intersects)
    unions = np.array(unions)
    return intersects, unions, intersects / (unions + 1e-7)


def get_accuracy_precision_recall_multi_class(preds, binimgs):
    tots = []
    cors = []
    tps = []
    fps = []
    fns = []
    with torch.no_grad():
        preds = onehot_encoding(preds).bool()
        tgts = binimgs.bool()
        for i in range(preds.shape[1]):
            pred = preds[:, i]
            tgt = tgts[:, i]
            tot = pred.shape.numel()
            cor = (pred == tgt).sum().float().item()
            tp = (pred & tgt).sum().float().item()
            fp = (pred & ~tgt).sum().float().item()
            fn = (~pred & tgt).sum().float().item()

            tots.append(tot)
            cors.append(cor)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)

    tots = np.array(tots)
    cors = np.array(cors)
    tps = np.array(tps)
    fps = np.array(fps)
    fns = np.array(fns)
    return tots, cors, tps, fps, fns, cors / tots, tps / (tps + fps + 1e-7), tps / (tps + fns + 1e-7)


def get_val_info(model, valloader, loss_fn, embedded_loss_fn, scale_seg=1.0, scale_var=1.0, scale_dist=1.0, use_tqdm=True):
    lane_seg_metric = LaneSegMetric()

    model.eval()
    total_seg_loss = 0.0
    total_reg_loss = 0.0
    total_dist_loss = 0.0
    total_var_loss = 0.0
    total_final_loss = 0.0
    total_intersect = None
    total_union = None
    total_pix = None
    total_cor = None
    total_tp = None
    total_fp = None
    total_fn = None
    total_CD = None
    print('running eval...')
    loader = tqdm(valloader) if use_tqdm else valloader
    with torch.no_grad():
        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll, binimgs, inst_mask = batch
            binimgs = binimgs.cuda()
            inst_mask = inst_mask.cuda()
            preds, embedded = model(allimgs.cuda(), rots.cuda(),
                          trans.cuda(), intrins.cuda(), post_rots.cuda(),
                          post_trans.cuda(), translation.cuda(), yaw_pitch_roll.cuda())
            onehot_preds = onehot_encoding(preds, dim=1)

            # loss
            bs = preds.shape[0]
            seg_loss = loss_fn(preds, binimgs).item() * bs
            var_loss, dist_loss, reg_loss = embedded_loss_fn(embedded, inst_mask)
            var_loss, dist_loss, reg_loss = var_loss.item() * bs, dist_loss.item() * bs, reg_loss.item() * bs
            final_loss = seg_loss * scale_seg + var_loss + scale_var + dist_loss * scale_dist

            total_seg_loss += seg_loss
            total_reg_loss += reg_loss
            total_dist_loss += dist_loss
            total_var_loss += var_loss
            total_final_loss += final_loss

            # iou
            intersect, union, _ = get_batch_iou_multi_class(preds, binimgs)
            tot, cor, tp, fp, fn, _, _, _ = get_accuracy_precision_recall_multi_class(preds, binimgs)
            CD = lane_seg_metric.semantic_mask_chamfer_dist(onehot_preds[:, 1:], binimgs[:, 1:])
            CD = CD.cpu().numpy()
            if total_intersect is None:
                total_intersect = intersect
                total_union = union
                total_pix = tot
                total_cor = cor
                total_tp = tp
                total_fp = fp
                total_fn = fn
                total_CD = CD
            else:
                total_intersect += intersect
                total_union += union
                total_pix += tot
                total_cor += cor
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_CD += CD

    model.train()
    return {
        'seg_loss': total_seg_loss / len(valloader.dataset),
        'reg_loss': total_reg_loss / len(valloader.dataset),
        'dist_loss': total_dist_loss / len(valloader.dataset),
        'var_loss': total_var_loss / len(valloader.dataset),
        'final_loss': total_final_loss / len(valloader.dataset),
        'iou': total_intersect / total_union,
        'accuracy': total_cor / total_pix,
        'precision': total_tp / (total_tp + total_fp),
        'recall': total_tp / (total_tp + total_fn),
        'chamfer_distance': total_CD / len(valloader.dataset),
    }


def add_ego(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084 / 2. + 0.5, W / 2.],
        [4.084 / 2. + 0.5, W / 2.],
        [4.084 / 2. + 0.5, -W / 2.],
        [-4.084 / 2. + 0.5, -W / 2.],
    ])
    pts = (pts - bx) / dx
    # pts[:, [0, 1]] = pts[:, [1, 0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')


def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                                       map_name=map_name) for map_name in [
                     "singapore-hollandvillage",
                     "singapore-queenstown",
                     "boston-seaport",
                     "singapore-onenorth",
                 ]}
    return nusc_maps


def plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

    poly_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,
                         50.0, poly_names, line_names)
    for name in poly_names:
        for la in lmap[name]:
            pts = (la - bx) / dx
            plt.fill(pts[:, 0], pts[:, 1], c=(1.00, 0.50, 0.31), alpha=0.2)
    for la in lmap['road_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 0], pts[:, 1], c=(0.0, 0.0, 1.0), alpha=0.5)
    for la in lmap['lane_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 0], pts[:, 1], c=(159. / 255., 0.0, 1.0), alpha=0.5)


def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names,
                                                 mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record['token']

            line = nmap.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
            )

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys
