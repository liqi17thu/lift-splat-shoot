"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import os
import cv2
import math
import numpy as np
import torch
from tqdm import tqdm
from pyquaternion import Quaternion
from PIL import Image
from shapely import affinity, ops
from shapely.geometry import LineString, MultiLineString, box, MultiPolygon, Polygon
from functools import reduce

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

# import matplotlib as mpl
# mpl.use('Agg')

import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion.map_api import NuScenesMap

from .metric import LaneSegMetric

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


def onehot_encoding_spread(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    one_hot.scatter_(dim, torch.clamp(max_idx-1, min=0), 1)
    one_hot.scatter_(dim, torch.clamp(max_idx-2, min=0), 1)
    one_hot.scatter_(dim, torch.clamp(max_idx+1, max=logits.shape[dim]-1), 1)
    one_hot.scatter_(dim, torch.clamp(max_idx+2, max=logits.shape[dim]-1), 1)

    return one_hot


def get_pred_top2_direction(direction, dim=1):
    direction = torch.softmax(direction, dim)
    idx1 = torch.argmax(direction, dim)
    idx1_onehot_spread = onehot_encoding_spread(direction, dim)
    idx1_onehot_spread = idx1_onehot_spread.bool()
    direction[idx1_onehot_spread] = 0
    idx2 = torch.argmax(direction, dim)
    direction = torch.stack([idx1, idx2], dim) - 1
    return direction


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

import random
from .postprocess import LaneNetPostProcessor

def get_val_info(model, valloader, loss_fn, embedded_loss_fn, direction_loss_fn, scale_seg=1.0, scale_var=1.0, scale_dist=1.0, angle_class=37, use_tqdm=True, eval_mAP=False):

    lane_seg_metric = LaneSegMetric()
    if eval_mAP:
        post_processor = LaneNetPostProcessor(dbscan_eps=1.5, postprocess_min_samples=50)
        thresholds = [1.33, 3.33, 6.67]

    color_map = []
    for i in range(30):
        color_map.append([random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)])

    model.eval()
    total_seg_loss = 0.0
    total_reg_loss = 0.0
    total_dist_loss = 0.0
    total_var_loss = 0.0
    total_direction_loss = 0.0
    total_final_loss = 0.0
    total_CD1 = None
    total_CD2 = None
    total_CD_num1 = None
    total_CD_num2 = None
    total_intersect = None
    total_union = None
    total_pix = None
    total_cor = None
    total_tp = None
    total_fp = None
    total_fn = None
    total_angle_diff = 0
    total_AP = 0
    print('running eval...')
    loader = tqdm(valloader) if use_tqdm else valloader
    with torch.no_grad():
        for batch in loader:
            points, points_mask, allimgs, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll, binimgs, inst_mask, direction_mask = batch
            binimgs = binimgs[..., -400:]
            inst_mask = inst_mask[..., -400:]
            direction_mask = direction_mask[..., -400:]

            binimgs = binimgs.cuda()
            inst_mask = inst_mask.cuda()
            direction_mask = direction_mask.cuda()
            preds, embedded, direction = model(
                points.cuda(), points_mask.cuda(),
                allimgs.cuda(), rots.cuda(),
                trans.cuda(), intrins.cuda(), post_rots.cuda(),
                post_trans.cuda(), translation.cuda(), yaw_pitch_roll.cuda())
            onehot_preds = onehot_encoding(preds, dim=1)

            # loss
            bs = preds.shape[0]
            seg_loss = loss_fn(preds, binimgs).item() * bs
            var_loss, dist_loss, reg_loss = embedded_loss_fn(embedded, inst_mask.sum(1))
            var_loss, dist_loss, reg_loss = var_loss.item() * bs, dist_loss.item() * bs, reg_loss.item() * bs
            direction_loss = direction_loss_fn(torch.softmax(direction, 1), direction_mask)
            lane_mask = (1 - direction_mask[:, 0]).unsqueeze(1)
            direction_loss = (direction_loss * lane_mask).sum() / (lane_mask.sum() * direction_loss.shape[1] + 1e-6)
            final_loss = seg_loss * scale_seg + var_loss + scale_var + dist_loss * scale_dist + direction_loss * 0.2

            total_seg_loss += seg_loss
            total_reg_loss += reg_loss
            total_dist_loss += dist_loss
            total_var_loss += var_loss
            total_direction_loss += direction_loss
            total_final_loss += final_loss

            # angle diff
            total_angle_diff += calc_angle_diff(direction, direction_mask, angle_class) * bs

            # iou
            intersect, union, _ = get_batch_iou_multi_class(preds, binimgs)
            tot, cor, tp, fp, fn, _, _, _ = get_accuracy_precision_recall_multi_class(preds, binimgs)
            CD1, CD2, num1, num2 = lane_seg_metric.semantic_mask_chamfer_dist_cum(onehot_preds[:, 1:], binimgs[:, 1:])

            if eval_mAP:
                inst_pred_mask = torch.zeros_like(inst_mask, dtype=torch.int)
                for si in range(allimgs.shape[0]):
                    count = 0
                    for i in range(1, onehot_preds.shape[1]):
                        single_mask = onehot_preds[si][i].bool()
                        single_embedded = embedded[si].permute(1, 2, 0)
                        single_class_inst_mask, single_class_inst_coords = post_processor.postprocess(single_mask.cpu().numpy(), single_embedded.cpu().numpy())
                        if single_class_inst_mask is None:
                            continue

                        num_inst = len(single_class_inst_coords)

                        single_class_inst_mask[single_class_inst_mask != 0] += count
                        inst_pred_mask[si, i] = torch.tensor(single_class_inst_mask).cuda()
                        count += num_inst

                AP = lane_seg_metric.instance_mask_AP(inst_pred_mask[:, 1:], inst_mask[:, 1:], preds.softmax(1)[:, 1:], thresholds).cpu().numpy() * bs

            if total_intersect is None:
                total_intersect = intersect
                total_union = union
                total_pix = tot
                total_cor = cor
                total_tp = tp
                total_fp = fp
                total_fn = fn
                total_CD1 = CD1
                total_CD2 = CD2
                total_CD_num1 = num1
                total_CD_num2 = num2
                if eval_mAP:
                    total_AP = AP
            else:
                total_intersect += intersect
                total_union += union
                total_pix += tot
                total_cor += cor
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_CD1 += CD1
                total_CD2 += CD2
                total_CD_num1 += num1
                total_CD_num2 += num2
                if eval_mAP:
                    total_AP += AP

    model.train()
    # print(full_CD.shape)
    # with open('CD_matrix.npy', 'wb') as f:
    #     np.save(f, full_CD)
    total_CD1 = total_CD1.cpu().numpy()
    total_CD2 = total_CD2.cpu().numpy()
    total_CD_num1 = total_CD_num1.cpu().numpy()
    total_CD_num2 = total_CD_num2.cpu().numpy()
    return {
        'seg_loss': total_seg_loss / len(valloader.dataset),
        'reg_loss': total_reg_loss / len(valloader.dataset),
        'dist_loss': total_dist_loss / len(valloader.dataset),
        'var_loss': total_var_loss / len(valloader.dataset),
        'direction_loss': total_dist_loss / len(valloader.dataset),
        'final_loss': total_final_loss / len(valloader.dataset),
        'iou': total_intersect / total_union,
        'accuracy': total_cor / total_pix,
        'precision': total_tp / (total_tp + total_fp),
        'recall': total_tp / (total_tp + total_fn),
        'angle_diff': total_angle_diff / len(valloader.dataset),
        'CD_pred (precision)': total_CD1 / total_CD_num1,
        'CD_label (recall)': total_CD2 / total_CD_num2,
        'chamfer_distance': (total_CD1 + total_CD2) / (total_CD_num1 + total_CD_num2),
        'Average_precision': total_AP / len(valloader.dataset),
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


def extract_contour(topdown_seg_mask, canvas_size, thickness=5):
    topdown_seg_mask[topdown_seg_mask != 0] = 255
    ret, thresh = cv2.threshold(topdown_seg_mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(topdown_seg_mask)
    patch = box(1, 1, canvas_size[1] - 2, canvas_size[0] - 2)
    idx = 0
    for cnt in contours:
        cnt = cnt.reshape((-1, 2))
        cnt = np.append(cnt, cnt[0].reshape(-1, 2), axis=0)
        line = LineString(cnt)
        line = line.intersection(patch)
        if isinstance(line, MultiLineString):
            line = ops.linemerge(line)

        if isinstance(line, MultiLineString):
            for l in line:
                idx += 1
                cv2.polylines(mask, [np.asarray(list(l.coords), np.int32).reshape((-1, 2))], False, color=idx, thickness=thickness)
        elif isinstance(line, LineString):
            idx += 1
            cv2.polylines(mask, [np.asarray(list(line.coords), np.int32).reshape((-1, 2))], False, color=idx, thickness=thickness)

    return mask, idx

def plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx, alpha_poly=0.6, alpha_line=1.):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

    poly_names = ['road_segment', 'lane', 'ped_crossing']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,
                         50.0, poly_names, line_names)

    for la in lmap['road_segment']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 0], pts[:, 1], c=(124./255., 179./255., 210./255.), alpha=alpha_poly, linewidth=5)
        # plt.plot(pts[:, 0], pts[:, 1], c=(0., 1., 0.), alpha=alpha_poly, linewidth=5)
    for la in lmap['lane']:
        pts = (la - bx) / dx
        plt.fill(pts[:, 0], pts[:, 1], c=(74./255., 163./255., 120./255.), alpha=alpha_poly)
    for la in lmap['ped_crossing']:
        dist = np.square(la[1:, :] - la[:-1, :]).sum(-1)
        x1, x2 = np.argsort(dist)[-2:]
        pts = (la - bx) / dx
        # plt.plot(pts[:, 0], pts[:, 1], c=(247./255., 129./255., 132./255.), alpha=alpha_poly, linewidth=5)
        plt.plot(pts[x1:x1+2, 0], pts[x1:x1+2, 1], c=(1., 0., 0.), alpha=alpha_poly, linewidth=5)
        plt.plot(pts[x2:x2+2, 0], pts[x2:x2+2, 1], c=(1., 0., 0.), alpha=alpha_poly, linewidth=5)
    for la in lmap['lane_divider']:
        pts = (la - bx) / dx
        # plt.plot(pts[:, 0], pts[:, 1], c=(159. / 255., 0.0, 1.0), alpha=alpha_line, linewidth=5)
        plt.plot(pts[:, 0], pts[:, 1], c=(0., 0., 1.), alpha=alpha_poly, linewidth=5)
    for la in lmap['road_divider']:
        pts = (la - bx) / dx
        # plt.plot(pts[:, 0], pts[:, 1], c=(0.0, 0.0, 1.0), alpha=alpha_line, linewidth=5)
        plt.plot(pts[:, 0], pts[:, 1], c=(0., 0., 1.), alpha=alpha_poly, linewidth=5)


def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg


def calc_angle_diff(pred_mask, gt_mask, angle_class):
    per_angle = float(360. / angle_class)
    eval_mask = 1 - gt_mask[:, 0]
    pred_direction = get_pred_top2_direction(pred_mask, dim=1)

    gt_direction = (torch.topk(gt_mask, 2, dim=1)[1] - 1).float()
    pred_direction *= per_angle
    gt_direction *= per_angle
    pred_direction = pred_direction[:, :, None, :, :].repeat(1, 1, 2, 1, 1)
    gt_direction = gt_direction[:, None, :, :, :].repeat(1, 2, 1, 1, 1)
    diff_mask = torch.abs(pred_direction - gt_direction)
    diff_mask = torch.min(diff_mask, 360 - diff_mask)
    diff_mask = torch.min(diff_mask[:, 0, 0] + diff_mask[:, 1, 1], diff_mask[:, 1, 0] + diff_mask[:, 0, 1])
    return ((eval_mask * diff_mask).sum() / (eval_mask.sum() + 1e-6)).item()


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

from copy import deepcopy

def sort_points_by_dist(coords):
    coords = coords.astype('float')
    num_points = coords.shape[0]
    diff_matrix = np.repeat(coords[:, None], num_points, 1) - coords
    # x_range = np.max(np.abs(diff_matrix[..., 0]))
    # y_range = np.max(np.abs(diff_matrix[..., 1]))
    # diff_matrix[..., 1] *= x_range / y_range
    dist_matrix = np.sqrt(((diff_matrix) ** 2).sum(-1))
    dist_matrix_full = deepcopy(dist_matrix)
    direction_matrix = diff_matrix / (dist_matrix.reshape(num_points, num_points, 1) + 1e-6)

    sorted_points = [coords[0]]
    sorted_indices = [0]
    dist_matrix[:, 0] = np.inf

    last_direction = np.array([0, 0])
    for i in range(num_points - 1):
        last_idx = sorted_indices[-1]
        dist_metric = dist_matrix[last_idx] - 0 * (last_direction * direction_matrix[last_idx]).sum(-1)
        idx = np.argmin(dist_metric) % num_points
        new_direction = direction_matrix[last_idx, idx]
        if dist_metric[idx] > 3 and min(dist_matrix_full[idx][sorted_indices]) < 5:
            dist_matrix[:, idx] = np.inf
            continue
        if dist_metric[idx] > 10 and i > num_points * 0.9:
            break
        sorted_points.append(coords[idx])
        sorted_indices.append(idx)
        dist_matrix[:, idx] = np.inf
        last_direction = new_direction

    return np.stack(sorted_points, 0)


def connect_by_step(coords, direction_mask, sorted_points, taken_direction, step=5, per_deg=10, ema=0.5):
    dn = None
    while True:
        last_point = tuple(np.flip(sorted_points[-1]))
        if not taken_direction[last_point][0]:
            direction = direction_mask[last_point][0]
            taken_direction[last_point][0] = True
        elif not taken_direction[last_point][1]:
            direction = direction_mask[last_point][1]
            taken_direction[last_point][1] = True
        else:
            break

        if direction == -1:
            continue

        # if (sorted_points[-1] == np.array([45, 43])).all():
        #     import ipdb; ipdb.set_trace()

        deg = per_deg * direction
        # if dn is not None:
        #     if max(deg, dn) - min(deg, dn) < 180:
        #         deg = deg * ema + dn * (1 - ema)
        #         dn = deg
        #     else:
        #         deg = ((deg * ema + dn * (1 - ema) - 360)/2 + 360) % 360
        #         dn = deg

        vector_to_target = step * np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])
        last_point = deepcopy(sorted_points[-1])

        # NMS
        coords = coords[np.linalg.norm(coords - last_point, axis=-1) > step-1]

        if len(coords) == 0:
            break

        # cosine_diff = vector_to_next.dot(vector_to_target) / (np.linalg.norm(vector_to_next, axis=-1) * np.linalg.norm(vector_to_target))
        # cosine_diff[cosine_diff < 1e-5] = 1e-5
        target_point = np.array([last_point[0] + vector_to_target[0], last_point[1] + vector_to_target[1]])
        # dist_metric = np.linalg.norm(coords - target_point, axis=-1) / cosine_diff
        dist_metric = np.linalg.norm(coords - target_point, axis=-1)
        idx = np.argmin(dist_metric)

        if dist_metric[idx] > 50:
           continue

        sorted_points.append(deepcopy(coords[idx]))

        vector_to_next = coords[idx] - last_point
        deg = np.rad2deg(math.atan2(vector_to_next[1], vector_to_next[0]))
        inverse_deg = (180 + deg) % 360
        target_direction = per_deg * direction_mask[tuple(np.flip(sorted_points[-1]))]
        tmp = np.abs(target_direction - inverse_deg)
        tmp = torch.min(tmp, 360 - tmp)
        taken = np.argmin(tmp)
        taken_direction[tuple(np.flip(sorted_points[-1]))][taken] = True


def connect_by_direction(coords, direction_mask, step=5, per_deg=10):
    # tmp = direction_mask[coords[:, 1], coords[:, 0]]
    # coords = coords[abs(tmp[:, 0] - tmp[:, 1]) > 30 / per_deg, :]
    sorted_points = [deepcopy(coords[random.randint(0, coords.shape[0]-1)])]
    taken_direction = np.zeros_like(direction_mask, dtype=np.bool)

    connect_by_step(coords, direction_mask, sorted_points, taken_direction, step, per_deg)
    sorted_points.reverse()
    connect_by_step(coords, direction_mask, sorted_points, taken_direction, step, per_deg)

    # if np.linalg.norm(sorted_points[0] - sorted_points[-1]) < step+3:
    #     sorted_points.append(deepcopy(sorted_points[0]))

    return np.stack(sorted_points, 0)


def test_function():
    H = 200
    W = 400
    num_points = 20
    coords = np.stack([np.random.randint(0, W, size=(num_points)), np.random.randint(0, H, size=(num_points))], -1)
    direction_mask = np.random.randint(0, 37, size=(H, W, 2))
    sorted_points = connect_by_direction(coords, direction_mask)
    print(sorted_points)


if __name__ == '__main__':
    test_function()
