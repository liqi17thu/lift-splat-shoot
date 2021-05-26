"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""
import os
import random
import cv2

import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
import tqdm

mpl.use('Agg')

from nuscenes import NuScenes
from .topdown_mask import MyNuScenesMap

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches
from shapely.geometry import LineString

from .data import compile_data, MAP, NuscData
from .tools import (ego_to_cam, get_only_in_img_mask, denormalize_img,
                    SimpleLoss, get_val_info, add_ego, gen_dx_bx,
                    get_nusc_maps, plot_nusc_map, DiscriminativeLoss)
from .tools import label_onehot_decoding, onehot_encoding
from .models import compile_model
from .hd_models import HDMapNet, TemporalHDMapNet
from .vpn_model import VPNet
from .postprocess import LaneNetPostProcessor
from .pointpillar import PointPillar
from .ori_vpn import VPNModel
from copy import deepcopy
from .metric import LaneSegMetric

from .tools import denormalize_img, sort_points_by_dist


def redundant_filter(mask, kernel=25):
    M, N = mask.shape
    for i in range(M):
        for j in range(N):
            if mask[i, j] != 0:
                var = deepcopy(mask[i, j])
                local_mask = mask[
                             max(0, i - kernel // 2):min(M, i + kernel // 2 + 1),
                             max(0, j - kernel // 2):min(N, j + kernel // 2 + 1)]
                local_mask[local_mask == mask[i, j]] = 0
                mask[i, j] = var
    return mask


def gen_data(version,
             dataroot='data/nuScenes',

             H=900, W=1600,
             resize_lim=(0.193, 0.225),
             final_dim=(128, 352),
             bot_pct_lim=(0.0, 0.22),
             rot_lim=(-5.4, 5.4),
             rand_flip=False,
             ncams=6,
             line_width=1,
             preprocess=False,
             overwrite=False,

             xbound=[-30.0, 30.0, 0.15],
             ybound=[-15.0, 15.0, 0.15],
             zbound=[-10.0, 10.0, 20.0],
             dbound=[4.0, 45.0, 1.0],
             ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'preprocess': preprocess,
        'line_width': line_width,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,
    }

    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=dataroot,
                    verbose=False)
    nusc_maps = {}
    for map_name in MAP:
        nusc_maps[map_name] = MyNuScenesMap(dataroot=dataroot, map_name=map_name)

    random.shuffle(nusc.sample)
    nusc_data = NuscData(nusc, nusc_maps, False, data_aug_conf, grid_conf)
    for rec in tqdm.tqdm(nusc.sample):
        lidar_top_path = nusc.get_sample_data_path(rec['data']['LIDAR_TOP'])
        seg_path = lidar_top_path.split('.')[0] + '_seg_mask.png'
        inst_path = lidar_top_path.split('.')[0] + '_inst_mask.png'
        forward_path = lidar_top_path.split('.')[0] + '_forward_mask.png'
        backward_path = lidar_top_path.split('.')[0] + '_backward_mask.png'
        direct_path = lidar_top_path.split('.')[0] + '_direct_mask.png'
        if os.path.exists(seg_path) and os.path.exists(inst_path) and not overwrite:
            continue

        seg_mask, inst_mask, forward_mask, backward_mask = nusc_data.get_lineimg(rec)
        seg_mask = label_onehot_decoding(seg_mask).numpy()
        seg_mask = cv2.medianBlur(seg_mask.astype('uint8') * 10, 3, cv2.BORDER_DEFAULT)

        Image.fromarray(seg_mask * 10).save(seg_path)
        inst_mask = (inst_mask * 30).numpy().sum(0).astype('uint8')
        Image.fromarray(inst_mask).save(inst_path)


        forward_mask = redundant_filter(forward_mask)
        coords = np.where(forward_mask != 0)
        coords = np.stack([coords[1], coords[0]], -1)
        plt.figure(figsize=(8, 4))
        plt.xticks([])
        plt.yticks([])
        plt.xlim(0, 800)
        plt.xlim(0, 400)
        R = 1
        arr_width= 0.5
        for coord in coords:
            x = coord[0]
            y = coord[1]
            angle = np.deg2rad((forward_mask[y, x] - 1) * 10)
            dx = R * np.cos(angle)
            dy = R * np.sin(angle)
            plt.arrow(x=x, y=y, dx=dx, dy=dy, width=arr_width, head_width=3 * arr_width, head_length=5 * arr_width, facecolor=(1, 0, 0, 0.5))
        plt.savefig(forward_path)
        # print(forward_path)

        backward_mask = redundant_filter(backward_mask)
        coords = np.where(backward_mask != 0)
        coords = np.stack([coords[1], coords[0]], -1)
        plt.figure(figsize=(8, 4))
        plt.xticks([])
        plt.yticks([])
        plt.xlim(0, 800)
        plt.xlim(0, 400)
        for coord in coords:
            x = coord[0]
            y = coord[1]
            angle = np.deg2rad((backward_mask[y, x] - 1) * 10)
            dx = R * np.cos(angle)
            dy = R * np.sin(angle)
            plt.arrow(x=x, y=y, dx=dx, dy=dy, width=arr_width, head_width=3 * arr_width, head_length=5 * arr_width, facecolor=(0, 0, 1, 0.5))
        plt.savefig(backward_path)
        # print(backward_path)

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        plt.xlim(0, 800)
        plt.xlim(0, 400)
        coords = np.where(forward_mask != 0)
        coords = np.stack([coords[1], coords[0]], -1)
        for coord in coords:
            x = coord[0]
            y = coord[1]
            angle = np.deg2rad((forward_mask[y, x] - 1) * 10)
            dx = R * np.cos(angle)
            dy = R * np.sin(angle)
            plt.arrow(x=x, y=y, dx=dx, dy=dy, width=arr_width, head_width=6 * arr_width, head_length=9 * arr_width, facecolor=(1, 0, 0, 0.5))

        coords = np.where(backward_mask != 0)
        coords = np.stack([coords[1], coords[0]], -1)
        for coord in coords:
            x = coord[0]
            y = coord[1]
            angle = np.deg2rad((backward_mask[y, x] - 1) * 10)
            dx = R * np.cos(angle)
            dy = R * np.sin(angle)
            plt.arrow(x=x, y=y, dx=dx, dy=dy, width=arr_width, head_width=6 * arr_width, head_length=9 * arr_width, facecolor=(0, 0, 1, 0.5))
        plt.savefig(direct_path)
        # print(direct_path)



def viz_ipm_with_label(version,
                       dataroot='data/nuScenes',

                       H=900, W=1600,
                       resize_lim=(0.193, 0.225),
                       final_dim=(128, 352),
                       # final_dim=(900, 1600),
                       bot_pct_lim=(0.0, 0.22),
                       rot_lim=(-5.4, 5.4),
                       rand_flip=False,
                       ncams=6,
                       line_width=5,
                       preprocess=False,
                       overwrite=False,
                       z_roll_pitch=False,

                       xbound=[-30.0, 30.0, 0.15],
                       ybound=[-15.0, 15.0, 0.15],
                       zbound=[-10.0, 10.0, 20.0],
                       dbound=[4.0, 45.0, 1.0],
                       ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'preprocess': preprocess,
        'line_width': line_width,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,
    }
    [trainloader, valloader], [train_sampler, val_sampler] = compile_data(version, dataroot,
                                                                          data_aug_conf=data_aug_conf,
                                                                          grid_conf=grid_conf, bsz=4, nworkers=10,
                                                                          parser_name='segmentationdata',
                                                                          distributed=False)

    val = 0.01
    fH, fW = final_dim
    fH, fW = 1, 2
    plt.figure(figsize=(3 * fW * val, (3.5 * fW + 2 * fH) * val))
    gs = mpl.gridspec.GridSpec(4, 3, height_ratios=(1.5 * fW, 1.5 * fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    ipm_with_pitch = HDMapNet(xbound, ybound, outC=3, cam_encoding=False, bev_encoding=False, z_roll_pitch=True)
    ipm_without_pitch = HDMapNet(xbound, ybound, outC=3, cam_encoding=False, bev_encoding=False, z_roll_pitch=False)

    # plt.figure(figsize=(4, 2))
    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll, binimgs,
                     inst_label) in enumerate(valloader):
            topdown_with_pitch = ipm_with_pitch(imgs.cuda(),
                                                rots.cuda(),
                                                trans.cuda(),
                                                intrins.cuda(),
                                                post_rots.cuda(),
                                                post_trans.cuda(),
                                                translation.cuda(),
                                                yaw_pitch_roll.cuda(),
                                                )

            topdown_without_pitch = ipm_without_pitch(imgs.cuda(),
                                                      rots.cuda(),
                                                      trans.cuda(),
                                                      intrins.cuda(),
                                                      post_rots.cuda(),
                                                      post_trans.cuda(),
                                                      translation.cuda(),
                                                      yaw_pitch_roll.cuda(),
                                                      )

            binimgs[binimgs < 0.1] = np.nan
            for si in range(binimgs.shape[0]):
                plt.clf()
                plt.axis('off')
                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                plt.imshow(np.array(denormalize_img(topdown_with_pitch[si])))
                plt.imshow(binimgs[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.4)
                plt.imshow(binimgs[si][2], vmin=0, cmap='Reds', vmax=1, alpha=0.4)
                plt.imshow(binimgs[si][3], vmin=0, cmap='Greens', vmax=1, alpha=0.4)
                plt.xlim((0, binimgs.shape[3]))
                plt.ylim((0, binimgs.shape[2]))

                ax = plt.subplot(gs[1, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                plt.imshow(np.array(denormalize_img(topdown_without_pitch[si])))
                plt.imshow(binimgs[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.4)
                plt.imshow(binimgs[si][2], vmin=0, cmap='Reds', vmax=1, alpha=0.4)
                plt.imshow(binimgs[si][3], vmin=0, cmap='Greens', vmax=1, alpha=0.4)
                plt.xlim((0, binimgs.shape[3]))
                plt.ylim((0, binimgs.shape[2]))

                for imgi, img in enumerate(imgs[si]):
                    ax = plt.subplot(gs[2 + imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    # flip the bottom images
                    if imgi > 2:
                        showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                    plt.imshow(showimg)
                    plt.axis('off')

                print(f'saving topdown_{batchi:06}_{si:03}.png')
                plt.savefig(f'topdown_{batchi:06}_{si:03}.png')


def lidar_check(version,
                dataroot='/data/nuscenes',
                show_lidar=True,
                viz_train=False,
                nepochs=1,

                H=900, W=1600,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,

                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[4.0, 45.0, 1.0],

                bsz=1,
                nworkers=10,
                ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': cams,
        'Ncams': 5,
    }
    [trainloader, valloader], [train_sampler, val_sampler] = compile_data(version, dataroot,
                                                                          data_aug_conf=data_aug_conf,
                                                                          grid_conf=grid_conf, bsz=bsz,
                                                                          nworkers=nworkers,
                                                                          parser_name='vizdata', distributed=False)

    loader = trainloader if viz_train else valloader

    model = compile_model(grid_conf, data_aug_conf, outC=1)

    rat = H / W
    val = 10.1
    fig = plt.figure(figsize=(val + val / 3 * 2 * rat * 3, val / 3 * 2 * rat))
    gs = mpl.gridspec.GridSpec(2, 6, width_ratios=(1, 1, 1, 2 * rat, 2 * rat, 2 * rat))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    for epoch in range(nepochs):
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, pts, binimgs) in enumerate(loader):

            img_pts = model.get_geometry(rots, trans, intrins, post_rots, post_trans)

            for si in range(imgs.shape[0]):
                plt.clf()
                final_ax = plt.subplot(gs[:, 5:6])
                for imgi, img in enumerate(imgs[si]):
                    ego_pts = ego_to_cam(pts[si], rots[si, imgi], trans[si, imgi], intrins[si, imgi])
                    mask = get_only_in_img_mask(ego_pts, H, W)
                    plot_pts = post_rots[si, imgi].matmul(ego_pts) + post_trans[si, imgi].unsqueeze(1)

                    ax = plt.subplot(gs[imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    plt.imshow(showimg)
                    if show_lidar:
                        plt.scatter(plot_pts[0, mask], plot_pts[1, mask], c=ego_pts[2, mask],
                                    s=5, alpha=0.1, cmap='jet')
                    # plot_pts = post_rots[si, imgi].matmul(img_pts[si, imgi].view(-1, 3).t()) + post_trans[si, imgi].unsqueeze(1)
                    # plt.scatter(img_pts[:, :, :, 0].view(-1), img_pts[:, :, :, 1].view(-1), s=1)
                    plt.axis('off')

                    plt.sca(final_ax)
                    plt.plot(img_pts[si, imgi, :, :, :, 0].view(-1), img_pts[si, imgi, :, :, :, 1].view(-1), '.',
                             label=cams[imgi].replace('_', ' '))

                plt.legend(loc='upper right')
                final_ax.set_aspect('equal')
                plt.xlim((-50, 50))
                plt.ylim((-50, 50))

                ax = plt.subplot(gs[:, 3:4])
                plt.scatter(pts[si, 0], pts[si, 1], c=pts[si, 2], vmin=-5, vmax=5, s=5)
                plt.xlim((-50, 50))
                plt.ylim((-50, 50))
                ax.set_aspect('equal')

                ax = plt.subplot(gs[:, 4:5])
                plt.imshow(binimgs[si].squeeze(0).T, origin='lower', cmap='Greys', vmin=0, vmax=1)

                imname = f'lcheck{epoch:03}_{batchi:05}_{si:02}.jpg'
                print('saving', imname)
                plt.savefig(imname)


def cumsum_check(version,
                 dataroot='/data/nuscenes',
                 gpuid=1,

                 H=900, W=1600,
                 resize_lim=(0.193, 0.225),
                 final_dim=(128, 352),
                 bot_pct_lim=(0.0, 0.22),
                 rot_lim=(-5.4, 5.4),
                 rand_flip=True,

                 xbound=[-50.0, 50.0, 0.5],
                 ybound=[-50.0, 50.0, 0.5],
                 zbound=[-10.0, 10.0, 20.0],
                 dbound=[4.0, 45.0, 1.0],

                 bsz=4,
                 nworkers=10,
                 ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': 6,
    }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
    loader = trainloader

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    model.eval()
    for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):
        model.use_quickcumsum = False
        model.zero_grad()
        out = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
        out.mean().backward()
        print('autograd:    ', out.mean().detach().item(), model.camencode.depthnet.weight.grad.mean().item())

        model.use_quickcumsum = True
        model.zero_grad()
        out = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
        out.mean().backward()
        print('quick cumsum:', out.mean().detach().item(), model.camencode.depthnet.weight.grad.mean().item())
        print()


def eval_model(version,
               modelf,
               dataroot='data/nuScenes',
               gpuid=0,
               outC=4,
               method='temporal_HDMapNet',
               preprocess=False,

               H=900, W=1600,
               resize_lim=(0.193, 0.225),
               final_dim=(128, 352),
               bot_pct_lim=(0.0, 0.22),
               rot_lim=(-5.4, 5.4),
               rand_flip=True,
               line_width=1,

               xbound=[-30.0, 30.0, 0.15],
               ybound=[-15.0, 15.0, 0.15],
               zbound=[-10.0, 10.0, 20.0],
               dbound=[4.0, 45.0, 1.0],
               eval_mAP=False,

               bsz=4,
               nworkers=10,
               ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'preprocess': preprocess,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'line_width': line_width,
        'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': 6,
    }
    [trainloader, valloader], [train_sampler, val_sampler] = compile_data(version, dataroot,
                                                                          data_aug_conf=data_aug_conf,
                                                                          grid_conf=grid_conf, bsz=bsz,
                                                                          nworkers=nworkers,
                                                                          parser_name='segmentationdata',
                                                                          distributed=False)

    if method == 'lift_splat':
        model = compile_model(grid_conf, data_aug_conf, outC=outC)
    elif method == 'HDMapNet':
        # model = HDMapNet(xbound, ybound, outC=outC)
        model = HDMapNet(xbound, ybound, outC=outC, cam_encoding=False, camC=3)
    elif method == 'temporal_HDMapNet':
        model = TemporalHDMapNet(xbound, ybound, outC=outC)
    elif method == 'VPN':
        model = VPNet(outC=outC)
    elif method == 'PP':
        model = PointPillar(outC, xbound, ybound, zbound)
    elif method == 'VPNPP':
        model = VPNet(outC, lidar=True, xbound=xbound, ybound=ybound, zbound=zbound)
    elif method == 'ori_VPN':
        model = VPNModel(outC)
    else:
        raise NotImplementedError

    print('loading', modelf)
    model.load_state_dict(torch.load(modelf))
    model.cuda()

    embedded_dim = 16
    delta_v = 0.5
    delta_d = 3.0
    loss_fn = SimpleLoss(1.0).cuda()
    embedded_loss_fn = DiscriminativeLoss(embedded_dim, delta_v, delta_d).cuda()

    model.eval()
    val_info = get_val_info(model, valloader, loss_fn, embedded_loss_fn, eval_mAP=eval_mAP)
    val_info['chamfer_distance'] *= 0.15
    val_info['CD_pred (precision)'] *= 0.15
    val_info['CD_label (recall)'] *= 0.15
    print(val_info)
    print('iou: ', end='')
    print(np.mean(val_info['iou'][1:]))
    print('accuracy: ', end='')
    print(np.mean(val_info['accuracy'][1:]))
    print('precision: ', end='')
    print(np.mean(val_info['precision'][1:]))
    print('recall: ', end='')
    print(np.mean(val_info['recall'][1:]))
    print('chamfer distance: ', end='')
    print(np.mean(val_info['chamfer_distance']))
    print('mAP: ', end='')
    print(np.mean(val_info['Average_precision']))


def viz_model_preds(version,
                    modelf,
                    dataroot='/data/nuscenes',
                    map_folder='/data/nuscenes/mini',
                    gpuid=1,
                    viz_train=False,
                    outC=3,
                    method='lift_splat',

                    H=900, W=1600,
                    resize_lim=(0.193, 0.225),
                    final_dim=(128, 352),
                    bot_pct_lim=(0.0, 0.22),
                    rot_lim=(-5.4, 5.4),
                    line_width=5,
                    rand_flip=True,

                    xbound=[-50.0, 50.0, 0.5],
                    ybound=[-15.0, 15.0, 0.15],
                    zbound=[-10.0, 10.0, 20.0],
                    dbound=[4.0, 45.0, 1.0],

                    bsz=4,
                    nworkers=10,
                    ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'line_width': line_width,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': cams,
        'Ncams': 5,
    }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')
    loader = trainloader if viz_train else valloader
    nusc_maps = get_nusc_maps(map_folder)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    if method == 'lift_splat':
        model = compile_model(grid_conf, data_aug_conf, outC=outC)
    else:
        model = HDMapNet(ybound, xbound, outC=outC)
    model.load_state_dict(torch.load(modelf))
    model.to(device)

    dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    scene2map = {}
    for rec in loader.dataset.nusc.scene:
        log = loader.dataset.nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']

    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3 * fW * val, (1.5 * fW + 2 * fH) * val))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5 * fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    model.eval()
    counter = 0
    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):
            out = model(imgs.to(device),
                        rots.to(device),
                        trans.to(device),
                        intrins.to(device),
                        post_rots.to(device),
                        post_trans.to(device),
                        )
            out = out.sigmoid().cpu()

            for si in range(imgs.shape[0]):
                plt.clf()
                for imgi, img in enumerate(imgs[si]):
                    ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    # flip the bottom images
                    if imgi > 2:
                        showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                    plt.imshow(showimg)
                    plt.axis('off')
                    plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction')

                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                # plt.setp(ax.spines.values(), color='b', linewidth=2)
                # plt.legend(handles=[
                #     mpatches.Patch(color=(0.0, 0.0, 1.0, 1.0), label='Output Vehicle Segmentation'),
                #     mpatches.Patch(color='#76b900', label='Ego Vehicle'),
                #     mpatches.Patch(color=(1.00, 0.50, 0.31, 0.8), label='Map (for visualization purposes only)')
                # ], loc=(0.01, 0.86))

                plt.imshow(binimgs[si].squeeze(0), vmin=0, vmax=1, cmap='Reds', alpha=0.6)
                plt.imshow(out[si].squeeze(0), vmin=0, vmax=1, cmap='Blues', alpha=0.6)

                # plot static map (improves visualization)
                rec = loader.dataset.ixes[counter]
                plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                plt.xlim((out.shape[3], 0))
                plt.ylim((0, out.shape[3]))
                add_ego(bx, dx)

                imname = f'eval{batchi:06}_{si:03}.jpg'
                print('saving', imname)
                plt.savefig(imname)
                counter += 1


def viz_model_preds_class3(version,
                           modelf,
                           dataroot='data/nuScenes',
                           map_folder='data/nuScenes',
                           gpuid=0,
                           viz_train=False,
                           outC=4,
                           method='temporal_HDMapNet',

                           preprocess=False,
                           H=900, W=1600,
                           resize_lim=(0.193, 0.225),
                           final_dim=(128, 352),
                           bot_pct_lim=(0.0, 0.22),
                           rot_lim=(-5.4, 5.4),
                           line_width=1,
                           rand_flip=True,

                           xbound=[-30.0, 30.0, 0.15],
                           ybound=[-15.0, 15.0, 0.15],
                           zbound=[-10.0, 10.0, 20.0],
                           dbound=[4.0, 45.0, 1.0],

                           bsz=4,
                           nworkers=10,
                           ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'line_width': line_width,
        'preprocess': preprocess,
        'bot_pct_lim': bot_pct_lim,
        'cams': cams,
        'Ncams': 6,
    }

    temporal = 'temporal' in method
    if temporal:
        parser_name = 'temporalsegmentationdata'
    else:
        parser_name = 'segmentationdata'

    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name=parser_name)
    loader = trainloader if viz_train else valloader
    nusc_maps = get_nusc_maps(map_folder)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    if method == 'lift_splat':
        model = compile_model(grid_conf, data_aug_conf, outC=outC)
    elif method == 'HDMapNet':
        model = HDMapNet(xbound, ybound, outC=outC)
    elif method == 'temporal_HDMapNet':
        model = TemporalHDMapNet(xbound, ybound, outC=outC)

    model.load_state_dict(torch.load(modelf))
    model.to(device)

    dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    scene2map = {}
    for rec in loader.dataset.nusc.scene:
        log = loader.dataset.nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']

    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3 * fW * val, (2 * fW + 2 * fH) * val))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(2 * fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    model.eval()
    counter = 0
    with torch.no_grad():
        for batchi, (
        imgs, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll, binimgs) in enumerate(loader):

            out = model(imgs.to(device),
                        rots.to(device),
                        trans.to(device),
                        intrins.to(device),
                        post_rots.to(device),
                        post_trans.to(device),
                        translation.to(device),
                        yaw_pitch_roll.to(device),
                        )
            out = out.softmax(1).cpu()

            if temporal:
                imgs = imgs[:, 0]
            # visualization
            binimgs[binimgs < 0.1] = np.nan
            out[out < 0.1] = np.nan
            for si in range(imgs.shape[0]):
                plt.clf()
                for imgi, img in enumerate(imgs[si]):
                    ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    # flip the bottom images
                    if imgi > 2:
                        showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                    plt.imshow(showimg)
                    plt.axis('off')
                    plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction')

                ax = plt.subplot(gs[0, :])
                # ax.get_xaxis().set_ticks([])
                # ax.get_yaxis().set_ticks([])
                # plt.setp(ax.spines.values(), color='b', linewidth=2)
                # plt.legend(handles=[
                #     mpatches.Patch(color=(0.0, 0.0, 1.0, 1.0), label='Output Vehicle Segmentation'),
                #     mpatches.Patch(color='#76b900', label='Ego Vehicle'),
                #     mpatches.Patch(color=(1.00, 0.50, 0.31, 0.8), label='Map (for visualization purposes only)')
                # ], loc=(0.01, 0.86))

                plt.imshow(out[si][1], vmin=0, vmax=1, cmap='Blues', alpha=0.8)
                plt.imshow(out[si][2], vmin=0, vmax=1, cmap='Reds', alpha=0.8)
                plt.imshow(out[si][3], vmin=0, vmax=1, cmap='Greens', alpha=0.8)

                plt.imshow(binimgs[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
                plt.imshow(binimgs[si][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
                plt.imshow(binimgs[si][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)

                # plt.imshow(out[si].transpose(0, -1), vmin=0, vmax=1, cmap='Blues', alpha=0.6)

                # plot static map (improves visualization)
                rec = loader.dataset.ixes[counter]
                plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                plt.xlim((0, binimgs.shape[3]))
                plt.ylim((0, binimgs.shape[2]))
                add_ego(bx, dx)

                imname = f'eval{batchi:06}_{si:03}.jpg'
                print('saving', imname)
                plt.savefig(imname)
                counter += 1




from .tools import connect_by_direction

def viz_model_preds_inst(version,
                            modelf,
                            dataroot='data/nuScenes',
                            map_folder='data/nuScenes',
                            gpuid=0,
                            viz_train=False,
                            outC=4,
                            method='temporal_HDMapNet',

                            preprocess=False,
                            H=900, W=1600,
                            resize_lim=(0.193, 0.225),
                            final_dim=(128, 352),
                            angle_class=36,
                            # final_dim=(300, 400),
                            bot_pct_lim=(0.0, 0.22),
                            rot_lim=(-5.4, 5.4),
                            line_width=5,
                            rand_flip=True,

                            xbound=[-30.0, 30.0, 0.15],
                            ybound=[-15.0, 15.0, 0.15],
                            zbound=[-10.0, 10.0, 20.0],
                            dbound=[4.0, 45.0, 1.0],

                            bsz=4,
                            nworkers=10,
                            ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'line_width': line_width,
                    'preprocess': preprocess,
                    'bot_pct_lim': bot_pct_lim,
                    'angle_class': angle_class,
                    'cams': cams,
                    'Ncams': 6,
                }

    temporal = 'temporal' in method
    if temporal:
        parser_name = 'temporalsegmentationdata'
    else:
        parser_name = 'segmentationdata'

    [trainloader, valloader], [_, _] = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name=parser_name, distributed=False)
    loader = trainloader if viz_train else valloader
    nusc_maps = get_nusc_maps(map_folder)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    if method == 'lift_splat':
        model = compile_model(grid_conf, data_aug_conf, outC=outC)
    elif method == 'HDMapNet':
        model = HDMapNet(xbound, ybound, outC=outC)
        # model = HDMapNet(xbound, ybound, outC=outC, cam_encoding=False, camC=3)
    elif method == 'temporal_HDMapNet':
        model = TemporalHDMapNet(xbound, ybound, outC=outC)
    elif method == 'VPN':
        model = VPNet(outC=outC)
    elif method == 'PP':
        model = PointPillar(outC, xbound, ybound, zbound)
    elif method == 'VPNPP':
        model = VPNet(outC, lidar=True, xbound=xbound, ybound=ybound, zbound=zbound)
    elif method == 'ori_VPN':
        model = VPNModel(outC)
    else:
        raise NotImplementedError

    # model.load_state_dict(torch.load(modelf), strict=False)
    model.load_state_dict(torch.load(modelf))
    model.to(device)

    dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    scene2map = {}
    for rec in loader.dataset.nusc.scene:
        log = loader.dataset.nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']

    val = 0.01
    fH, fW = final_dim
    plt.figure(figsize=(3*fW*val, (3*fW)*val))
    gs = mpl.gridspec.GridSpec(2, 3, height_ratios=(1.5*fW, 1.5*fW))

    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    max_pool_1 = nn.MaxPool2d((1, 5), padding=(0, 2), stride=1)
    avg_pool_1 = nn.AvgPool2d((9, 5), padding=(4, 2), stride=1)
    max_pool_2 = nn.MaxPool2d((5, 1), padding=(2, 0), stride=1)
    avg_pool_2 = nn.AvgPool2d((5, 9), padding=(2, 4), stride=1)
    post_processor = LaneNetPostProcessor(dbscan_eps=1.5, postprocess_min_samples=50)
    pca = PCA(n_components=3)

    color_map = []
    for i in range(30):
        color_map.append([random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)])

    car_img = Image.open('car_3.png')
    model.eval()
    # counter = 1204
    # counter = 80
    # counter = 72
    counter = 44
    counter = 0
    with torch.no_grad():
        for batchi, (points, points_mask, imgs, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll, binimgs, inst_label, direction_mask) in enumerate(loader):
            if batchi < 11:
                continue

            out, embedded, direction = model(
                    points.cuda(),
                    points_mask.cuda(),
                    imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    translation.to(device),
                    yaw_pitch_roll.to(device),
                    )
            origin_out = out
            # origin_out = binimgs
            out = out.softmax(1).cpu()
            direction = direction.permute(0, 2, 3, 1).cpu()
            direction = get_pred_top2_direction(direction, dim=1)

            _, direction_mask = torch.topk(direction_mask, 2, dim=1)
            direction_mask = direction_mask.cpu() - 1

            preds = onehot_encoding(out).cpu().numpy()
            embedded = embedded.cpu()

            N, C, H, W = embedded.shape
            # embedded_test = embedded.permute(0, 2, 3, 1).reshape(N*H*W, C)
            # embedded_fitted = pca.fit_transform(embedded_test)
            # embedded_fitted = torch.sigmoid(torch.tensor(embedded_fitted)).numpy()
            # embedded_fitted = embedded_fitted.reshape((N, H, W, 3))
            # alpha_channel = np.ones((N, H, W, 1))

            if temporal:
                imgs = imgs[:, 0]
            # visualization
            binimgs[binimgs < 0.1] = np.nan
            seg_mask = out.numpy()
            seg_mask[seg_mask < 0.1] = np.nan

            for si in range(imgs.shape[0]):
                plt.clf()

                # inst_mask = np.zeros((200, 400), dtype='int32')
                # inst_mask_pil = np.zeros((200, 400, 4), dtype='uint8')

                simplified_coords = []

                count = 0
                for i in range(1, preds.shape[1]):
                    single_mask = preds[si][i].astype('uint8')
                    single_embedded = embedded[si].permute(1, 2, 0)
                    single_class_inst_mask, single_class_inst_coords = post_processor.postprocess(single_mask, single_embedded)
                    if single_class_inst_mask is None:
                        continue

                    num_inst = len(single_class_inst_coords)
                    # GT
                    # single_class_inst_mask = inst_label[si][i].int().numpy()
                    # num_inst = np.max(single_class_inst_mask)

                    prob = origin_out[si][i]
                    prob[single_class_inst_mask == 0] = 0
                    nms_mask_1 = ((max_pool_1(prob.unsqueeze(0))[0] - prob) < 0.0001).cpu().numpy()
                    avg_mask_1 = avg_pool_1(prob.unsqueeze(0))[0].cpu().numpy()
                    nms_mask_2 = ((max_pool_2(prob.unsqueeze(0))[0] - prob) < 0.0001).cpu().numpy()
                    avg_mask_2 = avg_pool_2(prob.unsqueeze(0))[0].cpu().numpy()
                    vertical_mask = avg_mask_1 > avg_mask_2
                    horizontal_mask = ~vertical_mask
                    nms_mask = (vertical_mask & nms_mask_1) | (horizontal_mask & nms_mask_2)

                    for j in range(1, num_inst+1):
                        full_idx = np.where((single_class_inst_mask == j))
                        full_lane_coord = np.vstack((full_idx[1], full_idx[0])).transpose()

                        idx = np.where(nms_mask & (single_class_inst_mask == j))
                        if len(idx[0]) == 0:
                            continue
                        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

                        range_0 = np.max(full_lane_coord[:, 0]) - np.min(full_lane_coord[:, 0])
                        range_1 = np.max(full_lane_coord[:, 1]) - np.min(full_lane_coord[:, 1])
                        if range_0 > range_1:
                            lane_coordinate = sorted(lane_coordinate, key=lambda x: x[0])
                            full_lane_coord = sorted(full_lane_coord, key=lambda x: x[0])
                            if full_lane_coord[0][0] < nx[0] - full_lane_coord[-1][0]:
                                full_lane_coord.insert(0, lane_coordinate[0])
                            else:
                                full_lane_coord.insert(0, lane_coordinate[-1])
                        else:
                            lane_coordinate = sorted(lane_coordinate, key=lambda x: x[1])
                            full_lane_coord = sorted(full_lane_coord, key=lambda x: x[1])
                            if full_lane_coord[0][1] < nx[1] - full_lane_coord[-1][1]:
                                full_lane_coord.insert(0, lane_coordinate[0])
                            else:
                                full_lane_coord.insert(0, lane_coordinate[-1])

                        full_lane_coord = np.stack(full_lane_coord)
                        lane_coordinate = np.stack(lane_coordinate)
                        idx = np.where((full_lane_coord == full_lane_coord[0]).all(-1))[0][-1]
                        full_lane_coord = np.concatenate([full_lane_coord[:idx], full_lane_coord[idx+1:]])

                        # lane_coordinate = connect_by_direction(full_lane_coord, direction[si])
                        lane_coordinate = sort_points_by_dist(lane_coordinate)
                        # line = LineString(lane_coordinate)
                        # line = line.simplify(tolerance=1.5)
                        # lane_coordinate = np.asarray(list(line.coords)).reshape((-1, 2))
                        lane_coordinate = lane_coordinate.astype('int32')
                        lane_coordinate = connect_by_direction(lane_coordinate, direction[si], step=5, per_deg=360/angle_class)
                        simplified_coords.append(lane_coordinate)

                    # inst_mask[single_class_inst_mask != 0] += single_class_inst_mask[single_class_inst_mask != 0] + count
                    count += num_inst

                # for i in range(1, count+1):
                #     inst_mask_pil[inst_mask == i, :3] = color_map[i]
                #     inst_mask_pil[inst_mask == i, 3] = 150

                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

                R = 2
                arr_width = 0.5
                # plt.imshow(seg_mask[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
                # plt.imshow(seg_mask[si][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
                # plt.imshow(seg_mask[si][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
                # plt.imshow(binimgs[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
                # plt.imshow(binimgs[si][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
                # plt.imshow(binimgs[si][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)

                for coord in simplified_coords:
                    for i in range(len(coord)):
                        x, y = coord[i, 0], coord[i, 1]
                        angle = np.deg2rad((direction[si, y, x, 0] - 1)*10)
                        # angle = np.deg2rad((direction_mask[si, 0, y, x] - 1)*10)
                        dx = R * np.cos(angle)
                        dy = R * np.sin(angle)
                        plt.arrow(x=x+2, y=y+2, dx=dx, dy=dy, width=arr_width, head_width=5*arr_width, head_length=9*arr_width, overhang=0., facecolor=(1, 0, 0, 0.6))

                        x, y = coord[i, 0], coord[i, 1]
                        angle = np.deg2rad((direction[si, y, x, 1] - 1)*10)
                        # angle = np.deg2rad((direction_mask[si, 1, y, x] - 1)*10)
                        dx = R * np.cos(angle)
                        dy = R * np.sin(angle)
                        plt.arrow(x=x-2, y=y-2, dx=dx, dy=dy, width=arr_width, head_width=5*arr_width, head_length=9*arr_width, overhang=0., facecolor=(0, 0, 1, 0.6))

                # plot static map (improves visualization)
                rec = loader.dataset.ixes[counter]
                plt.xlim((0, binimgs.shape[3]))
                plt.ylim((0, binimgs.shape[2]))
                plt.imshow(car_img, extent=[200-15, 200+15, 100-12, 100+12])
                plt.setp(ax.spines.values(), linewidth=0)

                ax = plt.subplot(gs[1, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                plt.setp(ax.spines.values(), linewidth=0)
                for coord in simplified_coords:
                    plt.plot(coord[:, 0], coord[:, 1], linewidth=5)

                plt.xlim((0, binimgs.shape[3]))
                plt.ylim((0, binimgs.shape[2]))
                plt.imshow(car_img, extent=[200-15, 200+15, 100-12, 100+12])

                imname = f'eval{batchi:06}_{si:03}.jpg'
                print('saving', imname)
                plt.savefig(imname)
                counter += 1


def get_pc_from_mask(mask):
    pc = np.where(mask > 0)
    pc = np.stack([pc[1], pc[0]], axis=-1)
    pc[:, 0] -= 200
    pc[:, 1] -= 100
    z = np.zeros(pc.shape[0]).reshape(-1, 1)
    pc = np.c_[pc, z]
    return pc


import numpy as np
import os.path as osp

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from nuscenes import NuScenes


def render_sample_data(nusc,
                       sample_data_token: str,
                       nsweeps: int = 1,
                       use_flat_vehicle_coordinates: bool = True,
                       show_lidarseg: bool = False) -> None:
    sd_record = nusc.get('sample_data', sample_data_token)

    sample_rec = nusc.get('sample', sd_record['sample_token'])
    chan = sd_record['channel']
    ref_chan = 'LIDAR_TOP'
    ref_sd_token = sample_rec['data'][ref_chan]
    ref_sd_record = nusc.get('sample_data', ref_sd_token)

    if show_lidarseg:
        assert hasattr(nusc, 'lidarseg'), 'Error: nuScenes-lidarseg not installed!'

        # Ensure that lidar pointcloud is from a keyframe.
        assert sd_record['is_key_frame'], \
            'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

        assert nsweeps == 1, \
            'Error: Only pointclouds which are keyframes have lidar segmentation labels; nsweeps should ' \
            'be set to 1.'

        # Load a single lidar point cloud.
        pcl_path = osp.join(nusc.dataroot, ref_sd_record['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
    else:
        # Get aggregated lidar point cloud in lidar frame.
        pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)
    velocities = None

    # By default we render the sample_data top down in the sensor frame.
    # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
    # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
    if use_flat_vehicle_coordinates:
        # Retrieve transformation matrices for reference point cloud.
        cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
        ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                      rotation=Quaternion(cs_record["rotation"]))

        # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
        ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
        rotation_vehicle_flat_from_vehicle = np.dot(
            Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
            Quaternion(pose_record['rotation']).inverse.rotation_matrix)
        # rotation_vehicle_flat_from_vehicle = Quaternion(pose_record['rotation']).inverse.rotation_matrix
        vehicle_flat_from_vehicle = np.eye(4)
        vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
        viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
    else:
        viewpoint = np.eye(4)

    # Show point cloud.
    pc.points[:2, :] /= 0.15
    points = view_points(pc.points[:3, :], viewpoint, normalize=False)

    return points


from .tools import get_pred_top2_direction

def gen_pred_pc(version,
                modelf,
                dataroot='data/nuScenes',
                map_folder='data/nuScenes',
                gpuid=0,
                viz_train=False,
                outC=4,
                method='temporal_HDMapNet',

                preprocess=False,
                H=900, W=1600,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                # final_dim=(300, 400),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                line_width=2,
                angle_class=36,
                rand_flip=True,

                xbound=[-30.0, 30.0, 0.15],
                ybound=[-15.0, 15.0, 0.15],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[4.0, 45.0, 1.0],

                bsz=4,
                nworkers=10,
                ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'line_width': line_width,
                    'preprocess': preprocess,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': cams,
                    'angle_class': angle_class,
                    'Ncams': 6,
                }

    temporal = 'temporal' in method
    if temporal:
        parser_name = 'temporalsegmentationdata'
    else:
        parser_name = 'segmentationdata'

    [trainloader, valloader], [train_sampler, val_sampler] = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name=parser_name, distributed=False)
    loader = trainloader if viz_train else valloader
    nusc_maps = get_nusc_maps(map_folder)
    nusc = NuScenes(version='v1.0-'+version, dataroot=dataroot, verbose=False)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    if method == 'lift_splat':
        model = compile_model(grid_conf, data_aug_conf, outC=outC)
    elif method == 'HDMapNet':
        # model = HDMapNet(xbound, ybound, outC=outC)
        model = HDMapNet(xbound, ybound, outC=outC, cam_encoding=False, camC=3)
    elif method == 'temporal_HDMapNet':
        model = TemporalHDMapNet(xbound, ybound, outC=outC)
    elif method == 'VPN':
        model = VPNet(outC=outC)
    elif method == 'PP':
        model = PointPillar(outC, xbound, ybound, zbound)
    elif method == 'VPNPP':
        model = VPNet(outC, lidar=True, xbound=xbound, ybound=ybound, zbound=zbound)
    elif method == 'ori_VPN':
        model = VPNModel(outC)
    else:
        raise NotImplementedError

    model.load_state_dict(torch.load(modelf))
    model.to(device)

    val = 0.01
    fH, fW = final_dim
    plt.figure(figsize=(3*fW*val, (2*fH)*val))
    gs = mpl.gridspec.GridSpec(2, 3, height_ratios=(1, 1))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])

    scene2map = {}
    for rec in loader.dataset.nusc.scene:
        log = loader.dataset.nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']

    max_pool_1 = nn.MaxPool2d((1, 5), padding=(0, 2), stride=1)
    avg_pool_1 = nn.AvgPool2d((9, 5), padding=(4, 2), stride=1)
    max_pool_2 = nn.MaxPool2d((5, 1), padding=(2, 0), stride=1)
    avg_pool_2 = nn.AvgPool2d((5, 9), padding=(2, 4), stride=1)
    post_processor = LaneNetPostProcessor(dbscan_eps=1.5, postprocess_min_samples=50)

    counter = 0
    model.eval()
    with torch.no_grad():
        for batchi, (points, points_mask, imgs, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll, binimgs, inst_label, direction_mask) in enumerate(loader):
            # if batchi < 18:
            #     continue

            out, embedded, direction = model(
                    points.cuda(),
                    points_mask.cuda(),
                    imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    translation.to(device),
                    yaw_pitch_roll.to(device),
                    )
            origin_out = out
            out = out.softmax(1).cpu()

            direction = direction.permute(0, 2, 3, 1).cpu()
            direction = get_pred_top2_direction(direction, dim=-1)

            preds = onehot_encoding(out).cpu().numpy()
            embedded = embedded.cpu()

            for si in range(imgs.shape[0]):
                simplified_coords = []
                mask = np.zeros((4, preds.shape[2], preds.shape[3]))
                for i in range(1, preds.shape[1]):

                    single_mask = preds[si][i].astype('uint8')
                    single_embedded = embedded[si].permute(1, 2, 0)
                    single_class_inst_mask, single_class_inst_coords = post_processor.postprocess(single_mask, single_embedded)
                    if single_class_inst_mask is None:
                        continue

                    num_inst = len(single_class_inst_coords)
                    # GT
                    # single_class_inst_mask = inst_label[si][i].int().numpy()
                    # num_inst = np.max(single_class_inst_mask)

                    prob = origin_out[si][i]
                    prob[single_class_inst_mask == 0] = 0
                    nms_mask_1 = ((max_pool_1(prob.unsqueeze(0))[0] - prob) < 0.0001).cpu().numpy()
                    avg_mask_1 = avg_pool_1(prob.unsqueeze(0))[0].cpu().numpy()
                    nms_mask_2 = ((max_pool_2(prob.unsqueeze(0))[0] - prob) < 0.0001).cpu().numpy()
                    avg_mask_2 = avg_pool_2(prob.unsqueeze(0))[0].cpu().numpy()
                    vertical_mask = avg_mask_1 > avg_mask_2
                    horizontal_mask = ~vertical_mask
                    nms_mask = (vertical_mask & nms_mask_1) | (horizontal_mask & nms_mask_2)

                    for j in range(1, num_inst + 1):
                        full_idx = np.where((single_class_inst_mask == j))
                        full_lane_coord = np.vstack((full_idx[1], full_idx[0])).transpose()

                        idx = np.where(nms_mask & (single_class_inst_mask == j))
                        if len(idx[0]) == 0:
                            continue
                        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

                        range_0 = np.max(full_lane_coord[:, 0]) - np.min(full_lane_coord[:, 0])
                        range_1 = np.max(full_lane_coord[:, 1]) - np.min(full_lane_coord[:, 1])
                        if range_0 > range_1:
                            lane_coordinate = sorted(lane_coordinate, key=lambda x: x[0])
                            full_lane_coord = sorted(full_lane_coord, key=lambda x: x[0])
                            if full_lane_coord[0][0] < nx[0] - full_lane_coord[-1][0]:
                                full_lane_coord.insert(0, lane_coordinate[0])
                            else:
                                full_lane_coord.insert(0, lane_coordinate[-1])
                        else:
                            lane_coordinate = sorted(lane_coordinate, key=lambda x: x[1])
                            full_lane_coord = sorted(full_lane_coord, key=lambda x: x[1])
                            if full_lane_coord[0][1] < nx[1] - full_lane_coord[-1][1]:
                                full_lane_coord.insert(0, lane_coordinate[0])
                            else:
                                full_lane_coord.insert(0, lane_coordinate[-1])

                        lane_coordinate = np.stack(lane_coordinate)
                        lane_coordinate = sort_points_by_dist(lane_coordinate)
                        lane_coordinate = lane_coordinate.astype('int32')
                        lane_coordinate = connect_by_direction(lane_coordinate, direction[si], step=5, per_deg=360/angle_class)
                        cv2.polylines(mask[i], [lane_coordinate], False, color=1, thickness=1)
                        simplified_coords.append(lane_coordinate)

                # mask = preds[si]
                pc_divider = get_pc_from_mask(mask[1])
                pc_ped = get_pc_from_mask(mask[2])
                pc_boundary = get_pc_from_mask(mask[3])



                idx = f'eval{batchi:06}_{si:03}'
                rec = loader.dataset.ixes[counter]
                sample_data_token = rec['data']['LIDAR_TOP']
                pc_lidar = render_sample_data(nusc, sample_data_token).T

                with open(f'{idx}_divider.bin', 'wb') as f:
                    np.save(f, pc_divider)
                with open(f'{idx}_ped.bin', 'wb') as f:
                    np.save(f, pc_ped)
                with open(f'{idx}_boundary.bin', 'wb') as f:
                    np.save(f, pc_boundary)
                with open(f'{idx}_lidar.bin', 'wb') as f:
                    np.save(f, pc_lidar)
                print('saving', idx)
                counter += 1
