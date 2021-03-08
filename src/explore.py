"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""
import os
import random
import cv2

import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
import tqdm

# mpl.use('Agg')

from nuscenes import NuScenes
from .topdown_mask import MyNuScenesMap

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches

from .data import compile_data, MAP, NuscData
from .tools import (ego_to_cam, get_only_in_img_mask, denormalize_img,
                    SimpleLoss, get_val_info, add_ego, gen_dx_bx,
                    get_nusc_maps, plot_nusc_map, DiscriminativeLoss)
from .tools import label_onehot_decoding, onehot_encoding
from .models import compile_model
from .hd_models import HDMapNet, TemporalHDMapNet
from .vpn_model import VPNet
from .postprocess import LaneNetPostProcessor
from .metric import LaneSegMetric

from .tools import denormalize_img

def gen_data(version,
            dataroot='data/nuScenes',

            H=900, W=1600,
            resize_lim=(0.193, 0.225),
            final_dim=(128, 352),
            bot_pct_lim=(0.0, 0.22),
            rot_lim=(-5.4, 5.4),
            rand_flip=False,
            ncams=6,
            line_width=5,
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
        if os.path.exists(seg_path) and os.path.exists(inst_path) and not overwrite:
            continue

        seg_mask, inst_mask = nusc_data.get_lineimg(rec)
        seg_mask = label_onehot_decoding(seg_mask)

        Image.fromarray(seg_mask.numpy().astype('uint8')).save(seg_path)
        Image.fromarray(inst_mask.numpy().astype('uint8')).save(inst_path)



def viz_ipm_with_label(version,
            dataroot='data/nuScenes',

            H=900, W=1600,
            resize_lim=(0.193, 0.225),
            final_dim=(128, 352),
            bot_pct_lim=(0.0, 0.22),
            rot_lim=(-5.4, 5.4),
            rand_flip=False,
            ncams=6,
            line_width=5,
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

    [trainloader, valloader], [train_sampler, val_sampler] = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=4, nworkers=10,
                                          parser_name='segmentationdata', distributed=False)

    ipm = HDMapNet(xbound, ybound, outC=3, cam_encoding=False, bev_encoding=False)

    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll, binimgs, inst_label) in enumerate(valloader):
            topdown = ipm(imgs.cuda(),
                    rots.cuda(),
                    trans.cuda(),
                    intrins.cuda(),
                    post_rots.cuda(),
                    post_trans.cuda(),
                    translation.cuda(),
                    yaw_pitch_roll.cuda(),
                    )

            topdown = denormalize_img(topdown)
            for si in range(binimgs.shape[0]):
                plt.imshow(topdown[si])
                plt.imshow(inst_label[si], alpha=0.6)
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
    [trainloader, valloader], [train_sampler, val_sampler] = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                                                          parser_name='vizdata', distributed=False)

    loader = trainloader if viz_train else valloader

    model = compile_model(grid_conf, data_aug_conf, outC=1)

    rat = H / W
    val = 10.1
    fig = plt.figure(figsize=(val + val/3*2*rat*3, val/3*2*rat))
    gs = mpl.gridspec.GridSpec(2, 6, width_ratios=(1, 1, 1, 2*rat, 2*rat, 2*rat))
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
                    plt.plot(img_pts[si, imgi, :, :, :, 0].view(-1), img_pts[si, imgi, :, :, :, 1].view(-1), '.', label=cams[imgi].replace('_', ' '))

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
                    'Ncams': 5,
                }
    [trainloader, valloader], [train_sampler, val_sampler] = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata', distributed=False)

    if method == 'lift_splat':
        model = compile_model(grid_conf, data_aug_conf, outC=outC)
    elif method == 'HDMapNet':
        model = HDMapNet(xbound, ybound, outC=outC)
    elif method == 'temporal_HDMapNet':
        model = TemporalHDMapNet(xbound, ybound, outC=outC)
    elif method == 'VPN':
        model = VPNet(outC=outC)

    print('loading', modelf)
    model.load_state_dict(torch.load(modelf))
    model.cuda()

    embedded_dim = 16
    delta_v = 0.5
    delta_d = 3.0
    loss_fn = SimpleLoss(1.0).cuda()
    embedded_loss_fn = DiscriminativeLoss(embedded_dim, delta_v, delta_d).cuda()

    model.eval()
    val_info = get_val_info(model, valloader, loss_fn, embedded_loss_fn, eval_mAP=True)
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
    fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
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
                plt.setp(ax.spines.values(), color='b', linewidth=2)
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
                            line_width=2,
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
    fig = plt.figure(figsize=(3*fW*val, (2*fW + 2*fH)*val))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(2*fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    model.eval()
    counter = 0
    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll, binimgs) in enumerate(loader):

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
                plt.setp(ax.spines.values(), color='b', linewidth=2)
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
                            bot_pct_lim=(0.0, 0.22),
                            rot_lim=(-5.4, 5.4),
                            line_width=2,
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

    [trainloader, valloader], [train_sampler, val_sampler] = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name=parser_name, distributed=False)
    loader = trainloader if viz_train else valloader
    nusc_maps = get_nusc_maps(map_folder)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    if method == 'lift_splat':
        model = compile_model(grid_conf, data_aug_conf, outC=outC)
    elif method == 'HDMapNet':
        model = HDMapNet(xbound, ybound, outC=outC)
    elif method == 'temporal_HDMapNet':
        model = TemporalHDMapNet(xbound, ybound, outC=outC)
    elif method == 'VPN':
        model = VPNet(outC=outC)

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
    plt.figure(figsize=(3*fW*val, (4.5*fW + 2*fH)*val))
    gs = mpl.gridspec.GridSpec(5, 3, height_ratios=(1.5*fW, 1.5*fW, 1.5*fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    max_pool = nn.MaxPool2d(3, padding=1, stride=1)
    post_processor = LaneNetPostProcessor(dbscan_eps=1.5, postprocess_min_samples=50)
    pca = PCA(n_components=3)

    color_map = []
    for i in range(30):
        color_map.append([random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)])

    model.eval()
    counter = 0
    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll, binimgs, inst_label) in enumerate(loader):
            out, embedded = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    translation.to(device),
                    yaw_pitch_roll.to(device),
                    )
            out = out.softmax(1).cpu()
            preds = onehot_encoding(out).cpu().numpy()
            embedded = embedded.cpu()
            inst_label = inst_label.sum(1)

            N, C, H, W = embedded.shape
            embedded_test = embedded.permute(0, 2, 3, 1).reshape(N*H*W, C)
            embedded_fitted = pca.fit_transform(embedded_test)
            embedded_fitted = torch.sigmoid(torch.tensor(embedded_fitted)).numpy()
            embedded_fitted = embedded_fitted.reshape((N, H, W, 3))

            if temporal:
                imgs = imgs[:, 0]
            # visualization
            binimgs[binimgs < 0.1] = np.nan
            seg_mask = out.numpy()
            seg_mask[seg_mask < 0.1] = np.nan
            for si in range(imgs.shape[0]):
                plt.clf()
                for imgi, img in enumerate(imgs[si]):
                    ax = plt.subplot(gs[3 + imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    # flip the bottom images
                    if imgi > 2:
                        showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                    plt.imshow(showimg)
                    plt.axis('off')
                    plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction')

                inst_mask = np.zeros((200, 400), dtype='int32')
                inst_mask_pil = np.zeros((200, 400, 4), dtype='uint8')

                simplified_coords = []
                simplified_mask = np.zeros((200, 400), dtype='int32')
                simplified_mask_pil = np.zeros((200, 400, 4), dtype='uint8')

                count = 0
                for i in range(1, preds.shape[1]):
                    single_mask = preds[si][i].astype('uint8')
                    single_embedded = embedded[si].permute(1, 2, 0)
                    single_class_inst_mask, single_class_inst_coords = post_processor.postprocess(single_mask, single_embedded)
                    if single_class_inst_mask is None:
                        continue

                    num_inst = len(single_class_inst_coords)

                    prob = out[si][i]
                    prob[single_class_inst_mask == 0] = 0
                    max_pooled = max_pool(prob.unsqueeze(0))[0]
                    nms_mask = ((max_pooled - prob) < 1e-6).numpy()

                    for j in range(1, num_inst+1):
                        idx = np.where(nms_mask & (single_class_inst_mask == j))
                        # idx = np.where((single_class_inst_mask == j))
                        if len(idx[0]) == 0:
                            continue

                        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

                        range_0 = np.max(lane_coordinate[:, 0]) - np.min(lane_coordinate[:, 0])
                        range_1 = np.max(lane_coordinate[:, 1]) - np.min(lane_coordinate[:, 1])
                        if range_0 > range_1:
                            lane_coordinate = np.stack(sorted(lane_coordinate, key=lambda x: x[0]))
                        else:
                            lane_coordinate = np.stack(sorted(lane_coordinate, key=lambda x: x[1]))
                        simplified_coords.append(lane_coordinate)
                        simplified_mask = cv2.polylines(simplified_mask, [lane_coordinate], False, color=count+j, thickness=1)

                    inst_mask[single_class_inst_mask != 0] += single_class_inst_mask[single_class_inst_mask != 0] + count
                    count += num_inst

                for i in range(1, count+1):
                    inst_mask_pil[inst_mask == i, :3] = color_map[i]
                    inst_mask_pil[inst_mask == i, 3] = 150

                for i in range(1, count+1):
                    simplified_mask_pil[simplified_mask == i, :3] = color_map[-i]
                    simplified_mask_pil[simplified_mask == i, 3] = 255

                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                embedded_fitted[si][inst_label[si] == 0] = 0
                plt.imshow(embedded_fitted[si])
                plt.xlim((0, binimgs.shape[3]))
                plt.ylim((0, binimgs.shape[2]))
                add_ego(bx, dx)

                ax = plt.subplot(gs[1, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                plt.imshow(inst_mask_pil)
                plt.imshow(simplified_mask_pil)
                plt.xlim((0, binimgs.shape[3]))
                plt.ylim((0, binimgs.shape[2]))
                add_ego(bx, dx)


                ax = plt.subplot(gs[2, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                plt.setp(ax.spines.values(), color='b', linewidth=2)
                plt.imshow(seg_mask[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
                plt.imshow(seg_mask[si][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
                plt.imshow(seg_mask[si][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)

                plt.imshow(binimgs[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
                plt.imshow(binimgs[si][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
                plt.imshow(binimgs[si][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)

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
