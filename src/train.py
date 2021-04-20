"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os

import torch
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from .models import compile_model
from .data import compile_data
from .tools import get_batch_iou_multi_class, get_val_info
from .tools import get_accuracy_precision_recall_multi_class
from .tools import FocalLoss, SimpleLoss, DiscriminativeLoss
from .hd_models import HDMapNet, TemporalHDMapNet
from .vpn_model import VPNet, TemporalVPNet
# from .vit_model import VITNet
from .pointpillar import PointPillar
from .ori_vpn import VPNModel


import argparse
parser = argparse.ArgumentParser(description='Train HDMapNet', add_help=False)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--logdir', default='runs', type=str)
parser.add_argument('--bsz', default=4, type=int)
parser.add_argument('--version', default='mini', type=str)
parser.add_argument('--method', default='temporal_HDMapNet', type=str)
parser.add_argument('--distributed', default=True, type=bool)


def write_log(writer, ious, acces, precs, recalls, title, counter):
    writer.add_scalar(f'{title}/iou', np.mean(ious[1:]), counter)
    writer.add_scalar(f'{title}/acc', np.mean(acces[1:]), counter)
    writer.add_scalar(f'{title}/prec', np.mean(precs[1:]), counter)
    writer.add_scalar(f'{title}/recall', np.mean(recalls[1:]), counter)

    for i, iou in enumerate(ious):
        writer.add_scalar(f'{title}/class_{i}/iou', iou, counter)
    for i, acc in enumerate(acces):
        writer.add_scalar(f'{title}/class_{i}/acc', acc, counter)
    for i, prec in enumerate(precs):
        writer.add_scalar(f'{title}/class_{i}/prec', prec, counter)
    for i, recall in enumerate(recalls):
        writer.add_scalar(f'{title}/class_{i}/recall', recall, counter)


def train(version='mini',
          dataroot='/mnt/datasets/nuScenes',
          nepochs=30,
          gpuid=0,
          outC=4,
          method='temporal_HDMapNet',
          preprocess=False,

          H=900, W=1600,
          final_dim=(128, 352),
          ncams=6,
          line_width=5,
          max_grad_norm=5.0,
          pos_weight=2.13,
          logdir='./runs',

          xbound=[-30.0, 30.0, 0.15],
          ybound=[-15.0, 15.0, 0.15],
          zbound=[-10.0, 10.0, 20.0],
          dbound=[4.0, 45.0, 1.0],

          instance_seg=True,
          embedded_dim=16,
          delta_v=0.5,
          delta_d=3.0,

          scale_seg=1.0,
          scale_var=1.0,
          scale_dist=1.0,

          finetune=False,
          modelf='output/refine_data_HDMapNet/model130000.pt',

          bsz=4,
          nworkers=10,
          lr=1e-3,
          weight_decay=1e-7,

          distributed=False,
          local_rank=0,
          ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'final_dim': final_dim,
                    'H': H, 'W': W,
                    'rand_flip': False,
                    'preprocess': preprocess,
                    'line_width': line_width,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
                }

    if 'temporal' in method:
        parser_name = 'temporalsegmentationdata'
    else:
        parser_name = 'segmentationdata'

    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    [trainloader, valloader], [train_sampler, val_sampler] = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                                              grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                                              parser_name=parser_name, distributed=distributed)

    if method == 'lift_splat':
        model = compile_model(grid_conf, data_aug_conf, outC=outC, instance_seg=instance_seg, embedded_dim=embedded_dim)
    elif method == 'HDMapNet':
        model = HDMapNet(xbound, ybound, outC=outC, instance_seg=instance_seg, embedded_dim=embedded_dim)
    elif method == 'temporal_HDMapNet':
        model = TemporalHDMapNet(xbound, ybound, outC=outC, instance_seg=instance_seg, embedded_dim=embedded_dim)
    elif method == 'temporal_VPN':
        model = TemporalVPNet(xbound, ybound, outC, instance_seg=instance_seg, embedded_dim=embedded_dim)
    elif method == 'VPN':
        model = VPNet(outC, instance_seg=instance_seg, embedded_dim=embedded_dim)
    elif method == 'VIT':
        model = VITNet(outC, instance_seg=instance_seg, embedded_dim=embedded_dim)
    elif method == 'PP':
        model = PointPillar(outC, xbound, ybound, zbound, embedded_dim=embedded_dim)
    elif method == 'VPNPP':
        model = VPNet(outC, instance_seg=instance_seg, embedded_dim=embedded_dim, lidar=True, xbound=xbound, ybound=ybound, zbound=zbound)
    elif method == 'ori_VPN':
        model = VPNModel(outC, instance_seg=instance_seg, embedded_dim=embedded_dim)
    else:
        raise NotImplementedError

    if finetune:
        model.load_state_dict(torch.load(modelf), strict=False)
        for name, param in model.named_parameters():
            if 'bevencode.up' in name or 'bevencode.layer3' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    model.cuda()

    if distributed:
        model = NativeDDP(model, device_ids=[local_rank], find_unused_parameters=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = StepLR(opt, 10, 0.1)

    # loss_fn = FocalLoss(alpha=.25, gamma=2.).cuda(gpuid)
    loss_fn = SimpleLoss(pos_weight).cuda()
    embedded_loss_fn = DiscriminativeLoss(embedded_dim, delta_v, delta_d).cuda()
    direction_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

    writer = SummaryWriter(logdir=logdir)
    val_step = 1000 if version == 'mini' else 10000

    model.train()
    counter = 0
    for epoch in range(nepochs):
        if distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        np.random.seed()
        for batchi, (points, points_mask, imgs, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll, binimgs, inst_mask, direction_mask) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()
            preds, embedded, direction = model(points.cuda(),
                                    points_mask.cuda(),
                                    imgs.cuda(),
                                    rots.cuda(),
                                    trans.cuda(),
                                    intrins.cuda(),
                                    post_rots.cuda(),
                                    post_trans.cuda(),
                                    translation.cuda(),
                                    yaw_pitch_roll.cuda(),
                                    )
            binimgs = binimgs.cuda()
            inst_mask = inst_mask.cuda().sum(1)
            direction_mask = direction_mask.cuda()
            seg_loss = loss_fn(preds, binimgs)
            var_loss, dist_loss, reg_loss = embedded_loss_fn(embedded, inst_mask)
            direction_loss = direction_loss_fn(direction, direction_mask)
            lane_mask = (1 - direction_mask[:, 0]).unsqueeze(1)
            direction_loss = (direction_loss * lane_mask).sum() / (lane_mask.sum() * 361 + 1e-6)
            final_loss = seg_loss * scale_seg + var_loss * scale_var + dist_loss * scale_dist + direction_loss
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0 and local_rank == 0:
                print(counter, seg_loss.item(), t1 - t0)
                writer.add_scalar('train/seg_loss', seg_loss, counter)
                writer.add_scalar('train/var_loss', var_loss, counter)
                writer.add_scalar('train/dist_loss', dist_loss, counter)
                writer.add_scalar('train/reg_loss', reg_loss, counter)
                writer.add_scalar('train/direction_loss', direction_loss, counter)
                writer.add_scalar('train/final_loss', final_loss, counter)

            if counter % 50 == 0 and local_rank == 0:
                _, _, ious = get_batch_iou_multi_class(preds, binimgs)
                _, _, _, _, _, acces, precs, recalls = get_accuracy_precision_recall_multi_class(preds, binimgs)
                write_log(writer, ious, acces, precs, recalls, 'train', counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                val_info = get_val_info(model, valloader, loss_fn, embedded_loss_fn, direction_loss_fn, scale_seg, scale_var, scale_dist)
                if local_rank == 0:
                    print('VAL', val_info)
                    writer.add_scalar('val/seg_loss', val_info['seg_loss'], counter)
                    writer.add_scalar('val/var_loss', val_info['var_loss'], counter)
                    writer.add_scalar('val/dist_loss', val_info['dist_loss'], counter)
                    writer.add_scalar('val/reg_loss', val_info['reg_loss'], counter)
                    writer.add_scalar('val/direction_loss', val_info['direction_loss'], counter)
                    writer.add_scalar('val/final_loss', val_info['final_loss'], counter)
                    write_log(writer, val_info['iou'], val_info['accuracy'], val_info['precision'], val_info['recall'], 'val', counter)

            if counter % val_step == 0 and local_rank == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                if distributed:
                    torch.save(model.module.state_dict(), mname)
                else:
                    torch.save(model.state_dict(), mname)
                model.train()

        sched.step()


if __name__ == '__main__':
    args = parser.parse_args()

    train(local_rank=args.local_rank,
          logdir=args.logdir,
          bsz=args.bsz,
          version=args.version,
          distributed=args.distributed)
