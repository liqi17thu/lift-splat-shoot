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
from .hd_models import HDMapNet
from .hd_models import TemporalHDMapNet


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


def train(version,
          dataroot='data/nuScenes',
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
          finetune=False,
          modelf='output/refine_data_HDMapNet/model130000.pt',

          delta_v=0.5,
          delta_d=3.0,

          scale_seg=1.0,
          scale_var=1.0,
          scale_dist=1.0,

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
        world_size = torch.distributed.get_world_size()

    [trainloader, valloader], [train_sampler, val_sampler] = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                                              grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                                              parser_name=parser_name, distributed=distributed)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    if method == 'lift_splat':
        model = compile_model(grid_conf, data_aug_conf, outC=outC)
    elif method == 'HDMapNet':
        model = HDMapNet(xbound, ybound, outC=outC, instance_seg=instance_seg, embedded_dim=embedded_dim)
    elif method == 'temporal_HDMapNet':
        model = TemporalHDMapNet(xbound, ybound, outC=outC, instance_seg=instance_seg, embedded_dim=embedded_dim)

    if finetune:
        model.load_state_dict(torch.load(modelf), strict=False)
        for name, param in model.named_parameters():
            if 'bevencode.up' in name or 'bevencode.layer3' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    model.to(device)
    if distributed:
        model = NativeDDP(model, device_ids=[local_rank], find_unused_parameters=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = StepLR(opt, 10, 0.1)

    # loss_fn = FocalLoss(alpha=.25, gamma=2.).cuda(gpuid)
    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)
    embedded_loss_fn = DiscriminativeLoss(embedded_dim, delta_v, delta_d).cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)
    val_step = 1000 if version == 'mini' else 10000

    model.train()
    counter = 0
    for epoch in range(nepochs):
        if distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        np.random.seed()
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll, binimgs, inst_mask) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()
            preds, embedded = model(imgs.to(device),
                                    rots.to(device),
                                    trans.to(device),
                                    intrins.to(device),
                                    post_rots.to(device),
                                    post_trans.to(device),
                                    translation.to(device),
                                    yaw_pitch_roll.to(device),
                                    )
            binimgs = binimgs.to(device)
            inst_mask = inst_mask.to(device)
            seg_loss = loss_fn(preds, binimgs)
            var_loss, dist_loss, reg_loss = embedded_loss_fn(embedded, inst_mask)
            final_loss = seg_loss * scale_seg + var_loss * scale_var + dist_loss * scale_dist
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print(counter, seg_loss.item(), t1 - t0)
                writer.add_scalar('train/seg_loss', seg_loss, counter)
                writer.add_scalar('train/var_loss', var_loss, counter)
                writer.add_scalar('train/dist_loss', dist_loss, counter)
                writer.add_scalar('train/reg_loss', reg_loss, counter)
                writer.add_scalar('train/final_loss', final_loss, counter)

            if counter % 50 == 0:
                _, _, ious = get_batch_iou_multi_class(preds, binimgs)
                _, _, _, _, _, acces, precs, recalls = get_accuracy_precision_recall_multi_class(preds, binimgs)
                write_log(writer, ious, acces, precs, recalls, 'train', counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                val_info = get_val_info(model, valloader, loss_fn, embedded_loss_fn, scale_seg, scale_var, scale_dist, device)
                print('VAL', val_info)
                writer.add_scalar('val/seg_loss', val_info['seg_loss'], counter)
                writer.add_scalar('val/var_loss', val_info['var_loss'], counter)
                writer.add_scalar('val/dist_loss', val_info['dist_loss'], counter)
                writer.add_scalar('val/reg_loss', val_info['reg_loss'], counter)
                writer.add_scalar('val/final_loss', val_info['final_loss'], counter)
                write_log(writer, val_info['iou'], val_info['accuracy'], val_info['precision'], val_info['recall'], 'val', counter)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()

        sched.step(epoch)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import argparse
parser = argparse.ArgumentParser(description='Train HDMapNet', add_help=False)
parser.add_argument('--version', default='mini', help='data version')
parser.add_argument('--dataroot', default='data/nuScenes', help='data root')
parser.add_argument('--nepochs', default=30)
parser.add_argument('--gpuid', default=1)
parser.add_argument('--outC', default=4, help='classes of segmentation')
parser.add_argument('--method', default='temporal_HDMapNet', choices=['lift-splat', 'HDMapNet', 'temporal_HDMapNet'])
parser.add_argument('--preprocess', default=False, type=str2bool)
parser.add_argument('--H', default=900, type=int)
parser.add_argument('--W', default=1600, type=int)
parser.add_argument('--final_h', default=128, type=int)
parser.add_argument('--final_w', default=352, type=int)
parser.add_argument('--ncams', default=6, type=int)
parser.add_argument('--max_grad_norm', default=5.0, type=float)
parser.add_argument('--logdir', default='./runs', type=str)
parser.add_argument('--xbound', default=[-30.0, 30.0, 0.15], type=float, nargs=3)
parser.add_argument('--ybound', default=[-15.0, 15.0, 0.15], type=float, nargs=3)
parser.add_argument('--zbound', default=[-10.0, 10.0, 20.0], type=float, nargs=3)
parser.add_argument('--dbound', default=[-4.0, 45.0, 1.0], type=float, nargs=3)
parser.add_argument('--instance_seg', default=True, type=str2bool)
parser.add_argument('--embedded_dim', default=16)
parser.add_argument('--finetune', default=False, type=str2bool)
parser.add_argument('--modelf', default='output/refine_data_HDMapNet/model130000.pt', type=str)

parser.add_argument('--delta_v', default=0.5, type=float)
parser.add_argument('--delta_d', default=3.0, type=float)

parser.add_argument('--scale_seg', default=1.0, type=float)
parser.add_argument('--scale_var', default=1.0, type=float)
parser.add_argument('--scale_dist', default=1.0, type=float)
parser.add_argument('--bsz', default=4, type=int)
parser.add_argument('--nworkers', default=10, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--weight_decay', default=1e-7, type=float)

parser.add_argument('--distributed', default=False, type=str2bool)


parser.add_argument('--local_rank', default=0)


if __name__ == '__main__':
    args = parser.parse_args()
    train(version=args.version,
          dataroot=args.dataroot,
          nepochs=args.dataroot,
          gpuid=args.gpuid,
          outC=args.outC,
          method=args.method,
          preprocess=args.preprocess,
          H=args.H,
          W=args.W,
          final_dim=args.final_dim,
          ncams=args.ncams,
          line_width=args.line_width,
          max_grad_norm=args.max_grad_norm,
          pos_weight=args.pos_weight,
          logdir=args.logdir,

          xbound=args.xbound,
          ybound=args.ybound,
          zbound=args.zbound,
          dbound=args.dbound,

          instance_seg=args.instance_seg,
          embedded_dim=args.embedded_dim,
          finetune=args.finetune,
          modelf=args.modelf,

          delta_v=args.delta_v,
          delta_d=args.delta_d,

          scale_seg=args.scale_seg,
          scale_var=args.scale_var,
          scale_dist=args.scale_dist,

          bsz=args.bsz,
          nworkers=args.nworkers,
          lr=args.lr,
          weight_decay=args.weight_decay,
          distributed=args.distributed,
          local_rank=args.local_rank
          )