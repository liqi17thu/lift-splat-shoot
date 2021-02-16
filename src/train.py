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

from .models import compile_model
from .data import compile_data
from .tools import get_batch_iou, get_val_info
from .tools import get_batch_iou_multi_class, get_val_info
from .tools import get_accuracy_precision_recall_multi_class
from .tools import FocalLoss, SimpleLoss
from .hd_models import HDMapNet

def write_log(writer, loss, ious, acces, precs, recalls, title, counter):
    writer.add_scalar(f'{title}/loss', loss, counter)
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
          dataroot='/data/nuscenes',
          nepochs=30,
          gpuid=1,
          outC=3,

          H=900, W=1600,
          resize_lim=(0.193, 0.225),
          final_dim=(128, 352),
          bot_pct_lim=(0.0, 0.22),
          rot_lim=(-5.4, 5.4),
          rand_flip=True,
          ncams=5,
          max_grad_norm=5.0,
          pos_weight=2.13,
          logdir='./runs',

          xbound=[-50.0, 50.0, 0.5],
          # ybound=[-50.0, 50.0, 0.5],
          ybound=[-15.0, 15.0, 0.15],
          zbound=[-10.0, 10.0, 20.0],
          dbound=[4.0, 45.0, 1.0],

          bsz=4,
          nworkers=10,
          lr=1e-3,
          weight_decay=1e-7,
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
                    'Ncams': ncams,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = HDMapNet(xbound, ybound, outC=outC)
    # model = compile_model(grid_conf, data_aug_conf, outC=outC)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = StepLR(opt, 10, 0.1)

    # loss_fn = FocalLoss(alpha=.25, gamma=2.).cuda(gpuid)
    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)
    val_step = 1000 if version == 'mini' else 10000

    model.train()
    counter = 0
    for epoch in range(nepochs):
        np.random.seed()
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()
            preds = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
            binimgs = binimgs.to(device)
            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print(counter, loss.item())
                writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0:
                _, _, ious = get_batch_iou_multi_class(preds, binimgs)
                _, _, _, _, _, acces, precs, recalls = get_accuracy_precision_recall_multi_class(preds, binimgs)
                write_log(writer, loss, ious, acces, precs, recalls, 'train', counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                val_info = get_val_info(model, valloader, loss_fn, device)
                print('VAL', val_info)
                write_log(writer, val_info['loss'], val_info['iou'], val_info['accuracy'], val_info['precision'], val_info['recall'], 'val', counter)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()

        sched.step(epoch)
