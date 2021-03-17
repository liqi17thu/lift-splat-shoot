import argparse
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from catalyst import utils
from catalyst.runners import SupervisedRunner

from .dataset import FirstViewSegDataset, CrossViewSegDataset
from .transform import compose, pre_transforms, post_transforms
from .homography import IPM

parser = argparse.ArgumentParser(description='First-View Segmentation')
parser.add_argument('--seed', default=42, type=int)

parser.add_argument('--dataroot', default='data/nuScenes', type=str)
parser.add_argument('--version', default='v1.0-trainval', type=str)

parser.add_argument('--img-size', default=224, type=int)
parser.add_argument('--bs', default=4, type=int)
parser.add_argument('--num-workers', default=4, type=int)

parser.add_argument('--resume-path', default='logs/segmentation/checkpoints/best.pth', type=str)
parser.add_argument('--max-count', default=5, type=int)

import cv2
from shapely import affinity, ops
from shapely.geometry import LineString, MultiLineString, box, MultiPolygon, Polygon

def extract_contour(topdown_seg_mask, canvas_size, thickness=5):
    topdown_seg_mask[topdown_seg_mask != 0] = 255
    ret, thresh = cv2.threshold(topdown_seg_mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(topdown_seg_mask)
    patch = box(1, 1, canvas_size[1] - 2, canvas_size[0] - 2)
    lines = []
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
                coord = np.asarray(list(l.coords), np.int32).reshape((-1, 2))
                cv2.polylines(mask, [coord], False, color=idx, thickness=thickness)
                lines.append(coord)
        elif isinstance(line, LineString):
            idx += 1
            coord = np.asarray(list(line.coords), np.int32).reshape((-1, 2))
            cv2.polylines(mask, [coord], False, color=idx, thickness=thickness)
            lines.append(coord)

    return mask, lines, idx

def main():
    arg = parser.parse_args()

    # sample_transforms = compose([pre_transforms(arg.img_size)])
    sample_transforms = compose([pre_transforms([300, 400])])
    valid_transforms = compose([pre_transforms([arg.img_size, arg.img_size]), post_transforms()])

    # create test dataset
    test_dataset = CrossViewSegDataset(
        dataroot=arg.dataroot,
        version=arg.version,
        transforms=valid_transforms,
        sample_transforms=sample_transforms,
        num_samples=arg.max_count
    )

    infer_loader = DataLoader(
        test_dataset,
        batch_size=arg.bs,
        shuffle=False,
        num_workers=arg.num_workers
    )

    # model = smp.FPN(encoder_name="resnext50_32x4d", classes=1)
    model = torch.jit.load(arg.resume_path)

    # tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode="mean")

    tta_runner = SupervisedRunner(
        model=model,
        device=utils.get_device(),
        input_key="image"
    )

    predictions = np.vstack(list(map(
        lambda x: x["logits"].cpu().numpy(),
        tta_runner.predict_loader(loader=infer_loader)
    )))

    w, h = arg.img_size, arg.img_size
    w, h = 400, 300

    plt.figure(figsize=(w*3 / 100, (w*1.5+h*2) / 100))
    # gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*w, h, h))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(h, h, 1.5*w))

    # plt.figure(figsize=(w*3 / 100, (h*2) / 100))
    # gs = mpl.gridspec.GridSpec(2, 3, height_ratios=(h, h))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)
    sample_num = test_dataset.__len__() // 6
    max_pool = nn.MaxPool2d(7, padding=3, stride=1)


    xbound = [-30, 30, 0.15]
    ybound = [-15, 15, 0.15]
    ipm = IPM(xbound, ybound, z_roll_pitch=False, visual=True)

    color_map = []
    for i in range(2000):
        color_map.append([random.randint(0, 256) / 255., random.randint(0, 256) / 255., random.randint(0, 256) / 255.])

    for i in range(sample_num):
        # if i < 75:
        #     continue
        # logits = predictions[i]
        # mask_pred = torch.from_numpy(logits).sigmoid().squeeze()
        # image, mask_label = test_dataset.get_sample(i)
        # plt.axis('off')
        # plt.imshow(image)
        # mask_pred = mask_pred.numpy().astype('float')
        # mask_pred[mask_pred < 0.1] = np.nan
        # plt.imshow(mask_pred, cmap='Reds', vmin=0, vmax=1)
        # plt.savefig(f'{i}.jpg')
        # print(f'saving {i}.jpg')
        # continue

        images, masks_label, Ks, RTs = test_dataset.get_sample(i)
        logits = predictions[i*6:(i+1)*6]
        masks_pred = torch.from_numpy(logits).sigmoid().squeeze()
        # masks_pred = utils.detach(masks_pred).astype('float')
        # masks_pred = utils.detach(masks_pred > threshold).astype('float')
        # print(len(masks_pred[masks_pred != 0]))

        # post trans
        post_RTs = torch.tensor([
            [arg.img_size/1600, 0, 0, 0],
            [0, arg.img_size/900, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        topdown_lines = []
        contours = []
        plt.clf()
        for j in range(6):
            plt.subplot(gs[j//3, j % 3])
            if j <= 2:
                image = images[j]
                # mask_label = masks_label[j]
                mask_pred = masks_pred[j]
            else:
                image = torch.flip(images[j], [1])
                # mask_label = torch.flip(masks_label[j], [1])
                mask_pred = torch.flip(masks_pred[j], [1])

            plt.axis('off')
            plt.imshow(image)
            # mask_label = mask_label.numpy().astype('float')
            # mask_label[mask_label < 0.1] = np.nan
            mask_pred = mask_pred.numpy().astype('float')
            contour, lines, idx = extract_contour((mask_pred > 0.2).astype('uint8'), (224, 224))
            contours.append(contour)
            mask_pred = cv2.resize(mask_pred, (400, 300))
            mask_pred[mask_pred < 0.1] = np.nan
            # plt.imshow(mask_label, cmap='Blues', vmin=0, vmax=1, alpha=0.6)
            plt.imshow(mask_pred, cmap='Reds', vmin=0, vmax=1, alpha=0.3)
            for k, line in enumerate(lines):
                plt.plot(line[:, 0] * 400 / 224, line[:, 1] * 300 / 224, linewidth=5, color=color_map[6*i+j+k])
                line = np.concatenate([line, np.zeros(line.shape[0]).reshape(-1, 1), np.ones(line.shape[0]).reshape(-1, 1)], -1)
                # line = np.linalg.inv(post_RTs @ Ks[j] @ RTs[j]) @ np.moveaxis(line, 0, 1)
                line = np.linalg.inv(post_RTs @ Ks[j] @ RTs[j]) @ np.moveaxis(line, 0, 1)
                # line = line[:2] / line[2]
                topdown_lines.append(line)

        contours = np.stack(contours)
        contours = torch.tensor(contours)
        images = images.permute(0, 3, 1, 2)
        image_IPM = ipm(images[None], Ks, RTs, post_RTs=post_RTs)
        contour_IPM = ipm(contours[None, :, None, ...], Ks, RTs, post_RTs=post_RTs)
        # mask_label_IPM = ipm(masks_label[None, :, None, ...], Ks, RTs)
        mask_pred_IPM = ipm(masks_pred[None, :, None, ...], Ks, RTs, post_RTs=post_RTs, visual_sum=True)
        images = images.permute(0, 2, 3, 1)
        image_IPM = image_IPM[0].permute(1, 2, 0)
        # mask_label_IPM = mask_label_IPM.squeeze()
        mask_pred_IPM = mask_pred_IPM.squeeze()

        ax = plt.subplot(gs[2, :])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        # plt.imshow(image_IPM)
        contour_IPM = contour_IPM[0, 0].numpy().astype('float')
        contour_IPM[contour_IPM < 0.1] = np.nan
        plt.imshow(contour_IPM, cmap='Greens', vmin=0, vmax=1, alpha=0.6)
        # mask_label_IPM = mask_label_IPM.numpy().astype('float')
        # mask_label_IPM[mask_label_IPM < 0.1] = np.nan
        mask_pred_IPM = mask_pred_IPM.numpy().astype('float')
        mask_pred_IPM[mask_pred_IPM < 0.1] = np.nan
        # plt.imshow(mask_label_IPM, cmap='Blues', vmin=0, vmax=1, alpha=0.6)
        # plt.imshow(mask_pred_IPM, cmap='Reds', vmin=0, vmax=1, alpha=0.6)
        # for line in topdown_lines:
        #     plt.plot(line[0], line[1], linewidth=5)

        plt.savefig(f'{i}.jpg')
        print(f'{i}.jpg')

        if i >= arg.max_count:
            break


if __name__ == '__main__':
    main()