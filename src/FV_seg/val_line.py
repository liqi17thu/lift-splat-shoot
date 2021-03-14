import argparse
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from catalyst import utils
from catalyst.runners import SupervisedRunner

from .dataset import FirstViewSegDataset, CrossViewSegDataset
from .transform import compose, pre_transforms, post_transforms
from ..homography import IPM

parser = argparse.ArgumentParser(description='First-View Segmentation')
parser.add_argument('--seed', default=42, type=int)

parser.add_argument('--dataroot', default='data/nuScenes', type=str)
parser.add_argument('--version', default='v1.0-trainval', type=str)

parser.add_argument('--img-size', default=224, type=int)
parser.add_argument('--bs', default=4, type=int)
parser.add_argument('--num-workers', default=4, type=int)

parser.add_argument('--resume-path', default='logs/segmentation/checkpoints/best.pth', type=str)
parser.add_argument('--max-count', default=100, type=int)


def main():
    arg = parser.parse_args()

    sample_transforms = compose([pre_transforms(arg.img_size)])
    valid_transforms = compose([pre_transforms(arg.img_size), post_transforms()])

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

    plt.figure(figsize=(w*3 / 100, (w*1.5+h*2) / 100))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*w, h, h))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)
    sample_num = test_dataset.__len__() // 6

    xbound = [-30, 30, 0.15]
    ybound = [-15, 15, 0.15]
    ipm = IPM(xbound, ybound, z_roll_pitch=False, visual=True)
    for i in range(sample_num):
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

        images = images.permute(0, 3, 1, 2)
        image_IPM = ipm(images[None], Ks, RTs, post_RTs=post_RTs)
        # mask_label_IPM = ipm(masks_label[None, :, None, ...], Ks, RTs)
        mask_pred_IPM = ipm(masks_pred[None, :, None, ...], Ks, RTs, post_RTs=post_RTs, visual_sum=True)
        images = images.permute(0, 2, 3, 1)
        image_IPM = image_IPM[0].permute(1, 2, 0)
        # mask_label_IPM = mask_label_IPM.squeeze()
        mask_pred_IPM = mask_pred_IPM.squeeze()

        plt.clf()
        for j in range(6):
            plt.subplot(gs[j//3+1, (j % 3+1) % 3])
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
            mask_pred[mask_pred < 0.1] = np.nan
            # plt.imshow(mask_label, cmap='Blues', vmin=0, vmax=1, alpha=0.6)
            plt.imshow(mask_pred, cmap='Reds', vmin=0, vmax=1, alpha=0.6)

        ax = plt.subplot(gs[0, :])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plt.imshow(image_IPM)
        # mask_label_IPM = mask_label_IPM.numpy().astype('float')
        # mask_label_IPM[mask_label_IPM < 0.1] = np.nan
        mask_pred_IPM = mask_pred_IPM.numpy().astype('float')
        mask_pred_IPM[mask_pred_IPM < 0.1] = np.nan
        # plt.imshow(mask_label_IPM, cmap='Blues', vmin=0, vmax=1, alpha=0.6)
        plt.imshow(mask_pred_IPM, cmap='Reds', vmin=0, vmax=1, alpha=0.6)

        plt.savefig(f'{i}.jpg')
        print(f'{i}.jpg')

        if i >= arg.max_count:
            break


if __name__ == '__main__':
    main()
