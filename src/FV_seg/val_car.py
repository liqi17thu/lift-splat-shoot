import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from catalyst import utils
from catalyst.runners import SupervisedRunner

from .dataset import SegmentationDataset
from .transform import compose, pre_transforms, post_transforms
import ttach as tta

parser = argparse.ArgumentParser(description='First-View Segmentation')
parser.add_argument('--seed', default=42, type=int)

parser.add_argument('--img-size', default=224, type=int)
parser.add_argument('--bs', default=4, type=int)
parser.add_argument('--num-workers', default=4, type=int)

parser.add_argument('--resume-path', default='logs/segmentation/checkpoints/best.pth', type=str)
parser.add_argument('--max-count', default=5, type=int)


def main():
    args = parser.parse_args()

    valid_transforms = compose([pre_transforms(args.img_size), post_transforms()])

    from pathlib import Path

    ROOT = Path("segmentation_data/")

    test_image_path = ROOT / "test"
    ALL_IMAGES = sorted(test_image_path.glob("*.jpg"))

    # create test dataset
    test_dataset = SegmentationDataset(
        images=ALL_IMAGES,
        transforms=valid_transforms,
    )

    infer_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers
    )

    model = torch.jit.load(args.resume_path)

    tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode="mean")

    tta_runner = SupervisedRunner(
        model=tta_model,
        device=utils.get_device(),
        input_key="image"
    )

    predictions = np.vstack(list(map(
        lambda x: x["logits"].cpu().numpy(),
        tta_runner.predict_loader(loader=infer_loader)
    )))

    threshold = 0.5
    for i, (features, logits) in enumerate(zip(test_dataset, predictions)):
        image = utils.tensor_to_ndimage(features["image"])

        mask_ = torch.from_numpy(logits[0]).sigmoid()
        mask = utils.detach(mask_ > threshold).astype("float")
        mask[mask < 0.1] = np.nan

        plt.clf()
        plt.imshow(image)
        plt.imshow(mask, cmap='Reds', vmin=0, vmax=1)
        plt.savefig(f'car_{i}.jpg')

        if i >= args.max_count:
            break


if __name__ == '__main__':
    main()
