import argparse

from torch import nn
from torch import optim

from catalyst import utils
from catalyst.dl import SupervisedRunner
from catalyst.contrib.nn import RAdam, Lookahead, DiceLoss, IoULoss
from catalyst.dl import DiceCallback, IouCallback, CriterionCallback, MetricAggregationCallback
from catalyst.contrib.callbacks import DrawMasksCallback
import segmentation_models_pytorch as smp

from .dataset import get_loaders
from .transform import resize_transforms, hard_transforms, post_transforms, compose, pre_transforms

parser = argparse.ArgumentParser(description='First-View Segmentation')
parser.add_argument('--seed', default=42, type=int)

parser.add_argument('--dataroot', default='data/nuImages', type=str)
parser.add_argument('--version', default='trainval', type=str)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--wd', default=0.0003, type=float)
parser.add_argument('--encoder-lr', default=0.0005, type=float)
parser.add_argument('--encoder-wd', default=0.00003, type=float)


parser.add_argument('--bs', default=64, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--logdir', default='./logs/segmentation', type=str)


def main():
    args = parser.parse_args()

    utils.set_global_seed(args.seed)
    utils.prepare_cudnn(deterministic=True)

    train_transforms = compose([
        resize_transforms(),
        hard_transforms(),
        post_transforms()
    ])

    valid_transforms = compose([pre_transforms(), post_transforms()])

    loaders = get_loaders(
        dataroot=args.dataroot,
        version=args.version,
        train_transforms_fn=train_transforms,
        valid_transforms_fn=valid_transforms,
        batch_size=args.bs
    )

    # We will use Feature Pyramid Network with pre-trained ResNeXt50 backbone
    model = smp.FPN(encoder_name="resnext50_32x4d", classes=1)

    # we have multiple criterions
    criterion = {
        "dice": DiceLoss(),
        "iou": IoULoss(),
        "bce": nn.BCEWithLogitsLoss()
    }

    # Since we use a pre-trained encoder, we will reduce the learning rate on it.
    layerwise_params = {"encoder*": dict(lr=args.encoder_lr, weight_decay=args.encoder_wd)}

    # This function removes weight_decay for biases and applies our layerwise_params
    model_params = utils.process_model_params(model, layerwise_params=layerwise_params)

    # Catalyst has new SOTA optimizers out of box
    base_optimizer = RAdam(model_params, lr=args.lr, weight_decay=args.wd)
    optimizer = Lookahead(base_optimizer)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)

    device = utils.get_device()

    is_fp16_used = False
    if is_fp16_used:
        fp16_params = dict(opt_level="O1")  # params for FP16
    else:
        fp16_params = None

    # by default SupervisedRunner uses "features" and "targets",
    # in our case we get "image" and "mask" keys in dataset __getitem__
    runner = SupervisedRunner(device=device, input_key="image", input_target_key="mask")


    callbacks = [
        # Each criterion is calculated separately.
        CriterionCallback(
            input_key="mask",
            prefix="loss_dice",
            criterion_key="dice"
        ),
        CriterionCallback(
            input_key="mask",
            prefix="loss_iou",
            criterion_key="iou"
        ),
        CriterionCallback(
            input_key="mask",
            prefix="loss_bce",
            criterion_key="bce"
        ),

        # And only then we aggregate everything into one loss.
        MetricAggregationCallback(
            prefix="loss",
            mode="weighted_sum",  # can be "sum", "weighted_sum" or "mean"
            # because we want weighted sum, we need to add scale for each loss
            metrics={"loss_dice": 1.0, "loss_iou": 1.0, "loss_bce": 0.8},
        ),

        # metrics
        DiceCallback(input_key="mask"),
        IouCallback(input_key="mask"),
        # visualization
        DrawMasksCallback(output_key='logits',
                          input_image_key='image',
                          input_mask_key='mask',
                          summary_step=50
        )
    ]

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        # our dataloaders
        loaders=loaders,
        # We can specify the callbacks list for the experiment;
        callbacks=callbacks,
        # path to save logs
        logdir=args.logdir,
        num_epochs=args.epochs,
        # save our best checkpoint by IoU metric
        main_metric="iou",
        # IoU needs to be maximized.
        minimize_metric=False,
        # for FP16. It uses the variable from the very first cell
        fp16=fp16_params,
        # prints train logs
        verbose=True,
    )

    batch = next(iter(loaders["valid"]))
    # saves to `logdir` and returns a `ScriptModule` class
    runner.trace(model=model, batch=batch, logdir=args.logdir, fp16=is_fp16_used)

import matplotlib.pyplot as plt

if __name__ == '__main__':
    main()
