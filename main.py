"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

from fire import Fire

# import src
from src import explore, train


if __name__ == '__main__':
    Fire({
        'lidar_check': explore.lidar_check,
        'cumsum_check': explore.cumsum_check,

        'gen_data': explore.gen_data,
        'train': train.train,
        'eval_model': explore.eval_model,
        'viz_model_preds': explore.viz_model_preds,
        'viz_model_preds_class3': explore.viz_model_preds_class3,
        'viz_model_preds_inst': explore.viz_model_preds_inst,
    })
