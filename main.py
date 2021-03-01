"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

from fire import Fire

import src


if __name__ == '__main__':
    Fire({
        'lidar_check': src.explore.lidar_check,
        'cumsum_check': src.explore.cumsum_check,

        'gen_data': src.explore.gen_data,
        'train': src.train.train,
        'eval_model': src.explore.eval_model,
        'viz_model_preds': src.explore.viz_model_preds,
        'viz_model_preds_class3': src.explore.viz_model_preds_class3,
        'viz_model_preds_inst': src.explore.viz_model_preds_inst,
    })
