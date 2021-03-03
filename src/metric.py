import torch
from chamferdist import ChamferDistance
from nuscenes import NuScenes

from src.topdown_mask import MyNuScenesMap
from .data import NuscData, MAP


def lane_semantic_chamfer_distance(seg_label, seg_pred):
    # seg_label: N, C, H, W
    # seg_pred: N, C, H, W
    chamfer_dist = ChamferDistance()
    N, C, H, W = seg_label.shape

    CD = torch.zeros((N, C), device=seg_label.device)
    for n in range(N):
        for c in range(C):
            label_pc_x, label_pc_y = torch.where(seg_label[n, c] != 0)
            pred_pc_x, pred_pc_y = torch.where(seg_pred[n, c] != 0)
            label_pc_coords = torch.stack([label_pc_x, label_pc_y], -1).float()
            pred_pc_coords = torch.stack([pred_pc_x, pred_pc_y], -1).float()
            dist = chamfer_dist(label_pc_coords[None], pred_pc_coords[None], bidirectional=True)
            CD[n, c] = dist
    dist = torch.mean(CD, 0)

    return dist


if __name__ == '__main__':
    version = 'mini'
    dataroot = 'data/nuScenes'

    H = 900
    W = 1600
    resize_lim = (0.193, 0.225)
    final_dim = (128, 352)
    bot_pct_lim = (0.0, 0.22)
    rot_lim = (-5.4, 5.4)
    rand_flip = False
    ncams = 6
    line_width = 1
    preprocess = False
    overwrite = False

    xbound = [-30.0, 30.0, 0.15]
    ybound = [-15.0, 15.0, 0.15]
    zbound = [-10.0, 10.0, 20.0]
    dbound = [4.0, 45.0, 1.0]

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

    nusc_data = NuscData(nusc, nusc_maps, False, data_aug_conf, grid_conf)

    rec = nusc.sample[0]
    seg_mask, inst_mask = nusc_data.get_lineimg(rec)
    chamfer_distance = lane_semantic_chamfer_distance(seg_mask[None], seg_mask[None])
    print(chamfer_distance)