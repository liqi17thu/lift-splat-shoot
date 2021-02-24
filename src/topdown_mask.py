import cv2

import numpy as np
from typing import Tuple

from nuscenes.eval.common.utils import quaternion_yaw

from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from pyquaternion import Quaternion
from shapely.geometry import LineString


LINE_WIDTH = 5


class MyNuScenesMap(NuScenesMap):
    def __init__(self,
                 dataroot: str = '/data/sets/nuscenes',
                 map_name: str = 'singapore-onenorth'):
        super(MyNuScenesMap, self).__init__(dataroot, map_name)
        self.explorer = MyNuScenesMapExplorer(self)


class MyNuScenesMapExplorer(NuScenesMapExplorer):
    def __init__(self,
                 map_api: MyNuScenesMap,
                 representative_layers: Tuple[str] = ('drivable_area', 'lane', 'walkway'),
                 color_map: dict = None):
        super(MyNuScenesMapExplorer, self).__init__(map_api, representative_layers, color_map)

    @staticmethod
    def mask_for_lines(lines: LineString, mask: np.ndarray) -> np.ndarray:
        if lines.geom_type == 'MultiLineString':
            for line in lines:
                coords = np.asarray(list(line.coords), np.int32)
                coords = coords.reshape((-1, 2))
                cv2.polylines(mask, [coords], False, 1, LINE_WIDTH)
        else:
            coords = np.asarray(list(lines.coords), np.int32)
            coords = coords.reshape((-1, 2))
            cv2.polylines(mask, [coords], False, 1, LINE_WIDTH)
        return mask


def gen_topdown_mask(nuscene, nusc_maps, sample_record, patch_size, canvas_size, seg_layers):
    sample_record_data = sample_record['data']
    sample_data_record = nuscene.get('sample_data', sample_record_data['LIDAR_TOP'])

    pose_record = nuscene.get('ego_pose', sample_data_record['ego_pose_token'])
    map_pose = pose_record['translation'][:2]
    rotation = Quaternion(pose_record['rotation'])

    patch_box = (map_pose[0], map_pose[1], patch_size[0], patch_size[1])
    patch_angle = quaternion_yaw(rotation) / np.pi * 180

    scene_record = nuscene.get('scene', sample_record['scene_token'])
    log_record = nuscene.get('log', scene_record['log_token'])
    location = log_record['location']
    topdown_seg_mask = nusc_maps[location].get_map_mask(patch_box, patch_angle, seg_layers, canvas_size)
    topdown_seg_mask = np.flip(topdown_seg_mask, 1)  # left-right correction
    return topdown_seg_mask
