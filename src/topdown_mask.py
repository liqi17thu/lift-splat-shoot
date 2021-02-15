import cv2

import numpy as np
from typing import Tuple, List, Optional

from nuscenes.eval.common.utils import quaternion_yaw

from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer, Geometry
from pyquaternion import Quaternion
from shapely import affinity
from shapely.geometry import LineString


LINE_WIDTH = 5

def pickone_and_reorder(node1, node2, dist, node_num):
    if len(node1) == 2 and len(node2) == 2:
        node1, node2 = np.array(node1), np.array(node2)
        idx_array = abs(node1 - node2.reshape(-1, 1)) == dist
        node1 = node1[np.sum(idx_array, axis=0) == True][0]
        node2 = node2[np.sum(idx_array, axis=1) == True][0]
    elif len(node1) == 2:
        node2 = node2[0]
        node1 = node1[abs(np.array(node1) - node2) == dist][0]
    elif len(node2) == 2:
        node1 = node1[0]
        node2 = node2[abs(np.array(node2) - node1) == dist][0]
    else:
        node1, node2 = node1[0], node2[0]
    if node1 == (node2 + dist) % node_num:
        node1, node2 = node2, node1
    return node1, node2


def get_polygon_border(nusc_map, record):
    if not record['polygon_token']:
        return []

    polygon = nusc_map.explorer.map_api.extract_polygon(record['polygon_token'])
    from_edge_line = nusc_map.explorer.map_api.extract_line(record['from_edge_line_token'])
    to_edge_line = nusc_map.explorer.map_api.extract_line(record['to_edge_line_token'])
    polygon_xy = np.array(polygon.exterior.xy)
    from_edge_line_xy = np.array(from_edge_line.xy)
    to_edge_line_xy = np.array(to_edge_line.xy)
    node_num = polygon_xy.shape[1]

    n11 = np.where(polygon_xy[0] == from_edge_line_xy[0, 0])[0]
    n12 = np.where(polygon_xy[0] == from_edge_line_xy[0, -1])[0]
    dist1 = from_edge_line_xy.shape[1] - 1
    n21 = np.where(polygon_xy[0] == to_edge_line_xy[0, 0])[0]
    n22 = np.where(polygon_xy[0] == to_edge_line_xy[0, -1])[0]
    dist2 = to_edge_line_xy.shape[1] - 1
    n11, n12 = pickone_and_reorder(n11, n12, dist1, node_num)
    n21, n22 = pickone_and_reorder(n21, n22, dist2, node_num)

    max_n = np.max([n11, n12, n21, n22])
    if max_n == n22:
        l1 = polygon_xy[:, n12:n21 + 1]
        l2 = np.concatenate([polygon_xy[:, n22:], polygon_xy[:, :n11 + 1]], 1)
    elif max_n == n12:
        l1 = polygon_xy[:, n22:n11 + 1]
        l2 = np.concatenate([polygon_xy[:, n12:], polygon_xy[:, :n21 + 1]], 1)
    else:
        l1 = polygon_xy[:, n22:n11 + 1]
        l2 = polygon_xy[:, n12:n21 + 1]
    return [l1, l2]


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

    def _get_layer_geom(self,
                        patch_box: Tuple[float, float, float, float],
                        patch_angle: float,
                        layer_name: str) -> List[Geometry]:
        if layer_name in ['lane_border', 'road_block_border']:
            return self._get_border_line(patch_box, patch_angle, layer_name)
        elif layer_name in self.map_api.non_geometric_polygon_layers:
            return self._get_layer_polygon(patch_box, patch_angle, layer_name)
        elif layer_name in self.map_api.non_geometric_line_layers:
            return self._get_layer_line(patch_box, patch_angle, layer_name)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

    @staticmethod
    def mask_for_lines(lines: LineString, mask: np.ndarray) -> np.ndarray:
        """
        Convert a Shapely LineString back to an image mask ndarray.
        :param lines: List of shapely LineStrings to be converted to a numpy array.
        :param mask: Canvas where mask will be generated.
        :return: Numpy ndarray line mask.
        """
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


    def _get_border_line(self, patch_box, patch_angle, layer_name):
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.get_patch_coord(patch_box, patch_angle)

        line_list = []
        records = getattr(self.map_api, layer_name[:-7])
        for record in records:
            points_group = get_polygon_border(self.map_api, record)
            for points in points_group:
                points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
                line = LineString(points)
                if line.is_empty:  # Skip lines without nodes.
                    continue
                new_line = line.intersection(patch)

                if not new_line.is_empty:
                    new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                    new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    line_list.append(new_line)

        return line_list

    def _layer_geom_to_mask(self,
                            layer_name: str,
                            layer_geom: List[Geometry],
                            local_box: Tuple[float, float, float, float],
                            canvas_size: Tuple[int, int]) -> np.ndarray:
        if layer_name in ['lane_border', 'road_block_border']:
            return self._border_geom_to_mask(layer_geom, local_box, canvas_size)
        elif layer_name in self.map_api.non_geometric_polygon_layers:
            return self._polygon_geom_to_mask(layer_geom, local_box, layer_name, canvas_size)
        elif layer_name in self.map_api.non_geometric_line_layers:
            return self._line_geom_to_mask(layer_geom, local_box, layer_name, canvas_size)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

    def _border_geom_to_mask(self,
                             layer_geom: List[LineString],
                             local_box: Tuple[float, float, float, float],
                             canvas_size: Tuple[int, int]) -> Optional[np.ndarray]:
        patch_x, patch_y, patch_h, patch_w = local_box

        patch = self.get_patch_coord(local_box)

        canvas_h = canvas_size[0]
        canvas_w = canvas_size[1]
        scale_height = canvas_h / patch_h
        scale_width = canvas_w / patch_w

        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0

        map_mask = np.zeros(canvas_size, np.uint8)

        for line in layer_geom:
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
                new_line = affinity.scale(new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))

                map_mask = self.mask_for_lines(new_line, map_mask)
        return map_mask


def gen_topdown_mask(nuscene, nusc_maps, sample_record, patch_size, canvas_size, seg_layers):
    sample_record_data = sample_record['data']
    sample_data_record = nuscene.get('sample_data', sample_record_data['LIDAR_TOP'])

    pose_record = nuscene.get('ego_pose', sample_data_record['ego_pose_token'])
    map_pose = pose_record['translation'][:2]
    rotation = Quaternion(pose_record['rotation'])

    patch_box = (map_pose[0], map_pose[1], patch_size[0], patch_size[1])
    patch_angle = quaternion_yaw(rotation) / np.pi * 180 + 90

    scene_record = nuscene.get('scene', sample_record['scene_token'])
    log_record = nuscene.get('log', scene_record['log_token'])
    location = log_record['location']
    topdown_seg_mask = nusc_maps[location].get_map_mask(patch_box, patch_angle, seg_layers, canvas_size)
    topdown_seg_mask = np.flip(topdown_seg_mask, 1)  # left-right correction
    return topdown_seg_mask
