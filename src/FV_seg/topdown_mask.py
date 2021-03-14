import numpy as np
import cv2

from typing import Tuple, List

from PIL import Image
from nuscenes.map_expansion.map_api import NuScenesMapExplorer, NuScenesMap
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely import ops
from shapely.errors import TopologicalError
from shapely.geometry import box, Polygon, MultiPolygon, Point, LineString, MultiLineString

MASK_SHIFT = 50
MASK_STEP = 30


def mask_label_to_img(label):
    return Image.fromarray((MASK_SHIFT + MASK_STEP * np.array(label)).astype('uint8'), mode='L')


def mask_for_polygons(polygons: MultiPolygon, mask: np.ndarray) -> np.ndarray:
    def int_coords(x):
        # function to round and convert to int
        return np.array(x).round().astype(np.int32)

    for poly in polygons:
        exteriors = int_coords(poly.exterior.coords)
        cv2.fillPoly(mask, [exteriors], 1)
    return mask


def mask_for_lines(lines: LineString, mask: np.ndarray, thickness) -> np.ndarray:
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
            cv2.polylines(mask, [coords], False, 1, thickness=thickness)
    else:
        coords = np.asarray(list(lines.coords), np.int32)
        coords = coords.reshape((-1, 2))
        cv2.polylines(mask, [coords], False, 1, thickness=thickness)

    return mask


def extract_contour(topdown_seg_mask, canvas_size, thickness=5):
    topdown_seg_mask[topdown_seg_mask != 0] = 255
    ret, thresh = cv2.threshold(topdown_seg_mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(topdown_seg_mask)
    patch = box(1, 1, canvas_size[1] - 2, canvas_size[0] - 2)
    for cnt in contours:
        cnt = cnt.reshape((-1, 2))
        cnt = np.append(cnt, cnt[0].reshape(-1, 2), axis=0)
        line = LineString(cnt)
        line = line.intersection(patch)
        if isinstance(line, MultiLineString):
            line = ops.linemerge(line)

        if isinstance(line, MultiLineString):
            for l in line:
                cv2.polylines(mask, [np.asarray(list(l.coords), np.int32).reshape((-1, 2))], False, color=1, thickness=thickness)
        elif isinstance(line, LineString):
            cv2.polylines(mask, [np.asarray(list(line.coords), np.int32).reshape((-1, 2))], False,  color=1, thickness=thickness)

    return mask


def get_ped_crossing(nusc_map, token):
    def add_line(poly_xy, idx, line_list):
        points = np.array([(p0, p1) for p0, p1 in zip(poly_xy[0, idx:idx + 2], poly_xy[1, idx:idx + 2])]).T
        line_list.append(points)

    record = nusc_map.explorer.map_api.get('ped_crossing', token)
    polygon = nusc_map.explorer.map_api.extract_polygon(record['polygon_token'])
    poly_xy = np.array(polygon.exterior.xy)
    dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
    x1, x2 = np.argsort(dist)[-2:]

    line_group = []
    add_line(poly_xy, x1, line_group)
    add_line(poly_xy, x2, line_group)
    return line_group


def get_map_line_mask_in_image(nusc, nusc_map, cam_record,
                               im_size: Tuple[int, int],
                               patch_radius: float = 50.,
                               render_behind_cam: bool = True,
                               render_outside_im: bool = True,
                               layer_names: List[str] = None,
                               thickness=5):
    cs_record = nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])

    # Retrieve the current map.
    poserecord = nusc.get('ego_pose', cam_record['ego_pose_token'])
    ego_pose = poserecord['translation']
    box_coords = (
        ego_pose[0] - patch_radius,
        ego_pose[1] - patch_radius,
        ego_pose[0] + patch_radius,
        ego_pose[1] + patch_radius,
    )
    records_in_patch = nusc_map.explorer.get_records_in_patch(box_coords, layer_names, 'intersect')

    mask = np.zeros((len(layer_names), im_size[1], im_size[0]))
    patch = box(0, 0, im_size[0], im_size[1])
    # Retrieve and render each record.
    for layer_idx, layer_name in enumerate(layer_names):
        proj_lines = []
        for token in records_in_patch[layer_name]:
            if layer_name == 'ped_crossing':
                points_group = get_ped_crossing(nusc_map, token)
            else:
                record = nusc_map.explorer.map_api.get(layer_name, token)
                line_token = record['line_token']
                line = nusc_map.explorer.map_api.extract_line(line_token)
                points_group = [np.array(line.xy)]

            for points in points_group:
                near_plane = 1e-8
                points = np.vstack((points, np.zeros((1, points.shape[1]))))
                # Transform into the ego vehicle frame for the timestamp of the image.
                points = points - np.array(poserecord['translation']).reshape((-1, 1))
                points = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points)

                # Transform into the camera.
                points = points - np.array(cs_record['translation']).reshape((-1, 1))
                points = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, points)
                # Remove points that are partially behind the camera.
                depths = points[2, :]
                behind = depths < near_plane
                if np.all(behind):
                    continue

                if render_behind_cam:
                    # Perform clipping on polygons that are partially behind the camera.
                    points = NuScenesMapExplorer._clip_points_behind_camera(points, near_plane)
                elif np.any(behind):
                    # Otherwise ignore any polygon that is partially behind the camera.
                    continue
                points = points[:, :-1]
                # Ignore polygons with less than 2 points after clipping.
                if len(points) == 0 or points.shape[1] < 2:
                    continue

                # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
                points = view_points(points, cam_intrinsic, normalize=True)

                # Skip polygons where all points are outside the image.
                # Leave a margin of 1 pixel for aesthetic reasons.
                inside = np.ones(points.shape[1], dtype=bool)
                inside = np.logical_and(inside, points[0, :] > 1)
                inside = np.logical_and(inside, points[0, :] < im_size[0] - 1)
                inside = np.logical_and(inside, points[1, :] > 1)
                inside = np.logical_and(inside, points[1, :] < im_size[1] - 1)
                if render_outside_im:
                    if np.all(np.logical_not(inside)):
                        continue
                else:
                    if np.any(np.logical_not(inside)):
                        continue

                points = points[:2, :]
                points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
                line_proj = LineString(points)

                if not line_proj.is_valid:
                    try:
                        line_proj = line_proj.buffer(0)
                    except ValueError:
                        continue
                line_proj = line_proj.intersection(patch)

                if line_proj.is_empty:
                    continue
                if isinstance(line_proj, MultiLineString):
                    for line in line_proj:
                        proj_lines.append(line)
                elif isinstance(line_proj, LineString):
                    proj_lines.append(line_proj)

        if len(proj_lines) > 0:
            mask_for_lines(MultiLineString(proj_lines), mask[layer_idx], thickness=thickness)

    mask = mask.astype('uint8')
    return mask


def get_map_poly_mask_in_image(nusc, nusc_map, cam_record,
                               im_size: Tuple[int, int],
                               patch_radius: float = 50.,
                               min_polygon_area: float = 10.,
                               render_behind_cam: bool = True,
                               render_outside_im: bool = True,
                               layer_names: List[str] = None):
    near_plane = 1e-8

    cs_record = nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])

    # Retrieve the current map.
    poserecord = nusc.get('ego_pose', cam_record['ego_pose_token'])
    ego_pose = poserecord['translation']
    box_coords = (
        ego_pose[0] - patch_radius,
        ego_pose[1] - patch_radius,
        ego_pose[0] + patch_radius,
        ego_pose[1] + patch_radius,
    )
    records_in_patch = nusc_map.explorer.get_records_in_patch(box_coords, layer_names, 'intersect')

    mask = np.zeros((len(layer_names)+1, im_size[1], im_size[0]))
    patch = box(0, 0, im_size[0], im_size[1])
    # Retrieve and render each record.
    for layer_idx, layer_name in enumerate(layer_names):
        proj_polygons = []
        for token in records_in_patch[layer_name]:
            record = nusc_map.explorer.map_api.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = record['polygon_tokens']
            else:
                polygon_tokens = [record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nusc_map.explorer.map_api.extract_polygon(polygon_token)

                # Convert polygon nodes to pointcloud with 0 height.
                points = np.array(polygon.exterior.xy)
                points = np.vstack((points, np.zeros((1, points.shape[1]))))

                # Transform into the ego vehicle frame for the timestamp of the image.
                points = points - np.array(poserecord['translation']).reshape((-1, 1))
                points = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points)

                # Transform into the camera.
                points = points - np.array(cs_record['translation']).reshape((-1, 1))
                points = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, points)

                # Remove points that are partially behind the camera.
                depths = points[2, :]
                behind = depths < near_plane
                if np.all(behind):
                    continue

                if render_behind_cam:
                    # Perform clipping on polygons that are partially behind the camera.
                    points = NuScenesMapExplorer._clip_points_behind_camera(points, near_plane)
                elif np.any(behind):
                    # Otherwise ignore any polygon that is partially behind the camera.
                    continue

                # Ignore polygons with less than 3 points after clipping.
                if len(points) == 0 or points.shape[1] < 3:
                    continue

                # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
                points = view_points(points, cam_intrinsic, normalize=True)

                # Skip polygons where all points are outside the image.
                # Leave a margin of 1 pixel for aesthetic reasons.
                inside = np.ones(points.shape[1], dtype=bool)
                inside = np.logical_and(inside, points[0, :] > 1)
                inside = np.logical_and(inside, points[0, :] < im_size[0] - 1)
                inside = np.logical_and(inside, points[1, :] > 1)
                inside = np.logical_and(inside, points[1, :] < im_size[1] - 1)
                if render_outside_im:
                    if np.all(np.logical_not(inside)):
                        continue
                else:
                    if np.any(np.logical_not(inside)):
                        continue

                points = points[:2, :]
                points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
                polygon_proj = Polygon(points)

                # Filter small polygons
                if polygon_proj.area < min_polygon_area:
                    continue

                if not polygon_proj.is_valid:
                    try:
                        polygon_proj = polygon_proj.buffer(0)
                    except ValueError:
                        continue
                try:
                    polygon_proj = polygon_proj.intersection(patch)
                except TopologicalError:
                    continue
                if polygon_proj.is_empty:
                    continue
                if not polygon_proj.is_valid:
                    polygon_proj = polygon_proj.buffer(0)
                if isinstance(polygon_proj, MultiPolygon):
                    for poly in polygon_proj:
                        proj_polygons.append(poly)
                elif isinstance(polygon_proj, Polygon):
                    proj_polygons.append(polygon_proj)
                elif isinstance(polygon_proj, Point):
                    pass
                elif isinstance(polygon_proj, LineString):
                    pass
                else:
                    print(polygon_proj)
                    print(type(polygon_proj))

        mask_for_polygons(MultiPolygon(proj_polygons), mask[layer_idx+1])

    mask = mask.astype('uint8')
    return mask


def get_fv_mask(nuscene, nusc_maps, sample_record, pos, image_size=(1600, 900)):
    scene_record = nuscene.get('scene', sample_record['scene_token'])
    log_record = nuscene.get('log', scene_record['log_token'])
    location = log_record['location']

    sample_record_data = sample_record['data']
    sample_token = sample_record_data[pos]
    sample_data_record = nuscene.get('sample_data', sample_token)
    poly_mask = get_map_poly_mask_in_image(nuscene, nusc_maps[location], sample_data_record, image_size,
                                           layer_names=['road_segment', 'lane'])
    line_mask = get_map_line_mask_in_image(nuscene, nusc_maps[location], sample_data_record, image_size,
                                           layer_names=['lane_divider', 'road_divider', 'ped_crossing'])

    final_mask = np.zeros((4, image_size[1], image_size[0]), dtype='uint8')

    contour_mask = extract_contour(poly_mask[1:].any(0).astype('uint8'), (image_size[1], image_size[0]))
    final_mask[1] = line_mask[:2].any(0)
    final_mask[2] = line_mask[2]
    final_mask[3] = contour_mask
    final_mask[0] = 1 - np.any(final_mask, axis=0)
    return final_mask

def main():
    import os
    import random
    import tqdm
    from PIL import Image
    from nuscenes import NuScenes
    from nuscenes.map_expansion.map_api import NuScenesMap

    MAP = ['boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown']
    CAM_POSITION = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    overwrite = False
    version = 'v1.0-trainval'
    dataroot = 'data/nuScenes'
    nuscene = NuScenes(version=version, dataroot=dataroot, verbose=False)
    nusc_maps = {}
    for map_name in MAP:
        nusc_maps[map_name] = NuScenesMap(dataroot=dataroot, map_name=map_name)

    random.shuffle(nuscene.sample)
    for sample_record in tqdm.tqdm(nuscene.sample):
        for pos in CAM_POSITION:
            sample_token = sample_record['data'][pos]
            path = nuscene.get_sample_data_path(sample_token)
            image = Image.open(path).convert('RGB')
            image_size = image.size

            seg_mask_path = path.split('.')[0] + '_line_mask.png'
            if not overwrite and os.path.exists(seg_mask_path):
                continue

            final_mask = get_fv_mask(nuscene, nusc_maps, sample_record, pos, image_size=image_size)
            for i in range(len(final_mask)):
                final_mask[i] *= i
            final_mask = final_mask.max(0)
            final_mask = mask_label_to_img(final_mask)
            final_mask.save(seg_mask_path, 'PNG')


if __name__ == '__main__':
    main()
