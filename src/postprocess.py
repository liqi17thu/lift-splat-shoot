#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-30 上午10:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_postprocess.py
# @IDE: PyCharm Community Edition
"""
LaneNet model post process
"""
import os.path as ops
import math

import cv2
import numpy as np
import random

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


class _LaneFeat(object):
    """

    """
    def __init__(self, feat, coord, class_id=-1):
        """
        lane feat object
        :param feat: lane embeddng feats [feature_1, feature_2, ...]
        :param coord: lane coordinates [x, y]
        :param class_id: lane class id
        """
        self._feat = feat
        self._coord = coord
        self._class_id = class_id

    @property
    def feat(self):
        """

        :return:
        """
        return self._feat

    @feat.setter
    def feat(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        if value.dtype != np.float32:
            value = np.array(value, dtype=np.float64)

        self._feat = value

    @property
    def coord(self):
        """

        :return:
        """
        return self._coord

    @coord.setter
    def coord(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        if value.dtype != np.int32:
            value = np.array(value, dtype=np.int32)

        self._coord = value

    @property
    def class_id(self):
        """

        :return:
        """
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.int64):
            raise ValueError('Class id must be integer')

        self._class_id = value


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self, dbscan_eps=0.35, postprocess_min_samples=200):
        """

        """

        self._color_map = []
        for i in range(30):
            self._color_map.append([random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)])

        # self._color_map = [np.array([255, 0, 0]),
        #                    np.array([0, 255, 0]),
        #                    np.array([0, 0, 255]),
        #                    np.array([125, 125, 0]),
        #                    np.array([0, 125, 125]),
        #                    np.array([125, 0, 125]),
        #                    np.array([50, 100, 50]),
        #                    np.array([0, 50, 100]),
        #                    np.array([50, 100, 0]),
        #                    np.array([50, 0, 100]),
        #                    ]
        self.dbscan_eps = dbscan_eps
        self.postprocess_min_samples = postprocess_min_samples

    def _embedding_feats_dbscan_cluster(self, embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.postprocess_min_samples)
        try:
            features = StandardScaler().fit_transform(embedding_image_feats)
            db.fit(features)
        except Exception as err:
            # print(err)
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)

        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 255)
        lane_embedding_feats = instance_seg_ret[idx]
        # idx_scale = np.vstack((idx[0] / 256.0, idx[1] / 512.0)).transpose()
        # lane_embedding_feats = np.hstack((lane_embedding_feats, idx_scale))
        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """
        # get embedding feats and coords
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )

        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
        )

        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']
        coord = get_lane_embedding_feats_result['lane_coordinates']

        if db_labels is None:
            return None, None

        lane_coords = []

        for index, label in enumerate(unique_labels.tolist()):
            if label == -1:
                continue
            idx = np.where(db_labels == label)
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))
            mask[pix_coord_idx] = self._color_map[index]
            lane_coords.append(coord[idx])

        return mask, lane_coords


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """
    def __init__(self, dbscan_eps=0.35, postprocess_min_samples=200):
        """

        :param ipm_remap_file_path: ipm generate file path
        """

        self._cluster = _LaneNetCluster(dbscan_eps, postprocess_min_samples)

    def postprocess(self, binary_seg_result, instance_seg_result=None, min_area_threshold=100):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param data_source:
        :return:
        """
        # convert binary_seg_result
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # apply image morphology operation to fill in the hold and reduce the small area
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)

        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]
        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        # apply embedding features cluster
        mask_image, lane_coords = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result
        )

        return mask_image, lane_coords


if __name__ == '__main__':
    from nuscenes import NuScenes
    from src.topdown_mask import MyNuScenesMap
    from src.data import compile_data, MAP, NuscData
    from PIL import Image

    version = 'mini'
    dataroot = '../data/nuScenes'

    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=dataroot,
                    verbose=False)
    nusc_maps = {}
    for map_name in MAP:
        nusc_maps[map_name] = MyNuScenesMap(dataroot=dataroot, map_name=map_name)

    H = 900
    W = 1600
    resize_lim = (0.193, 0.225)
    final_dim = (128, 352)
    bot_pct_lim = (0.0, 0.22)
    rot_lim = (-5.4, 5.4)
    rand_flip = False
    ncams = 6
    line_width = 5
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

    nusc_data = NuscData(nusc, nusc_maps, False, data_aug_conf, grid_conf)

    post_processor = LaneNetPostProcessor()

    rec = nusc.sample[200]
    seg_mask, inst_mask = nusc_data.get_lineimg(rec)
    seg_mask = seg_mask.numpy()[1:].any(0).astype('uint8')
    inst_mask = inst_mask.unsqueeze(-1).repeat(1, 1, 16)

    mask_image, lane_coords = post_processor.postprocess(seg_mask, inst_mask)
