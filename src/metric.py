import torch
from chamferdist import ChamferDistance
# from nuscenes import NuScenes
#
# from src.topdown_mask import MyNuScenesMap
# from .data import NuscData, MAP


class LaneSegMetric(object):
    def __init__(self):
        self.chamfer_distance = ChamferDistance()
        self.sampled_recalls = torch.linspace(0, 1, 11)

    def semantic_mask_chamfer_dist(self, seg_pred, seg_label):
        # seg_label: N, C, H, W
        # seg_pred: N, C, H, W
        N, C, H, W = seg_label.shape

        CD = torch.zeros((N, C), device=seg_label.device)
        for n in range(N):
            for c in range(C):
                pred_pc_x, pred_pc_y = torch.where(seg_pred[n, c] != 0)
                pred_pc_coords = torch.stack([pred_pc_x, pred_pc_y], -1).float()

                label_pc_x, label_pc_y = torch.where(seg_label[n, c] != 0)
                label_pc_coords = torch.stack([label_pc_x, label_pc_y], -1).float()
                dist = self.chamfer_distance(pred_pc_coords[None], label_pc_coords[None], bidirectional=True)
                CD[n, c] = dist
        dist = torch.mean(CD, 0)
        return dist

    def single_instance_line_AP(self, inst_pred_lines, inst_pred_confidence, inst_label_lines, thresholds):
        # inst_label_line: a list of points [(N1, 2), (N2, 2), ..., (N_k1, 2)]
        # inst_pred_confidence: a list of confidence [c1, c2, ..., ck2]
        # inst_pred_line: a list of points [(M1, 2), (M2, 2), ..., (M_k2, 2)]
        # thresholds: threshold of chamfer distance to identify TP
        num_thres = len(thresholds)
        AP_thres = torch.zeros(num_thres)
        matching_list = self._instance_line_matching(inst_pred_lines, inst_pred_confidence, inst_label_lines)
        for t in range(num_thres):
            precision, recall = self._get_precision_recall_curve_by_confidence(matching_list, len(inst_label_lines), thresholds[t])
            precision, recall = self._smooth_PR_curve(precision, recall)
            AP = self._calc_AP_from_precision_recall(precision, recall, self.sampled_recalls)
            AP_thres[t] = AP
        return AP_thres

    def instance_mask_AP(self, inst_pred_mask, inst_label_mask, confidence_mask, thresholds):
        # inst_pred: N, C, H, W
        # inst_label: N, C, H, W
        # confidence_mask: N, C, H, W
        N, C, H, W = inst_label_mask.shape
        AP_matrix = torch.zeros((N, C, len(thresholds)))
        for n in range(N):
            for c in range(C):
                inst_pred_lines = self._get_line_instances_from_mask(inst_pred_mask[n, c])
                inst_label_lines = self._get_line_instances_from_mask(inst_label_mask[n, c])
                inst_pred_confidence = []
                for line in inst_label_lines:
                    confidence = torch.mean(confidence_mask[n, c][line[:, 0], line[:, 1]])
                    inst_pred_confidence.append(confidence)
                AP_matrix[n, c] = self.single_instance_line_AP(inst_pred_lines, inst_pred_confidence, inst_label_lines, thresholds)
        return AP_matrix

    def _get_line_instances_from_mask(self, mask):
        # mask: H, W
        # instance: [(N1, 2), (N2, 2), ..., (N_k1, 2)]
        indices = torch.unique(mask)
        instances = []
        for idx in indices:
            if idx == 0:
                continue
            pc_x, pc_y = torch.where(mask == idx)
            coords = torch.stack([pc_x, pc_y], -1).float()
            instances.append(coords)
        return instances

    def _instance_line_matching(self, inst_pred_lines, inst_pred_confidence, inst_label_lines):
        # inst_label_line: a list of points [(N1, 2), (N2, 2), ..., (N_k1, 2)]
        # inst_pred_confidence: a list of confidence [c1, c2, ..., ck2]
        # inst_pred_line: a list of points [(M1, 2), (M2, 2), ..., (M_k2, 2)]
        # return: a list of {'pred': (M, 2), 'label': (N, 2), 'confidence': scalar}
        label_num = len(inst_label_lines)
        pred_num = len(inst_pred_lines)
        CD = torch.zeros((pred_num, label_num), device=inst_label_lines.device)
        for i, inst_pc in enumerate(inst_pred_lines):
            for j, label_pc in enumerate(inst_label_lines):
                CD[i, j] = self.chamfer_distance(inst_pc[None], label_pc[None])  # TODO: direction ?
        sorted_idx = torch.argsort(CD, dim=-1)[:, 0]

        label_picked = torch.zeros(label_num, dtype=torch.bool)
        matched_list = []
        for i in range(pred_num):
            matched_list.append({
                    'pred': inst_pred_lines[i],
                    'confidence': inst_pred_confidence[i],
                    'label': inst_label_lines[sorted_idx[i]]
                })
            label_picked[sorted_idx[i]] = True

        for i in range(label_num):
            if not label_picked[i]:
                matched_list.append({
                    'pred': None,
                    'confidence': 0.,
                    'label': inst_label_lines[i]
                })
        return matched_list

    def _get_precision_recall_curve_by_confidence(self, matching_list, num_gt, threshold):
        matching_list = sorted(matching_list, key=lambda x: x['confidence'])

        TP = [0]
        FP = [0]
        for match_item in matching_list:
            pred = match_item['pred']
            label = match_item['label']

            dist = self.chamfer_distance(pred[None], label[None], bidirectional=True)
            if dist < threshold:
                TP.append(TP[-1] + 1)
                FP.append(FP[-1])
            else:
                TP.append(TP[-1])
                FP.append(FP[-1] + 1)
        TP = torch.tensor(TP)
        FP = torch.tensor(FP)

        precision = TP / (TP + FP)
        recall = TP / num_gt
        return precision, recall

    def _smooth_PR_curve(self, precision, recall):
        idx = torch.argsort(recall)
        recall = recall[idx]
        precision = precision[idx]
        length = len(precision)
        for i in range(length-1, 0, -1):
            precision[:i][precision[:i] < precision[i]] = precision[i]
        return precision, recall

    def _calc_AP_from_precision_recall(self, precision, recall, sampled_recalls):
        acc_precision = 0.
        total = len(sampled_recalls)
        for r in sampled_recalls:
            idx = torch.where(recall > r)[0]
            if len(idx) == 0:
                continue
            idx = idx[0]
            acc_precision += precision[idx]
        return acc_precision / total


if __name__ == '__main__':
    from chamfer_distance import ChamferDistance

    chamfer_dist = ChamferDistance()

    points1 = torch.randn(10, 3)
    points2 = torch.randn(5, 3)

    dist1, dist2 = chamfer_dist(points1, points2)

    print(dist1.shape)
    print(dist2.shape)

    # version = 'mini'
    # dataroot = 'data/nuScenes'
    #
    # H = 900
    # W = 1600
    # resize_lim = (0.193, 0.225)
    # final_dim = (128, 352)
    # bot_pct_lim = (0.0, 0.22)
    # rot_lim = (-5.4, 5.4)
    # rand_flip = False
    # ncams = 6
    # line_width = 1
    # preprocess = False
    # overwrite = False
    #
    # xbound = [-30.0, 30.0, 0.15]
    # ybound = [-15.0, 15.0, 0.15]
    # zbound = [-10.0, 10.0, 20.0]
    # dbound = [4.0, 45.0, 1.0]
    #
    # grid_conf = {
    #     'xbound': xbound,
    #     'ybound': ybound,
    #     'zbound': zbound,
    #     'dbound': dbound,
    # }
    # data_aug_conf = {
    #                 'resize_lim': resize_lim,
    #                 'final_dim': final_dim,
    #                 'rot_lim': rot_lim,
    #                 'H': H, 'W': W,
    #                 'rand_flip': rand_flip,
    #                 'bot_pct_lim': bot_pct_lim,
    #                 'preprocess': preprocess,
    #                 'line_width': line_width,
    #                 'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    #                          'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    #                 'Ncams': ncams,
    #             }
    #
    # nusc = NuScenes(version='v1.0-{}'.format(version),
    #                 dataroot=dataroot,
    #                 verbose=False)
    # nusc_maps = {}
    # for map_name in MAP:
    #     nusc_maps[map_name] = MyNuScenesMap(dataroot=dataroot, map_name=map_name)
    #
    # nusc_data = NuscData(nusc, nusc_maps, False, data_aug_conf, grid_conf)
    #
    # rec = nusc.sample[0]
    # seg_mask, inst_mask = nusc_data.get_lineimg(rec)
    #
    # lane_seg_metric = LaneSegMetric()
    #
    # chamfer_distance = lane_seg_metric.semantic_mask_chamfer_dist(seg_mask[None], seg_mask[None])
    # print(chamfer_distance)