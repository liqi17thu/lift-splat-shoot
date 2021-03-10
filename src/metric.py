import torch
import torch.nn as nn


class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, source_pc, target_pc, threshold=33.33, cum=False, bidirectional=True):
        dist = torch.cdist(source_pc.float(), target_pc.float())
        dist1, _ = torch.min(dist, 2)
        dist1[dist1 > threshold] = threshold
        dist2, _ = torch.min(dist, 1)
        dist2[dist2 > threshold] = threshold
        if cum:
            len1 = dist1.shape[-1]
            len2 = dist2.shape[-1]
            dist1 = dist1.sum(-1)
            dist2 = dist2.sum(-1)
            return dist1, dist2, len1, len2
        dist1 = dist1.mean(-1)
        dist2 = dist2.mean(-1)
        if bidirectional:
            return (dist1 + dist2) / 2
        else:
            return dist1, dist2


class LaneSegMetric(object):
    def __init__(self):
        self.chamfer_distance = ChamferDistance()
        self.sampled_recalls = torch.linspace(0, 1, 11)

    def semantic_mask_chamfer_dist_cum(self, seg_pred, seg_label, threshold=33.33):
        # seg_label: N, C, H, W
        # seg_pred: N, C, H, W
        N, C, H, W = seg_label.shape

        cum_CD1 = torch.zeros(C, device=seg_label.device)
        cum_CD2 = torch.zeros(C, device=seg_label.device)
        cum_num1 = torch.zeros(C, device=seg_label.device)
        cum_num2 = torch.zeros(C, device=seg_label.device)
        for n in range(N):
            for c in range(C):
                pred_pc_x, pred_pc_y = torch.where(seg_pred[n, c] != 0)
                label_pc_x, label_pc_y = torch.where(seg_label[n, c] != 0)
                if len(pred_pc_x) == 0 and len(label_pc_x) == 0:
                    continue

                if len(label_pc_x) == 0:
                    cum_CD1[c] += len(pred_pc_x) * threshold
                    cum_num1[c] += len(pred_pc_x)
                    continue

                if len(pred_pc_x) == 0:
                    cum_CD2[c] += len(label_pc_x) * threshold
                    cum_num2[c] += len(label_pc_x)
                    continue

                pred_pc_coords = torch.stack([pred_pc_x, pred_pc_y], -1).float()
                label_pc_coords = torch.stack([label_pc_x, label_pc_y], -1).float()
                CD1, CD2, len1, len2 = self.chamfer_distance(pred_pc_coords[None], label_pc_coords[None], threshold=threshold, cum=True)
                cum_CD1[c] += CD1[0]
                cum_CD2[c] += CD2[0]
                cum_num1[c] += len1
                cum_num2[c] += len2
        return cum_CD1, cum_CD2, cum_num1, cum_num2

    def semantic_mask_chamfer_dist(self, seg_pred, seg_label, threshold=33.33):
        # seg_label: N, C, H, W
        # seg_pred: N, C, H, W
        N, C, H, W = seg_label.shape

        CD = torch.zeros((N, C), device=seg_label.device)
        for n in range(N):
            for c in range(C):
                pred_pc_x, pred_pc_y = torch.where(seg_pred[n, c] != 0)
                label_pc_x, label_pc_y = torch.where(seg_label[n, c] != 0)
                if len(pred_pc_x) == 0 and len(label_pc_x) == 0:
                    continue

                if len(pred_pc_x) == 0 or len(label_pc_x) == 0:
                    CD[n, c] = threshold
                    continue
                pred_pc_coords = torch.stack([pred_pc_x, pred_pc_y], -1).float()
                label_pc_coords = torch.stack([label_pc_x, label_pc_y], -1).float()
                CD[n, c] = self.chamfer_distance(pred_pc_coords[None], label_pc_coords[None], threshold=threshold, bidirectional=True)
                CD[n, c] = CD[n, c] if CD[n, c] <= threshold else threshold
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
                for line in inst_pred_lines:
                    confidence = torch.mean(confidence_mask[n, c][line[:, 0], line[:, 1]])
                    inst_pred_confidence.append(confidence)
                AP_matrix[n, c] = self.single_instance_line_AP(inst_pred_lines, inst_pred_confidence, inst_label_lines, thresholds)
        return AP_matrix.mean(0)

    def _get_line_instances_from_mask(self, mask):
        # mask: H, W
        # instance: [(N1, 2), (N2, 2), ..., (N_k1, 2)]
        indices = torch.unique(mask)
        instances = []
        for idx in indices:
            if idx == 0:
                continue
            pc_x, pc_y = torch.where(mask == idx)
            coords = torch.stack([pc_x, pc_y], -1)
            instances.append(coords)
        return instances

    def _instance_line_matching(self, inst_pred_lines, inst_pred_confidence, inst_label_lines):
        # inst_label_line: a list of points [(N1, 2), (N2, 2), ..., (N_k1, 2)]
        # inst_pred_confidence: a list of confidence [c1, c2, ..., ck2]
        # inst_pred_line: a list of points [(M1, 2), (M2, 2), ..., (M_k2, 2)]
        # return: a list of {'pred': (M, 2), 'label': (N, 2), 'confidence': scalar}
        label_num = len(inst_label_lines)
        pred_num = len(inst_pred_lines)
        CD = torch.zeros((pred_num, label_num)).cuda()
        for i, inst_pc in enumerate(inst_pred_lines):
            for j, label_pc in enumerate(inst_label_lines):
                CD[i, j] = self.chamfer_distance(inst_pc[None], label_pc[None], bidirectional=True)  # TODO: direction ?

        if label_num > 0 and pred_num > 0:
            sorted_idx = torch.argsort(CD, dim=-1)[:, 0]

        label_picked = torch.zeros(label_num, dtype=torch.bool)

        matched_list = []
        for i in range(pred_num):
            if label_num > 0:
                label = inst_label_lines[sorted_idx[i]]
                label_picked[sorted_idx[i]] = True
            else:
                label = None
            matched_list.append({
                    'pred': inst_pred_lines[i],
                    'confidence': inst_pred_confidence[i],
                    'label': label
                })

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

            if pred is None:
                continue
            if label is None:
                TP.append(TP[-1])
                FP.append(FP[-1] + 1)
                continue

            dist = self.chamfer_distance(pred[None], label[None], bidirectional=True)
            if dist < threshold:
                TP.append(TP[-1] + 1)
                FP.append(FP[-1])
            else:
                TP.append(TP[-1])
                FP.append(FP[-1] + 1)
        TP = torch.tensor(TP[1:])
        FP = torch.tensor(FP[1:])

        # print(f'TP: {TP}')
        # print(f'FP: {FP}')

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
            idx = torch.where(recall >= r)[0]
            if len(idx) == 0:
                continue
            idx = idx[0]
            acc_precision += precision[idx]
        return acc_precision / total


if __name__ == '__main__':
    from nuscenes import NuScenes
    from src.data import MAP, NuscData
    from src.topdown_mask import MyNuScenesMap
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
    seg_mask, inst_mask = seg_mask[1:].cuda(), inst_mask[1:].cuda()

    lane_seg_metric = LaneSegMetric()

    chamfer_distance = lane_seg_metric.semantic_mask_chamfer_dist(seg_mask[None], seg_mask[None])
