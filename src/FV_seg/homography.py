import numpy as np
import torch
import torch.nn as nn
import cv2

CAM_FL = 0
CAM_F = 1
CAM_FR = 2
CAM_BL = 3
CAM_B = 4
CAM_BR = 5


# =========================================================
# Projections
# =========================================================
def rotation_from_euler(rolls, pitchs, yaws):
    """
    Get rotation matrix
    Args:
        roll, pitch, yaw:       In degrees

    Returns:
        R:          [B, 4, 4]
    """
    B = len(rolls)

    # si, sj, sk = torch.sin(torch.deg2rad(rolls)), torch.sin(torch.deg2rad(pitchs)), torch.sin(torch.deg2rad(yaws))
    # ci, cj, ck = torch.cos(torch.deg2rad(rolls)), torch.cos(torch.deg2rad(pitchs)), torch.cos(torch.deg2rad(yaws))
    si, sj, sk = np.sin(np.deg2rad(rolls)), np.sin(np.deg2rad(pitchs)), np.sin(np.deg2rad(yaws))
    ci, cj, ck = np.cos(np.deg2rad(rolls)), np.cos(np.deg2rad(pitchs)), np.cos(np.deg2rad(yaws))
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    R = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    R[:, 0, 0] = cj * ck
    R[:, 0, 1] = sj * sc - cs
    R[:, 0, 2] = sj * cc + ss
    R[:, 1, 0] = cj * sk
    R[:, 1, 1] = sj * ss + cc
    R[:, 1, 2] = sj * cs - sc
    R[:, 2, 0] = -sj
    R[:, 2, 1] = cj * si
    R[:, 2, 2] = cj * ci
    return R


def perspective(cam_coords, proj_mat, h, w):
    """
    P = proj_mat @ (x, y, z, 1)
    Project cam2pixel

    Args:
        cam_coords:         [B, 4, npoints]
        proj_mat:           [B, 4, 4]

    Returns:
        pix coords:         [B, h, w, 2]
    """
    eps = 1e-7
    pix_coords = proj_mat @ cam_coords

    N, _, _ = pix_coords.shape

    pix_coords = pix_coords[:, :2, :] / (pix_coords[:, 2, :][:, None, :] + eps)
    pix_coords = pix_coords.view(N, 2, h, w)
    pix_coords = pix_coords.permute(0, 2, 3, 1).contiguous()
    return pix_coords


def bilinear_sampler(imgs, pix_coords):
    """
    Construct a new image by bilinear sampling from the input image.
    Args:
        imgs:                   [B, H, W, C]
        pix_coords:             [B, h, w, 2]
    :return:
        sampled image           [B, h, w, c]
    """
    B, img_h, img_w, img_c = imgs.shape
    B, pix_h, pix_w, pix_c = pix_coords.shape
    out_shape = (B, pix_h, pix_w, img_c)

    pix_x, pix_y = torch.split(pix_coords, 1, dim=-1)  # [B, pix_h, pix_w, 1]

    # Rounding
    pix_x0 = torch.floor(pix_x)
    pix_x1 = pix_x0 + 1
    pix_y0 = torch.floor(pix_y)
    pix_y1 = pix_y0 + 1

    # Clip within image boundary
    y_max = (img_h - 1)
    x_max = (img_w - 1)

    pix_x0 = torch.clip(pix_x0, 0, x_max)
    pix_y0 = torch.clip(pix_y0, 0, y_max)
    pix_x1 = torch.clip(pix_x1, 0, x_max)
    pix_y1 = torch.clip(pix_y1, 0, y_max)

    # Weights [B, pix_h, pix_w, 1]
    wt_x0 = pix_x1 - pix_x
    wt_x1 = pix_x - pix_x0
    wt_y0 = pix_y1 - pix_y
    wt_y1 = pix_y - pix_y0

    # indices in the image to sample from
    dim = img_w

    # Apply the lower and upper bound pix coord
    base_y0 = pix_y0 * dim
    base_y1 = pix_y1 * dim

    # 4 corner vert ices
    idx00 = (pix_x0 + base_y0).view(B, -1, 1).repeat(1, 1, img_c).long()
    idx01 = (pix_x0 + base_y1).view(B, -1, 1).repeat(1, 1, img_c).long()
    idx10 = (pix_x1 + base_y0).view(B, -1, 1).repeat(1, 1, img_c).long()
    idx11 = (pix_x1 + base_y1).view(B, -1, 1).repeat(1, 1, img_c).long()

    # Gather pixels from image using vertices
    imgs_flat = imgs.reshape([B, -1, img_c])

    im00 = torch.gather(imgs_flat, 1, idx00).reshape(out_shape)
    im01 = torch.gather(imgs_flat, 1, idx01).reshape(out_shape)
    im10 = torch.gather(imgs_flat, 1, idx10).reshape(out_shape)
    im11 = torch.gather(imgs_flat, 1, idx11).reshape(out_shape)

    # Apply weights [pix_h, pix_w, 1]
    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1
    output = w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11
    return output


def plane_grid(xbound, ybound, zs, yaws, rolls, pitchs):
    B = len(zs)

    xmin, xmax = xbound[0], xbound[1]
    num_x = int((xbound[1] - xbound[0]) / xbound[2])
    ymin, ymax = ybound[0], ybound[1]
    num_y = int((ybound[1] - ybound[0]) / ybound[2])

    # y = torch.linspace(xmin, xmax, num_x, dtype=torch.double).cuda()
    # x = torch.linspace(ymin, ymax, num_y, dtype=torch.double).cuda()
    y = torch.linspace(xmin, xmax, num_x)
    x = torch.linspace(ymin, ymax, num_y)

    y, x = torch.meshgrid(x, y)

    x = x.flatten()
    y = y.flatten()

    x = x.unsqueeze(0).repeat(B, 1)
    y = y.unsqueeze(0).repeat(B, 1)

    # z = torch.ones_like(x, dtype=torch.double).cuda() * zs.view(-1, 1)
    # d = torch.ones_like(x, dtype=torch.double).cuda()
    z = torch.ones_like(x) * zs.view(-1, 1)
    d = torch.ones_like(x)
    coords = torch.stack([x, y, z, d], axis=1)
    # rotation_matrix = rotation_from_euler(rolls, pitchs, yaws)
    # rotation_matrix = rotation_from_euler(rolls, yaws, pitchs)
    rotation_matrix = rotation_from_euler(pitchs, rolls, yaws)
    coords = rotation_matrix @ coords
    return coords


def ipm_from_parameters(image, xyz, K, RT, target_h, target_w, post_RT=None):
    """
    :param image: [B, H, W, C]
    :param xyz: [B, 4, npoints]
    :param K: [B, 4, 4]
    :param RT: [B, 4, 4]
    :param target_h: int
    :param target_w: int
    :return: warped_images: [B, target_h, target_w, C]
    """
    P = K @ RT
    if post_RT is not None:
        P = post_RT @ P
    P = P.reshape(-1, 4, 4)
    pixel_coords = perspective(xyz, P, target_h, target_w)
    image2 = bilinear_sampler(image, pixel_coords)
    image2 = image2.type_as(image)
    return image2


class PlaneEstimationModule(nn.Module):
    def __init__(self, N, C):
        super(PlaneEstimationModule, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.linear = nn.Linear(N*C, 3)

        self.linear.weight.data.fill_(0.)
        self.linear.bias.data.fill_(0.)

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B*N, C, H, W)
        x = self.max_pool(x)
        x = x.view(B, N*C)
        x = self.linear(x)
        z, pitch, roll = x[:, 0], x[:, 1], x[:, 2]
        return z, pitch, roll


class IPM(nn.Module):
    def __init__(self, xbound, ybound, z_roll_pitch=True, visual=True):
        super(IPM, self).__init__()
        self.visual = visual
        self.z_roll_pitch = z_roll_pitch
        self.xbound = xbound
        self.ybound = ybound
        self.w = int((xbound[1] - xbound[0]) / xbound[2])
        self.h = int((ybound[1] - ybound[0]) / ybound[2])

        if z_roll_pitch:
            pass
        else:
            zs = torch.tensor([0.])
            yaws = torch.tensor([0.])
            rolls = torch.tensor([0.])
            pitchs = torch.tensor([0.])
            self.planes = plane_grid(self.xbound, self.ybound, zs, yaws, rolls, pitchs)[0]

        tri_mask = np.zeros((self.h, self.w))
        vertices = np.array([[0, 0], [0, self.h], [self.w, self.h]], np.int32)
        pts = vertices.reshape((-1, 1, 2))
        cv2.fillPoly(tri_mask, [pts], color=1.)
        self.tri_mask = torch.tensor(tri_mask[None, :, :, None])
        self.flipped_tri_mask = torch.flip(self.tri_mask, [2])

    def mask_warped(self, warped_fv_images):
        warped_fv_images[:, CAM_F, :, :self.w//2, :] *= 0  # CAM_FRONT
        warped_fv_images[:, CAM_FL] *= self.flipped_tri_mask.bool()  # CAM_FRONT_LEFT
        warped_fv_images[:, CAM_FR] *= ~self.tri_mask.bool()  # CAM_FRONT_RIGHT
        warped_fv_images[:, CAM_B, :, self.w//2:, :] *= 0  # CAM_BACK
        warped_fv_images[:, CAM_BL] *= self.tri_mask.bool()  # CAM_BACK_LEFT
        warped_fv_images[:, CAM_BR] *= ~self.flipped_tri_mask.bool()  # CAM_BACK_RIGHT
        return warped_fv_images

    def forward(self, images, Ks, RTs, translation=None, yaw_roll_pitch=None, post_RTs=None, visual_sum=False):
        images = images.permute(0, 1, 3, 4, 2).contiguous()
        B, N, H, W, C = images.shape

        if self.z_roll_pitch:
            zs = translation[:, 2]
            rolls = yaw_roll_pitch[:, 1]
            pitchs = yaw_roll_pitch[:, 2]
            planes = plane_grid(self.xbound, self.ybound, zs, torch.zeros_like(rolls), rolls, pitchs)
            planes = planes.repeat(N, 1, 1)
        else:
            planes = self.planes

        images = images.reshape(B*N, H, W, C)
        warped_fv_images = ipm_from_parameters(images, planes, Ks, RTs, self.h, self.w, post_RTs)
        warped_fv_images = warped_fv_images.reshape((B, N, self.h, self.w, C))
        warped_fv_images = self.mask_warped(warped_fv_images)

        if self.visual:
            if visual_sum:
                warped_topdown = warped_fv_images.sum(1)
            else:
                warped_topdown = warped_fv_images[:, CAM_F] + warped_fv_images[:, CAM_B]  # CAM_FRONT + CAM_BACK
                warped_mask = warped_topdown == 0
                warped_topdown[warped_mask] = warped_fv_images[:, CAM_FL][warped_mask] + warped_fv_images[:, CAM_FR][warped_mask]
                warped_mask = warped_topdown == 0
                warped_topdown[warped_mask] = warped_fv_images[:, CAM_BL][warped_mask] + warped_fv_images[:, CAM_BR][warped_mask]
            return warped_topdown.permute(0, 3, 1, 2).contiguous()
        else:
            warped_topdown, _ = warped_fv_images.max(1)
            warped_topdown = warped_topdown.permute(0, 3, 1, 2).contiguous()
            warped_topdown = warped_topdown.view(B, C, self.h, self.w)
            return warped_topdown


if __name__ == '__main__':
    from PIL import Image
    from dataset import MyNuScenes
    from pyquaternion import Quaternion
    from dataset import CAM_POSITION, translation_matrix, mask_label_to_img
    from nuscenes.map_expansion.map_api import NuScenesMap
    from dataset_const import MAP

    dataroot = 'data/nuScenes'
    nusc = MyNuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)

    nusc_maps = {}
    for map_name in MAP:
        nusc_maps[map_name] = NuScenesMap(dataroot=dataroot, map_name=map_name)

    images = []
    seg_labels = []
    trans = []
    rotation = []
    intri = []
    Ks = []
    RTs = []
    translations = []
    yaw_pitch_rolls = []

    sample = nusc.sample[280]
    sample_data = sample['data']

    sample_data_record = nusc.get('sample_data', sample_data['LIDAR_TOP'])

    pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])
    map_pose = pose_record['translation'][:2]
    translation = pose_record['translation']
    pos_rotation = Quaternion(pose_record['rotation'])
    yaw_pitch_roll = np.array(pos_rotation.yaw_pitch_roll) * 180 / np.pi

    translations.append(translation)
    yaw_pitch_rolls.append(yaw_pitch_roll)

    xbound = [-30., 30., 0.15]
    ybound = [-15., 15., 0.15]
    ipm = IPM(xbound, ybound)

    for pos in CAM_POSITION:
        sample_data_record = nusc.get('sample_data', sample_data[pos])
        path = nusc.get_sample_data_path(sample_data[pos])
        img = Image.open(path)
        images.append(np.array(img))
        seg_labels.append(np.array(Image.open(path.split('.')[0] + '_mask.png')))
        cali_sensor = nusc.get('calibrated_sensor', sample_data_record['calibrated_sensor_token'])
        trans.append(cali_sensor['translation'])
        rotation.append(cali_sensor['rotation'])
        intri.append(cali_sensor['camera_intrinsic'])

        K = np.eye(4)
        K[:3, :3] = cali_sensor['camera_intrinsic']
        R_veh2cam = Quaternion(cali_sensor['rotation']).transformation_matrix.T
        T_veh2cam = translation_matrix(-np.array(cali_sensor['translation']))
        RT = R_veh2cam @ T_veh2cam
        Ks.append(K)
        RTs.append(RT)

    images = np.stack(images, 0)
    seg_labels = np.stack(seg_labels, 0)
    trans = np.stack(trans, 0)
    rotation = np.stack(rotation, 0)
    intri = np.stack(intri, 0)
    Ks = np.stack(Ks, 0)
    RTs = np.stack(RTs, 0)

    images = torch.ByteTensor(images)
    seg_labels = torch.ByteTensor(seg_labels)
    Ks = torch.Tensor(Ks)
    RTs = torch.Tensor(RTs)
    translations = torch.Tensor(translations)
    yaw_pitch_rolls = torch.Tensor(yaw_pitch_rolls)

    images = images.permute(0, 3, 1, 2)
    warped_img = ipm(
        torch.stack([images, images]),
        torch.stack([Ks, Ks]),
        torch.stack([RTs, RTs]),
        torch.cat([translations, translations]),
        torch.cat([yaw_pitch_rolls, yaw_pitch_rolls]),
    )[1]
    warped_img = warped_img.permute(1, 2, 0)

    warped_labels_img = ipm(seg_labels[None, :, None, ...], Ks[None], RTs[None], translations, yaw_pitch_rolls)[0][0]
    # warped_img = to_pil(warped_img)

    # test
    from dataset_const import MAP, SIMPLE_NON_GEOMETRIC_POLYGON_LAYERS
    import matplotlib.pyplot as plt
    patch_size = (30, 60)
    canvas_size = (200, 400)

    lidar_top_path = nusc.get_sample_data_path(sample_data['LIDAR_TOP'])

    patch_box = (map_pose[0], map_pose[1], patch_size[0], patch_size[1])
    print(yaw_pitch_rolls)
    patch_angle = yaw_pitch_rolls[0, 0]

    scene_record = nusc.get('scene', sample['scene_token'])
    log_record = nusc.get('log', scene_record['log_token'])
    location = log_record['location']
    topdown_seg_mask = nusc_maps[location].get_map_mask(patch_box, patch_angle, SIMPLE_NON_GEOMETRIC_POLYGON_LAYERS, canvas_size)
    # background
    topdown_seg_mask = np.concatenate((np.zeros((1, canvas_size[0], canvas_size[1])), topdown_seg_mask))

    for i, m in enumerate(topdown_seg_mask):
        m *= i

    topdown_seg_mask = np.argmax(topdown_seg_mask, 0)
    topdown_seg_img = mask_label_to_img(topdown_seg_mask)

    fig, ax = plt.subplots(1, 3)
    plt.grid(b=None)
    plt.axis('off')

    ax[0].imshow(warped_img)
    ax[0].set_title('IPM')
    ax[1].imshow(warped_labels_img)
    ax[1].set_title('Label IPM')
    ax[2].imshow(topdown_seg_img)
    ax[2].set_title('Topdown Label')
    plt.show()
