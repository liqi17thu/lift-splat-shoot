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

    si, sj, sk = torch.sin(torch.deg2rad(rolls)), torch.sin(torch.deg2rad(pitchs)), torch.sin(torch.deg2rad(yaws))
    ci, cj, ck = torch.cos(torch.deg2rad(rolls)), torch.cos(torch.deg2rad(pitchs)), torch.cos(torch.deg2rad(yaws))
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    R = torch.eye(4, dtype=torch.double).unsqueeze(0).repeat(B, 1, 1).cuda()
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
    with open('master_cam_coords.npy', 'wb') as f:
        np.save(f, cam_coords.cpu().detach().numpy())
    with open('master_proj_mat.npy', 'wb') as f:
        np.save(f, proj_mat.cpu().detach().numpy())
    eps = 1e-7
    pix_coords = proj_mat @ cam_coords.float()
    N, _, _ = pix_coords.shape

    pix_coords = pix_coords[:, :2, :] / (pix_coords[:, 2, :][:, None, :] + eps)
    pix_coords = pix_coords.view(N, 2, h, w)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
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

    y = torch.linspace(xmin, xmax, num_x, dtype=torch.double).cuda()
    x = torch.linspace(ymin, ymax, num_y, dtype=torch.double).cuda()

    y, x = torch.meshgrid(x, y)

    x = x.flatten()
    y = y.flatten()

    x = x.unsqueeze(0).repeat(B, 1)
    y = y.unsqueeze(0).repeat(B, 1)

    z = torch.ones_like(x).cuda() * zs.view(-1, 1)
    d = torch.ones_like(x).cuda()

    coords = torch.stack([y, x, z, d], axis=1)

    with open('master_coords_before.npy', 'wb') as f:
        np.save(f, coords.cpu().detach().numpy())

    rotation_matrix = rotation_from_euler(rolls, pitchs, yaws)

    coords = rotation_matrix @ coords
    with open('master_coords_after.npy', 'wb') as f:
        np.save(f, coords.cpu().detach().numpy())
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

    with open('master_P.npy', 'wb') as f:
        np.save(f, P.detach().cpu().numpy())
    with open('master_pixel_coords.npy', 'wb') as f:
        np.save(f, pixel_coords.detach().cpu().numpy())

    image2 = bilinear_sampler(image, pixel_coords)
    image2 = image2.type_as(image)
    return image2


def plane_esti(images):
    B, N, H, W, C = images.shape
    z = torch.zeros(B)
    yaw = torch.zeros(B)
    roll = torch.zeros(B)
    pitch = torch.zeros(B)
    return z, yaw, roll, pitch


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
    def __init__(self, xbound, ybound, N, C):
        super(IPM, self).__init__()
        # self.plane_esti = PlaneEstimationModule(N, C)

        self.xbound = xbound
        self.ybound = ybound

        self.w = int((xbound[1] - xbound[0]) / xbound[2])
        self.h = int((ybound[1] - ybound[0]) / ybound[2])
        self.half_mask = np.zeros((1, self.h // 2, self.w, 1))

        tri_mask = np.zeros((self.h, self.w))
        vertices = np.array([[0, 0], [0, self.h], [self.w, self.h]], np.int32)
        pts = vertices.reshape((-1, 1, 2))
        cv2.fillPoly(tri_mask, [pts], color=1.)
        self.tri_mask = tri_mask[None, :, :, None]

    def forward(self, images, Ks, RTs, zs, yaws, rolls, pitchs, post_RTs=None):
        # z, pitch, roll = self.plane_esti(images)
        # zs += z
        # pitchs += pitch
        # rolls += roll

        images = images.permute(0, 1, 3, 4, 2)
        B, N, H, W, C = images.shape
        planes = plane_grid(self.xbound, self.ybound, zs, yaws, rolls, pitchs)
        planes = planes.repeat(N, 1, 1)
        images = images.reshape(B * N, H, W, C)

        with open('master_images.npy', 'wb') as f:
            np.save(f, images.detach().cpu().numpy())
        with open('master_planes.npy', 'wb') as f:
            np.save(f, planes.detach().cpu().numpy())
        with open('master_Ks.npy', 'wb') as f:
            np.save(f, Ks.detach().cpu().numpy())
        with open('master_RTs.npy', 'wb') as f:
            np.save(f, RTs.detach().cpu().numpy())
        with open('master_post_RTs.npy', 'wb') as f:
            np.save(f, post_RTs.detach().cpu().numpy())

        warped_fv_images = ipm_from_parameters(images, planes, Ks, RTs, self.h, self.w, post_RTs)
        warped_fv_images = warped_fv_images.reshape((B, N, self.h, self.w, C))

        warped_topdown = torch.max(warped_fv_images, 1)[0]
        return warped_topdown.permute(0, 3, 1, 2)

        half_mask = torch.Tensor(self.half_mask).type_as(warped_fv_images)
        tri_mask = torch.Tensor(self.tri_mask).type_as(warped_fv_images)
        fliped_tri_mask = torch.flip(tri_mask, [2])
        if warped_fv_images.is_cuda:
            half_mask = half_mask.cuda()
            tri_mask = tri_mask.cuda()
            fliped_tri_mask = fliped_tri_mask.cuda()

        warped_fv_images[:, CAM_F, :self.h // 2, :, :] = half_mask  # CAM_FRONT
        warped_fv_images[:, CAM_FL] *= fliped_tri_mask  # CAM_FRONT_LEFT
        warped_fv_images[:, CAM_FR] *= tri_mask  # CAM_FRONT_RIGHT
        warped_fv_images[:, CAM_B, self.h // 2:, :, :] *= half_mask  # CAM_BACK
        warped_fv_images[:, CAM_BL] *= 1 - tri_mask  # CAM_BACK_LEFT
        warped_fv_images[:, CAM_BR] *= 1 - fliped_tri_mask  # CAM_BACK_RIGHT

        # max pooling
        warped_topdown, _ = warped_fv_images.max(1)
        return warped_topdown

        warped_topdown = warped_fv_images[:, CAM_F] + warped_fv_images[:, CAM_B]  # CAM_FRONT + CAM_BACK
        warped_mask = warped_topdown == 0
        warped_topdown[warped_mask] = warped_fv_images[:, CAM_FL][warped_mask] + warped_fv_images[:, CAM_FR][warped_mask]
        warped_mask = warped_topdown == 0
        warped_topdown[warped_mask] = warped_fv_images[:, CAM_BL][warped_mask] + warped_fv_images[:, CAM_BR][warped_mask]

        return warped_topdown.permute(0, 3, 1, 2)

