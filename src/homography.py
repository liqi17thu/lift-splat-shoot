import numpy as np
import torch
import cv2

CAM_FL = 0
CAM_F = 1
CAM_FR = 2
CAM_BL = 3
CAM_B = 4
CAM_BR = 5

import numpy as np
import torch
import torch.nn as nn
import cv2

# =========================================================
# Projections
# =========================================================
def rotation_from_euler(roll=1., pitch=1., yaw=1.):
    """
    Get rotation matrix
    Args:
        roll, pitch, yaw:       In degrees

    Returns:
        R:          [4, 4]
    """
    si, sj, sk = np.sin(np.deg2rad(roll)), np.sin(np.deg2rad(pitch)), np.sin(np.deg2rad(yaw))
    ci, cj, ck = np.cos(np.deg2rad(roll)), np.cos(np.deg2rad(pitch)), np.cos(np.deg2rad(yaw))
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    R = np.identity(4)
    R[0, 0] = cj * ck
    R[0, 1] = sj * sc - cs
    R[0, 2] = sj * cc + ss
    R[1, 0] = cj * sk
    R[1, 1] = sj * ss + cc
    R[1, 2] = sj * cs - sc
    R[2, 0] = -sj
    R[2, 1] = cj * si
    R[2, 2] = cj * ci
    return R


def perspective(cam_coords, proj_mat, h, w):
    """
    P = proj_mat @ (x, y, z, 1)
    Project cam2pixel

    Args:
        cam_coords:         [4, npoints]
        proj_mat:           [B, 4, 4]

    Returns:
        pix coords:         [B, h, w, 2]
    """
    eps = 1e-7
    pix_coords = proj_mat @ cam_coords
    N, _, _ = pix_coords.shape

    pix_coords = pix_coords[:, :2, :] / (pix_coords[:, 2, :][:, None, :] + eps)
    if isinstance(pix_coords, torch.Tensor):
        pix_coords = pix_coords.view(N, 2, h, w)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
    else:
        pix_coords = np.reshape(pix_coords, (N, 2, h, w))
        pix_coords = np.transpose(pix_coords, (0, 2, 3, 1))
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

    if isinstance(pix_coords, torch.Tensor):
        pix_x, pix_y = torch.split(pix_coords, 1, dim=-1)  # [B, pix_h, pix_w, 1]
    else:
        pix_x, pix_y = np.split(pix_coords, [1], axis=-1)  # [B, pix_h, pix_w, 1]
    # pix_x = pix_x.astype(np.float32)
    # pix_y = pix_y.astype(np.float32)

    # Rounding
    if isinstance(pix_x, torch.Tensor):
        pix_x0 = torch.floor(pix_x)
        pix_x1 = pix_x0 + 1
        pix_y0 = torch.floor(pix_y)
        pix_y1 = pix_y0 + 1
    else:
        pix_x0 = np.floor(pix_x)
        pix_x1 = pix_x0 + 1
        pix_y0 = np.floor(pix_y)
        pix_y1 = pix_y0 + 1


    # Clip within image boundary
    y_max = (img_h - 1)
    x_max = (img_w - 1)
    zero = np.zeros([1])

    if isinstance(pix_x0, torch.Tensor):
        pix_x0 = torch.clip(pix_x0, 0, x_max)
        pix_y0 = torch.clip(pix_y0, 0, y_max)
        pix_x1 = torch.clip(pix_x1, 0, x_max)
        pix_y1 = torch.clip(pix_y1, 0, y_max)

    else:
        pix_x0 = np.clip(pix_x0, zero, x_max)
        pix_y0 = np.clip(pix_y0, zero, y_max)
        pix_x1 = np.clip(pix_x1, zero, x_max)
        pix_y1 = np.clip(pix_y1, zero, y_max)

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
    if isinstance(pix_x0, torch.Tensor):
        idx00 = (pix_x0 + base_y0).view(B, -1, 1).repeat(1, 1, img_c).long()
        idx01 = (pix_x0 + base_y1).view(B, -1, 1).repeat(1, 1, img_c).long()
        idx10 = (pix_x1 + base_y0).view(B, -1, 1).repeat(1, 1, img_c).long()
        idx11 = (pix_x1 + base_y1).view(B, -1, 1).repeat(1, 1, img_c).long()
    else:
        idx00 = (pix_x0 + base_y0).reshape(B, -1, 1).astype(np.int)
        idx01 = (pix_x0 + base_y1).reshape(B, -1, 1).astype(np.int)
        idx10 = (pix_x1 + base_y0).reshape(B, -1, 1).astype(np.int)
        idx11 = (pix_x1 + base_y1).reshape(B, -1, 1).astype(np.int)

    # Gather pixels from image using vertices
    imgs_flat = imgs.reshape([B, -1, img_c])

    if isinstance(imgs_flat, torch.Tensor):
        im00 = torch.gather(imgs_flat, 1, idx00).reshape(out_shape)
        im01 = torch.gather(imgs_flat, 1, idx01).reshape(out_shape)
        im10 = torch.gather(imgs_flat, 1, idx10).reshape(out_shape)
        im11 = torch.gather(imgs_flat, 1, idx11).reshape(out_shape)
    else:
        im00 = np.take_along_axis(imgs_flat, idx00, 1).reshape(out_shape)
        im01 = np.take_along_axis(imgs_flat, idx01, 1).reshape(out_shape)
        im10 = np.take_along_axis(imgs_flat, idx10, 1).reshape(out_shape)
        im11 = np.take_along_axis(imgs_flat, idx11, 1).reshape(out_shape)

    # Apply weights [pix_h, pix_w, 1]
    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1
    output = w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11
    return output


class Plane:
    """
    Defines a plane in the world
    """

    def __init__(self, xbound, ybound, z=0., yaw=0., roll=0., pitch=0.):
        self.xbound = xbound
        self.ybound = ybound
        self.z = z
        self.yaw, self.roll, self.pitch = yaw, roll, pitch
        self.xyz = self.xyz_coord()
        self.xyz = torch.Tensor(self.xyz).cuda()


    def xyz_coord(self):
        """
        Returns:
            Grid coordinate: [b, 3/4, row*cols]
        """
        xmin = self.xbound[0]
        xmax = self.xbound[1]
        num_x = int((self.xbound[1] - self.xbound[0]) / self.xbound[2])
        ymin = self.ybound[0]
        ymax = self.ybound[1]
        num_y = int((self.ybound[1] - self.ybound[0]) / self.ybound[2])
        grid = meshgrid(xmin, xmax, num_x, ymin, ymax, num_y, self.z)
        rotation_matrix = rotation_from_euler(self.roll, self.pitch, self.yaw)
        grid = rotation_matrix @ grid
        return grid


def meshgrid(xmin, xmax, num_x, ymin, ymax, num_y, z, is_homogeneous=True):
    """
    Grid is parallel to z-axis

    Returns:
        array x,y,z,[1] coordinate   [3/4, num_x * num_y]
    """
    x = np.linspace(xmin, xmax, num_x)
    y = np.linspace(ymin, ymax, num_y)
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    z = np.ones_like(x) * z

    if is_homogeneous:
        coords = np.stack([y, x, z, np.ones_like(x)], axis=0)
    else:
        coords = np.stack([y, x, z], axis=0)
    return coords


def ipm_from_parameters(image, xyz, K, RT, target_h, target_w, post_RT=None):
    """
    :param image: [B, H, W, C]
    :param xyz: [4, npoints]
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
    if isinstance(image2, np.ndarray):
        image2 = image2.astype(image.dtype)
    else:
        image2 = image2.type_as(image)
    return image2


class IPM(nn.Module):
    def __init__(self, xbound, ybound, z=0., yaw=0., roll=0., pitch=0.):
        super(IPM, self).__init__()

        self.plane = Plane(xbound, ybound, z, yaw, roll, pitch)

        self.w = int((xbound[1] - xbound[0]) / xbound[2])
        self.h = int((ybound[1] - ybound[0]) / ybound[2])
        self.half_mask = np.zeros((1, self.h//2, self.w, 1))

        tri_mask = np.zeros((self.h, self.w))
        vertices = np.array([[0, 0], [0, self.h], [self.w, self.h]], np.int32)
        pts = vertices.reshape((-1, 1, 2))
        cv2.fillPoly(tri_mask, [pts], color=1.)
        self.tri_mask = tri_mask[None, :, :, None]

    def forward(self, images, Ks, RTs, post_RTs=None):
        B, N, H, W, C = images.shape

        images = images.reshape(B * N, H, W, C)
        warped_fv_images = ipm_from_parameters(images, self.plane.xyz, Ks, RTs, self.h, self.w, post_RTs)
        warped_fv_images = warped_fv_images.reshape((B, N, self.h, self.w, C))

        # max pooling
        warped_topdown, _ = warped_fv_images.max(1)
        return warped_topdown

        if isinstance(warped_fv_images, np.ndarray):
            half_mask = self.half_mask.astype(warped_fv_images.dtype)
            tri_mask = self.tri_mask.astype(warped_fv_images.dtype)
            fliped_tri_mask = np.flip(tri_mask, 2)
        elif isinstance(warped_fv_images, torch.Tensor):
            half_mask = torch.Tensor(self.half_mask).type_as(warped_fv_images)
            tri_mask = torch.Tensor(self.tri_mask).type_as(warped_fv_images)
            fliped_tri_mask = torch.flip(tri_mask, [2])
            if warped_fv_images.is_cuda:
                half_mask = half_mask.cuda()
                tri_mask = tri_mask.cuda()
                fliped_tri_mask = fliped_tri_mask.cuda()
        else:
            raise NotImplementedError

        warped_fv_images[:, CAM_F, :self.h // 2, :, :] = half_mask  # CAM_FRONT
        warped_fv_images[:, CAM_FL] *= fliped_tri_mask  # CAM_FRONT_LEFT
        warped_fv_images[:, CAM_FR] *= tri_mask  # CAM_FRONT_RIGHT
        warped_fv_images[:, CAM_B, self.h // 2:, :, :] *= half_mask  # CAM_BACK
        warped_fv_images[:, CAM_BL] *= 1 - tri_mask  # CAM_BACK_LEFT
        warped_fv_images[:, CAM_BR] *= 1 - fliped_tri_mask  # CAM_BACK_RIGHT

        warped_topdown = warped_fv_images[:, CAM_F] + warped_fv_images[:, CAM_B]  # CAM_FRONT + CAM_BACK
        warped_mask = warped_topdown == 0
        warped_topdown[warped_mask] = warped_fv_images[:, CAM_FL][warped_mask] + warped_fv_images[:, CAM_FR][
            warped_mask]
        warped_mask = warped_topdown == 0
        warped_topdown[warped_mask] = warped_fv_images[:, CAM_BL][warped_mask] + warped_fv_images[:, CAM_BR][
            warped_mask]

        # if isinstance(warped_topdown, np.ndarray):
        #     warped_topdown = np.flip(warped_topdown, 1)
        # else:
        #     warped_topdown = torch.flip(warped_topdown, [1])
        # warped_topdown, _ = warped_fv_images.max(1)
        return warped_topdown