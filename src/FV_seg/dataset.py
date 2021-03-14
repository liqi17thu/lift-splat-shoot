import os
import collections
import numpy as np
import cv2
import torch

from PIL import Image
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion

from torch.utils.data import Dataset, DataLoader

from nuimages import NuImages

from shapely.geometry import LineString, MultiLineString, box
from shapely import ops

from .topdown_mask import get_fv_mask

MAP = ['boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown']
CAM_POSITION = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


def translation_matrix(vector):
    M = np.identity(4)
    M[:3, 3] = vector[:3]
    return M


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
            cv2.polylines(mask, [np.asarray(list(line.coords), np.int32).reshape((-1, 2))], False, color=1, thickness=thickness)

    return mask


class CrossViewSegDataset(Dataset):
    def __init__(self, dataroot, version, transforms=None, sample_transforms=None, num_samples=999999999):
        self.nuscene = NuScenes(dataroot=dataroot, version=version, verbose=False)
        self.nusc_maps = {}
        for map_name in MAP:
            self.nusc_maps[map_name] = NuScenesMap(dataroot=dataroot, map_name=map_name)
        self.transforms = transforms
        self.sample_transforms = sample_transforms
        self.num_samples = num_samples

    def __len__(self):
        return min(len(self.nuscene.sample), self.num_samples) * 6

    def get_sample(self, index):
        sample_record = self.nuscene.sample[index]
        sample_record_data = sample_record['data']
        images = []
        masks = []
        Ks = []
        RTs = []
        for pos in CAM_POSITION:
            sample_token = sample_record_data[pos]
            sample_data_record = self.nuscene.get('sample_data', sample_record_data[pos])

            path = self.nuscene.get_sample_data_path(sample_token)
            image = np.array(Image.open(path).convert('RGB'))
            mask = get_fv_mask(self.nuscene, self.nusc_maps, sample_record, pos)
            if self.sample_transforms:
                sample = self.sample_transforms(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            image = torch.ByteTensor(image)
            mask = torch.ByteTensor(mask)

            cali_sensor = self.nuscene.get('calibrated_sensor', sample_data_record['calibrated_sensor_token'])
            K = torch.eye(4)
            K[:3, :3] = torch.tensor(cali_sensor['camera_intrinsic'])
            R_veh2cam = torch.FloatTensor(Quaternion(cali_sensor['rotation']).transformation_matrix.T)
            T_veh2cam = torch.FloatTensor(translation_matrix(-np.array(cali_sensor['translation'])))
            RT = R_veh2cam @ T_veh2cam

            images.append(image)
            masks.append(mask)
            Ks.append(K)
            RTs.append(RT)

        images = torch.stack(images, 0)
        masks = torch.stack(masks, 0)
        Ks = torch.stack(Ks, 0)
        RTs = torch.stack(RTs, 0)
        return images, masks, Ks, RTs

    def __getitem__(self, index):
        sample_record = self.nuscene.sample[index // 6]
        pos = CAM_POSITION[index % 6]
        sample_record_data = sample_record['data']

        sample_token = sample_record_data[pos]
        path = self.nuscene.get_sample_data_path(sample_token)
        image = np.array(Image.open(path).convert('RGB'))
        mask = get_fv_mask(self.nuscene, self.nusc_maps, sample_record, pos)
        if self.transforms:
            sample = self.sample_transforms(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return {'image': image, 'mask': mask}


class FirstViewSegDataset(Dataset):
    def __init__(self, dataroot, version, transforms=None, sample_transforms=None, num_samples=999999999):
        self.dataroot = dataroot
        self.nuimage = NuImages(dataroot=dataroot, version=version, verbose=False, lazy=False)
        self.transforms = transforms
        self.sample_transforms = sample_transforms
        self.num_samples = num_samples

    def __len__(self):
        return min(self.num_samples, len(self.nuimage.sample))

    def get_sample(self, index):

        sample = self.nuimage.sample[index]
        kd_token = sample['key_camera_token']
        sample_data = self.nuimage.get('sample_data', kd_token)

        cam_path = os.path.join(self.dataroot, sample_data['filename'])

        image = np.array(Image.open(cam_path).convert('RGB'))

        semantic_mask, _ = self.nuimage.get_segmentation(kd_token)
        mask = (semantic_mask == 24).astype('float')
        # mask = extract_contour((semantic_mask == 24).astype('uint8'), (900, 1600), thickness=5)

        if self.transforms:
            sample = self.sample_transforms(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        mask[mask != 0] = 1

        return image, mask


    def __getitem__(self, index):
        sample = self.nuimage.sample[index]
        kd_token = sample['key_camera_token']
        sample_data = self.nuimage.get('sample_data', kd_token)

        cam_path = os.path.join(self.dataroot, sample_data['filename'])

        image = np.array(Image.open(cam_path).convert('RGB'))

        semantic_mask, _ = self.nuimage.get_segmentation(kd_token)
        mask = (semantic_mask == 24).astype('float')
        # mask = extract_contour((semantic_mask == 24).astype('uint8'), (900, 1600), thickness=5)

        if self.transforms:
            sample = self.transforms(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        mask[mask != 0] = 1

        return {'image': image, 'mask': mask}


def get_loaders(
        dataroot: str = 'data/nuImages',
        version: str = 'mini',
        batch_size: int = 32,
        num_workers: int = 16,
        train_transforms_fn=None,
        valid_transforms_fn=None):

    if version == 'mini':
        train_version = 'v1.0-mini'
        val_version = 'v1.0-mini'
    else:
        train_version = 'v1.0-train'
        val_version = 'v1.0-val'

    # Creates our train dataset
    train_dataset = FirstViewSegDataset(
        dataroot=dataroot,
        version=train_version,
        transforms=train_transforms_fn
    )

    # Creates our valid dataset
    valid_dataset = FirstViewSegDataset(
        dataroot=dataroot,
        version=val_version,
        transforms=valid_transforms_fn
    )

    # Catalyst uses normal torch.data.DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders



from catalyst import utils
from typing import List
from pathlib import Path
from skimage.io import imread as gif_imread
from sklearn.model_selection import train_test_split


class SegmentationDataset(Dataset):
    def __init__(
            self,
            images: List[Path],
            masks: List[Path] = None,
            transforms=None
    ) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        image = utils.imread(image_path)

        result = {"image": image}

        if self.masks is not None:
            mask = gif_imread(self.masks[idx])
            result["mask"] = mask

        if self.transforms is not None:
            result = self.transforms(**result)

        result["filename"] = image_path.name

        return result


def get_car_loaders(
    images: List[Path],
    masks: List[Path],
    random_state: int,
    valid_size: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transforms_fn = None,
    valid_transforms_fn = None,
) -> dict:

    indices = np.arange(len(images))

    # Let's divide the data set into train and valid parts.
    train_indices, valid_indices = train_test_split(
      indices, test_size=valid_size, random_state=random_state, shuffle=True
    )

    np_images = np.array(images)
    np_masks = np.array(masks)

    # Creates our train dataset
    train_dataset = SegmentationDataset(
      images = np_images[train_indices].tolist(),
      masks = np_masks[train_indices].tolist(),
      transforms = train_transforms_fn
    )

    # Creates our valid dataset
    valid_dataset = SegmentationDataset(
      images = np_images[valid_indices].tolist(),
      masks = np_masks[valid_indices].tolist(),
      transforms = valid_transforms_fn
    )

    # Catalyst uses normal torch.data.DataLoader
    train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      drop_last=True,
    )

    valid_loader = DataLoader(
      valid_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      drop_last=True,
    )

    # And excpect to get an OrderedDict of loaders
    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders
