import time
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

from .topdown_mask import mask_img_to_label


MAP = ['boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown']
CAM_POSITION = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


class MyNuScenes(NuScenes):
    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/data/sets/nuScenes',
                 verbose: bool = True,
                 map_resolution: float = 0.1):
        super(MyNuScenes, self).__init__(version, dataroot, verbose, map_resolution)

    def __make_reverse_index__(self, verbose: bool) -> None:
        """
        De-normalizes database to create reverse indices for common cases.
        :param verbose: Whether to print outputs.
        """

        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()

            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member['token']] = ind

        # Decorate (adds short-cut) sample_annotation table with for category name.
        for record in self.sample_annotation:
            inst = self.get('instance', record['instance_token'])
            record['category_name'] = self.get('category', inst['category_token'])['name']

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get('calibrated_sensor', record['calibrated_sensor_token'])
            sensor_record = self.get('sensor', cs_record['sensor_token'])
            record['sensor_modality'] = sensor_record['modality']
            record['channel'] = sensor_record['channel']

        # Reverse-index samples with sample_data and annotations.
        for record in self.sample:
            record['data'] = {}
            record['anns'] = []

        for record in self.sample_data:
            if record['is_key_frame']:
                try:
                    sample_record = self.get('sample', record['sample_token'])
                except KeyError:
                    continue
                sample_record['data'][record['channel']] = record['token']

        for ann_record in self.sample_annotation:
            try:
                sample_record = self.get('sample', ann_record['sample_token'])
            except KeyError:
                continue
            sample_record['anns'].append(ann_record['token'])

        # Add reverse indices from log records to map records.
        if 'log_tokens' not in self.map[0].keys():
            raise Exception('Error: log_tokens not in map table. This code is not compatible with the teaser dataset.')
        log_to_map = dict()
        for map_record in self.map:
            for log_token in map_record['log_tokens']:
                log_to_map[log_token] = map_record['token']
        for log_record in self.log:
            log_record['map_token'] = log_to_map[log_record['token']]

        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))



def label_onehot_encoding(label, num_classes=4):
    H, W = label.shape
    onehot = np.zeros((H, W, num_classes))
    np.put_along_axis(onehot, label[..., None], 1, axis=-1)
    # onehot.scatter_(0, label[None].long(), 1)
    return onehot

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
        self.nuscene = MyNuScenes(dataroot=dataroot, version=version, verbose=False)
        self.nusc_maps = {}
        for map_name in MAP:
            self.nusc_maps[map_name] = NuScenesMap(dataroot=dataroot, map_name=map_name)
        self.transforms = transforms
        self.sample_transforms = sample_transforms
        self.num_samples = num_samples
        self.nuscene.sample.sort(key=lambda x: (x['scene_token'], x['timestamp']))
        self.nuscene.sample = self.nuscene.sample[74:]

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
            mask = mask_img_to_label(Image.open(path))

            # mask = get_fv_mask(self.nuscene, self.nusc_maps, sample_record, pos)
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
        mask_path = path.split('.')[0] + '_line_mask.png'

        image = np.array(Image.open(path).convert('RGB'))
        mask = mask_img_to_label(Image.open(mask_path))
        mask = label_onehot_encoding(mask)
        if self.transforms:
            sample = self.transforms(image=image, mask=mask)
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

def get_cv_loaders(
        dataroot: str = 'data/nuImages',
        version: str = 'mini',
        batch_size: int = 32,
        num_workers: int = 10,
        train_transforms_fn=None,
        valid_transforms_fn=None):

    if version == 'mini':
        train_version = 'v1.0-mini'
        val_version = 'v1.0-mini'
    else:
        train_version = 'v1.0-trainval-train'
        val_version = 'v1.0-trainval-val'

    # Creates our train dataset
    train_dataset = CrossViewSegDataset(
        dataroot=dataroot,
        version=train_version,
        transforms=train_transforms_fn
    )

    # Creates our valid dataset
    valid_dataset = CrossViewSegDataset(
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
