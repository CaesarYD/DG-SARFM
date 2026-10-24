import os
import random
import re

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from .filter.add_noise import Speckle_Noise
from .datasets_for_mae import GaussianNoise
from torchvision.transforms import functional as F
from torch import nn

class MultiScaleCropsTransform:
    def __init__(self, scales, min_scales, max_scales, same_position=True):
        self.scales = list(scales)
        self.min_scales = list(min_scales)
        self.max_scales = list(max_scales)
        assert len(self.scales) == len(self.min_scales)
        assert len(self.scales) > 0
        self.same_position = same_position

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        torch.manual_seed(seed & 0xFFFFFFFF)

    def creat_noise_transform_fn(self):
        trans = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.2),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0)], p=0.4),
            transforms.ToTensor(),
            # Speckle_Noise(),
            transforms.RandomApply([GaussianNoise(sigma=0.1)], p=0.2),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return trans
        

    def creat_clean_transform_fn(self):
        trans = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.2),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0)], p=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return trans
        
    def creat_output_transform_fn(self):
        trans = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.2),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0)], p=0.4),
            transforms.ToTensor(),
            transforms.RandomApply([GaussianNoise(sigma=0.1)], p=0.2),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return trans

    def compute_iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou

    def __call__(self, ori_img: Image.Image, denoised_img: Image.Image):
        input_views = []
        ious = []
        t_noise = self.creat_noise_transform_fn()
        t_clean = self.creat_clean_transform_fn()
        t_output = self.creat_output_transform_fn()

        contrast_bbox = None

        for idx, (size, min_scale, max_scale) in enumerate(zip(self.scales, self.min_scales, self.max_scales)):
            # build transforms for this scale
            randomresize = transforms.RandomResizedCrop(size, scale=(min_scale, max_scale), ratio=(0.8, 1.25), interpolation=InterpolationMode.BICUBIC)

            if idx == 0:
                resize_seed = random.randint(0, int(4e10))
                aug_seed = random.randint(0, int(4e10))
                if self.same_position:
                    self._set_seed(resize_seed)
                    i1, j1, w1, h1 = randomresize.get_params(ori_img, randomresize.scale, randomresize.ratio)
                    in_view = F.resized_crop(ori_img, i1, j1, h1, w1, randomresize.size, randomresize.interpolation)
                    self._set_seed(resize_seed)
                    i2, j2, w2, h2 = randomresize.get_params(denoised_img, randomresize.scale, randomresize.ratio)
                    ct_view = F.resized_crop(denoised_img, i2, j2, h2, w2, randomresize.size, randomresize.interpolation)
                    contrast_bbox = (i2, j2, w2, h2)
                else:
                    self._set_seed(resize_seed)
                    i1, j1, w1, h1 = randomresize.get_params(ori_img, randomresize.scale, randomresize.ratio)
                    in_view = F.resized_crop(ori_img, i1, j1, h1, w1, randomresize.size, randomresize.interpolation)

                    i2, j2, w2, h2 = randomresize.get_params(denoised_img, randomresize.scale, randomresize.ratio)
                    ct_view = F.resized_crop(denoised_img, i2, j2, h2, w2, randomresize.size,
                                             randomresize.interpolation)
                    contrast_bbox = (i2, j2, w2, h2)

                output_view = F.resized_crop(denoised_img, i1, j1, h1, w1, randomresize.size, randomresize.interpolation)
                self._set_seed(aug_seed)
                output_global = t_output(output_view)

                self._set_seed(aug_seed)
                in_view = t_noise(in_view)
                contrast_view = t_clean(ct_view)

                if self.same_position:
                    iou = 1.0
                else:
                    input_bbox = (i1, j1, h1, w1)
                    iou = self.compute_iou(input_bbox, contrast_bbox)
                ious.append(iou)

            else:
                i1, j1, w1, h1 = randomresize.get_params(ori_img, randomresize.scale, randomresize.ratio)
                in_view = F.resized_crop(ori_img, i1, j1, h1, w1, randomresize.size, randomresize.interpolation)
                in_view = t_noise(in_view)
                input_bbox = (i1, j1, h1, w1)
                iou = self.compute_iou(input_bbox, contrast_bbox)
                ious.append(iou)


            input_views.append(in_view)
        ious = torch.tensor(ious)
        ious[1:] = nn.functional.normalize(ious[1:], p=1, dim=-1)

        return input_views, output_global, contrast_view, ious


class MultiScaleDataset(Dataset):
    def __init__(self, image_list, label_list, dataset_list, output_file_dir,
                 scales, min_scales, max_scales, same_position=True):
        self.image_list = image_list
        self.label_list = label_list
        self.dataset_list = dataset_list
        self.output_file_dir = output_file_dir

        self.transform = MultiScaleCropsTransform(
            scales=scales,
            min_scales=min_scales,
            max_scales=max_scales,
            same_position=same_position,
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        ori_path = self.image_list[index]
        ori_img = cv2.imread(ori_path, cv2.IMREAD_COLOR)
        ori_img = Image.fromarray(ori_img)

        label = self.label_list[index]
        denoised_path = os.path.join(self.output_file_dir, self.dataset_list[index], label)
        denoised_img = cv2.imread(denoised_path, cv2.IMREAD_COLOR)
        denoised_img = Image.fromarray(denoised_img)

        input_views, output_global, contrast_views, ious = self.transform(ori_img, denoised_img)
        return [input_views, output_global, contrast_views, ious], label


def load_data(file_dir, output_file_dir, scales, min_scales, max_scales,
              same_position=True):
    pic_list = []
    label_list = []
    dataset_list = []
    for root, dirs, files in os.walk(file_dir):
        files = sorted(files)
        for file in files:
            jpeg_path = os.path.join(root, file)
            pic_list.append(jpeg_path)
            path_spilt = re.split('[/\\\]', jpeg_path)
            label_list.append(path_spilt[-1])
            img_dir = re.sub(file_dir, '', root).strip(r'/\\')
            dataset_list.append(img_dir)

    dataset = MultiScaleDataset(pic_list, label_list, dataset_list, output_file_dir,
                                scales=scales,
                                min_scales=min_scales,
                                max_scales=max_scales,
                                same_position=same_position)
    return dataset


if __name__ == '__main__':
    # example builder functions for noise / clean transform

    # usage example
    path = 'D:/dateset/SAR/pretraining_datasets'
    path_output = 'D:/dateset/SAR/SAR_pretraining_datasets_despeckled/fans'
    scales = [224, 96, 64]  # global first
    min_scales = [0.25, 0.25, 0.25]
    max_scales = [0.5, 0.85, 0.85]
    dataset = load_data(path, path_output, scales, min_scales, max_scales,
                        same_position=True)
    (inp_views, ct_views, ious), label = dataset[0]
    print([v.shape for v in inp_views], ct_views.shape)
    topil = transforms.ToPILImage()
    img0 = topil(inp_views[0])
    img2 = topil(ct_views)
    plt.imshow(img0, cmap='gray')
    plt.show()
    plt.imshow(img2, cmap='gray')
    plt.show()
    print(ious.shape)