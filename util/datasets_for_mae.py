# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random

import PIL
import torch
from torch import nn

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from .filter.add_noise import Speckle_Noise

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma  # 预设的最大噪声强度

    def forward(self, x):
        # 生成高斯噪声并添加
        noise = torch.randn_like(x) * self.sigma
        noisy_x = x + noise

        # 限制值域
        noisy_x = torch.clamp(noisy_x, 0.0, 1.0)
        return noisy_x


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MyDataSet(Dataset):
    def __init__(self, image_list, label_list, transform=None):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = Image.fromarray(img)
        label = self.label_list[index]
        if self.transform:
            img = self.transform(img)
        return img, label


from PIL import Image, ImageFilter
import re
import cv2


def load_data(file_dir, transform):
    pic_list = []
    label_list = []
    for root, dirs, files in os.walk(file_dir):
        files = sorted(files)
        for file in files:
            jpeg_path = os.path.join(root, file)
            # jpeg = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)
            # jpeg = Image.fromarray(cv2.cvtColor(jpeg,cv2.COLOR_BGR2RGB))
            # pic_list.append(jpeg)
            pic_list.append(jpeg_path)
            label_list.append(re.split('[/\\\]', jpeg_path)[-1])

    dataset = MyDataSet(pic_list, label_list, transform)
    return dataset


def load_data_fine_tune(file_dir, transform):
    pic_list = []
    label_list = []
    label_name = os.listdir(file_dir)
    for i, label in enumerate(label_name):
        imgs_path = os.path.join(file_dir, label)
        imgs_list = os.listdir(imgs_path)
        for img_file in imgs_list:
            img_path = os.path.join(imgs_path, img_file)
            pic_list.append(img_path)
            label_list.append(i)

    dataset = MyDataSet(pic_list, label_list, transform)
    # for i in range(len(label_list)):
    #     print(re.split('[/\\\]', pic_list[i])[-2], label_list[i])
    return dataset


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = load_data_fine_tune(root, transform=transform)

    return dataset


def build_transform(is_train, args):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        # transform = create_transform(
        #     input_size=args.input_size,
        #     is_training=True,
        #     color_jitter=args.color_jitter,
        #     auto_augment=args.aa,
        #     interpolation='bicubic',
        #     re_prob=args.reprob,
        #     re_mode=args.remode,
        #     re_count=args.recount,
        #     mean=mean,
        #     std=std,
        # )
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(args.input_size, scale=(0.85, 1.0), interpolation=InterpolationMode.BICUBIC),
        #     # 3 is bicubic
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.2),
        #     transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0)], p=0.4),
        #    transforms.ToTensor(),
        #     # Speckle_Noise(enl=(1.0, 1.0)),
        #     transforms.RandomApply([GaussianNoise(sigma=0.1)], p=0.2),
        #     # transforms.Normalize(mean=[0.182], std=[0.112])
        #     transforms.Normalize(mean=[0.5], std=[0.5])
        #     # transforms.Normalize(mean=[0.001132057125], std=[0.00077576914])
        # ])
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        #     transforms.RandomApply(
        #         [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
        #         p=0.2,  # not strengthened
        #     ),
        #     # transforms.RandomGrayscale(p=0.2),
        #     # transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.ToTensor(),
        #     transforms.RandomApply([GaussianNoise(sigma=0.1)], p=0.2),
        #     ]
        # )
        size = random.randint(256, 256)
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    # t.append(Speckle_Noise(enl=(1.0, 1.0)))
    # t.append(transforms.Normalize(mean=[0.182], std=[0.112]))
    t.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    return transforms.Compose(t)


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, noise_transform, free_noise_transform):
        self.free_noise_transform = free_noise_transform
        self.noise_transform = noise_transform

    def __call__(self, ori_img):
        seed = random.randint(0, 4e12)
        torch.manual_seed(seed)
        noise_img = self.noise_transform(ori_img)
        torch.manual_seed(seed)
        ori_img = self.free_noise_transform(ori_img)
        torch.manual_seed(seed)
        img = torch.stack([noise_img, ori_img], dim=1)
        return img

from .filter.add_noise import Speckle_Noise

def build_transform2(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        # transform = create_transform(
        #     input_size=args.input_size,
        #     is_training=True,
        #     color_jitter=args.color_jitter,
        #     auto_augment=args.aa,
        #     interpolation='bicubic',
        #     re_prob=args.reprob,
        #     re_mode=args.remode,
        #     re_count=args.recount,
        #     mean=mean,
        #     std=std,
        # )
        noise_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0), interpolation=InterpolationMode.BICUBIC),
            # 3 is bicubic
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.2),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0)], p=0.4),
            transforms.ToTensor(),
            Speckle_Noise(),
            transforms.RandomApply([GaussianNoise(sigma=0.1)], p=0.2),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        free_noise_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0), interpolation=InterpolationMode.BICUBIC),
            # 3 is bicubic
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.2),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0)], p=0.4),
            transforms.ToTensor(),
            transforms.RandomApply([GaussianNoise(sigma=0.1)], p=0.2),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        transform = TwoCropsTransform(noise_transform, free_noise_transform)
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        #     transforms.RandomApply(
        #         [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
        #         p=0.2,  # not strengthened
        #     ),
        #     # transforms.RandomGrayscale(p=0.2),
        #     # transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.ToTensor(),
        #     transforms.RandomApply([GaussianNoise(sigma=0.1)], p=0.2),
        #     ]
        # )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    noise_transform = transforms.Compose([
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(args.input_size),
        # 3 is bicubic
        transforms.ToTensor(),
        Speckle_Noise(),
        # transforms.RandomApply([datasets_for_mae.GaussianNoise(sigma=0.1)], p=0.2),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    free_noise_transform = transforms.Compose([
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(args.input_size),
        # 3 is bicubic
        transforms.ToTensor(),
        # transforms.RandomApply([datasets_for_mae.GaussianNoise(sigma=0.1)], p=0.2),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    transform = TwoCropsTransform(noise_transform, free_noise_transform)
    return transform
