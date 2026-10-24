import random

import cv2
import matplotlib.pyplot as plt
import numpy
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torch import distributions


class Add_Noise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, imgs, sigma=(0.1, 0.5)):
        noise = torch.randn_like(imgs, device=imgs.device) * 0.1
        I_log = torch.log(imgs + 1.0)
        max_I = I_log.max()
        min_I = I_log.min()
        I_log = (I_log - min_I)/(max_I - min_I)

        print(I_log)

        imgs = I_log + noise
        # imgs = torch.clip(imgs, 0., 1.)
        return imgs


class Speckle_Noise(nn.Module):
    def __init__(self, enl=(2.0, 6.0)):
        super().__init__()
        self.enl = random.random() * (enl[1] - enl[0]) + enl[0]

    def forward(self, imgs):
        concentration = torch.tensor(self.enl)
        gamma = distributions.Gamma(concentration, concentration)
        noise = gamma.sample(imgs.shape)

        # noise = numpy.random.gamma(concentration, 1./concentration, imgs.shape)
        noisy_img = imgs * noise

        return noisy_img


if __name__ == '__main__':
    filter = Speckle_Noise()
    img_path = r'D:\project_in_hit\project_from_others\mae-main\mae-main\1_9.png'

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = Image.fromarray(img)

    transform = transforms.Compose([
        transforms.CenterCrop(256),

        transforms.ToTensor(),
        filter,
        # transforms.Normalize([0.5], [0.5])
    ])

    img_ = transform(img)
    to_pil = transforms.ToPILImage()
    # img_ = to_pil(img_)
    img_ = img_.permute(1, 2, 0).numpy()
    plt.imshow(img, )
    plt.show()
    plt.imshow(img_ , vmin=numpy.percentile(img_, 1), vmax=numpy.percentile(img_, 99))
    plt.show()

    img_de_path = r'D:\project_in_hit\project_from_others\mae-main\mae-main\1_9_de.png'
    img_de = cv2.imread(img_de_path, cv2.IMREAD_GRAYSCALE)
    img_de = Image.fromarray(img_de)
    plt.imshow(img_de)
    plt.show()
