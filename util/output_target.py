import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from torchvision import transforms


def create_kernel(w):
    m11 = np.concatenate((np.ones([w + 1, 2 * w + 1]), np.zeros([w, 2 * w + 1])), axis=0)
    m21 = np.concatenate((np.zeros([w, 2 * w + 1]), np.ones([w + 1, 2 * w + 1])), axis=0)
    m13 = np.concatenate((np.ones([2 * w + 1, w + 1]), np.zeros([2 * w + 1, w])), axis=1)
    m23 = np.concatenate((np.zeros([2 * w + 1, w]), np.ones([2 * w + 1, w + 1])), axis=1)
    m11 = torch.from_numpy(m11).float()
    m21 = torch.from_numpy(m21).float()
    m13 = torch.from_numpy(m13).float()
    m23 = torch.from_numpy(m23).float()
    return m11, m21, m13, m23


class SAR_Hog(nn.Module):
    def __init__(self, kernel_size, p):
        super().__init__()
        self.kernel_size = kernel_size
        self.p = p
        m11, m21, m13, m23 = create_kernel(self.kernel_size)
        self.register_buffer('m11', m11.view(1, 1, self.kernel_size*2+1, self.kernel_size*2+1))
        self.register_buffer('m21', m21.view(1, 1, self.kernel_size*2+1, self.kernel_size*2+1))
        self.register_buffer('m13', m13.view(1, 1, self.kernel_size*2+1, self.kernel_size*2+1))
        self.register_buffer('m23', m23.view(1, 1, self.kernel_size*2+1, self.kernel_size*2+1))

    def forward(self, x):
        # x: （B, 1, h, w）
        B, _, H, W = x.shape
        x = F.pad(x, (self.kernel_size,) * 4, mode='reflect') + 1e-2
        m11 = F.conv2d(x, weight=self.m11)
        m21 = F.conv2d(x, weight=self.m21)
        m13 = F.conv2d(x, weight=self.m13)
        m23 = F.conv2d(x, weight=self.m23)
        gh = torch.log(m11 / m21)
        gv = torch.log(m13 / m23)
        gm = torch.sqrt(gh ** 2 + gv ** 2).reshape(B, 1, H * W).reshape(B, 1, H, W)
        return gm

class Multi_SAR_Hog(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_target1 = SAR_Hog(kernel_size=3, p=2)
        self.output_target2 = SAR_Hog(kernel_size=5, p=2)
        self.output_target3 = SAR_Hog(kernel_size=7, p=2)

    def forward(self, x):
        x = torch.stack(
            [self.output_target1(x), self.output_target2(x), self.output_target3(x)], dim=0
        )
        return x

if __name__ == "__main__":
    path_img = r'D:\project_in_hit\project_from_others\mae-main\mae-main\1_9_de.png'
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.CenterCrop(size=224),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor()
    ])

    img = transform(img).unsqueeze(0)
    hog = SAR_Hog(kernel_size=3, p=2)
    gm = hog(img)
    print(gm, img)

    to_pil = transforms.ToPILImage()
    img_pil = to_pil(gm.squeeze(0))
    img = to_pil(img.squeeze(0))
    plt.imshow(img_pil, 'gray')
    plt.show()
    plt.imshow(img, 'gray')
    plt.show()
