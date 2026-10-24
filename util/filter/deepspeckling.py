import matplotlib.pyplot as plt
from deepdespeckling.utils.load_cosar import cos2mat
from deepdespeckling.utils.constants import PATCH_SIZE, STRIDE_SIZE
from deepdespeckling.sar2sar.sar2sar_denoiser import Sar2SarDenoiser

# Load you image
image = r'D:\project_in_hit\project_from_others\mae-main\mae-main\0.png'
# NB: image must be in AMPLITUDE format and have the following shape: [img_height,img_width]

# Denoise the image with SAR2SAR
denoiser = Sar2SarDenoiser()
denoised_image = denoiser.denoise_image(
                image, patch_size=PATCH_SIZE, stride_size=STRIDE_SIZE)

noise = denoised_image['noisy']
denoise = denoised_image['denoised']

plt.imshow(image)
plt.show()
plt.imshow(noise)
plt.show()
plt.imshow(denoise)
plt.show()