import numpy as np


def darkchannel(Image):             #计算每个通道中的最小值，输入Image图像，输出最小值img_min
    rows,cols,channels=Image.shape
    img=np.array(Image)
    for i in range(0,rows-1):
      for j in range(0,cols-1):
        min_rgb = Image[i][j][0]
        if min_rgb  > Image[i][j][1]:
          min_rgb = Image[i][j][1]
        elif min_rgb  > Image[i][j][2]:
          min_rgb = Image[i][j][2]
        for c in range(channels):
          img[i][j][c] = min_rgb
    return img

def light_channel(Image):
    rows,cols,channels=Image.shape
    img=np.array(Image)
    for i in range(0,rows-1):
        for j in range(0,cols-1):
            max_rgb = Image[i][j][0]
            if max_rgb  < Image[i][j][1]:
                max_rgb = Image[i][j][1]
            elif max_rgb  < Image[i][j][2]:
                max_rgb = Image[i][j][2]
            for c in range(channels):
              img[i][j][c] = max_rgb
    return img

import torch
import torchvision.transforms as transforms

def enhance_contrast(images, clip=0.10):
    """
    Enhances the contrast of a batch of images by clipping their pixel values.

    Args:
        images (torch.Tensor): A batch of input images as a 4D tensor with shape (batch_size, channels, height, width).
        clip (float): The fraction of pixel values to be clipped.

    Returns:
        A 4D tensor with enhanced contrast and the same shape as the input tensor.
    """

    # Compute the clipping threshold for each channel
    n_pixels = images.size(2) * images.size(3)
    topk_values, _ = torch.topk(images.view(images.size(0), images.size(1), -1), int(n_pixels * clip), dim=-1)
    clip_values = topk_values[:, :, -1]

    # Clip the pixel values in the input tensor
    clipped_images = torch.clamp(images, max=clip_values[:, :, None, None])

    # Normalize the pixel values to the range [0, 1]
    transformed_images = (clipped_images - clipped_images.min()) / (clipped_images.max() - clipped_images.min())

    return transformed_images