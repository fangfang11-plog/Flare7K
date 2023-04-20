import os

from dehaze import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.nn as nn
from skimage.filters import threshold_otsu
import torchvision


class ExpansionConvNet(nn.Module):
    def __init__(self,img_size=512,img_ch=3,output_ch=6,use_se=False):
        super().__init__()

        # First channel
        self.conv11 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu12 = nn.ReLU()
        self.expand1 = 1

        # Second channel
        self.conv21 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu21 = nn.ReLU()
        self.conv22 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=2, dilation=2)
        self.relu22 = nn.ReLU()
        self.expand2 = 2

        # Third channel
        self.conv31 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2, dilation=2)
        self.relu31 = nn.ReLU()
        self.conv32 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=3, dilation=3)
        self.relu32 = nn.ReLU()
        self.expand3 = 2

        # Connection layer
        self.conv_connect = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=1)

        # Optimization layer
        self.conv_optimize = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        self.no_linear = nn.Sigmoid()

    def forward(self, x):
        # First channel
        out11 = self.conv11(x)
        out11 = self.relu11(out11)
        out12 = self.conv12(out11)
        out12 = self.relu12(out12)
        # out1 = out12

        # Second channel
        out21 = self.conv21(x)
        out21 = self.relu21(out21)
        out22 = self.conv22(out21)
        out22 = self.relu22(out22)

        # Third channel
        out31 = self.conv31(x)
        out31 = self.relu31(out31)
        out32 = self.conv32(out31)
        out32 = self.relu32(out32)

        # Concatenate outputs from three channels
        out = torch.cat([out12, out22, out32], dim=1)

        # Connection layer
        out = self.conv_connect(out)

        # Optimization layer
        out = self.conv_optimize(out)

        out = self.no_linear(out)

        adjust_out = enhance_contrast(out)
        return adjust_out



import torch
import torchvision.transforms as transforms

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import torch
import torchvision.transforms as transforms

import torch
import torchvision.transforms as transforms

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




# cnn = ExpansionConvNet()
# print(cnn)
# import torchvision.transforms as transforms
# import cv2 as cv
# import sys
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
#
#
# sys.path.append('D:\\data\\tunnel\\Flare')
# img = cv.imread('D:\\data\\tunnel\\Flare\\test\\test_images\\input1.png')
#
#
# # 获取原图
# transf = transforms.ToTensor()
# gray_transform = transforms.Grayscale()
# img_tensor = transf(img).unsqueeze(0)  # tensor数据格式是torch(C,H,W)
#
# # 获取差别图
# img_min=darkchannel(img) * 0.8     #计算每个通道的最小值
# img_max=light_channel(img) * 1.2
# diff = img_max - img_min
# diff_tensor = transf(diff).float().unsqueeze(0) /255
# diff_tensor = enhance_contrast(diff_tensor)
#
# atmospheres = torch.tensor([[0.8, 0.6, 0.4], [0.7, 0.5, 0.3]])
# dehaze_img = dehaze(img_tensor,atmospheres)
#
# out,adjust_out= cnn(img_tensor)
# diff_out,adjust_diff_out = cnn(diff_tensor)
#
# origin=img_tensor.squeeze(0)
# diff_tensor=diff_tensor.squeeze(0)
# out = out.squeeze(0)
# diff_out = diff_out.squeeze(0)
# adjust_out = adjust_out.squeeze(0)
# adjust_diff_out = adjust_diff_out.squeeze(0)
# dehaze_img = dehaze_img.squeeze(0)
#
# plt.subplots(3,3)
# plt.subplot(3,3,1)
# plt.imshow(origin.detach().permute(1,2,0)[:,:,:3])
# plt.subplot(3,3,2)
# plt.imshow(out.detach().permute(1,2,0))
# # plt.subplot(2,3,3)
# # plt.imshow(out.detach().permute(1,2,0)[:,:,3:])
# plt.subplot(3,3,3)
# plt.imshow(adjust_out.detach().permute(1,2,0))
#
# plt.subplot(3,3,4)
# plt.imshow(diff_tensor.detach().permute(1,2,0)[:,:,:3])
# plt.subplot(3,3,5)
# # plt.imshow(diff_out.squeeze(0).detach())
# plt.imshow(diff_out.detach().permute(1,2,0),cmap='Greys')
#
# plt.subplot(3,3,6)
# plt.imshow(adjust_diff_out.detach().permute(1,2,0),cmap='Greys')
#
# plt.subplot(3,3,7)
# plt.imshow(dehaze_img.detach().permute(1,2,0)[:,:,:3])
#
#
# # plt.subplot(2,3,5)
# # plt.imshow(mask_tensor.detach().permute(1,2,0))
#
# plt.show()