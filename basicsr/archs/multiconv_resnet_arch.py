import torch
from torch import nn
# from basicsr.archs.test import ExpansionConvNet
import torch.functional as F

from basicsr.utils.channel_transform import darkchannel
from basicsr.utils.registry import ARCH_REGISTRY
from test_ours.dark_channel_show import light_channel


class conv_bn_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_bn_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


# 定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out
class ResNet18(nn.Module):
    def __init__(self,img_ch=3,mid_ch=64,output_ch=3,num_blocks=2):
        super(ResNet18, self).__init__()
        self.inchannel = mid_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=img_ch,out_channels=mid_ch,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = self.make_layer(ResBlock, mid_ch, num_blocks, stride=1)
        self.layer2 = self.make_layer(ResBlock, mid_ch, num_blocks, stride=1)
        self.layer3 = self.make_layer(ResBlock, mid_ch, num_blocks, stride=1)
        self.layer4 = self.make_layer(ResBlock, mid_ch, num_blocks, stride=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=mid_ch, out_channels=output_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

    #这个函数主要是用来，重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        #在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.conv2(out)
        return out

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

class ExpansionConvNet(nn.Module):
    def __init__(self,img_size=512,img_ch=3,output_ch=6,use_se=False):
        super(ExpansionConvNet,self).__init__()

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

# @ARCH_REGISTRY.register()
class DeflareNet(nn.Module):
    def __init__(self,img_size=512, img_ch=3, output_ch=6, use_se=False):
        super(DeflareNet, self).__init__()
        self.diff_conv = ExpansionConvNet(img_size=512,img_ch=3,output_ch=3,use_se=False)

        self.resnet18 = ResNet18()

    def forward(self,diff,x):
        # 差异通道卷积
        diff = self.diff_conv(diff)

        x_diff = x / diff

        re_x_diff = self.resnet18(x_diff)

        output = x_diff - re_x_diff

        return output

if __name__ == "__main__":
    arch = DeflareNet()

    print(arch)

    import torchvision.transforms as transforms
    import cv2 as cv
    import sys
    import matplotlib.pyplot as plt
    import torch.nn.functional as F


    sys.path.append('D:\\data\\tunnel\\Flare')
    img = cv.imread('D:\\data\\tunnel\\Flare\\test\\test_images\\input1.png')

    # 获取原图
    transf = transforms.ToTensor()
    gray_transform = transforms.Grayscale()
    img_tensor = transf(img).unsqueeze(0)  # tensor数据格式是torch(C,H,W)

    # 获取差别图
    img_min=darkchannel(img) * 0.8     #计算每个通道的最小值
    img_max=light_channel(img) * 1.2
    diff = img_max - img_min
    diff_tensor = transf(diff).float().unsqueeze(0) /255
    diff_tensor = enhance_contrast(diff_tensor)

    output = arch(diff_tensor,img_tensor)

    img_tensor = img_tensor.squeeze(0)
    diff_tensor = diff_tensor.squeeze(0)
    output = output.squeeze(0)

    plt.subplots(3,1)

    plt.subplot(3,1,1)
    plt.imshow(img_tensor.detach().permute(1,2,0))
    plt.subplot(3,1,2)
    plt.imshow(diff_tensor.detach().permute(1,2,0))
    plt.subplot(3,1,3)
    plt.imshow(output.detach().permute(1,2,0))

    plt.show()