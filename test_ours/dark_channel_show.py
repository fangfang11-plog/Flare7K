import cv2 as cv
import numpy
import numpy as np

def zmMinFilterGray(src, r=5):
  '''最小值滤波，r是滤波器半径'''
  return cv.erode(src, np.ones((2 * r + 1, 2 * r + 1)))

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


def min_filter(Image,r):                # 最小值滤波，输入最小值图像，在2*r+1的矩形窗口内寻找最小值
    rows, cols, channels = Image.shape    # 输出为暗通道图像
    img = np.array(Image)
    for i in range(0, rows):
      for j in range(0, cols):
        for c in range(0, channels):
          if i == 0 or j == 0 or i == rows - 1 or j == cols - 1:
            img[i][j][c] = Image[i][j][c]
          elif j == 0:
            img[i][j][c] = Image[i][j][c]
          else:
            min = 255
            for m in range(i - r, i + r):         # 寻找像素点(i,j)为中心的5*5窗口内的每个通道的最小值
              for n in range(j - r, j + r):
                if min > Image[m][n][c]:
                  min = Image[m][n][c]
            img[i][j][c] = min
    return img


def guided_filter(Image,p,r,eps):     # 基于导向滤波进行暗通道图像的变换
    #Image归一化之后的原图，p最小值图像，r导向滤波搜索范围，eps为惩罚项，输出导向滤波后的图像
    # q = a * I + b
    mean_I = cv.blur(Image, (r, r))  # I的均值平滑
    mean_p = cv.blur(p, (r, r))  # p的均值平滑
    mean_II = cv.blur(Image*Image, (r, r))  # I*I的均值平滑
    mean_Ip = cv.blur(Image*p, (r, r))  # I*p的均值平滑
    var_I = mean_II - mean_I * mean_I  # 方差
    cov_Ip = mean_Ip - mean_I * mean_p  # 协方差
    a = cov_Ip / (var_I +eps)
    b = mean_p - a *mean_I
    mean_a = cv.blur(a, (r, r))  # 对a、b进行均值平滑
    mean_b = cv.blur(b, (r, r))
    q = mean_a*Image + mean_b
    return q

def select_bright(Image,img_origin,w,t0,V):          #计算大气光A和折射图t
    #输入：Image最小值图像，img_origion原图，w是t之前的修正参数，t0阈值，V导向滤波结果
    rows,cols,channels=Image.shape
    size=rows*cols
    order = [0 for i in range(size)]
    m = 0
    for t in range(0,rows):
      for j in range(0,cols):
        order[m] = Image[t][j][0]
        m = m+1
    order.sort(reverse=True)
    index =int(size * 0.001) #从暗通道中选取亮度最大的前0.1%
    mid = order[index]
    A = 0
    img_hsv = cv.cvtColor(img_origin,cv.COLOR_RGB2HLS)
    for i in range(0,rows):
      for j in range(0,cols):
        if Image[i][j][0]>mid and img_hsv[i][j][1]>A:
          A = img_hsv[i][j][1]
    V = V * w
    t = 1 - V/A
    t = np.maximum(t,t0)
    return t,A

def repair(Image,t,A):
    rows, cols, channels = Image.shape
    J = np.zeros(Image.shape)
    for i in range(0,rows):
      for j in range(0,cols):
        for c in range(0,channels):
          t[i][j][c] = t[i][j][c]-0.25 # 不知道为什么这里减掉0.25效果才比较好
          J[i][j][c] = (Image[i][j][c]-A/255.0)/t[i][j][c]+A/255.0
    return J

def get_flare(Image,threshold):
    rows,cols,channels= Image.shape
    J = np.zeros(Image.shape)
    for i in range(0,rows):
        for j in range(0,cols):
            for k in range(0,channels):
                if Image[i][j][k] > threshold:
                    J[i][j][k] = Image[i][j][k]
    return J

def del_flare(threshold_Image,Image):
    rows,cols,channels= Image.shape
    result_image = np.zeros(Image.shape)
    for i in range(0,rows):
        for j in range(0,cols):
            for k in range(0,channels):
                if threshold_Image[i][j][k] > 0:
                    result_image[i][j][k]= Image[i][j][k] = 0
                else:
                    result_image[i][j][k]= Image[i][j][k]
    return result_image


# import matplotlib.pyplot as plt
#
# # img = cv.imread('../dataset/Flare7k/Scattering_Flare/Glare_with_shimmer/glare_000007.png')
# img = cv.imread('../dataset/Flare7k/test_data/real/input/input_000000.png')
# gt = cv.imread('../dataset/Flare7k/test_data/real/gt/gt_000000.png')
# # 读入图片
# # img = cv.imread('../test/test_images/input5.png')
#
# img_arr=np.array(img/255.0)                     #归一化
#
# # 获取diff
# img_min=darkchannel(img_arr) * 0.8    #计算每个通道的最小值
# img_max=light_channel(img_arr) * 1.2
# diff = img_max - img_min
# diff_mean = numpy.mean(diff[:,:,0])
# width,height,channels = diff.shape
# for i in range(0,width):
#     for j in range(0,height):
#         for k in range(0,channels):
#             if diff[i,j,k] > diff_mean * 2:
#                 diff[i][j][k] = diff[i,j,k] * (1 + (diff[i,j,k]))
#             else:
#                 diff[i][j][k] = diff[i,j,k] * (1 - (diff[i,j,k]))


# generate dehaze
# img_dark=min_filter(img_min,2)       #计算暗通道图像
# img_guided=guided_filter(img_arr,img_min,r=200,eps=0.001)
# t,A=select_bright(img_min,img,w=0.95,t0=0.1,V=img_guided)
# dehaze=repair(img_arr,t,A)
#
# # 获取diff
# img_min_dehaze = darkchannel(dehaze) * 0.8
# img_max_dehaze = light_channel(dehaze) * 0.8
# diff_dehaze = img_max_dehaze - img_min_dehaze
#
# # generate deflare
# img_aug_diff=min_filter(diff,2)       #计算暗通道图像
# img_guided=guided_filter(img_arr,diff,r=75,eps=0.001)
# t,A=select_bright(diff,img,w=0.95,t0=0.1,V=img_guided)
# deflare=repair(img_arr,t,A)
#
# # generate deflare from dehaze
# img_aug_diff_dehaze=min_filter(diff_dehaze,2)       #计算暗通道图像
# img_guided=guided_filter(img_arr,diff_dehaze,r=75,eps=0.001)
# t,A=select_bright(diff,img,w=0.95,t0=0.1,V=img_guided)
# deflare_from_dehaze=repair(img_arr,t,A)
#
# plt.subplots(2,2)
#
# plt.subplot(2,2,1)
# plt.imshow(img_arr)
# plt.title('origin')
#
# plt.subplot(2,2,2)
# plt.imshow(img_min)
# plt.title('img_min')
#
# plt.subplot(2,2,3)
# plt.imshow(diff)
# plt.title('diff')
#
# plt.subplot(2,2,4)
# plt.imshow(gt)
# plt.title('deflare')

# plt.subplots(3,5)
#
# plt.subplot(3,5,1)
# plt.imshow(img_arr)
# plt.title('origin')
#
# plt.subplot(3,5,2)
# plt.imshow(img_min)
# plt.title('img_min')
#
# plt.subplot(3,5,3)
# plt.imshow(img_max)
# plt.title('img_max')
#
# plt.subplot(3,5,4)
# plt.imshow(diff)
# plt.title('diff')
#
# plt.subplot(3,5,5)
# plt.imshow(img_aug_diff)
# plt.title('aug_diff')
#
# plt.subplot(3,5,6)
# plt.imshow(dehaze)
# plt.title('dehaze')
#
# plt.subplot(3,5,7)
# plt.imshow(img_min_dehaze)
# plt.title('img_min_haze')
#
# plt.subplot(3,5,8)
# plt.imshow(img_max_dehaze)
# plt.title('img_max_dehaze')
#
# plt.subplot(3,5,9)
# plt.imshow(diff_dehaze)
# plt.title('diff_dehaze')
#
# plt.subplot(3,5,10)
# plt.imshow(img_aug_diff_dehaze)
# plt.title('aug_diff_dehaze')
#
# plt.subplot(3,5,11)
# plt.imshow(deflare)
# plt.title('deflare')
#
# plt.subplot(3,5,12)
# plt.imshow(deflare_from_dehaze)
# plt.title('deflare_from_dehaze')

# plt.figure
# plt.subplots(2,2)
# plt.subplot(2,2,1)
# plt.imshow(img_arr)
# plt.title('origin')
#
# plt.subplot(2,2,2)
# plt.imshow(diff)
# plt.title('diff')
#
# plt.subplot(2,2,3)
# plt.imshow(deflare)
# plt.title('deflare')
#
# plt.subplot(2,2,4)
# plt.imshow(dehaze)
# plt.title('gt')

# plt.show()