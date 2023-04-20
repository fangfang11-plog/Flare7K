import torch
import torch.nn.functional as F
import cv2


class Dehaze:
    def __init__(self, r=5, eps=0.001, t0=0.1, w=0.95):
        self.r = r
        self.eps = eps
        self.t0 = t0
        self.w = w

    def zm_min_filter_gray(self, src):
        kernel_size = 2 * self.r + 1
        return F.max_pool2d(src, kernel_size, stride=1, padding=self.r)

    def dark_channel(self, image):
        height, width, _ = image.shape
        min_rgb = torch.zeros_like(image)
        for i in range(height):
            for j in range(width):
                min_value, _ = torch.min(image[i, j], dim=0)
                min_rgb[i, j] = min_value
        return min_rgb

    def light_channel(self, image):
        height, width, _ = image.shape
        max_rgb = torch.zeros_like(image)
        for i in range(height):
            for j in range(width):
                max_value, _ = torch.max(image[i, j], dim=0)
                max_rgb[i, j] = max_value
        return max_rgb

    def min_filter(self, image):
        height, width, channels = image.shape
        dst = torch.zeros_like(image)
        for i in range(height):
            for j in range(width):
                if i < self.r or j < self.r or i >= height - self.r or j >= width - self.r:
                    dst[i, j] = image[i, j]
                else:
                    patch = image[i - self.r:i + self.r + 1, j - self.r:j + self.r + 1]
                    min_value, _ = torch.min(patch, dim=(0, 1))
                    dst[i, j] = min_value
        return dst

    def guided_filter(self, image, p):
        mean_I = F.avg_pool2d(image, self.r, stride=1, padding=self.r // 2)
        mean_p = F.avg_pool2d(p, self.r, stride=1, padding=self.r // 2)
        mean_II = F.avg_pool2d(image * image, self.r, stride=1, padding=self.r // 2)
        mean_Ip = F.avg_pool2d(image * p, self.r, stride=1, padding=self.r // 2)
        var_I = mean_II - mean_I * mean_I
        cov_Ip = mean_Ip - mean_I * mean_p
        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        mean_a = F.avg_pool2d(a, self.r, stride=1, padding=self.r // 2)
        mean_b = F.avg_pool2d(b, self.r, stride=1, padding=self.r // 2)
        q = mean_a * image + mean_b
        return q

    def select_bright(self, image, img_origin, V):
        batch_size, height, width, _ = image.shape
        size = batch_size * height * width
        order = torch.zeros(size)
        m = 0
        for n in range(batch_size):
            for i in range(height):
                for j in range(width):
                    order[m] = image[n, i, j, 0]
                    m += 1
        _, indices = torch.sort(order, descending=True)
        index = int(size * 0.001)
        mid = order[indices[index]]
        A = torch.zeros((batch_size, 3))
        img_hsv = cv2.cvtColor(img_origin.numpy(), cv2.COLOR_RGB2HLS)
        for n in range(batch_size):
            for c in range(3):
                mask = (image[n, :, :, c] > mid).type(torch.float32)
                A[n, c] = torch.max(img_hsv[:, :, 1] * mask)
        V = V * self.w
        t = 1 - V / A
        t = torch.max(t, self.t0 * torch.ones_like(t))
        return t, A

    def repair(self, image, t, A):
        batch_size, height, width, channels = image.shape
        J = torch.zeros_like(image)
        for n in range(batch_size):
            for c in range(channels):
                t[n, :, :, c] = t[n, :, :, c] - 0.25
                J[n, :, :, c] = (image[n, :, :, c] - A[n, c] / t[n, :, :, c] + A[n, c])
        return J

    def get_flare(self, image, threshold):
        mask = (image > threshold).type(torch.float32)
        return image * mask

    def del_flare(self, image, threshold):
        flare = self.get_flare(image, threshold)
        _, flare_gray = cv2.threshold(flare.numpy(), 0, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        flare_mask = cv2.morphologyEx(flare_gray.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        flare_mask = cv2.erode(flare_mask, kernel, iterations=2)
        mask = torch.from_numpy(flare_mask.astype(np.float32) / 255.0).unsqueeze(2)
        J = image * (1 - mask) + mask * self.guided_filter(image, mask)
        return J

    def process_batch(self, images):
        batch_size, height, width, channels = images.shape
        images = images.permute(0, 3, 1, 2)
        images = images.type(torch.float32) / 255.0
        dark = self.dark_channel(images)
        V = self.min_filter(dark)
        t, A = self.select_bright(images, images.permute(0, 2, 3, 1), V)
        J = self.repair(images, t, A)
        J = J.permute(0, 2, 3, 1)
        J = torch.clamp(J * 255.0, 0, 255).type(torch.uint8)
        J = self.del_flare(J, 240)
        return J
