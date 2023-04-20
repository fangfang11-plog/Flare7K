import torch
import torch.nn as nn

class GuidedFilter(nn.Module):
    """
    Implements a guided filter for smoothing images.

    Args:
        radius (int): The radius of the filter.
        eps (float): The regularization parameter of the filter.
    """

    def __init__(self, radius, eps):
        super(GuidedFilter, self).__init__()

        # Define the convolutional layers used for filtering
        kernel_size = radius * 2 + 1
        self.mean_filter = nn.Conv2d(3, 3, kernel_size, padding=radius)
        self.mean_filter.weight.data.fill_(1.0 / kernel_size**2)
        self.mean_filter.bias.data.fill_(0.0)

        self.correlation_filter = nn.Conv2d(3, 9, kernel_size, padding=radius)
        self.correlation_filter.weight.data.fill_(0.0)
        for i in range(3):
            self.correlation_filter.weight.data[i, i, radius, radius] = 1.0
            self.correlation_filter.weight.data[3+i, i, radius, radius] = 1.0
            self.correlation_filter.weight.data[6+i, i, radius, radius] = 1.0
        self.correlation_filter.bias.data.fill_(0.0)

        self.eps = eps

    def forward(self, input, guide):
        """
        Applies the guided filter to a given input image using a guide image.

        Args:
            input (torch.Tensor): A 3D tensor with shape (channels, height, width).
            guide (torch.Tensor): A 3D tensor with the same shape as the input tensor.

        Returns:
            A 3D tensor with the same shape as the input tensor.
        """

        # Compute the mean and covariance of the guide image
        mean_guide = self.mean_filter(guide)
        mean_input = self.mean_filter(input)
        cov_guide_input = self.correlation_filter(guide * input) - mean_guide * mean_input

        # Compute the mean and variance of the guide image
        mean_guide_sq = self.mean_filter(guide * guide)
        var_guide = mean_guide_sq - mean_guide**2

        # Compute the local covariance of the guide and input images
        a = cov_guide_input / (var_guide + self.eps)

        # Compute the mean of the output image
        b = mean_input - a * mean_guide

        # Apply the filter to the input image
        mean_a = self.mean_filter(a)
        mean_b = self.mean_filter(b)
        output = mean_a * guide + mean_b

        return output

def dark_channel(image, window_size=15):
    """
    Computes the dark channel prior for a given image.

    Args:
        image (torch.Tensor): A 3D tensor with shape (channels, height, width).
        window_size (int): The size of the sliding window used to compute the minimum value.

    Returns:
        A 2D tensor with the same height and width as the input image.
    """

    # Add a batch dimension to the input tensor
    image = image.unsqueeze(0)

    # Compute the minimum value in each sliding window
    padded_image = nn.functional.pad(image, (window_size // 2, window_size // 2, window_size // 2, window_size // 2), mode='reflect')
    windows = nn.functional.unfold(padded_image, kernel_size=window_size, stride=1)
    min_values, _ = torch.min(windows, dim=1)

    # Remove the batch dimension from the output tensor
    min_values = min_values.squeeze(0)

    return min_values

def estimate_transmission(image, atmosphere, omega=0.95, window_size=15):
    """
    Estimates the transmission map for a given image using the dark channel prior.

    Args:
        image (torch.Tensor): A 3D tensor with shape (channels, height, width).
        atmosphere (torch.Tensor): A 1D tensor with length equal to the number of channels in the input image.
        omega (float): The weight given to the transmission estimate.
        window_size (int): The size of the sliding window used to compute the dark channel.

    Returns:
        A 2D tensor with the same height and width as the input image.
    """

    # Compute the normalized dark channel prior
    min_values = dark_channel(image, window_size=window_size)
    normalized_min = min_values / atmosphere.view(-1, 1, 1)

    # Estimate the transmission map
    transmission_estimate = 1 - omega * normalized_min

    return transmission_estimate

def dehaze(images, atmospheres, tmin=0.1, guided_filter_radius=16, guided_filter_eps=1e-6):
    """
    Performs single-image dehazing using the dark channel prior and guided filtering.

    Args:
        images (torch.Tensor): A 4D tensor with shape (batch_size, channels, height, width).
        atmospheres (torch.Tensor): A 2D tensor with shape (batch_size, channels) containing the atmosphere light for each image in the batch.
        tmin (float): The minimum allowed value for the transmission map.
        guided_filter_radius (int): The radius of the guided filter.
        guided_filter_eps (float): The regularization parameter of the guided filter.

    Returns:
        A 4D tensor with the same shape as the input tensor.
    """

    # Convert the input images and atmospheres to float tensors
    images = images.float()
    atmospheres = atmospheres.float().view(-1, 1, 1, images.size(3))

    # Estimate the transmission maps
    transmission_estimates = []
    for i in range(images.size(0)):
        transmission_estimate = estimate_transmission(images[i], atmospheres[i])
        transmission_estimate = torch.clamp(transmission_estimate, min=tmin, max=1)
        transmission_estimates.append(transmission_estimate)
    transmission_maps = torch.stack(transmission_estimates, dim=0)

    # Perform guided filtering on the estimated transmission maps
    guided_filter = GuidedFilter(guided_filter_radius, guided_filter_eps)
    transmission_maps = guided_filter(transmission_maps, images)

    # Dehaze the input images using the estimated transmission maps and atmospheres
    dehazed_images = (images - atmospheres.view(-1, images.size(1), 1, 1)) / transmission_maps + atmospheres.view(-1, images.size(1), 1, 1)

    # Clamp the output tensor to the range [0, 1]
    dehazed_images = torch.clamp(dehazed_images, min=0, max=1)

    return dehazed_images
