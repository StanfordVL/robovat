"""Utilities for depth images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
import scipy.stats

from robovat.perception import image_utils


transform = image_utils.transform
crop = image_utils.crop


def inpaint(data, rescale_factor=0.5):
    """Fills in the zero pixels in the depth image.

    Parameters:
        data: The raw depth image.
        rescale_factor: Amount to rescale the image for inpainting, smaller
            numbers increase speed.

    Returns:
        new_data: The inpainted depth imaga.
    """
    # Form inpaint kernel.
    inpaint_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    # Resize the image.
    resized_data = scipy.misc.imresize(data, rescale_factor,
                                       interp='nearest', mode='F')

    # Inpaint the smaller image.
    cur_data = resized_data.copy()
    zeros = (cur_data == 0)

    while np.any(zeros):
        neighbors = scipy.signal.convolve2d(
            (cur_data != 0), inpaint_kernel, mode='same', boundary='symm')
        avg_depth = scipy.signal.convolve2d(
            cur_data, inpaint_kernel, mode='same', boundary='symm')
        avg_depth[neighbors > 0] = (avg_depth[neighbors > 0] /
                                    neighbors[neighbors > 0])
        avg_depth[neighbors == 0] = 0
        avg_depth[resized_data > 0] = resized_data[resized_data > 0]
        cur_data = avg_depth

        zeros = (cur_data == 0)

    inpainted_data = cur_data

    # Fill in zero pixels with inpainted and resized image.
    filled_data = scipy.misc.imresize(inpainted_data, 1.0 / rescale_factor,
                                      interp='bilinear')

    new_data = np.copy(data)
    new_data[data == 0] = filled_data[data == 0]

    return new_data


def threshold_gradients(data, threshold):
    """Get the threshold gradients.

    Creates a new DepthImage by zeroing out all depths
    where the magnitude of the gradient at that point is
    greater than threshold.

    Args:
        data: The raw depth image.
        threhold: A threshold for the gradient magnitude.

    Returns:
        A new DepthImage created from the thresholding operation.
    """
    data = np.copy(data)
    gx, gy = np.gradient(data.astype(np.float32))
    gradients = np.zeros([gx.shape[0], gx.shape[1], 2])
    gradients[:, :, 0] = gx
    gradients[:, :, 1] = gy
    gradient_magnitudes = np.linalg.norm(gradients, axis=2)
    ind = np.where(gradient_magnitudes > threshold)
    data[ind[0], ind[1]] = 0.0
    return data


def gamma_noise(data, gamma_shape=1000):
    """Apply multiplicative denoising to the images.

    Args:
        data: A numpy array of 3 or 4 dimensions.

    Returns:
        The corrupted data with the applied noise.
    """
    if data.ndim == 3:
        images = data[np.newaxis, :, :, :]
    else:
        images = data

    num_images = images.shape[0]
    gamma_scale = 1.0 / gamma_shape

    mult_samples = scipy.stats.gamma.rvs(gamma_shape, scale=gamma_scale,
                                         size=num_images)
    mult_samples = mult_samples[:, np.newaxis, np.newaxis, np.newaxis]
    new_images = data * mult_samples

    if data.ndim == 3:
        return new_images[0]
    else:
        return new_images


def gaussian_noise(data,
                   prob=0.5,
                   rescale_factor=4.0,
                   sigma=0.005):
    """Add correlated Gaussian noise.

    Args:
        data: A numpy array of 3 or 4 dimensions.

    Returns:
        The corrupted data with the applied noise.
    """
    if data.ndim == 3:
        images = data[np.newaxis, :, :, :]
    else:
        images = data

    num_images = images.shape[0]
    image_height = images.shape[1]
    image_width = images.shape[2]
    sample_height = int(image_height / rescale_factor)
    sample_width = int(image_width / rescale_factor)
    num_pixels = sample_height * sample_width

    new_images = []

    for i in range(num_images):
        image = images[i, :, :, 0]

        if np.random.rand() < prob:
            gp_noise = scipy.stats.norm.rvs(scale=sigma, size=num_pixels)
            gp_noise = gp_noise.reshape(sample_height, sample_width)
            gp_noise = scipy.misc.imresize(gp_noise, rescale_factor,
                                           interp='bicubic', mode='F')
            image[image > 0] += gp_noise[image > 0]

        new_images.append(image[:, :, np.newaxis])

    new_images = np.stack(new_images)

    if data.ndim == 3:
        return new_images[0]
    else:
        return new_images
