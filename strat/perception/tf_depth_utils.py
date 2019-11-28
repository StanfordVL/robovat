"""Tensorflow utilities for depth images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _random_bernoulli(shape, prob):
    """Random Bernoulli variable.

    Args:
        shape: The shape of samples.
        prob: The probability of the Bernoulli distribution.

    Returns:
        A boolean tensor of the specified shape.
    """
    return tf.less(tf.random_uniform(shape=shape, minval=0.0, maxval=1.0), prob)


def random_flip(tensor, prob=0.5):
    """Apply random flipping to the images.

    Args:
        tensor: A 4-dimensional tensor.

    Returns:
        The corrupted tensor with the applied noise.
    """
    num_images = tf.shape(tensor)[0]
    shape = [num_images]
    use_flip = _random_bernoulli(shape, 0.5)
    flip_up_down = _random_bernoulli(shape, 0.5)
    flip_left_right = _random_bernoulli(shape, 0.5)

    flipped_tensor = tf.where(
            flip_up_down,
            tf.reverse(tensor, [1]),
            tensor)
    flipped_tensor = tf.where(
            flip_left_right,
            tf.reverse(tensor, [2]),
            flipped_tensor)
    new_tensor = tf.where(
            use_flip,
            flipped_tensor,
            tensor)

    return new_tensor


def gamma_noise(tensor, gamma_shape=1000):
    """Apply multiplicative denoising to the images.

    Args:
        tensor: A 4-dimensional tensor.

    Returns:
        The corrupted tensor with the applied noise.
    """
    num_images = tf.shape(tensor)[0]
    shape = [num_images, 1, 1, 1]
    noise = tf.random_gamma(shape=shape, alpha=gamma_shape, beta=gamma_shape)
    new_tensor = tensor * noise
    return new_tensor


def gaussian_noise(tensor,
                   prob=0.5,
                   rescale_factor=4.0,
                   sigma=0.02):
    """Apply correlated Gaussian noise.

    Args:
        tensor: A 4-dimensional tensor.

    Returns:
        The corrupted tensor with the applied noise.
    """
    shape = tf.shape(tensor)
    sample_height = tf.cast(
            tf.divide(tf.cast(shape[1], tf.float32), rescale_factor), tf.int32)
    sample_width = tf.cast(
            tf.divide(tf.cast(shape[2], tf.float32), rescale_factor), tf.int32)
    noise_shape = [shape[0], sample_height, sample_width, shape[3]]
    noise = tf.random_normal(shape=noise_shape, stddev=sigma)
    noise = tf.image.resize_images(noise, [shape[1], shape[2]],
                                   method=tf.image.ResizeMethod.BICUBIC)

    use_noise = _random_bernoulli([shape[0], 1, 1, 1], 0.5)
    use_noise = tf.tile(use_noise, [1, shape[1], shape[2], shape[3]])
    pixel_greater_then_zero = tf.greater(tensor, 0.0)

    new_tensor = tf.where(
        tf.logical_and(use_noise, pixel_greater_then_zero),
        tensor + noise,
        tensor)

    return new_tensor
