import tensorflow as tf
import numpy as np


def apply_blur(img):
    return tf.image.gaussian_filter2d(img, filter_shape=(3, 3), sigma=1.0)
    
def downsample(img, scale):
    return tf.image.resize(img, (img.shape[0]//scale, img.shape[1]//scale), method='bilinear')

def upsample(img, scale):
    return tf.image.resize(img, (img.shape[0]*scale, img.shape[1]*scale), method='bilinear')

def apply_noise(img, sigma):
    return img + tf.random.normal(shape=img.shape, mean=0.0, stddev=sigma)

