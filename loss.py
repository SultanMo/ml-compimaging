import tensorflow as tf
import numpy as np
from scipy import signal
from math import log2, log10
from scipy.ndimage import generic_laplace,uniform_filter,correlate,gaussian_filter

import numpy as np
from scipy.ndimage.filters import uniform_filter,gaussian_filter
from scipy import signal
import warnings
from enum import Enum
from PIL import Image

def SSIM(x_train, x_train_pred):
    return tf.reduce_mean(tf.image.ssim(x_train, x_train_pred, 1.0))

def PSNR(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, 1.0)

def L1(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def L2(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def adversarial_loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true))

def loss_fn(y_train, y_pred, y_pred_adversarial):
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_train, y_pred) * 0.8 + (1 - tf.keras.losses.sparse_categorical_crossentropy(y_train, y_pred_adversarial)) * 0.2

    return loss

def CE(y_train, y_pred):
    return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(y_train, y_pred))

def BCE(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy(from_logits=False)

def MSE(x_true, x_pred):
    return tf.reduce_mean(tf.keras.losses.mean_squared_error(x_true, x_pred))

def MAE(x_true, x_pred):
    return tf.reduce_mean(tf.keras.losses.mean_absolute_error(x_true, x_pred))

def TV(x_psf):
    return tf.reduce_mean(tf.image.total_variation(x_psf))

# Implementation source: https://stackoverflow.com/questions/72195156/correct-implementation-of-dice-loss-in-tensorflow-keras
def dice_coef_loss(y_true, y_pred, smooth=10e-12):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return 1 - dice

def binary_iou(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou_score = tf.cast(intersection + 1e-15, tf.float32) / tf.cast(union + 1e-15, tf.float32)
    
    return iou_score



## Implementation source https://pypi.org/project/sewar/ ##
class Filter(Enum):
	UNIFORM = 0
	GAUSSIAN = 1

def _initial_check(GT,P):
	assert GT.shape == P.shape, "Supplied images have different sizes " + \
	str(GT.shape) + " and " + str(P.shape)
	if GT.dtype != P.dtype:
		msg = "Supplied images have different dtypes " + \
			str(GT.dtype) + " and " + str(P.dtype)
		warnings.warn(msg)
	

	if len(GT.shape) == 2:
		GT = GT[:,:,np.newaxis]
		P = P[:,:,np.newaxis]

	return GT.astype(np.float64),P.astype(np.float64)

def _replace_value(array,value,replace_with):
    array[array == value] = replace_with
    return array

def _get_sums(GT,P,win,mode='same'):
	mu1,mu2 = (filter2(GT,win,mode),filter2(P,win,mode))
	return mu1*mu1, mu2*mu2, mu1*mu2

def _get_sigmas(GT,P,win,mode='same',**kwargs):
	if 'sums' in kwargs:
		GT_sum_sq,P_sum_sq,GT_P_sum_mul = kwargs['sums']
	else:
		GT_sum_sq,P_sum_sq,GT_P_sum_mul = _get_sums(GT,P,win,mode)

	return filter2(GT*GT,win,mode)  - GT_sum_sq,\
			filter2(P*P,win,mode)  - P_sum_sq, \
			filter2(GT*P,win,mode) - GT_P_sum_mul

def fspecial(fltr,ws,**kwargs):
	if fltr == Filter.UNIFORM:
		return np.ones((ws,ws))/ ws**2
	elif fltr == Filter.GAUSSIAN:
		x, y = np.mgrid[-ws//2 + 1:ws//2 + 1, -ws//2 + 1:ws//2 + 1]
		g = np.exp(-((x**2 + y**2)/(2.0*kwargs['sigma']**2)))
		g[ g < np.finfo(g.dtype).eps*g.max() ] = 0
		assert g.shape == (ws,ws)
		den = g.sum()
		if den !=0:
			g/=den
		return g
	return None

def filter2(img,fltr,mode='same'):
	return signal.convolve2d(img, np.rot90(fltr,2), mode=mode)

def _str_to_array(str):
	pattern = r'''# Match (mandatory) whitespace between...
			(?<=\]) # ] and
			\s+
			(?= \[) # [, or
			|
			(?<=[^\[\]\s]) 
			\s+
			(?= [^\[\]\s]) # two non-bracket non-whitespace characters
			'''
	return np.array(ast.literal_eval(str))

def _power_complex(a,b):
	return a.astype('complex') ** b

def imresize(arr,size):
	return np.array(Image.fromarray(arr).resize(size))

def _compute_bef(im, block_size=8):
	"""Calculates Blocking Effect Factor (BEF) for a given grayscale/one channel image
	C. Yim and A. C. Bovik, "Quality Assessment of Deblocked Images," in IEEE Transactions on Image Processing,
		vol. 20, no. 1, pp. 88-98, Jan. 2011.
	:param im: input image (numpy ndarray)
	:param block_size: Size of the block over which DCT was performed during compression
	:return: float -- bef.
	"""
	if len(im.shape) == 3:
		height, width, channels = im.shape
	elif len(im.shape) == 2:
		height, width = im.shape
		channels = 1
	else:
		raise ValueError("Not a 1-channel/3-channel grayscale image")

	if channels > 1:
		raise ValueError("Not for color images")

	h = np.array(range(0, width - 1))
	h_b = np.array(range(block_size - 1, width - 1, block_size))
	h_bc = np.array(list(set(h).symmetric_difference(h_b)))

	v = np.array(range(0, height - 1))
	v_b = np.array(range(block_size - 1, height - 1, block_size))
	v_bc = np.array(list(set(v).symmetric_difference(v_b)))

	d_b = 0
	d_bc = 0

	# h_b for loop
	for i in list(h_b):
		diff = im[:, i] - im[:, i+1]
		d_b += np.sum(np.square(diff))

	# h_bc for loop
	for i in list(h_bc):
		diff = im[:, i] - im[:, i+1]
		d_bc += np.sum(np.square(diff))

	# v_b for loop
	for j in list(v_b):
		diff = im[j, :] - im[j+1, :]
		d_b += np.sum(np.square(diff))

	# V_bc for loop
	for j in list(v_bc):
		diff = im[j, :] - im[j+1, :]
		d_bc += np.sum(np.square(diff))

	# N code
	n_hb = height * (width/block_size) - 1
	n_hbc = (height * (width - 1)) - n_hb
	n_vb = width * (height/block_size) - 1
	n_vbc = (width * (height - 1)) - n_vb

	# D code
	d_b /= (n_hb + n_vb)
	d_bc /= (n_hbc + n_vbc)

	# Log
	if d_b > d_bc:
		t = np.log2(block_size)/np.log2(min(height, width))
	else:
		t = 0

	# BEF
	bef = t*(d_b - d_bc)

	return bef

def ssim (GT,P,ws=11,K1=0.01,K2=0.03,MAX=None,fltr_specs=None,mode='valid'):
	"""calculates structural similarity index (ssim).
	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param ws: sliding window size (default = 8).
	:param K1: First constant for SSIM (default = 0.01).
	:param K2: Second constant for SSIM (default = 0.03).
	:param MAX: Maximum value of datarange (if None, MAX is calculated using image dtype).
	:returns:  tuple -- ssim value, cs value.
	"""
	if MAX is None:
		MAX = np.iinfo(GT.dtype).max

	GT,P = _initial_check(GT,P)

	if fltr_specs is None:
		fltr_specs=dict(fltr=Filter.UNIFORM,ws=ws)

	C1 = (K1*MAX)**2
	C2 = (K2*MAX)**2

	ssims = []
	css = []
	for i in range(GT.shape[2]):
		ssim,cs = _ssim_single(GT[:,:,i],P[:,:,i],ws,C1,C2,fltr_specs,mode)
		ssims.append(ssim)
		css.append(cs)
	return np.mean(ssims),np.mean(css)

def msssim (GT,P,weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],ws=11,K1=0.01,K2=0.03,MAX=None):
	"""calculates multi-scale structural similarity index (ms-ssim).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param weights: weights for each scale (default = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).
	:param ws: sliding window size (default = 11).
	:param K1: First constant for SSIM (default = 0.01).
	:param K2: Second constant for SSIM (default = 0.03).
	:param MAX: Maximum value of datarange (if None, MAX is calculated using image dtype).

	:returns:  float -- ms-ssim value.
	"""
	if MAX is None:
		MAX = np.iinfo(GT.dtype).max

	GT,P = _initial_check(GT,P)

	scales = len(weights)

	fltr_specs = dict(fltr=Filter.GAUSSIAN,sigma=1.5,ws=11)

	if isinstance(weights, list):
		weights = np.array(weights)

	mssim = []
	mcs = []
	for _ in range(scales):
		_ssim, _cs = ssim(GT, P, ws=ws,K1=K1,K2=K2,MAX=MAX,fltr_specs=fltr_specs)
		mssim.append(_ssim)
		mcs.append(_cs)

		filtered = [uniform_filter(im, 2) for im in [GT, P]]
		GT, P = [x[::2, ::2, :] for x in filtered]

	mssim = np.array(mssim,dtype=np.float64)
	mcs = np.array(mcs,dtype=np.float64)

	return np.prod(_power_complex(mcs[:scales-1],weights[:scales-1])) * _power_complex(mssim[scales-1],weights[scales-1])
