from scipy import ndimage
from scipy.misc import imread
from scipy import ndimage, signal
from skimage.color import rgb2gray
import numpy as np

GREY_LEVEL_MAX_VAL = 256
LOWEST_SIZE = 16

def expand_single_pic(image, filter_vec, x_flag, y_flag):
    x_size, y_size = image.shape
    new_img = np.zeros((2 * x_size - x_flag, 2 * y_size - y_flag))
    for row in range(x_size):
        for pixel in range(y_size):
            new_img[2 * row][2 * pixel] = image[row][pixel]
    conv_vec = ndimage.filters.convolve(new_img, filter_vec)
    cur_im = ndimage.filters.convolve(conv_vec, filter_vec.T)
    x_flag = x_size % 2
    y_flag = y_size % 2
    return cur_im, x_flag, y_flag

def expand(reduced_im, filter_vec):
    expanded_im = [reduced_im[0]]
    filter_vec *= 2
    x_flag = reduced_im[0].shape[0]% 2
    y_flag = reduced_im[0].shape[1]% 2
    for image in reduced_im[1:]:
        cur_im, x_flag, y_flag = expand_single_pic(image, filter_vec, x_flag, y_flag)
        expanded_im.append(cur_im)
    return expanded_im


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    construct a Laplacian pyramid of a given image.
    :param im: a grayscale image with double values in [0, 1] (e.g. the output of ex1’s
    read_image with the representation set to 1).
    :param max_levels: the maximal number of levels1 in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that
    represents a squared filter) to be used in constructing the pyramid filter
    :return: pyr, filter_vec
    """
    filter_vec = create_filter(filter_size)
    reduced_im = reduce(im, max_levels, filter_vec)
    expand_im = expand(reduced_im, filter_vec)
    pyr = []
    for image in range(len(expand_im) - 1):
        lap_img = reduced_im[image] - expand_im[image + 1]
        pyr.append(lap_img)
    pyr.append(reduced_im[len(reduced_im) - 1])
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    the reconstruction of an image from its Laplacian Pyramid.
    """
    expanded_pic = np.dot(lpyr[len(lpyr) - 1], coeff[len(lpyr) - 1])
    for i in range(len(lpyr) - 2, -1, -1):
        mult_lp = np.multiply(lpyr[i], coeff[i])
        cur_img = expand_single_pic(expanded_pic, filter_vec, mult_lp.shape[0] % 2, mult_lp.shape[1] % 2)[0]
        expanded_pic = cur_img + mult_lp
    return expanded_pic


def create_filter(filter_size):
  """
  :param filter_size
  :return: the filter.
  """
  filter = [1 / 2, 1 / 2]
  conv_with = [1 / 2, 1 / 2]
  for _ in range(filter_size - 2):
    filter = signal.convolve(filter, conv_with)
  return filter.reshape(1, len(filter))


"""
Function that derives a row of binomial coefficient.
"""
def gaussian_maker(kernel_size):
    gaussian = [1 / 2, 1 / 2]
    consequent = [1 / 2, 1 / 2]
    for i in range(kernel_size - 2):
        gaussian = signal.convolve(gaussian, consequent)
    gaussian_2D = signal.convolve(np.asarray(gaussian).reshape(1, kernel_size),
                                  np.asarray(gaussian).reshape(kernel_size, 1))
    return gaussian_2D


def reduce(im, max_levels, filter_vec):
  """
  finds the pyr.
  :return: pyr
  """
  pyr = [im]
  cur_im = im
  for _ in range(max_levels - 1):
    conv_vec = ndimage.filters.convolve(filter_vec, filter_vec.T)
    cur_im = ndimage.filters.convolve(cur_im, conv_vec)
    sampled_pic = cur_im[0: cur_im.shape[0]: 2]
    sampled_pic = sampled_pic.T[0: cur_im.shape[1]: 2]
    x_size, y_size = sampled_pic.shape
    if x_size < LOWEST_SIZE or y_size < LOWEST_SIZE:
      break
    cur_im = sampled_pic.T
    pyr.append(cur_im)
  return pyr


"""
Function that performs image blurring using 2D convolution between the image f and a gaussian
kernel g.
im - is the input image to be blurred (grayscale float64 image).
kernel_Size - is the size of the gaussian kernel in each dimension (an odd integer).
The function returns the output blurry image (grayscale float64 image).
"""
def blur_spatial(im, kernel_size):
    image_blurred = im
    if kernel_size > 1:
        g = gaussian_maker(kernel_size)
        image_blurred = signal.convolve2d(im, g)
    return image_blurred


def read_image(filename, representation):
  """
  function which reads an image file and converts it into a given representation.
  This function returns an image, normalized to the range [0, 1].
  """
  im = imread(filename).astype(np.float64) / (GREY_LEVEL_MAX_VAL - 1)
  if (representation == 1):
    im_g = rgb2gray(im)
    return im_g
  return im

def build_gaussian_pyramid(im, max_levels, filter_size):
  """
  construct a Gaussian pyramid pyramid of a given image.
  :param im: a grayscale image with double values in [0, 1]
  (e.g. the output of ex1’s read_image with the representation set to 1).
  :param max_levels: the maximal number of levels1 in the resulting pyramid
  :param filter_size: the size of the Gaussian filter (an odd scalar that
  represents a squared filter) to be used in constructing the pyramid filter.
  :return: pyr, filter_vec
  """
  filter_vec = create_filter(filter_size)
  pyr = reduce(im, max_levels, filter_vec)
  return pyr, filter_vec

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
  laplacian_im1, filter = build_laplacian_pyramid(im1, max_levels, filter_size_im)
  laplacian_im2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
  gaussian_m = build_gaussian_pyramid(mask.astype(np.double), max_levels, filter_size_mask)[0]
  mult_pics = np.multiply(gaussian_m, laplacian_im1) + np.multiply((1 - np.array(gaussian_m)), laplacian_im2)
  coeff = np.ones(len(mult_pics))
  sum_levels = laplacian_to_image(mult_pics, filter, coeff)
  im_blend = np.clip(sum_levels, 0, 1)
  return im_blend

