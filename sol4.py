# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged


import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, map_coordinates
import sol4_utils
from scipy.ndimage.filters import convolve


PYR_LEVEL = 3
DESK_RAD_DEFAULT = 3


def non_maximum_suppression(image):
  """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
  # Find local maximas.
  neighborhood = generate_binary_structure(2,2)
  local_max = maximum_filter(image, footprint=neighborhood)==image
  local_max[image<(image.max()*0.1)] = False

  # Erode areas to single points.
  lbs, num = label(local_max)
  centers = center_of_mass(local_max, lbs, np.arange(num)+1)
  centers = np.stack(centers).round().astype(np.int)
  ret = np.zeros_like(image, dtype=np.bool)
  ret[centers[:,0], centers[:,1]] = True

  return ret


def harris_corner_detector(im):
  """
  Detects harris corners.
  Make sure the returned coordinates are x major!!!
  :param im: A 2D array representing an image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
  k = 0.04
  filter = np.reshape(np.asarray([1, 0, -1]), (1, 3))
  filter_transpose = np.reshape(filter, (3, 1))
  I_x = convolve(im, filter)
  I_y = convolve(im, filter_transpose)
  I_xx = np.multiply(I_x, I_x)
  I_yy = np.multiply(I_y, I_y)
  I_xy = np.multiply(I_x, I_y)
  blur_x = sol4_utils.blur_spatial(I_xx, 3)
  blur_y = sol4_utils.blur_spatial(I_yy, 3)
  blur_xy = sol4_utils.blur_spatial(I_xy, 3)

  det_M = blur_x * blur_y - np.square(blur_xy)
  trace = blur_x + blur_y
  R = det_M - k * np.square(trace)
  maxima_points = non_maximum_suppression(R)
  corner_points_arr = np.argwhere(maxima_points.T)
  return corner_points_arr


def sample_descriptor(im, pos, desc_rad):
  """
  Samples descriptors at the given corners.
  :param im: A 2D array representing an image.
  :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
  :param desc_rad: "Radius" of descriptors to compute.
  :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
  """
  K = 1 + 2 * desc_rad
  level_i, level_j = 1, PYR_LEVEL
  transform_points_coordinates = pos * 2 ** (level_i - level_j)
  descriptors_lst = []
  for point in transform_points_coordinates:
      x = np.arange(point[1] - desc_rad, point[1] + desc_rad + 1, 1)
      samp_desc = []
      for j in range(K):
          y = np.asarray(K * [point[0] + j - desc_rad])
          sampled = map_coordinates(im, [x, y], order=1, prefilter=False)
          samp_desc.append(sampled)
      samp_desc = np.asarray(samp_desc - np.mean(samp_desc))
      norm = np.linalg.norm(samp_desc)
      samp_desc = np.divide(samp_desc, norm, where=norm != 0)
      descriptors_lst.append(samp_desc)
  return np.asarray(descriptors_lst)


def find_features(pyr):
  """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
  im = pyr[0]
  feature_locations = spread_out_corners(im, PYR_LEVEL, PYR_LEVEL, DESK_RAD_DEFAULT -1 )
  return feature_locations, sample_descriptor(pyr[2], feature_locations, DESK_RAD_DEFAULT)



def match_features(desc1, desc2, min_score):
  """
  Return indices of matching descriptors.
  :param desc1: A feature descriptor array with shape (N1,K,K).
  :param desc2: A feature descriptor array with shape (N2,K,K).
  :param min_score: Minimal match score.
  :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
  """
  N1, N2 = desc1.shape[0], desc2.shape[0]
  desc1_matches, desc2_matches = [], []

  S = np.dot(np.reshape(desc1, (N1, desc1.shape[1] * desc1.shape[2])),
             np.reshape(desc2, (N2, desc2.shape[1] * desc2.shape[2])).T)
  sorted_rows = np.sort(S)
  sorted_cols = np.sort(S.T)
  two_max_rows = sorted_rows[:, -2:]
  two_max_cols = sorted_cols[:, -2:]
  for i in range(N1):
      for j in range(N2):
          if S[i][j] >= two_max_rows[i][0] and S[i][j] >= two_max_cols[j][0] and S[i][j] >= min_score:
              desc1_matches.append(i)
              desc2_matches.append(j)
  return [np.asarray(desc1_matches), np.asarray(desc2_matches)]



def apply_homography(pos1, H12):
  """
  Apply homography to inhomogenous points.
  :param pos1: An array with shape (N,2) of [x,y] point coordinates.
  :param H12: A 3x3 homography matrix.
  :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
  """
  N = pos1.shape[0]
  h_i = np.ones((N, 3))
  h_i[:, :2] = pos1
  first = np.dot(H12, h_i.T)
  sec = [x2_y2 for x2_y2 in first[:2] / first[2]]
  homography = np.asarray(sec).T
  return homography


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
  """
  Computes homography between two sets of points using RANSAC.
  :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
  :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
  :param num_iter: Number of RANSAC iterations to perform.
  :param inlier_tol: inlier tolerance threshold.
  :param translation_only: see estimate rigid transform
  :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
  """
  largest_inlier_set = 0
  inliers_set = []
  N = points1.shape[0]
  for _ in range(num_iter):
      j = np.random.randint(0, N, 1) if translation_only else np.random.randint(0, N, 2)
      p_1j, p_2j = points1[j], points2[j]
      H12 = estimate_rigid_transform(p_1j, p_2j, translation_only)
      P2_tag = apply_homography(points1, H12)
      E_j = np.square(np.linalg.norm(P2_tag - points2, axis=1))
      inliar_matches = np.where(E_j < inlier_tol)[0]
      if len(inliar_matches) > largest_inlier_set:
          largest_inlier_set = len(inliar_matches)
          inliers_set = inliar_matches
  return estimate_rigid_transform(np.asarray(points1[inliers_set]), np.asarray(points2[inliers_set]), translation_only), inliers_set


def display_matches(im1, im2, points1, points2, inliers):
  """
  Dispalay matching points.
  :param im1: A grayscale image.
  :param im2: A grayscale image.
  :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
  :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
  :param inliers: An array with shape (S,) of inlier matches.
  """
  N = im1.shape[1]
  im = np.hstack((im1, im2))
  x1, y1 = points1.T
  x2, y2 = points2.T
  for i in range(len(points2)):
      if i in inliers:
          plt.plot([x1[i], N + x2[i]], [y1[i], y2[i]], lw=1, marker=',', mfc='red', color='yellow')
      else:
          plt.plot([x1[i], N + x2[i]], [y1[i], y2[i]], lw=0.5, marker=',', mfc='red', color='blue')
  plt.imshow(im, 'gray')
  plt.show()


def accumulate_homographies(H_succesive, m):
  """
  Convert a list of succesive homographies to a
  list of homographies to a common reference frame.
  :param H_successive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
  """
  M = len(H_succesive) + 1
  H2m = M * [np.eye(3)]
  for i in range(m, -1, -1):
      H2m_i = np.dot(H_succesive[i - 1], H2m[i])
      H2m_i /= H2m_i[2][2]
      H2m[i - 1] = H2m_i
  for j in range(m + 1, M):
      H2m_j = np.dot(H2m[j - 1], np.linalg.inv(H_succesive[j - 1]))
      H2m_j /= H2m_j[2][2]
      H2m[j] = H2m_j
  return H2m


def compute_bounding_box(homography, w, h):
  """
  computes bounding box of warped image under homography, without actually warping the image
  :param homography: homography
  :param w: width of the image
  :param h: height of the image
  :return: 2x2 array, where the first row is [x,y] of the top left corner,
   and the second row is the [x,y] of the bottom right corner
  """
  corners_pixels = [[0, 0], [0, h], [w, 0], [w, h]]  # top-left, top-right, bottom-right, bottom-left coordinates
  corners = apply_homography(np.array(corners_pixels), homography)
  top_left_and_top_right = [np.min(corners[:, 0]), np.min(corners[:, 1])]
  bottom_left_and_bottom_rihjt = [np.max(corners[:, 0]), np.max(corners[:, 1])]
  bounding_box = np.array([top_left_and_top_right, bottom_left_and_bottom_rihjt]).astype(np.int)
  return bounding_box


def warp_channel(image, homography):
  """
  Warps a 2D image with a given homography.
  :param image: a 2D image.
  :param homography: homograhpy.
  :return: A 2d warped image.
  """
  bounding_box = compute_bounding_box(homography, image.shape[0], image.shape[1])
  x_cords, y_cords = np.meshgrid(np.arange(bounding_box[0, 0], bounding_box[1, 0]), np.arange(bounding_box[0, 1], bounding_box[1, 1]))
  normalized_inv_homography = np.linalg.inv(homography) / np.linalg.inv(homography)[2, 2]
  pos_array = np.asarray([x_cords.flatten(), y_cords.flatten()]).T
  back_coords = apply_homography(pos_array, normalized_inv_homography)
  cols_len = bounding_box[1][0] - bounding_box[0][0]
  rows_len = bounding_box[1][1] - bounding_box[0][1]
  coordinates = [back_coords[:, 0], back_coords[:, 1]]
  map_coords = map_coordinates(image, coordinates, order=1, prefilter=False).reshape((rows_len, cols_len)).T
  return map_coords


def warp_image(image, homography):
  """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
  return np.dstack([warp_channel(image[...,channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
  """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
  translation_over_thresh = [0]
  last = homographies[0][0,-1]
  for i in range(1, len(homographies)):
    if homographies[i][0,-1] - last > minimum_right_translation:
      translation_over_thresh.append(i)
      last = homographies[i][0,-1]
  return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
  """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
  centroid1 = points1.mean(axis=0)
  centroid2 = points2.mean(axis=0)

  if translation_only:
    rotation = np.eye(2)
    translation = centroid2 - centroid1

  else:
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2

    sigma = centered_points2.T @ centered_points1
    U, _, Vt = np.linalg.svd(sigma)

    rotation = U @ Vt
    translation = -rotation @ centroid1 + centroid2

  H = np.eye(3)
  H[:2,:2] = rotation
  H[:2, 2] = translation
  return H





def spread_out_corners(im, m, n, radius):
  """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
  corners = [np.empty((0,2), dtype=np.int)]
  x_bound = np.linspace(0, im.shape[1], n+1, dtype=np.int)
  y_bound = np.linspace(0, im.shape[0], m+1, dtype=np.int)
  for i in range(n):
    for j in range(m):
      # Use Harris detector on every sub image.
      sub_im = im[y_bound[j]:y_bound[j+1], x_bound[i]:x_bound[i+1]]
      sub_corners = harris_corner_detector(sub_im)
      sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis,:]
      corners.append(sub_corners)
  corners = np.vstack(corners)
  legit = ((corners[:,0]>radius) & (corners[:,0]<im.shape[1]-radius) &
           (corners[:,1]>radius) & (corners[:,1]<im.shape[0]-radius))
  ret = corners[legit,:]
  return ret



class PanoramicVideoGenerator:
  """
  Generates panorama from a set of images.
  """

  def __init__(self, data_dir, file_prefix, num_images):
    """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
    self.file_prefix = file_prefix
    self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
    self.files = list(filter(os.path.exists, self.files))
    self.panoramas = None
    self.homographies = None
    print('found %d images' % len(self.files))

  def align_images(self, translation_only=False):
    """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
    # Extract feature point locations and descriptors.
    points_and_descriptors = []
    for file in self.files:
      image = sol4_utils.read_image(file, 1)
      self.h, self.w = image.shape
      pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
      points_and_descriptors.append(find_features(pyramid))

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(len(points_and_descriptors) - 1):
      points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
      desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

      # Find matching feature points.
      ind1, ind2 = match_features(desc1, desc2, .7)
      points1, points2 = points1[ind1, :], points2[ind2, :]

      # Compute homography using RANSAC.
      H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

      # Uncomment for debugging: display inliers and outliers among matching points.
      # In the submitted code this function should be commented out!
      # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

      Hs.append(H12)

    # Compute composite homographies from the central coordinate system.
    accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
    self.homographies = np.stack(accumulated_homographies)
    self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
    self.homographies = self.homographies[self.frames_for_panoramas]


  def generate_panoramic_images(self, number_of_panoramas):
    """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
    assert self.homographies is not None

    # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
    self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
    for i in range(self.frames_for_panoramas.size):
      self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

    # change our reference coordinate system to the panoramas
    # all panoramas share the same coordinate system
    global_offset = np.min(self.bounding_boxes, axis=(0, 1))
    self.bounding_boxes -= global_offset

    slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
    warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
    # every slice is a different panorama, it indicates the slices of the input images from which the panorama
    # will be concatenated
    for i in range(slice_centers.size):
      slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
      # homography warps the slice center to the coordinate system of the middle image
      warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
      # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
      warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

    panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

    # boundary between input images in the panorama
    x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
    x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                  x_strip_boundary,
                                  np.ones((number_of_panoramas, 1)) * panorama_size[0]])
    x_strip_boundary = x_strip_boundary.round().astype(np.int)

    self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
    for i, frame_index in enumerate(self.frames_for_panoramas):
      # warp every input image once, and populate all panoramas
      image = sol4_utils.read_image(self.files[frame_index], 2)
      warped_image = warp_image(image, self.homographies[i])
      x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
      y_bottom = y_offset + warped_image.shape[0]

      for panorama_index in range(number_of_panoramas):
        # take strip of warped image and paste to current panorama
        boundaries = x_strip_boundary[panorama_index, i:i + 2]
        image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
        x_end = boundaries[0] + image_strip.shape[1]
        self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

    # crop out areas not recorded from enough angles
    # assert will fail if there is overlap in field of view between the left most image and the right most image
    crop_left = int(self.bounding_boxes[0][1, 0])
    crop_right = int(self.bounding_boxes[-1][0, 0])
    assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
    print(crop_left, crop_right)
    self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]



  def save_panoramas_to_video(self):
    assert self.panoramas is not None
    out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
    try:
      shutil.rmtree(out_folder)
    except:
      print('could not remove folder')
      pass
    os.makedirs(out_folder)
    # save individual panorama images to 'tmp_folder_for_panoramic_frames'
    for i, panorama in enumerate(self.panoramas):
      plt.imsave('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
    if os.path.exists('%s.mp4' % self.file_prefix):
      os.remove('%s.mp4' % self.file_prefix)
    # write output video to current folder
    os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
              (out_folder, self.file_prefix))


  def show_panorama(self, panorama_index, figsize=(20, 20)):
    assert self.panoramas is not None
    plt.figure(figsize=figsize)
    plt.imshow(self.panoramas[panorama_index].clip(0, 1))
    plt.show()
