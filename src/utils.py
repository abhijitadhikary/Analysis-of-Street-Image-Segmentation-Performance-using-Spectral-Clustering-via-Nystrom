import numpy as np
from time import time
from tqdm import tqdm
from scipy.linalg import sqrtm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from sklearn.cluster import KMeans
import os
import argparse

def get_args():
    args = argparse.Namespace()
    args.num_clusters = 3
    args.num_eigen_vectors = 3
    args.sigma_color = 0.5
    args.sigma_distance = 5
    args.height = 100
    args.width = 100
    args.num_channels = 0
    args.num_dimensions = 0
    args.num_elements_flat = 0
    args.use_numpy_eigen_decompose = True
    args.dim_low = 100
    return args

def convert(source, min_value=0, max_value=1):
    source = np.array(source).astype(np.float64)
    smin = source.min()
    smax = source.max()
    a = (max_value - min_value) / (smax - smin)
    b = max_value - a * smax
    target = (a * source + b)
    return target

def get_file_names(root=os.path.join('..', 'data')):
    filenames = []
    for root, _, files in os.walk(root):
        for f in files:
            filename = os.path.join(root, f)
            filenames.append(filename)
    return filenames

def imshow(image, title=''):
    convert(image, 0, 255)
    plt.figure(figsize=(5, 5))
    plt.title(title)

    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


def get_image_array(image, args):
    image_array = np.zeros((args.num_elements_flat, (args.num_channels + 2)))

    image_array_index = 0
    for index_row in range(args.height):
        for index_col in range(args.width):
            current_pixel = image[index_row, index_col]
            if args.num_channels == 3:
                image_array[image_array_index] = np.array([current_pixel[0],
                                                           current_pixel[1],
                                                           current_pixel[2],
                                                           index_row,
                                                           index_col])
            elif args.num_channels == 1:
                image_array[image_array_index] = np.array([current_pixel,
                                                           index_row,
                                                           index_col])
            image_array_index += 1

    return image_array

def get_exponential_bump(distance, sigma=1):
    exponential_bump = np.exp(-np.abs(distance) / sigma ** 2)
    return exponential_bump

def get_eucledian_distance_vectorized(point_1, point_2_array):
    euclidean_distance = np.sqrt(np.sum(np.power((point_1 - point_2_array), 2), axis=1))
    return euclidean_distance

def get_color_weight_vectorized(point_1, point_2_array, sigma_color):
    point_1 = point_1.reshape(-1, point_2_array.shape[1])
    point_2_array = point_2_array.reshape(-1, point_2_array.shape[1])
    difference_color = get_eucledian_distance_vectorized(point_1, point_2_array)
    color_weight = get_exponential_bump(difference_color, sigma_color)
    return color_weight

def get_distance_weight_vectorized(point_1, point_2_array, sigma_distance):
    point_1 = point_1.reshape(-1, point_2_array.shape[1])
    point_2_array = point_2_array.reshape(-1, point_2_array.shape[1])
    distance = get_eucledian_distance_vectorized(point_1, point_2_array)
    distance_weight = get_exponential_bump(distance, sigma_distance)
    return distance_weight


def get_k_eig_vectors_nystrom(image_array, args):
    dim_low = args.dim_low

    indices_low = np.random.choice(args.num_elements_flat, size=dim_low, replace=False)
    image_low = image_array[indices_low]
    distances_colour = np.linalg.norm(np.expand_dims(image_array[:,:args.num_channels], axis=1) - image_low[:,:args.num_channels], axis=-1, ord=2)
    distances_position = np.linalg.norm(np.expand_dims(image_array[:,args.num_channels:], axis=1) - image_low[:,args.num_channels:], axis=-1, ord=2)
    weight_matrix = (get_exponential_bump(distances_colour,args.sigma_color)*get_exponential_bump(distances_position,args.sigma_distance)).T
    row = [i for i in range(dim_low)]
    weight_matrix[row, row] = 1

    A = weight_matrix[:, list(indices_low)]  # nxn
    B = np.delete(weight_matrix, list(indices_low), axis=1)  # nxm
    n, m = B.shape
    #     print(A.shape)
    #     print(B.shape)
    A_pinv = np.linalg.pinv(A)
    d1 = np.sum(np.vstack((A, B.T)), axis=0).reshape(1, -1)
    d2 = np.sum(B, axis=0) + (np.sum(B.T, axis=0).reshape(1, -1) @ (A_pinv @ B))
    dhat = np.sqrt(1 / np.hstack((d1, d2))).reshape(-1, 1)
    A = A * (dhat[:n].reshape(-1, 1) @ dhat[:n].reshape(-1, 1).T)
    B = B * (dhat[:n].reshape(-1, 1) @ dhat[n:].reshape(-1, 1).T)

    pinv_A = np.linalg.pinv(A)

    Asi = sqrtm(pinv_A)

    Q = A + Asi @ B @ B.T @ Asi
    U, L, T = np.linalg.svd(Q)

    L = np.diag(L)

    V = np.hstack((A, B)).T @ Asi @ U @ np.linalg.pinv(np.sqrt(L))

    V = V[:, 1:args.num_eigen_vectors + 1]
    # reordering V appropriately
    all_idx = list(np.arange(args.num_elements_flat))
    rem_idx = [idx for idx in all_idx if idx not in indices_low]
    top_matrix = np.zeros((args.num_elements_flat, args.num_eigen_vectors))
    top_matrix[list(indices_low), :] = V[:dim_low, :]
    top_matrix[rem_idx, :] = V[dim_low:, :]
    return top_matrix

def get_segmented_image(image, clustered_image, clustered_labels, args, use_median=True):
    '''
        abhi's color
    '''

    if args.num_channels == 3:
        label_values = np.unique(clustered_labels)
        segmented_image = np.zeros_like(image)
        if use_median:
            factor = 255 / (np.max(image) - np.min(image))
        else:
            factor = 1

        for index in label_values:
            current_mask = (clustered_image == index).astype(np.float64)
            current_segment = image * np.repeat(current_mask[..., None], args.num_channels, axis=2) * factor

            for channel_index in range(args.num_channels):
                current_channel = current_segment[:, :, channel_index]
                cluster_total = np.count_nonzero(current_channel)

                if use_median:
                    ################################
                    non_zero_current_channel = np.sort(current_channel[current_channel != 0]) # Sort values to find median
                    cluster_median = non_zero_current_channel[cluster_total // 2]  # Median of non-0 elements
                    current_segment[:, :, channel_index] = np.where(current_channel > 0, cluster_median, current_channel)
                    ################################
                else:
                    cluster_sum = np.sum(current_channel)
                    cluster_mean = cluster_sum / cluster_total
                    current_segment[:, :, channel_index] = np.where(current_segment[:, :, channel_index] > 0,
                                                                    cluster_mean,
                                                                    current_segment[:, :, channel_index])

                segmented_image[:, :, channel_index] += current_segment[:, :, channel_index].astype(np.float64)

    elif args.num_channels == 1:
        label_values = np.unique(clustered_labels)
        segmented_image = np.zeros_like(image)
        if use_median:
            factor = 255 / (np.max(image) - np.min(image))
        else:
            factor = 1

        for index in label_values:
            current_mask = (clustered_image == index).astype(np.float64)
            current_segment = image * current_mask * factor

            current_channel = current_segment
            cluster_total = np.count_nonzero(current_channel)
            if use_median:
                non_zero_current_segment = current_segment[current_segment != 0]
                cluster_center = non_zero_current_segment[cluster_total // 2]
                current_segment = np.where(current_segment > 0, cluster_center, current_segment)
            else:
                cluster_sum = np.sum(current_channel)
                cluster_mean = cluster_sum / cluster_total
                current_segment = np.where(current_segment > 0, cluster_mean, current_segment)
            segmented_image += current_segment.astype(np.float64) #* np.random.rand(1)

    return segmented_image

# def get_segmented_image(image, clustered_image, clustered_labels, args, use_median=False):
#     label_values = np.unique(clustered_labels)
#     segmented_image = np.zeros_like(image)
#
#     if use_median:
#         factor = 255 / (np.max(image) - np.min(image))
#         for index in label_values:
#             current_mask = (clustered_image == index).astype(np.uint8)
#             current_segment = factor * image * current_mask
#
#             cluster_total = np.count_nonzero(current_segment)
#             #cluster_sum = np.sum(current_segment)
#             #cluster_mean = cluster_sum / cluster_total
#             non_zero_current_segment = current_segment[current_segment != 0]
#             cluster_center = non_zero_current_segment[cluster_total//2]
#             current_segment = np.where(current_segment > 0, cluster_center, current_segment)
#             segmented_image += current_segment.astype(np.uint8)
#
#     else:
#         # factor = 255 / (np.max(image) - np.min(image))
#         for index in label_values:
#             current_mask = (clustered_image == index).astype(np.uint8)
#             factor = (index+1) * (255/len(label_values))
#             current_segment = factor * image * current_mask
#             # cluster_total = np.count_nonzero(current_segment)
#             # cluster_sum = np.sum(current_segment)
#             # cluster_mean = cluster_sum / cluster_total
#             current_segment = np.where(current_segment > 0, factor, current_segment)
#             segmented_image += current_segment.astype(np.uint8)
#
#     return segmented_image

def get_dummy_image():
    l = 100
    x, y = np.indices((l, l))

    center1 = (28, 24)
    center2 = (40, 50)
    center3 = (67, 58)
    center4 = (24, 70)

    radius1, radius2, radius3, radius4 = 16, 14, 15, 14

    circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
    circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
    circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
    circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2

    # #############################################################################
    # 4 circles
    image = circle1 + circle2 + circle3 + circle4

    # We use a mask that limits to the foreground: the problem that we are
    # interested in here is not separating the objects from the background,
    # but separating them one from the other.
    mask = image.astype(bool)

    image = image.astype(float)
    image += 1 + 0.2 * np.random.randn(*image.shape)
    return image
