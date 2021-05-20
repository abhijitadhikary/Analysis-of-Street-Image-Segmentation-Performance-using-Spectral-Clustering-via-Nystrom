import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans

args = {
    'num_clusters': 5,
    'num_eigen_vectors': 4,
    'sigma_color': 2,
    'sigma_distance': 3,
    'height': 30,
    'width': 30,
    'num_channels': 0,
    'num_dimensions': 0,
    'num_elements_flat': 0,
    'use_numpy_eigen_decompose': True,
    'dim_low': 5
}

# data_dir = os.path.join('..', 'data')
# image_dir = os.path.join(data_dir, 'idd20k_lite', 'leftImg8bit', 'train', '1')
# filename = '502201_image.jpg'
# image_path_full = os.path.join(image_dir, filename)
# image = cv2.imread(image_path_full)

# image = cv2.imread('bw.jpg')
image = cv2.imread('../notebooks/vegetables.png')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB
# image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # convert to Grayscale
image = cv2.resize(image, (args['height'], args['width']))

image_shape = len(image.shape)
if image_shape == 3:
    height, width, num_channels = image.shape
elif image_shape == 2:
    height, width = image.shape
    num_channels = 1

args['height'] = height
args['width'] = width
args['num_channels'] = num_channels
args['num_elements_flat'] = height * width
args['num_dimensions'] = num_channels + 2


def visualize_matrix(matrix, title=''):
    plt.figure(figsize=(5, 5))
    plt.title(title)
    plt.imshow(matrix)
    plt.colorbar()
    plt.show()


def imshow(image, title=''):
    plt.figure(figsize=(5, 5))
    plt.title(title)
    plt.imshow(image)
    plt.show()


def get_image_array(image):
    image_array = np.zeros((args['num_elements_flat'], (args['num_channels'] + 2)))

    image_array_index = 0
    for index_row in range(args['height']):
        for index_col in range(args['width']):
            current_pixel = image[index_row, index_col]
            image_array[image_array_index] = np.array([current_pixel[0],
                                                       current_pixel[1],
                                                       current_pixel[2],
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

def get_segmented_image(image, clustered_image, clustered_labels):
    label_values = np.unique(clustered_labels)
    segmented_image = np.zeros_like(image)
    factor = 255 / (np.max(image) - np.min(image))
    for index in label_values:
        current_mask = (clustered_image == index).astype(np.uint8)
        current_segment = factor * image * current_mask

        cluster_total = np.count_nonzero(current_segment)
        cluster_sum = np.sum(current_segment)
        cluster_mean = cluster_sum / cluster_total
        current_segment = np.where(current_segment > 0, cluster_mean, current_segment)
        segmented_image += current_segment.astype(np.uint8)

    return segmented_image


def get_k_eig_vectors_nystrom(image_array):
    dim_low = args['dim_low']
    weight_matrix = np.zeros((dim_low, args['num_elements_flat']))

    indices_low = np.random.choice(args['num_elements_flat'], size=dim_low, replace=False)
    image_low = image_array[indices_low]

    for index_1 in range(len(image_low)):
        point_1 = image_low[index_1]
        weight_color = get_color_weight_vectorized(point_1[:args['num_channels']], image_array[:, :args['num_channels']], args['sigma_color'])
        weight_distance = get_distance_weight_vectorized(point_1[-2:], image_array[:, -2:], args['sigma_distance'])
        weight_matrix[index_1] = weight_color * weight_distance
        weight_matrix[index_1, index_1] = 1

    A = weight_matrix[:, list(indices_low)]  # nxn
    B = np.delete(weight_matrix, list(indices_low), axis=1)  # nxm
    n, m = B.shape
    #     print(A.shape)
    #     print(B.shape)
    A_pinv = np.linalg.pinv(A)
    d1 = np.sum(np.vstack((A, B.T)), axis=0).reshape(1, -1)
    d2 = np.sum(B, axis=0) + (np.sum(B.T, axis=0).reshape(1, -1) @ (A_pinv @ B))
    print(d1.shape)
    print(d2.shape)
    dhat = np.sqrt(1 / np.hstack((d1, d2))).reshape(-1, 1)
    print(dhat.shape)
    print(np.count_nonzero(np.argwhere(d2 < 0)))
    A = A * (dhat[:n].reshape(-1, 1) @ dhat[:n].reshape(-1, 1).T)
    B = B * (dhat[:n].reshape(-1, 1) @ dhat[n:].reshape(-1, 1).T)

    pinv_A = np.linalg.pinv(A)

    Asi = sqrtm(pinv_A)

    Q = A + Asi @ B @ B.T @ Asi
    U, L, T = np.linalg.svd(Q)

    L = np.diag(L)

    V = np.hstack((A, B)).T @ Asi @ U @ np.linalg.pinv(np.sqrt(L))

    V = V[:, 1:args['num_eigen_vectors'] + 1]
    # reordering V appropriately
    all_idx = list(np.arange(args['num_elements_flat']))
    rem_idx = [idx for idx in all_idx if idx not in indices_low]
    top_matrix = np.zeros((args['num_elements_flat'], args['num_eigen_vectors']))
    top_matrix[list(indices_low), :] = V[:dim_low, :]
    top_matrix[rem_idx, :] = V[dim_low:, :]
    return top_matrix


image_array = get_image_array(image)
print(image_array.shape)
eigen_vecs_k = get_k_eig_vectors_nystrom(image_array)
print(eigen_vecs_k.shape)
visualize_matrix(eigen_vecs_k, 'eigen_vecs_k')