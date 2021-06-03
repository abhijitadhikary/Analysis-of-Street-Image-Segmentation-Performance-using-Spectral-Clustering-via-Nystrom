import numpy as np
from nystrom import run_nystrom
from utils import get_image_array
from clusters import get_clustered_image
from utils import get_segmented_image


def get_exponential_bump(distance, sigma=1):
    '''
    Applies an exponential function to each element of the array with a supplied variance
    :param distance: a matrix of distance measure
    :param sigma: for the exponential bum, default = 1
    :return: exponential function applied to distances, works as a similarity measure
    '''
    exponential_bump = np.exp(-np.abs(distance) / sigma ** 2)
    return exponential_bump


def get_euclidean_distance(point_1, point_2_array):
    '''
    Returns the Euclidean distance between each row of two arrays
    :param point_1:
    :param point_2_array:
    :return:
    '''
    euclidean_distance = np.sqrt(np.sum(np.power((point_1 - point_2_array), 2), axis=1))
    return euclidean_distance


def get_hsv_weights(array):
    h, s, v = array[:, 0].reshape(-1, 1), array[:, 1].reshape(-1, 1), array[:, 2].reshape(-1, 1)
    output = np.hstack((
        v,
        v * s * np.sin(h),
        v * s * np.cos(h),
    ))
    return output


def get_color_weight(image_array, image_low, args):
    '''
    Returns the weight of the color information for calculating the Adjacency martix
    :param image_array:
    :param image_low:
    :param args:
    :return:
    '''
    # ***********************************************************
    # ***********************************************************
    # EXPERIMENT WITH RGB, HSV and DOOG, experiments section of the paper, page 7
    # ***********************************************************
    # ***********************************************************
    # 0: RGB, 1: constant(1), 2: HSV, 1: DOOG

    if args.color_weight_mode == 0:
        # intensity distance (RGB)
        color_weight = np.linalg.norm(
            np.expand_dims(image_array[:, :args.num_channels], axis=1) - image_low[:, :args.num_channels], axis=-1,
            ord=2)
    elif args.color_weight_mode == 1:
        # not a good idea for images
        color_weight = np.ones((image_array.shape[0], image_low.shape[0])).astype(np.float64)
    elif args.color_weight_mode == 2:
        # for color images
        if args.num_channels == 3:
            image_array = get_hsv_weights(image_array)
            image_low = get_hsv_weights(image_low)
            color_weight = np.linalg.norm(
                np.expand_dims(image_array[:, :args.num_channels], axis=1) - image_low[:, :args.num_channels], axis=-1,
                ord=2)
        # for grayscale images
        elif args.num_channels == 1:
            raise NotImplementedError('HSV weight mode not implemented for Grayscale images')
    elif args.color_weight_mode == 3:
        pass
    else:
        raise NotImplementedError('Please choose one of the specified values for args.color_weight_mode')

    return color_weight


def get_distance_weight(image_array, image_low, args):
    '''
    Returns the weight of the pixel location for calculating the Adjacency martix
    :param point_1:
    :param point_2_array:
    :param sigma_distance:
    :return:
    '''
    distance_weight = np.linalg.norm(
        np.expand_dims(image_array[:, args.num_channels:], axis=1) - image_low[:, args.num_channels:], axis=-1, ord=2)
    return distance_weight


def get_weight_martix_partial(image_array, indices_random_low_dim, args):
    '''
    Returns a dim_low x len(image_array) weight matrix
    :param image_array:
    :param indices_random_low_dim:
    :param args:
    :return:
    '''
    image_low = image_array[indices_random_low_dim]
    color_weight = get_color_weight(image_array, image_low, args)
    distance_weight = get_distance_weight(image_array, image_low, args)
    weight_matrix = (get_exponential_bump(color_weight, args.sigma_color)
                     * get_exponential_bump(distance_weight, args.sigma_distance)).T
    # set the diagonal entries to 0
    row = [i for i in range(args.dim_low)]
    weight_matrix[row, row] = 1

    return weight_matrix


def get_top_matrix(V, indices_random_low_dim, args):
    '''
        Returns the correctly ordered eigenvectors
        :param V: the unordered eigenvectors
        :param indices_random_low_dim: the random indices chosen for nystorm
        :param args: the set of arguments
        :return top_matrix: correctly ordered eigenvectors taking random indices into account
        '''
    # reordering V appropriately
    all_idx = list(np.arange(args.num_elements_flat))
    rem_idx = [idx for idx in all_idx if idx not in indices_random_low_dim]
    top_matrix = np.zeros((args.num_elements_flat, args.num_clusters - 1))
    top_matrix[list(indices_random_low_dim), :] = V[:args.dim_low, :]
    top_matrix[rem_idx, :] = V[args.dim_low:, :]

    return top_matrix


def get_k_smallest_eigen_vectors(eigen_vectors, args):
    '''
    Returns the 1-k smallest eigen vectors of eigen_vectors
    :param eigen_vectors: all the eigenvectors
    :param args: number of eigenvectors needed = number of clusters - 1
    :return: The 1-k smallest eigen vectors of eigen_vectors
    '''
    # ***********************************************************
    # ***********************************************************
    # EXPERIMENT WITH HISTOGRAMS of min:max eigen vectors, page 5 point 4 of the paper
    # ***********************************************************
    # ***********************************************************
    return eigen_vectors[:, 1:args.num_clusters]


def get_random_indices(array, num_indices, replace=False):
    '''
    Returns specified number of random indices (uniform distribution) from the supplied array
    :param array: The range from which the indices are to be chosen
    :param num_indices: The number of random indices chosen
    :param replace: if we want the random points chosen to be repeated or not, default = False
    :return: array of size [num_indices,]
    '''
    return np.random.choice(array, size=num_indices, replace=replace)


def run_spectral_segmentation(scaled_image, image, args):
    '''
    Returns the k (dim_low) smallest eigenvectors after performing spectral clustering
    (Uses Nystrom to efficiently computer the eigenvectors)
    :param scaled_image: The input image of shape [height, width, num_channels]
    :param args: the arguments with the hyper-parameters
    :return segmented_image: The segmented image of shape [height, width, num_channels]
    '''
    # start = time()
    image_array = get_image_array(scaled_image, args)
    # k (dim_low) random indices for nystrom
    indices_random_low_dim = get_random_indices(args.num_elements_flat, args.dim_low)

    # get weight matrix corresponding to the randomly generated k indices
    weight_matrix_partial = get_weight_martix_partial(image_array, indices_random_low_dim, args)

    # calculate eigenvectors using the Nystrom method
    eigen_vectors_dim_low = run_nystrom(weight_matrix_partial, indices_random_low_dim)

    # extract the smallest k eigen vectors (ingoring the first one)
    eigen_vectors_k = get_k_smallest_eigen_vectors(eigen_vectors_dim_low, args)

    # get top matrix - SID PLEASE EXPLAIN WHAT THIS DOES
    eigen_vecs_k = get_top_matrix(eigen_vectors_k, indices_random_low_dim, args)

    # end = time()
    # print('Run time', end - start)
    print(eigen_vecs_k.shape)

    # segment the eigenvectors using
    clustered_image, clustered_labels = get_clustered_image(eigen_vecs_k, args)

    # segment the image according to cluster mean/medians color values
    segmented_image = get_segmented_image(image, clustered_image, clustered_labels, args)

    unique, counts = np.unique(segmented_image[:, :, 0], return_counts=True)
    print(f'After Segmentation\n{dict(zip(unique, counts))}')
    return segmented_image
