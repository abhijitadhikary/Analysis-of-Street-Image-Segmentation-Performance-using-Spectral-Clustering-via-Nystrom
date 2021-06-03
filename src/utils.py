import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import cv2
import torch


def get_args():
    '''
    Returns all the hyper-parameters
    :return args: the hyper-parameters
    '''
    args = argparse.Namespace()
    args.num_clusters = 8
    # args.num_clusters = 8
    args.sigma_color = 0.6 # 0.4
    args.sigma_distance = 5 # 20
    args.height = 100
    args.width = 100
    args.num_channels = 0
    args.num_dimensions = 0
    args.num_elements_flat = 0
    args.use_numpy_eigen_decompose = True
    args.dim_low = 100
    args.color_weight_mode = 0 # 0: RGB Intensity, 1: constant(1), 2: HSV, 1: DOOG
    return args


def convert(source, min_value=0, max_value=1):
    '''
    Rescales the values of an array between min_value and max_value
    :param source: Input image if size [height, width, num_channels]
    :param min_value: Minimum scalar value in the target
    :param max_value: Maximum scalar value in the target
    :return target: Rescaled target with same shape as the source
    '''
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


def imshow(image, args, title=''):
    '''
    Displays an image using matplotlib
    :param image: The image to show ( of shape [height, width, 3]-> colour or [height, width]-> grayscale)
    :param title: A string. Title to be put as the heading
    :return:
    '''
    # rearrange pixel values between 0 and 255 and convert dtype to uint8
    # image = convert(image, 0, 255).astype(np.uint8)
    image = image.astype(np.uint8)
    # # if input image is hsv format, convert it to rgb
    # if args.color_weight_mode == 2 and args.num_channels == 3:
    #     image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    plt.figure(figsize=(5, 5))
    plt.title(title)

    if args.num_channels == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


def process_image_attributes(image, args):
    '''
    Update args's parameters corresponding to the image dimension
    :param image: The input image of shape [height, width, num_channels]
    :param args: The arguments with all the hyper-parameters
    :return image: The normalized image of shape [height, width, num_channels]
            args: Updated arguments with image dimensions
    '''
    image_shape = len(image.shape)
    if image_shape == 3:
        height, width, num_channels = image.shape
    elif image_shape == 2:
        height, width = image.shape
        num_channels = 1
    else:
        raise NotImplementedError('Invalid image format')

    args.height = height
    args.width = width
    args.num_channels = num_channels
    args.num_elements_flat = height * width
    args.num_dimensions = num_channels + 2

    if args.color_weight_mode == 2:
        # convert to HSV
        image = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HSV)

    image = convert(image, min_value=0, max_value=1)
    return image, args


def get_image_array(image, args):
    # noinspection PyPackageRequirements
    '''
        Converts an image into a long vector where the first three (or 1: for grayscale)
        elements of each row are the color intensity information, and the last two are the
        X and Y coordinates of the pixel
        :param image: The input image of shape [height, width, num_channels]
        :param args: The arguments with all the hyper-parameters
        :return image_array: flattened image array of shape [height*width, num_channels + 2]
        '''
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


def get_segmented_image(image, clustered_image, clustered_labels, args, use_median=True):
    '''
    Returns a segmented image based on the supplied labels, either using median
    or mean each cluster
    :param image : The input image of shape [height, width, num_channels]
    :param clustered_image: The cluster label of each pixel in the shape of the image [height, width]
    :param clustered_labels: The cluster label of each pixel flattened, shape = [height*width,]
    :param args: The arguments with all the hyper-parameters
    :param use_median: Use median value as the cluster intensity or not (False -> mean is used, True -> median is used)
    :return segmented_image:  The segmented image of shape [height, width, num_channels]
    '''
    image = convert(image, 0, 1)
    label_values = np.unique(clustered_labels)
    segmented_image = np.zeros_like(image)
    if use_median:
        factor = 255 / (np.max(image) - np.min(image))
    else:
        factor = 1

    if args.num_channels == 3:
        for index in label_values:
            current_mask = (clustered_image == index).astype(np.float64)
            current_segment = image * np.repeat(current_mask[..., None], args.num_channels, axis=2) * factor

            for channel_index in range(args.num_channels):
                current_channel = current_segment[:, :, channel_index]
                cluster_total = np.count_nonzero(current_channel)

                if use_median:
                    non_zero_current_channel = np.sort(
                        current_channel[current_channel != 0])  # Sort values to find median
                    cluster_median = non_zero_current_channel[cluster_total // 2]  # Median of non-0 elements
                    current_segment[:, :, channel_index] = np.where(current_channel > 0, cluster_median,
                                                                    current_channel)
                else:
                    cluster_sum = np.sum(current_channel)
                    cluster_mean = cluster_sum / cluster_total
                    current_segment[:, :, channel_index] = np.where(current_segment[:, :, channel_index] > 0,
                                                                    cluster_mean,
                                                                    current_segment[:, :, channel_index])
                segmented_image[:, :, channel_index] += current_segment[:, :, channel_index].astype(np.float64)

    elif args.num_channels == 1:
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
            segmented_image += current_segment.astype(np.float64)  # * np.random.rand(1)

    return segmented_image

def get_dummy_image():
    '''
    To make the dummy noise image, for low level debugging and testing
    :return image: a single channel image with 4 circles in a 100x100 grid
    '''
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

def get_IOU(ground_truth, predicted):
    # for channel_index in range(ground_truth.shape[2]):
    #     unique_values = np.unique(ground_truth[:, :, channel_index])
    #     intersection = np.logical_and(ground_truth[:, :, channel_index], predicted[:, :, channel_index])
    #     union = np.logical_or(ground_truth[:, :, channel_index], predicted[:, :, channel_index])
    #     iou = np.sum(intersection) / np.sum(union)
    #     print(iou)

    unique_color_values = np.unique(ground_truth.reshape(-1, ground_truth.shape[2]), axis=0)

    return unique_color_values

# def get_IOU(label, pred, num_classes=8):
#     pred = torch.tensor(pred)
#     pred = pred.permute(2, 0, 1)
#     pred = torch.softmax(pred, dim=1)
#     pred = torch.argmax(pred, dim=1).squeeze(1)
#     iou_list = list()
#     present_iou_list = list()
#
#     pred = pred.view(-1)
#     label = torch.tensor(label)
#     label = label.view(-1)
#     # Note: Following for loop goes from 0 to (num_classes-1)
#     # and ignore_index is num_classes, thus ignore_index is
#     # not considered in computation of IoU.
#     for sem_class in range(num_classes):
#         pred_inds = (pred == sem_class)
#         target_inds = (label == sem_class)
#         if target_inds.long().sum().item() == 0:
#             iou_now = float('nan')
#         else:
#             intersection_now = (pred_inds[target_inds]).long().sum().item()
#             union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
#             iou_now = float(intersection_now) / float(union_now)
#             present_iou_list.append(iou_now)
#         iou_list.append(iou_now)
#     return np.mean(present_iou_list)