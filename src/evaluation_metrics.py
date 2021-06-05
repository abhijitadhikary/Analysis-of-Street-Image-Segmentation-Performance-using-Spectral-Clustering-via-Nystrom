import numpy as np
import os
import cv2
from tqdm import tqdm
from scipy.special import comb
import matplotlib.pyplot as plt

def get_mean_absolute_error(image_a, image_b):
    '''
    Returns the mean absolute error between two arrays
    :param image_a:
    :param image_b:
    :return:
    '''
    mean_absolute_error = np.mean(np.abs(image_a - image_b))
    return mean_absolute_error

def get_peak_signal_to_noise_ratio(image_a, image_b):
    '''
    Returns the PSNR value between two arrays of images
    :param image_a:
    :param image_b:
    :return:
    '''
    mean_absolute_error = get_mean_absolute_error(image_a, image_b)
    peak_signal_to_noise_ratio = 20 * np.log10(255 ** 2 / mean_absolute_error)
    return peak_signal_to_noise_ratio

def metrics_np(y_true, y_pred, metric_name,
    metric_type='standard', drop_last = True, mean_per_class=False, verbose=False):
    """
    This function was copied from: https://ilmonteux.github.io/2019/05/10/segmentation-metrics.html
    and is not our own implementation.
    Compute mean metrics of two segmentation masks, via numpy.

    IoU(A,B) = |A & B| / (| A U B|)
    Dice(A,B) = 2*|A & B| / (|A| + |B|)

    Args:
        y_true: true masks, one-hot encoded.
        y_pred: predicted masks, either softmax outputs, or one-hot encoded.
        metric_name: metric to be computed, either 'iou' or 'dice'.
        metric_type: one of 'standard' (default), 'soft', 'naive'.
          In the standard version, y_pred is one-hot encoded and the mean
          is taken only over classes that are present (in y_true or y_pred).
          The 'soft' version of the metrics are computed without one-hot
          encoding y_pred.
          The 'naive' version return mean metrics where absent classes contribute
          to the class mean as 1.0 (instead of being dropped from the mean).
        drop_last = True: boolean flag to drop last class (usually reserved
          for background class in semantic segmentation)
        mean_per_class = False: return mean along batch axis for each class.
        verbose = False: print intermediate results such as intersection, union
          (as number of pixels).
    Returns:
        IoU/Dice of y_true and y_pred, as a float, unless mean_per_class == True
          in which case it returns the per-class metric, averaged over the batch.

    Inputs are B*W*H*N tensors, with
        B = batch size,
        W = width,
        H = height,
        N = number of classes
    """

    assert y_true.shape == y_pred.shape, 'Input masks should be same shape, instead are {}, {}'.format(y_true.shape, y_pred.shape)
    assert len(y_pred.shape) == 4, 'Inputs should be B*W*H*N tensors, instead have shape {}'.format(y_pred.shape)

    flag_soft = (metric_type == 'soft')
    flag_naive_mean = (metric_type == 'naive')

    num_classes = y_pred.shape[-1]
    # if only 1 class, there is no background class and it should never be dropped
    drop_last = drop_last and num_classes>1

    if not flag_soft:
        if num_classes>1:
            # get one-hot encoded masks from y_pred (true masks should already be in correct format, do it anyway)
            y_pred = np.array([ np.argmax(y_pred, axis=-1)==i for i in range(num_classes) ]).transpose(1,2,3,0)
            y_true = np.array([ np.argmax(y_true, axis=-1)==i for i in range(num_classes) ]).transpose(1,2,3,0)
        else:
            y_pred = (y_pred > 0).astype(int)
            y_true = (y_true > 0).astype(int)

    # intersection and union shapes are batch_size * n_classes (values = area in pixels)
    axes = (1,2) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) # or, np.logical_and(y_pred, y_true) for one-hot
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum  - intersection # or, np.logical_or(y_pred, y_true) for one-hot

    if verbose:
        print('intersection (pred*true), intersection (pred&true), union (pred+true-inters), union (pred|true)')
        print(intersection, np.sum(np.logical_and(y_pred, y_true), axis=axes), union, np.sum(np.logical_or(y_pred, y_true), axis=axes))

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    dice = 2*(intersection + smooth)/(mask_sum + smooth)

    metric = {'iou': iou, 'dice': dice}[metric_name]

    # define mask to be 0 when no pixels are present in either y_true or y_pred, 1 otherwise
    mask =  np.not_equal(union, 0).astype(int)
    # mask = 1 - np.equal(union, 0).astype(int) # True = 1

    if drop_last:
        metric = metric[:,:-1]
        mask = mask[:,:-1]

    # return mean metrics: remaining axes are (batch, classes)
    # if mean_per_class, average over batch axis only
    # if flag_naive_mean, average over absent classes too
    if mean_per_class:
        if flag_naive_mean:
            return np.mean(metric, axis=0)
        else:
            # mean only over non-absent classes in batch (still return 1 if class absent for whole batch)
            return (np.sum(metric * mask, axis=0) + smooth)/(np.sum(mask, axis=0) + smooth)
    else:
        if flag_naive_mean:
            return np.mean(metric)
        else:
            # mean only over non-absent classes
            class_count = np.sum(mask, axis=0)
            return np.mean(np.sum(metric * mask, axis=0)[class_count!=0]/(class_count[class_count!=0]))

def mean_iou_np(y_true, y_pred, **kwargs):
    """
    Compute mean Intersection over Union of two segmentation masks, via numpy.

    Calls metrics_np(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
    """
    return metrics_np(y_true, y_pred, metric_name='iou', **kwargs)

def mean_dice_np(y_true, y_pred, **kwargs):
    """
    Compute mean Dice coefficient of two segmentation masks, via numpy.

    Calls metrics_np(y_true, y_pred, metric_name='dice'), see there for allowed kwargs.
    """
    return metrics_np(y_true, y_pred, metric_name='dice', **kwargs)

def get_PRI_score(clusters, classes):
    '''
    The implementation followed from:
    https://stackoverflow.com/questions/49586742/rand-index-function-clustering-performance-evaluation
    :param clusters:
    :param classes:
    :return:
    '''
    clusters, classes = clusters.reshape(-1), classes.reshape(-1)
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def get_voi_score(array_a, array_b):
    '''
    Returns the variaiton of information metric between two arrays/images
    The implementation followed from:
    https://gist.github.com/jwcarr/626cbc80e0006b526688
    :param array_a:
    :param array_b:
    :return:
    '''
    array_a = np.squeeze(np.squeeze(array_a, axis=0), axis=0)
    array_b = np.squeeze(np.squeeze(array_b, axis=0), axis=0)
    n = len(array_a.reshape(-1))

    sigma = 0.0
    for x in array_a:
        p = len(x) / n
        for y in array_b:
            q = len(y) / n
            r = len(set(x) & set(y)) / n
            if r > 0.0:
                sigma += r * (np.log(r / p) + np.log(r / q))

    res = abs(sigma)

    p = array_a.shape[1] / n
    q = array_b.shape[1] / n

    r_all = np.array([len(list(set(row_a) & set(row_b)))/n for row_a, row_b in zip(array_a, array_b)])
    r_all = r_all.reshape(-1, 1)
    r_all = r_all @ r_all.T
    res_ = np.abs(np.sum(np.where(r_all > 0.0, r_all * (np.log2(r_all / p) + np.log2(r_all / q)), 0)))

    return res

def get_gce_score(array_a, array_b):
    '''
    Returns the global consistency error between two arrays/images
    :param array_a:
    :param array_b:
    :return:
    '''

    N = np.sum(np.stack((array_a, array_b)))

    marginal_1 = np.sum(array_a)
    marginal_2 = np.sum(array_b)

    E1 = 1 - np.sum(np.sum(array_a*array_a) / (marginal_1 + (marginal_1 == 0))) / N
    E2 = 1 - np.sum(np.sum(array_b*array_b) / (marginal_2 + (marginal_2 == 0))) / N
    gce = min(E1, E2)

    return gce

def get_interclass_iou_score(array_a, array_b):
    '''
    Returns the interclass intersection over union scores betwen two arrays/images
    :param array_a:
    :param array_b:
    :return:
    '''
    unique_vals_a = np.unique(array_a)
    unique_vals_b = np.unique(array_b)
    num_unique_a = len(unique_vals_a)
    num_unique_b = len(unique_vals_b)

    num_pixels = array_b.shape[2] * array_b.shape[3]
    score_array = np.zeros((num_unique_a, num_unique_b))

    for index_a in range(num_unique_a):
        for index_b in range(num_unique_b):
            segment_a = (array_a == unique_vals_a[index_a])
            segment_b = (array_b == unique_vals_b[index_b])

            method = 0
            # bad
            if method == 0:
                intersection = (segment_a * segment_b) # only looks into the true values
                intersection_score = np.sum(intersection)
                union_score = (np.sum(segment_a) + np.sum(segment_b)) - intersection_score
                score = (intersection_score / (union_score+1e-6))
                score_array[index_a, index_b] = score
            elif method == 1:
                intersection = (segment_a == segment_b)  # looks into both true and false values
                score = np.sum(intersection) / num_pixels
                score_array[index_a, index_b] = score
    return score_array

def get_mean(array):
    '''
    Returns the mean value of an array
    :param array:
    :return:
    '''
    return np.mean(np.array(array))

def get_median(array):
    '''
    Returns the median value of an array
    :param array:
    :return:
    '''
    return np.median(np.array(array))

def print_mean_and_median(mean, median, metric_type):
    '''
    Prints the mean and median of a metric
    :param mean:
    :param median:
    :param metric_type:
    :return:
    '''
    print(f'{metric_type}\t\t{mean:.4f}\t\t{median:.4f}')

def print_interclass_iou(array):
    plt.figure(figsize=(5, 5))
    plt.imshow(array)
    plt.title('Inter-class IOU')
    plt.show()

def run_evaluation(mode, use_color_eval=False):
    '''
    Given a mode (train, val, test) the function runs evaluation metrics and prints the mean results
    :param mode:
    :return:
    '''
    path_stacked = os.path.join('..', 'output', 'stacked', mode)
    try:
        filename_list = os.listdir(path_stacked)
    except FileNotFoundError as error:
        print(f'Exiting evaluation. Directory not found: {path_stacked} in mode {mode.upper()}')
        return
    if len(filename_list) == 0:
        print(f'Exiting evaluation. No files exist in directory: {path_stacked} in mode {mode.upper()}')
        return
    mae_list = []
    psnr_list = []
    iou_list = []
    dice_list = []
    pri_list = []
    voi_list = []
    gce_list = []
    interclass_iou_list = np.zeros((8, 8))
    print(f'Running evaluation metrics in mode {mode.upper()} .....')
    for index_filename, curent_filename in tqdm(enumerate(filename_list), leave=True, total=len(filename_list)):
        image_path_full = os.path.join(path_stacked, curent_filename)
        try:
            image_stacked = cv2.cvtColor(cv2.imread(image_path_full), cv2.COLOR_BGR2RGB)
        except cv2.error as error:
            print(f'Invalid file format: {image_path_full} in mode {{mode.upper()}}')
            continue

        height, width_stacked, num_channels = image_stacked.shape

        if width_stacked > 640:
            # for train and eval mode, compares the predicted segmentations against the true segmentations
            width = width_stacked // 3
            image = image_stacked[:, :width]

            if use_color_eval:
                label_gt = image_stacked[:, width:width * 2]
                label_pred = image_stacked[:, width * 2:]
                label_gt = np.expand_dims(np.transpose(label_gt, (2, 1, 0)), axis=0)
                label_pred = np.expand_dims(np.transpose(label_pred, (2, 1, 0)), axis=0)
            else:
                label_gt = cv2.cvtColor(image_stacked[:, width:width * 2], cv2.COLOR_RGB2GRAY)
                label_pred = cv2.cvtColor(image_stacked[:, width * 2:], cv2.COLOR_RGB2GRAY)
                label_gt = np.expand_dims(label_gt, axis=0)
                label_gt = np.expand_dims(label_gt, axis=0)
                label_pred = np.expand_dims(label_pred, axis=0)
                label_pred = np.expand_dims(label_pred, axis=0)
        else:
            # for testing, although this does not make much sense to compare actual image with predicted labels
            width = width_stacked // 2
            image = image_stacked[:, :width]
            if use_color_eval:
                label_gt = np.expand_dims(np.transpose(image, (2, 1, 0)), axis=0)
                label_pred = image_stacked[:, width:width*2]
                label_pred = np.expand_dims(np.transpose(label_pred, (2, 1, 0)), axis=0)
            else:
                label_gt = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                label_pred = cv2.cvtColor(image_stacked[:, width:width * 2], cv2.COLOR_RGB2GRAY)
                label_gt = np.expand_dims(label_gt, axis=0)
                label_gt = np.expand_dims(label_gt, axis=0)
                label_pred = np.expand_dims(label_pred, axis=0)
                label_pred = np.expand_dims(label_pred, axis=0)


        mean_absolute_error = get_mean_absolute_error(label_gt, label_pred)
        mae_list.append(mean_absolute_error)

        peak_signal_to_noise_ratio = get_peak_signal_to_noise_ratio(label_gt, label_pred)
        psnr_list.append(peak_signal_to_noise_ratio)

        intersection_over_union = metrics_np(label_gt, label_pred, metric_name="iou", metric_type="soft")
        iou_list.append(intersection_over_union)

        dice_score = metrics_np(label_gt, label_pred, metric_name="dice", metric_type="soft")
        dice_list.append(dice_score)

        probabilistic_rand_index = get_PRI_score(label_gt, label_pred)
        pri_list.append(probabilistic_rand_index)

        gce_score = get_gce_score(label_gt, label_pred)
        gce_list.append(gce_score)

        # takes a long time to run
        # variation_of_information = get_voi_score(label_gt, label_pred)
        # voi_list.append(variation_of_information)

        interclass_iou_score = get_interclass_iou_score(label_gt, label_pred)
        height, width = interclass_iou_score.shape
        if mode in ['train', 'val']:
            interclass_iou_list[:height, :width] += interclass_iou_score

    if len(mae_list) == 0:
        print(f'Exiting evaluation. No compatible files found to run evaluation metrics in {path_stacked}')
        return

    mae_mean, mae_median = get_mean(mae_list), get_median(mae_list)
    psnr_mean, psnr_median = get_mean(psnr_list), get_median(psnr_list)
    iou_mean, iou_median = get_mean(iou_list), get_median(iou_list)
    dice_mean, dice_median = get_mean(dice_list), get_median(dice_list)
    pri_mean, pri_median = get_mean(pri_list), get_median(pri_list)
    # voi_mean, voi_median = get_mean(voi_list), get_median(voi_list)
    gce_mean, gce_median = get_mean(gce_list), get_median(gce_list)

    if mode in ['train', 'val']:
        interclass_iou_list /= len(mae_list)

    num_samples = len(mae_list)
    print(f'Total number of samples in mode {mode.upper()}: {num_samples}')
    print(f'Metric\t\tMean\t\tMedian')
    print_mean_and_median(mae_mean, mae_median, 'MAE ')
    print_mean_and_median(psnr_mean, psnr_median, 'PSNR')
    print_mean_and_median(iou_mean, iou_median, 'IOU ')
    print_mean_and_median(dice_mean, dice_median, 'DICE')
    print_mean_and_median(pri_mean, pri_median, 'PRI ')
    # print_mean_and_median(voi_mean, voi_median, 'VoI ')
    print_mean_and_median(gce_mean, gce_median, 'GCE ')
    if mode in ['train', 'val']:
        print_interclass_iou(interclass_iou_list)