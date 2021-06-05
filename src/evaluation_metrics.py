import numpy as np
import os
import cv2
from tqdm import tqdm

def get_mean_absolute_error(image_a, image_b):
    mean_absolute_error = np.mean(np.abs(image_a - image_b))
    return mean_absolute_error

def get_peak_signal_to_noise_ratio(image_a, image_b):
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

def run_evaluation(mode):
    path_stacked = os.path.join('..', 'output', 'stacked', mode)
    filename_list = os.listdir(path_stacked)

    mae_list = []
    psnr_list = []
    iou_list = []
    dice_list = []
    print(f'Running evaluation for mode {mode.upper()} .....')
    for index_filename, curent_filename in tqdm(enumerate(filename_list), leave=True, total=len(filename_list)):
        image_path_full = os.path.join(path_stacked, curent_filename)
        image_stacked = cv2.cvtColor(cv2.imread(image_path_full), cv2.COLOR_BGR2RGB)

        height, width_stacked, num_channels = image_stacked.shape

        if width_stacked > 640:
            width = width_stacked // 3
            image = image_stacked[:, :width]
            label_gt = image_stacked[:, width:width * 2]
            label_pred = image_stacked[:, width * 2:]
            label_gt = np.expand_dims(np.transpose(label_gt, (2, 1, 0)), axis=0)
            label_pred = np.expand_dims(np.transpose(label_pred, (2, 1, 0)), axis=0)
        else:
            width = width_stacked // 2
            image = image_stacked[:, :width]
            label_pred = image_stacked[:, width:width*2]
            label_gt = np.expand_dims(np.transpose(image, (2, 1, 0)), axis=0)
            label_pred = np.expand_dims(np.transpose(label_pred, (2, 1, 0)), axis=0)

        mean_absolute_error = get_mean_absolute_error(label_gt, label_pred)
        peak_signal_to_noise_ratio = get_peak_signal_to_noise_ratio(label_gt, label_pred)
        intersection_over_union = metrics_np(label_gt, label_pred, metric_name="iou", metric_type="soft")
        dice = metrics_np(label_gt, label_pred, metric_name="dice", metric_type="soft")

        mae_list.append(mean_absolute_error)
        psnr_list.append(peak_signal_to_noise_ratio)
        iou_list.append(intersection_over_union)
        dice_list.append(dice)

    def get_mean(array):
        return np.mean(np.array(array))

    def get_median(array):
        return np.median(np.array(array))

    def print_mean_and_median(mean, median, metric_type):
        print(f'{metric_type}\t\t{mean:.4f}\t\t{median:.4f}')

    mae_mean, mae_median = get_mean(mae_list), get_median(mae_list)
    psnr_mean, psnr_median = get_mean(psnr_list), get_median(psnr_list)
    iou_mean, iou_median = get_mean(iou_list), get_median(iou_list)
    dice_mean, dice_median = get_mean(dice_list), get_median(dice_list)

    print(f'Metric\t\tMean\t\tMedian')
    print_mean_and_median(mae_mean, mae_median, 'MAE')
    print_mean_and_median(psnr_mean, psnr_median, 'PSNR')
    print_mean_and_median(iou_mean, iou_median, 'IOU')
    print_mean_and_median(dice_mean, dice_median, 'DICE')