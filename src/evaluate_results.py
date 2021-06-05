import numpy as np
import os
import cv2
import torch
from evaluation_metrics import metrics_np
import matplotlib.pyplot as plt

def get_mean_absolute_error(image_a, image_b):
    mean_absolute_error = np.mean(np.abs(image_a - image_b))
    return mean_absolute_error

def get_peak_signal_to_noise_ratio(image_a, image_b):
    mean_absolute_error = get_mean_absolute_error(image_a, image_b)
    peak_signal_to_noise_ratio = 20 * np.log10(255 ** 2 / mean_absolute_error)
    return peak_signal_to_noise_ratio

index_image = 0
image_path = os.path.join('..', 'output', 'stacked', 'train')
image_path_full = os.path.join(image_path, f'{index_image}.png')
image_stacked = cv2.cvtColor(cv2.imread(image_path_full), cv2.COLOR_BGR2RGB)

height, width_stacked, num_channels = image_stacked.shape
width = width_stacked // 3 if width_stacked > 500 else width_stacked // 2

image = image_stacked[:, :width]
label_gt = image_stacked[:, width:width*2]
label_pred = image_stacked[:, width*2:]

mean_absolute_error = get_mean_absolute_error(label_gt, label_pred)
peak_signal_to_noise_ratio = get_peak_signal_to_noise_ratio(label_gt, label_pred)
print(f'MAE (-): {mean_absolute_error}')
print(f'PSNR (-): {peak_signal_to_noise_ratio}')



# n = image
# N = np.sum(n)
# marginal_1 = sum(n, 1)
# marginal_2 = sum(n, 0)

# E1 = 1 - np.sum( np.sum(n*n, 1) / (marginal_1 + (marginal_1 == 0)) ) / N
# E2 = 1 - np.sum( np.sum(n*n, 0) / (marginal_2 + (marginal_2 == 0)) ) / N
# gce = np.min( E1, E2 )
# print(gce)

# plt.figure()
# plt.imshow(image)
# plt.show()
#
# plt.figure()
# plt.imshow(label_gt)
# plt.show()
#
# plt.figure()
# plt.imshow(label_pred)
# plt.show()




y_true = np.expand_dims(np.transpose(label_gt, (2, 1, 0)), axis=0)
y_pred = np.expand_dims(np.transpose(label_pred, (2, 1, 0)), axis=0)

print(f'IOU: {metrics_np(y_true, y_pred, metric_name="iou", metric_type="soft")}')
print(f'Dice: {metrics_np(y_true, y_pred, metric_name="dice", metric_type="soft")}')

