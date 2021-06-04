from utils import setup_model_parameters, get_args, unzip_dataset, imshow, get_segmented_image, get_stacked_image_horizontal, create_dirs, get_IOU, save_image
from spectral_clustering import run_spectral_clustering
import numpy as np
from dataloader import get_datasets
import os
from model_runner import run

if __name__ == '__main__':
    # initialize variables, create directories, unzip dataset
    args = setup_model_parameters()

    # get datasets
    dataset_train, dataset_val, dataset_test = get_datasets()

    if args.train_condition:
        run(dataset_train, args, mode='train')
    if args.val_condition:
        run(dataset_val, args, mode='val')
    if args.test_condition:
        run(dataset_test, args, mode='test')