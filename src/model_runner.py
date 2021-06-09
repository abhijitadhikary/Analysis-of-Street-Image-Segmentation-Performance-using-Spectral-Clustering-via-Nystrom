import os
from tqdm import tqdm
from time import time
from utils import process_image_attributes, print_duration_information, print_cluster_memberships, get_segmented_image, get_stacked_image_horizontal, save_image
from spectral_clustering import run_spectral_clustering
from kmeans import run_kmeans
from gmm import run_gmm
from evaluation_metrics import run_evaluation

def run(dataset, args, mode=None):
    '''
    Runs the model in either train, val or test mode
    :param dataset:
    :param args:
    :param mode:
    :return:
    '''
    print(f'Model running in {(mode.upper())} mode')
    if mode in ['train', 'val']:
        image_array, label_array, label_array_inst = dataset
    elif mode == 'test':
        image_array = dataset
    else:
        raise NotImplementedError('Invalid value for mode')

    # total number of images
    num_images = len(image_array)
    # list to keep track of the length of each iteration
    duration_list = []
    for image_index in tqdm(range(num_images), leave=True):
        if image_index == 100:
            break
        # uncomment the following to only visualize the selected images in the paper, validation
        # if not image_index in [18, 107, 120, 125, 192]:
        #     continue

        # uncomment the following to only visualize the selected images in the paper, test
        # if not image_index in [1, 4, 28, 68, 80]:
        #     continue

        image = image_array[image_index]
        time_start = time()
        if args.segmentation_mode == 0:
            # cluster the image using Spectral Clustering
            clustered_image = run_spectral_clustering(image, args)
        elif args.segmentation_mode == 1:
            # cluster the image using K-Means++
            image_scaled, args = process_image_attributes(image, args)
            image_scaled = image_scaled.reshape(image.shape[0] * image.shape[1], -1)
            clustered_image, _ = run_kmeans(image_scaled, args)
            clustered_image = clustered_image.reshape(args.height, args.width)
        elif args.segmentation_mode == 2:
            # cluster the image using GMM
            image_scaled, args = process_image_attributes(image, args)
            image_scaled = image_scaled.reshape(image.shape[0] * image.shape[1], -1)
            clustered_image = run_gmm(image_scaled, args)
            clustered_image = clustered_image.reshape(args.height, args.width)
        else:
            raise NotImplementedError('Please select a valud clustering method [0: Spectral, 1: K-Means++, 2: GMM]')

        time_end = time()
        duration = time_end - time_start
        duration_list.append(duration)

        # segment the image according to cluster mean/medians color values
        segmented_image_pred = get_segmented_image(image, clustered_image.reshape(-1), args)

        # print how many points belongs to each cluster
        if args.print_cluster_memberships:
            print_cluster_memberships(segmented_image_pred, 'Membership after Segmentation')

        if mode in ['train', 'val']:
            # for train and val
            labels_image_gt = label_array[image_index]
            # segment the image using the ground truth labels
            segmented_image_gt = get_segmented_image(image, labels_image_gt.reshape(-1), args)
            # stack the three images horizontally
            stacked_image_horizontal = get_stacked_image_horizontal(image, segmented_image_gt, segmented_image_pred)
            # save the image
            title = f'Image{" " * 30}Label (GT){" " * 30}Label (Pred)' if args.save_stacked_title else None
            save_path_stacked_full = os.path.join('..', 'output', 'stacked', mode, f'{image_index}.png')
            save_image(stacked_image_horizontal, save_path_stacked_full, title)
        elif mode == 'test':
            stacked_image_horizontal = get_stacked_image_horizontal(image, segmented_image_pred)
            title = f'Image{" " * 30}Label (Pred)' if args.save_stacked_title else None
            save_path_stacked_full = os.path.join('..', 'output', 'stacked', mode, f'{image_index}.png')
            save_image(stacked_image_horizontal, save_path_stacked_full, title)
        else:
            raise NotImplementedError('Invalid value for mode')

    if args.print_duration_information:
        print_duration_information(duration_list, mode)

    if args.run_evaluation_metric:
        run_evaluation(mode)

