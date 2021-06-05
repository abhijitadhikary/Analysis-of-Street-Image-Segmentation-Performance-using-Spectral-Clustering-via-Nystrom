from utils import print_cluster_memberships, get_segmented_image, get_stacked_image_horizontal, save_image
from spectral_clustering import run_spectral_clustering
from evaluation_metrics import run_evaluation
import os
from tqdm import tqdm

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

    for image_index in tqdm(range(num_images), leave=True):
        if image_index > 10:
            break
        image = image_array[image_index]
        # cluster the image using Spectral Clustering
        clustered_image = run_spectral_clustering(image, args)
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
    if args.run_evaluation_metric:
        run_evaluation(mode)

