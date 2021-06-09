import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import cv2
import zipfile

def get_args():
    '''
    Returns all the hyper-parameters
    :return args: the hyper-parameters
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0, help='sets the seed for numpy')
    parser.add_argument('--num_clusters', type=int, default=8, help='desired number of image clusters')
    parser.add_argument('--dim_low', type=int, default=8, help='number of randomly chosen rows to run Nystrom')
    parser.add_argument('--segmentation_mode', type=int, default=0, help='0: Spectral, 1: K-Means++, 2: GMM (Very Slow)')
    parser.add_argument('--centroid_type', type=int, default=1, help='0: mean, 1: median')
    parser.add_argument('--color_weight_mode', type=int, default=0, help='0: RGB Intensity, 1: constant(1), 2: HSV, 1: DOOG (Not implemented)')
    parser.add_argument('--sigma_color', type=float, default=0.8, help='BEST: 0.8 (RGB), 0.8 (HSV)') # increasing creates superpixels, decreasing increases detail
    parser.add_argument('--sigma_distance', type=float, default=17, help='BEST (RGB): 17, 18 (HSV)') # decreasing causes segments to be highly localized
    parser.add_argument('--height', type=int, default=100, help='change height of the image, initialized to 100, updated later according to the actual image size')
    parser.add_argument('--width', type=int, default=100, help='change width of the image, initialized to 100, updated later according to the actual image size')
    parser.add_argument('--save_path_stacked', type=str, default=os.path.join('..', 'results', 'stacked'), help='path to save stacked images')
    parser.add_argument('--num_channels', type=int, default=0, help='number of channels of the image, initialized to 0, updated according to the image')
    parser.add_argument('--num_dimensions', type=int, default=0, help='number of dimensions of the image, initialized to 0, updated according to the image')
    parser.add_argument('--num_elements_flat', type=int, default=0, help='shape of flattened image, initialized to 0, updated according to the image')
    parser.add_argument('--train_condition', type=bool, default=True, help='whether to train')
    parser.add_argument('--val_condition', type=bool, default=True, help='whether to validate')
    parser.add_argument('--test_condition', type=bool, default=True, help='whether to test')
    parser.add_argument('--save_stacked_title', type=bool, default=False, help='whether to save title in the saved image')
    parser.add_argument('--print_cluster_memberships', type=bool, default=False, help='whether to print how many members belong to each classs')
    parser.add_argument('--run_evaluation_metric', type=bool, default=False, help='whether to run evaluation after train/val/test')
    parser.add_argument('--print_duration_information', type=bool, default=True, help='whether to print time duration information')
    args = parser.parse_args()
    return args

def setup_model_parameters():
    '''
    Sets up the model parameters, parses the namespace and returns all variables in args
    :return:
    '''
    # load hyperparameters
    args = get_args()
    # apply seed to numpy
    np.random.seed(args.seed)
    # create the required directories if they don't exist
    create_dirs()
    # unzip the iid dataset if it hasn't already been unzipped
    unzip_dataset()
    print('Model parameters successfully set up.')
    print_model_parameters(args)
    return args

def create_dirs():
    '''
    Creates the required directories and sub-directories required for the project
    :return:
    '''
    dir_list = [
        ['..', 'data'],
        ['..', 'documents'],
        ['..', 'documents', 'docs'],
        ['..', 'documents', 'references'],
        ['..', 'notebooks'],
        ['..', 'output'],
        ['..', 'output', 'stacked'],
        ['..', 'output', 'stacked', 'train'],
        ['..', 'output', 'stacked', 'val'],
        ['..', 'output', 'stacked', 'test']
    ]
    for current_dir in dir_list:
        current_path = current_dir[0]
        if len(current_dir) > 1:
            for sub_dir_index in range(1, len(current_dir)):
                current_path = os.path.join(current_path, current_dir[sub_dir_index])
        if not os.path.exists(current_path):
            try:
                os.makedirs(current_path)
            except:
                # implement to handle exception
                pass

def unzip_dataset():
    '''
    Unzips the idd20k_lite dataset (need to put idd20k_lite.zip) in the data folder
    :return:
    '''
    data_path = os.path.join('..', 'data')
    if not os.path.exists(os.path.join(data_path, 'idd20k_lite')):
        zip_file_path = os.path.join(data_path, 'idd20k_lite.zip')
        try:
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                print('Unzipping dataset.......')
                zip_ref.extractall(data_path)
                print('Dataset successfully unzipped.')
        except FileNotFoundError as error:
            print(f'Exiting program: File does not exist: {zip_file_path}.\nPlease put the idd20k_lite.zip file in {data_path} folder and try again.')
            quit(1)

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

def imshow(image, title='', save_path_full=None):
    '''
    Displays an image using matplotlib, additionally saves the image if save_path_full is provided
    :param image: The image to show ( of shape [height, width, 3]-> colour or [height, width]-> grayscale)
    :param title: A string. Title to be put as the heading
    :return:
    '''
    # convert image datatype to uint8
    image = image.astype(np.uint8)
    height, width = image.shape[0], image.shape[1]

    # deal with horizontally stacked images
    plt_height = 3
    plt_width = plt_height if width < 500 else plt_height*3

    plt.figure(figsize=(plt_width, plt_height))
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)

    if save_path_full is not None:
        try:
            plt.savefig(save_path_full)
        except:
            # raise an error if file can not be saved
            pass
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
    image_flat = image.reshape(args.height*args.width, -1)
    image_array = np.zeros((args.num_elements_flat, (args.num_channels+2)))
    image_array[:, 0:3] = image_flat

    # coordinate information
    image_array[:, 3] = np.repeat(np.arange(args.height), args.width)
    image_array[:, 4] = np.tile(np.arange(args.width), args.height)

    return image_array

def get_segmented_image(image, clustered_labels, args):
    '''
    Returns a segmented image based on the supplied labels, either using median
    or mean each cluster
    :param image: The input image of shape [height, width, num_channels]
    :param clustered_labels: The cluster label of each pixel flattened, shape = [height*width,]
    :param args: The arguments with all the hyper-parameters
    :return: The segmented image of shape [height, width, num_channels]
    '''
    image = convert(image, 0, 1)
    image_flat = image.reshape(-1, args.num_channels)

    label_list = np.unique(clustered_labels)
    for current_label in label_list:
        selected_indices = np.where(clustered_labels == current_label)

        if args.centroid_type == 0:
            # mean value
            centroid = np.mean(image_flat[selected_indices], axis=0)
        elif args.centroid_type == 1:
            # median value
            centroid = np.mean(image_flat[selected_indices], axis=0)
        else:
            raise NotImplementedError('Invalid centroid type selected, choose between 0 and 1')
        image_flat[selected_indices] = centroid
    image_flat = image_flat.reshape(image.shape)
    image_flat = convert(image_flat, 0, 255)
    return image_flat

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

    # 4 circles
    image = circle1 + circle2 + circle3 + circle4
    image = image.astype(float)
    image += 1 + 0.2 * np.random.randn(*image.shape)
    return image

def print_cluster_memberships(segmented_image_pred, state):
    '''
    Prints how many points belong to which cluster
    :param segmented_image_pred:
    :param state:
    :return:
    '''
    unique, counts = np.unique(segmented_image_pred[:, :, 0], return_counts=True)
    print(f'{state}\n{dict(zip(unique, counts))}')

def get_stacked_image_horizontal(*image_list):
    '''
    Stacks three images side by side
    :param image_list:
    :return:
    '''
    stacked_image = np.copy(image_list[0])
    for image in image_list[1:]:
        stacked_image = np.hstack((stacked_image, image))
    return stacked_image

def save_image(image, image_path_full, title=None):
    '''
    Saves image, if a title is provided, it is augmented on top of the images
    :param image:
    :param image_path_full:
    :param title:
    :return:
    '''

    if title is not None:
        imshow(image, title, image_path_full)
    else:
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path_full, image)

def print_duration_information(duration_list, mode):
    num_images = len(duration_list)
    duration_list = np.array(duration_list)
    duration_mean = np.mean(duration_list)
    duration_sum = np.sum(duration_list)
    print(f'Mode: {mode.upper()}\t\tTotal samples: {num_images}\t\tTotal Duration: {duration_sum:.4f} s\t\tMean Duration: {duration_mean:.4f} s')

def print_model_parameters(args):
    print(f'{"*"*57}')
    print(f'{"*"*20} Model Parameters {"*"*20}')
    print(f'{"*"*57}')
    method_list = ['Spectral', 'K-Means++', 'GMM']
    print(f'Segmentation Method:\t\t\t{method_list[args.segmentation_mode]}')
    print(f'Number of clusters:\t\t\t\t{args.num_clusters}')
    if args.segmentation_mode == 0:
        color_mode_list = ['RGB', 'Constant (1)', 'DOOG', 'HSV']
        print(f'Color mode:\t\t\t\t\t\t{color_mode_list[args.color_weight_mode]}')
        print(f'Sigma color:\t\t\t\t\t{args.sigma_color}')
        print(f'Sigma distance:\t\t\t\t\t{args.sigma_distance}')
        print(f'Nystrom (n):\t\t\t\t\t{args.sigma_distance}')

    centroid_list = ['Mean', 'Median']
    print(f'Centroid type:\t\t\t\t\t{centroid_list[args.centroid_type]}')

    print(f'{"*"*57}')


