import os
import cv2
import numpy as np
from utils import convert

def get_image_full_path(dir_path, mode):
    '''
    Returns the full path of images/labels of the supplied path
    :param dir_path:
    :param mode:
    :return:
    '''
    image_full_path_list = []
    if mode == 'label':
        image_full_path_list_inst = []
    sub_dir_name_list = os.listdir(dir_path)
    for sub_dir_name in sub_dir_name_list:
        # print(sub_dir_name)
        current_dir_path = os.path.join(dir_path, sub_dir_name)
        if not os.path.isdir(current_dir_path):
            continue
        file_name_list = os.listdir(current_dir_path)
        for file_name in file_name_list:
            image_full_path = os.path.join(current_dir_path, file_name)
            # print(image_full_path)
            # if inst label
            if mode == 'label' and 'inst' in file_name:
                image_full_path_list_inst.append(image_full_path)
            else:
                image_full_path_list.append(image_full_path)
    if mode == 'label':
        return image_full_path_list, image_full_path_list_inst
    else:
        return image_full_path_list

def read_image_as_array(image_path_list_full, is_label):
    '''
    Takes a list of path of images as input, returns the images as a numpy array (RGB or grayscale)
    :param image_path_list_full:
    :param is_label:
    :return:
    '''
    image_list = []
    for image_path_full in image_path_list_full:
        try:
            image = cv2.imread(image_path_full, cv2.IMREAD_UNCHANGED)

            if is_label:
                # convert labels between [0, 255]
                image = convert(image, 0, 255)
            else:
                # convert images from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_list.append(image)
        except cv2.error as e:
            raise FileNotFoundError(f'{e}\tFile not found: {image_path_full}')
    image_array = np.array(image_list)
    return image_array

def load_data(variant):
    '''
    Returns images and labels (if applicable) of train, test and val data
    :param variant:
    :return:
    '''
    dir_path_image = os.path.join('..', 'data', 'idd20k_lite', 'leftImg8bit', variant)
    dir_path_label = os.path.join('..', 'data', 'idd20k_lite', 'gtFine', variant)

    # extract full paths
    image_path_list_full_image = get_image_full_path(dir_path_image, mode='image')
    image_array = read_image_as_array(image_path_list_full_image, is_label=False)
    if variant in ['train', 'val']:
        image_path_list_full_label, image_path_list_full_train_label_inst = get_image_full_path(dir_path_label, mode='label')
        label_array = read_image_as_array(image_path_list_full_label, is_label=True)
        label_array_inst = read_image_as_array(image_path_list_full_train_label_inst, is_label=True)
        return image_array, label_array, label_array_inst
    else:
        return image_array