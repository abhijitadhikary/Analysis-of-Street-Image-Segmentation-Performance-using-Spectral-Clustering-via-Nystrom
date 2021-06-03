from utils import get_args, imshow, get_segmented_image, get_IOU
from spectral_clustering import run_spectral_clustering
import numpy as np
from dataloader import load_data

if __name__ == '__main__':
    args = get_args()
    # load data
    image_array_train, label_array_train, label_array_inst_train = load_data(variant='train')
    image_array_val, label_array_val, label_array_inst_val = load_data(variant='val')
    image_array_test = load_data(variant='test')

    image_index = 100
    image = image_array_val[image_index]
    labels_image_gt = label_array_val[image_index]

    # cluster image using spectral clustering
    clustered_image = run_spectral_clustering(image, args)
    # segment the image according to cluster mean/medians color values
    segmented_image = get_segmented_image(image, clustered_image, clustered_image.reshape(-1), args)
    imshow(segmented_image, args, 'Segmented Image (Pred)')

    unique, counts = np.unique(segmented_image[:, :, 0], return_counts=True)
    print(f'After Segmentation\n{dict(zip(unique, counts))}')

    # iid GT labels
    segmented_image_gt = get_segmented_image(image, labels_image_gt, labels_image_gt.reshape(-1), args)
    imshow(segmented_image_gt, args, 'Segmented Image (GT)')



    # #iid GT Images
    # image = cv2.imread('../data/idd20k_lite/leftImg8bit/train/0/024541_image.jpg')
    # # labels_image_gt = cv2.imread('../data/idd20k_lite/gtFine/train/0/024541_label.png')
    # labels_image_gt = cv2.imread('../data/idd20k_lite/gtFine/train/0/024541_inst_label.png', cv2.IMREAD_UNCHANGED)
    # labels_image_gt = convert(labels_image_gt, 0, 255)
    #
    #
    # # vegetable
    # # image = cv2.imread('../notebooks/vegetables.png')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # # image = get_dummy_image()
    #
    # # imshow(image, args, 'Input Image')
    #
    # # cluster image using spectral clustering
    # clustered_image = run_spectral_clustering(image, args)
    # # segment the image according to cluster mean/medians color values
    # segmented_image = get_segmented_image(image, clustered_image, clustered_image.reshape(-1), args)
    # imshow(segmented_image, args, 'Segmented Image (Pred)')
    #
    # unique, counts = np.unique(segmented_image[:, :, 0], return_counts=True)
    # print(f'After Segmentation\n{dict(zip(unique, counts))}')
    #
    #
    # # iid GT labels
    # segmented_image_gt = get_segmented_image(image, labels_image_gt, labels_image_gt.reshape(-1), args)
    # imshow(segmented_image_gt, args, 'Segmented Image (GT)')
    #
    # iou = get_IOU(segmented_image_gt, segmented_image)
    # print(f'IOU: {iou:.4f}')

