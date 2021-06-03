from utils import get_args, imshow, process_image_attributes
import cv2
from spectral_clustering import run_spectral_segmentation

if __name__=='__main__':
    args = get_args()
    image = cv2.imread('../notebooks/vegetables.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = get_dummy_image()

    # imshow(image, args, 'Input Image')

    scaled_image, args = process_image_attributes(image, args)
    # cluster image using spectral clustering
    segmented_image = run_spectral_segmentation(scaled_image, image, args)
    imshow(segmented_image, args, 'Segmented Image')